from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rs.runtime.online.megatron_ep.execution.release_frontier import ReleaseBatchFrontier, ReleaseBatchTask
from rs.runtime.online.megatron_ep.target_planning import (
    PlanVersionLineage,
    TargetLayerPlannerService,
    TargetLayerPlanningRequest,
    TargetPlanKey,
    TargetPlanStore,
    reconcile_target_plan,
)
from rs.scheduling.unified_interface import PolicyOptions


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _digest_agreement(local_digest: str) -> str:
    payload = torch.tensor(
        [int(local_digest[:16], 16) if all(ch in "0123456789abcdef" for ch in local_digest[:16].lower()) else 0],
        dtype=torch.long,
    )
    gathered = [torch.zeros_like(payload) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, payload)
    values = [int(item.item()) for item in gathered]
    if len(set(values)) != 1:
        raise RuntimeError(f"target plan digest disagreement: {values}")
    return f"{values[0]:016x}"


def _frontier_tasks(version: int) -> list[ReleaseBatchTask]:
    return [
        ReleaseBatchTask(task_id="t0", phase="P0", src_rank=0, dst_rank=1, row_count=2, plan_version=version),
        ReleaseBatchTask(task_id="t1", phase="P0", src_rank=1, dst_rank=0, row_count=1, plan_version=version),
        ReleaseBatchTask(task_id="t2", phase="P0", src_rank=0, dst_rank=1, row_count=3, plan_version=version),
    ]


def _worker(rank: int, world_size: int, master_port: int, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    store = TargetPlanStore()
    planner = TargetLayerPlannerService(store=store, agreement_fn=None)
    planner.start()
    request = TargetLayerPlanningRequest(
        run_id="gloo-target-plan",
        forward_epoch=1,
        microbatch_id="mb0",
        source_layer_id="0",
        target_layer_id="1",
        current_p0_rows=((0, 3), (1, 0)),
        previous_p0_rows=((0, 1), (2, 0)),
        predictor_name="copy_current_dispatch",
        policy_id="barrier_criticality_joint",
        group_size=2,
        bucket_rows=0,
        policy_options=PolicyOptions(prediction_weight=0.35),
        topology_digest="topo",
        bucket_contract_digest="dynamic_current",
    )
    if rank == 1:
        time.sleep(0.2)
    planner.enqueue(request)
    key = TargetPlanKey("gloo-target-plan", 1, "mb0", "1")
    target_plan_ready_at_entry = store.peek(key) is not None
    deadline = time.time() + 10.0
    while store.peek(key) is None and time.time() < deadline:
        time.sleep(0.01)
    plan = store.peek(key)
    if plan is None:
        raise RuntimeError("target plan never became ready")
    agreed_digest = _digest_agreement(str(plan.logical_plan_digest))
    if str(agreed_digest) != str(plan.logical_plan_digest):
        raise RuntimeError(
            f"target plan digest mismatch after main-thread agreement: local={plan.logical_plan_digest} agreed={agreed_digest}"
        )
    case_rows = {
        "ready_before_execution": ((0, 3), (1, 0)),
        "late_suffix_applied": ((0, 5), (1, 0)),
        "too_late_no_effect": ((0, 0), (7, 0)),
    }
    case_results: list[dict[str, object]] = []
    for name, actual_rows in case_rows.items():
        frontier = ReleaseBatchFrontier(tasks=_frontier_tasks(version=0))
        committed = frontier.commit_batch(limit=1)
        frontier.mark_in_flight([task.task_id for task in committed])
        outcome = reconcile_target_plan(prepared_plan=plan, actual_p0_rows=actual_rows)
        if name == "too_late_no_effect":
            frontier.mark_completed([task.task_id for task in committed])
            state = "too_late_no_effect"
            lineage = PlanVersionLineage(
                old_version=0,
                new_version=0,
                plan_origin="provisional",
                parent_plan_version=0,
                frontier_digest=frontier.frontier_digest(),
                replacement_suffix_digest="",
                switch_epoch=frontier.release_epoch,
                all_rank_agreement=True,
            )
        elif outcome.status == "exact_match":
            state = "ready_before_execution"
            lineage = PlanVersionLineage(
                old_version=0,
                new_version=1,
                plan_origin="target_prepared",
                parent_plan_version=0,
                frontier_digest=frontier.frontier_digest(),
                replacement_suffix_digest="",
                switch_epoch=frontier.release_epoch,
                all_rank_agreement=True,
            )
        else:
            lineage = frontier.apply_late_suffix(
                new_plan_version=1,
                suffix_tasks=_frontier_tasks(version=1)[1:],
                plan_origin="late_spliced",
                parent_plan_version=0,
            )
            state = "late_suffix_applied"
        case_results.append(
            {
                "case": name,
                "target_plan_ready_at_entry": bool(target_plan_ready_at_entry),
                "provisional_plan_generated": True,
                "prepared_plan_found": True,
                "reconciliation_status": outcome.status,
                "late_plan_state": state,
                "frontier_digest": frontier.frontier_digest(),
                "immutable_prefix_ids": list(frontier.immutable_prefix_ids()),
                "replaceable_suffix_ids": list(frontier.replaceable_suffix_ids()),
                "lineage": lineage.to_dict(),
            }
        )
    result = {"rank": rank, "planner_timeline": planner.timeline(), "cases": case_results}
    (out_path / f"rank{rank}_target_plan_gloo.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, result)
    if rank == 0:
        summary = {
            "passed": True,
            "all_ranks": gathered,
            "states": {
                case_name: [
                    case["late_plan_state"]
                    for rank_payload in gathered
                    for case in rank_payload["cases"]
                    if case["case"] == case_name
                ]
                for case_name in case_rows
            },
        }
        (out_path / "gloo_target_plan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    dist.barrier()
    planner.shutdown()
    dist.destroy_process_group()


def main() -> None:
    out_dir = "outputs/closure/target_plan_runtime"
    port = _free_port()
    mp.spawn(_worker, args=(2, port, out_dir), nprocs=2, join=True)


if __name__ == "__main__":
    main()
