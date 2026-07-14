from __future__ import annotations

import json
import os
import socket
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rs.core.contracts.execution import ActualPhaseContext, ExecutionContext
from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.execution import PayloadInvocation, PhaseSyncExecutor, P2PReleaseExecutor, RuntimeExecutionPipeline
from tests.contract.megatron_ep.helpers import make_phase_context_generic


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _matrix_for_group(group_ranks: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if len(group_ranks) == 4:
        return ((0, 2, 0, 0), (0, 0, 3, 0), (0, 0, 0, 1), (4, 0, 0, 0))
    if len(group_ranks) == 2:
        return ((0, 3), (2, 0))
    raise ValueError(f"unsupported group size {group_ranks!r}")


def _window_plan(group_ranks: tuple[int, ...], phase: str) -> WindowPlan:
    matrix = _matrix_for_group(group_ranks)
    flow_phase = {"P0": "p0_dispatch", "P1": "p1_return"}[phase]
    flows = []
    for src_index, row in enumerate(matrix):
        for dst_index, rows in enumerate(row):
            if src_index == dst_index or int(rows) <= 0:
                continue
            flows.append(
                PlannedFlow(
                    flow_id=f"{flow_phase}_{group_ranks[src_index]}_{group_ranks[dst_index]}",
                    phase=flow_phase,
                    src_rank=int(group_ranks[src_index]),
                    dst_rank=int(group_ranks[dst_index]),
                    row_count=int(rows),
                    release_state="ready",
                    executable=True,
                )
            )
    return WindowPlan(
        planner_id="m2-formal",
        planner_family="joint",
        request_digest=f"{group_ranks[0]}->{group_ranks[-1]}:{phase}",
        waves=(PlanWave(wave_id=0, flows=tuple(flows), estimated_duration=float(sum(sum(row) for row in matrix))),),
        metadata={"source_layer_id": "0", "target_layer_id": "1"},
    )


def _context_for_rank(rank: int, *, group_ranks: tuple[int, ...], phase: str):
    matrix = _matrix_for_group(group_ranks)
    group_index = group_ranks.index(int(rank))
    row = tuple(int(value) for value in matrix[group_index])
    col = tuple(int(matrix[src][group_index]) for src in range(len(matrix)))
    if phase == "P0":
        input_splits = row
        output_splits = col
    else:
        input_splits = col
        output_splits = row
    return make_phase_context_generic(
        rank=int(rank),
        phase=str(phase),
        input_splits=input_splits,
        output_splits=output_splits,
        ep_group_ranks=group_ranks,
        p2_hint_mode="none",
    )


def _actual_phase_context(rank: int, *, phase_context) -> ActualPhaseContext:
    return ActualPhaseContext(
        layer_id=str(phase_context.layer_id),
        phase=str(phase_context.phase),
        world_size=len(phase_context.ep_group_ranks),
        rank_space="global",
        layout_digest=str(phase_context.canonical_receive_layout_id),
        metadata={"phase_ready_context": phase_context.to_dict()},
    )


def _input_tensor(spec) -> torch.Tensor:
    rows = int(spec.row_count)
    hidden = int(spec.shape_suffix[0]) if spec.shape_suffix else 1
    return torch.arange(max(rows, 1) * max(hidden, 1), dtype=torch.float16).reshape(max(rows, 1), max(hidden, 1))[:rows]


def _serializable_outcome(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    tensor = result.get("output_payload")
    if isinstance(tensor, torch.Tensor):
        result["output_payload"] = {
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": str(tensor.dtype),
            "sum": float(tensor.float().sum().item()),
        }
    return result


def _execute_for_rank(rank: int, *, group_ranks: tuple[int, ...], phase: str):
    phase_context = _context_for_rank(rank, group_ranks=group_ranks, phase=phase)
    published = CanonicalPlanPublisher(
        rank_map=RankMap(group_ranks=group_ranks, root_rank=group_ranks[0])
    ).build(
        publication_slot={
            "run_id": "m2-gloo",
            "forward_generation": 0,
            "microbatch_id": "mb",
            "source_layer_id": "0",
            "target_layer_id": "1",
            "planning_slot": f"{group_ranks[0]}->{group_ranks[-1]}",
        },
        window_plan=_window_plan(group_ranks, phase),
    )
    pipeline = RuntimeExecutionPipeline()
    prepared = pipeline.prepare(published, _actual_phase_context(rank, phase_context=phase_context))
    results = {}
    for spec in prepared.materialized_plan.payload_specs:
        tensor = _input_tensor(spec)
        sync_outcome = PhaseSyncExecutor().execute(
            plan=prepared.materialized_plan,
            invocation=PayloadInvocation(
                run_id="m2-gloo",
                forward_generation=0,
                layer_id=str(phase_context.layer_id),
                phase=str(phase_context.phase),
                payload_role=str(spec.payload_role),
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=str(spec.dtype),
                layout_digest=str(prepared.materialized_plan.layout_digest),
                invocation_id=f"sync:{rank}:{spec.payload_role}",
                input_tensor=tensor,
            ),
            context=ExecutionContext(
                run_id="m2-gloo",
                forward_generation=0,
                layer_id=str(phase_context.layer_id),
                phase=str(phase_context.phase),
                rank_space="global",
            ),
        )
        p2p_outcome = P2PReleaseExecutor().execute(
            plan=prepared.materialized_plan,
            invocation=PayloadInvocation(
                run_id="m2-gloo",
                forward_generation=0,
                layer_id=str(phase_context.layer_id),
                phase=str(phase_context.phase),
                payload_role=str(spec.payload_role),
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=str(spec.dtype),
                layout_digest=str(prepared.materialized_plan.layout_digest),
                invocation_id=f"p2p:{rank}:{spec.payload_role}",
                input_tensor=tensor,
            ),
            context=ExecutionContext(
                run_id="m2-gloo",
                forward_generation=0,
                layer_id=str(phase_context.layer_id),
                phase=str(phase_context.phase),
                rank_space="global",
                metadata={"max_inflight_batches": 2},
            ),
        )
        results[str(spec.payload_role)] = {
            "sync": _serializable_outcome(sync_outcome.to_dict()),
            "p2p": _serializable_outcome(p2p_outcome.to_dict()),
        }
    return {
        "rank": int(rank),
        "group_ranks": list(group_ranks),
        "phase": str(phase),
        "published_plan_digest": str(published.published_plan_digest),
        "logical_plan_digest": str(published.logical_plan_digest),
        "materialized_plan_digest": str(prepared.materialized_plan.materialized_plan_digest),
        "validation": prepared.validation.to_dict(),
        "results": results,
    }


def _worker(rank: int, world_size: int, master_port: int, out_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{int(master_port)}",
        rank=rank,
        world_size=world_size,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    subgroup = dist.new_group(ranks=[2, 3], backend="gloo")
    try:
        local = {
            "rank": int(rank),
            "full_group": _execute_for_rank(rank, group_ranks=(0, 1, 2, 3), phase="P0"),
        }
        dist.barrier()
        if rank in {2, 3}:
            local["subgroup"] = _execute_for_rank(rank, group_ranks=(2, 3), phase="P0")
        else:
            local["subgroup"] = {"rank": int(rank), "skipped": True}
        dist.barrier()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
        (out_path / f"rank{rank}_m2_formal_gloo.json").write_text(json.dumps(local, indent=2), encoding="utf-8")
        if rank == 0:
            summary = {
                "status": "completed",
                "world_size": int(world_size),
                "all_ranks": gathered,
            }
            (out_path / "m2_formal_execution_gloo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
    except Exception as exc:
        failure = {
            "rank": int(rank),
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (out_path / f"rank{rank}_m2_formal_gloo_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    finally:
        dist.destroy_process_group(subgroup)
        dist.destroy_process_group()


def main() -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    port = _free_port()
    out_dir = "outputs/closure/m2_formal_execution_gloo"
    mp.spawn(_worker, args=(4, port, out_dir), nprocs=4, join=True)


if __name__ == "__main__":
    main()
