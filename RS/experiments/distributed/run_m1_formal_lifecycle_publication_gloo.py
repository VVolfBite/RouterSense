from __future__ import annotations

import json
import os
import socket
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from rs.planning import PlannerPolicyConfig
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.control.communication_lane import GlooControlCommunicationLane, slot_from_request
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.public_types import (
    LocalPreparationToken,
    LocalPublicationCandidate,
    PublicationPollResult,
    PublicationPollStatus,
    PublicationSlot,
)
from rs.runtime.online.megatron_ep.target_planning.contracts import TwoHorizonPrediction
from rs.runtime.online.megatron_ep.target_planning.predictor import TwoHorizonPredictionBundle
from rs.runtime.online.megatron_ep.target_planning.planner_service import TargetLayerPlannerService, TargetLayerPlanningRequest
from rs.runtime.online.megatron_ep.target_planning.store import TargetPlanStore


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _TracingLane:
    def __init__(self, lane: GlooControlCommunicationLane, trace: list[dict[str, object]], rank: int) -> None:
        self._lane = lane
        self._trace = trace
        self._rank = int(rank)

    def poll(self, slot: PublicationSlot, local_candidate: LocalPublicationCandidate | None) -> PublicationPollResult:
        result = self._lane.poll(slot, local_candidate)
        self._trace.append(
            {
                "rank": int(self._rank),
                "slot_digest": str(slot.semantic_digest()),
                "safe_status": str(result.status.value),
                "published_plan_digest": str(result.published_plan_digest or ""),
            }
        )
        return result

    def cancel_before_generation(self, *, run_id: str, microbatch_id: str, current_generation: int) -> None:
        self._lane.cancel_before_generation(
            run_id=str(run_id),
            microbatch_id=str(microbatch_id),
            current_generation=int(current_generation),
        )


class _DelayedPredictor:
    def __init__(self, *, rank: int, delay_seconds: float = 0.0) -> None:
        self.rank = int(rank)
        self.delay_seconds = float(delay_seconds)

    def predict_two_horizon(
        self,
        *,
        source_layer_id: str,
        current_dispatch_matrix,
        previous_dispatch_matrix=None,
        history_matrices=(),
    ) -> TwoHorizonPredictionBundle:
        if self.delay_seconds > 0.0:
            time.sleep(self.delay_seconds)
        current = tuple(tuple(int(value) for value in row) for row in current_dispatch_matrix)
        next_layer_id = str(int(source_layer_id) + 1)
        return TwoHorizonPredictionBundle(
            h1=TwoHorizonPrediction(
                forecast_horizon=1,
                source_layer_id=str(source_layer_id),
                target_layer_id=str(next_layer_id),
                matrix_unit="rows",
                matrix_rows=current,
                matrix_digest=f"h1:{self.rank}:{source_layer_id}",
                predictor="copy_current",
                confidence=1.0,
                created_at_ns=1,
                prediction_us=10.0,
            ),
            h2=TwoHorizonPrediction(
                forecast_horizon=2,
                source_layer_id=str(next_layer_id),
                target_layer_id=str(int(next_layer_id) + 1),
                matrix_unit="rows",
                matrix_rows=current,
                matrix_digest=f"h2:{self.rank}:{source_layer_id}",
                predictor="copy_current",
                confidence=1.0,
                created_at_ns=2,
                prediction_us=10.0,
            ),
        )


class _FailingPlanner:
    planner_id = "U_barrier_criticality_global_matching"
    planner_family = "joint"

    def plan(self, _request):
        raise RuntimeError("intentional_planner_failure")


def _matrix_for_world_size(world_size: int) -> tuple[tuple[int, ...], ...]:
    if int(world_size) == 2:
        return ((0, 4), (3, 0))
    if int(world_size) == 4:
        return ((0, 4, 0, 2), (1, 0, 3, 0), (0, 2, 0, 5), (4, 0, 1, 0))
    raise ValueError(f"unsupported world_size {world_size!r}")


def _runtime(
    *,
    rank: int,
    world_size: int,
    group_ranks: tuple[int, ...],
    root_rank: int,
    process_group,
    planner_factory=None,
    predictor_delay_seconds: float = 0.0,
) -> tuple[RouterSenseInjectionRuntime, list[dict[str, object]]]:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="U_barrier_criticality_global_matching",
            scheduler_mode="disabled",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            online_p2_predictor="copy_current_dispatch",
            p2_hint_mode="none",
            observation_profile="execution",
            safe_projection_mode="disabled",
        ),
        rank=int(rank),
        local_rank=int(rank),
        run_id="m1-formal-gloo",
        step_id="step",
        microbatch_id="mb0",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="localhost",
        ep_group_ranks=tuple(int(item) for item in group_ranks),
        ep_group_root_global_rank=int(root_rank),
        ep_process_group=process_group,
    )
    runtime.target_plan_store = TargetPlanStore()
    runtime.target_plan_control_group = process_group
    trace: list[dict[str, object]] = []
    lane = GlooControlCommunicationLane(
        rank=int(rank),
        world_size=int(len(group_ranks)),
        root_rank=int(root_rank),
        process_group=process_group,
        group_ranks=tuple(int(item) for item in group_ranks),
    )
    runtime.control_communication_lane = _TracingLane(lane, trace, rank)
    service_kwargs = {
        "store": runtime.target_plan_store,
        "two_horizon_predictor_factory": lambda _name: _DelayedPredictor(rank=rank, delay_seconds=predictor_delay_seconds),
    }
    if planner_factory is not None:
        service_kwargs["planner_factory"] = planner_factory
    runtime.target_planner_service = TargetLayerPlannerService(**service_kwargs)
    runtime.target_planner_service.start()
    runtime._forward_epoch = 1  # noqa: SLF001
    return runtime, trace


def _submit_request(runtime: RouterSenseInjectionRuntime, *, source_layer_id: str = "0", target_layer_id: str = "1") -> PublicationSlot:
    matrix = _matrix_for_world_size(len(runtime.ep_group_ranks))
    result = runtime.target_planner_service.submit(  # type: ignore[union-attr]
        TargetLayerPlanningRequest(
            run_id=str(runtime.run_id),
            forward_epoch=int(runtime._forward_epoch),  # noqa: SLF001
            microbatch_id=str(runtime.microbatch_id),
            source_layer_id=str(source_layer_id),
            target_layer_id=str(target_layer_id),
            current_p0_rows=matrix,
            previous_p0_rows=matrix,
            predictor_name="copy_current_dispatch",
            policy_id="U_barrier_criticality_global_matching",
            raw_u_policy_id="U_barrier_criticality_global_matching",
            paired_b_policy_id="",
            safe_projection_mode="disabled",
            group_size=len(runtime.ep_group_ranks),
            bucket_rows=0,
            policy_options=PlannerPolicyConfig(),
            topology_digest="topo",
            bucket_contract_digest="dynamic_current",
        )
    )
    runtime._runtime_state.write("latest_target_plan_submit_status", str(result.status.value))  # noqa: SLF001
    slot = slot_from_request(
        run_id=str(runtime.run_id),
        forward_generation=int(runtime._forward_epoch),  # noqa: SLF001
        microbatch_id=str(runtime.microbatch_id),
        source_layer_id=str(source_layer_id),
        target_layer_id=str(target_layer_id),
    )
    runtime._expected_publication_slots[(str(runtime.run_id), int(runtime._forward_epoch), str(runtime.microbatch_id), str(target_layer_id))] = slot  # noqa: SLF001
    return slot


def _wait_until(local_predicate, *, timeout_seconds: float = 5.0) -> bool:
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        if bool(local_predicate()):
            return True
        time.sleep(0.01)
    return bool(local_predicate())


def _wait_for_slot_status(
    runtime: RouterSenseInjectionRuntime,
    slot: PublicationSlot,
    expected_status: str,
    *,
    timeout_seconds: float = 10.0,
) -> LocalPublicationCandidate:
    candidate: LocalPublicationCandidate | None = None
    if not _wait_until(
        lambda: (
            (candidate := runtime.target_planner_service.publication_state_for_slot(slot)) is not None  # type: ignore[union-attr]
            and str(candidate.status) == str(expected_status)
        ),
        timeout_seconds=timeout_seconds,
    ):
        candidate = runtime.target_planner_service.publication_state_for_slot(slot)  # type: ignore[union-attr]
        raise AssertionError(
            f"slot {slot.semantic_digest()} did not reach {expected_status}; current={None if candidate is None else candidate.status}"
        )
    return runtime.target_planner_service.publication_state_for_slot(slot)  # type: ignore[union-attr]


def _scenario_all_ready(rank: int, process_group) -> dict[str, object]:
    runtime, trace = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group)
    try:
        slot = _submit_request(runtime)
        _wait_for_slot_status(runtime, slot, "READY")
        dist.barrier(group=process_group)
        runtime._pump_target_planner_publications()  # noqa: SLF001
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="target_dispatch_ready")  # noqa: SLF001
        state = runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        return {"scenario": "all_ready", "trace": trace, "state": None if state is None else state.to_dict()}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_not_ready_then_ready(rank: int, process_group) -> dict[str, object]:
    delay = 0.4 if rank == 3 else 0.0
    runtime, trace = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group, predictor_delay_seconds=delay)
    try:
        slot = _submit_request(runtime)
        dist.barrier(group=process_group)
        runtime._pump_target_planner_publications()  # noqa: SLF001
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="source_combine_complete")  # noqa: SLF001
        assert _wait_until(lambda: any(str(item.get("safe_status")) == "not_ready" for item in trace))
        _wait_for_slot_status(runtime, slot, "READY")
        runtime._pump_target_planner_publications()  # noqa: SLF001
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="target_dispatch_ready")  # noqa: SLF001
        state = runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        return {"scenario": "not_ready_then_ready", "trace": trace, "state": None if state is None else state.to_dict()}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_failed_rank(rank: int, process_group) -> dict[str, object]:
    planner_factory = (lambda _planner_id, _config: _FailingPlanner()) if rank == 2 else None
    runtime, trace = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group, planner_factory=planner_factory)
    try:
        slot = _submit_request(runtime)
        if rank == 2:
            _wait_for_slot_status(runtime, slot, "FAILED")
        else:
            assert _wait_until(lambda: runtime.target_planner_service.publication_state_for_slot(slot) is not None, timeout_seconds=10.0)  # type: ignore[union-attr]
        dist.barrier(group=process_group)
        runtime._pump_target_planner_publications()  # noqa: SLF001
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="target_dispatch_ready")  # noqa: SLF001
        terminal = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        return {"scenario": "failed_rank", "trace": trace, "terminal": None if terminal is None else terminal.to_dict()}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_cancelled(rank: int, process_group) -> dict[str, object]:
    runtime, trace = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group, predictor_delay_seconds=0.4)
    try:
        slot = _submit_request(runtime)
        runtime.target_planner_service.cancel_generation(run_id=str(runtime.run_id), forward_epoch=1, microbatch_id=str(runtime.microbatch_id))  # type: ignore[union-attr]
        _wait_for_slot_status(runtime, slot, "CANCELLED")
        dist.barrier(group=process_group)
        runtime._pump_target_planner_publications()  # noqa: SLF001
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="target_dispatch_ready")  # noqa: SLF001
        terminal = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        return {"scenario": "cancelled", "trace": trace, "terminal": None if terminal is None else terminal.to_dict()}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_slot_mismatch(rank: int, process_group) -> dict[str, object]:
    runtime, trace = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group)
    try:
        slot = _submit_request(runtime)
        candidate = _wait_for_slot_status(runtime, slot, "READY")
        dist.barrier(group=process_group)
        runtime._pump_target_planner_publications()  # noqa: SLF001
        if rank == 1:
            ready_pair = runtime._ready_target_plan_candidates[str(slot.semantic_digest())]  # type: ignore[attr-defined]  # noqa: SLF001
            tampered = replace(
                ready_pair[1],
                token=LocalPreparationToken(
                    service_session_id=int(candidate.token.service_session_id),
                    forward_generation=int(candidate.token.forward_generation),
                    target_layer_id=str(candidate.token.target_layer_id),
                    task_version=int(candidate.token.task_version),
                    publication_slot_digest="tampered-slot-digest",
                ),
            )
            runtime._ready_target_plan_candidates[str(slot.semantic_digest())] = (ready_pair[0], tampered)  # type: ignore[attr-defined]  # noqa: SLF001
        dist.barrier(group=process_group)
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="target_dispatch_ready")  # noqa: SLF001
        terminal = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        return {"scenario": "slot_mismatch", "trace": trace, "terminal": None if terminal is None else terminal.to_dict()}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_plan_digest_mismatch(rank: int, process_group) -> dict[str, object]:
    runtime, trace = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group)
    try:
        slot = _submit_request(runtime)
        _wait_for_slot_status(runtime, slot, "READY")
        dist.barrier(group=process_group)
        runtime._pump_target_planner_publications()  # noqa: SLF001
        if rank == 0:
            ready_pair = runtime._ready_target_plan_candidates[str(slot.semantic_digest())]  # type: ignore[attr-defined]  # noqa: SLF001
            tampered_metadata = dict(ready_pair[1].metadata)
            tampered_plan = dict(tampered_metadata["plan"])
            tampered_plan["logical_plan_digest"] = "tampered-plan-digest"
            tampered_metadata["plan"] = tampered_plan
            runtime._ready_target_plan_candidates[str(slot.semantic_digest())] = (  # type: ignore[attr-defined]  # noqa: SLF001
                ready_pair[0],
                replace(ready_pair[1], metadata=tampered_metadata),
            )
        dist.barrier(group=process_group)
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="target_dispatch_ready")  # noqa: SLF001
        terminal = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        return {"scenario": "plan_digest_mismatch", "trace": trace, "terminal": None if terminal is None else terminal.to_dict()}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_subgroup(rank: int, world_group, subgroup) -> dict[str, object]:
    if rank not in {2, 3}:
        dist.barrier(group=world_group)
        dist.barrier(group=world_group)
        return {"scenario": "subgroup", "rank": rank, "skipped": True}
    runtime, trace = _runtime(rank=rank, world_size=2, group_ranks=(2, 3), root_rank=2, process_group=subgroup)
    try:
        slot = _submit_request(runtime)
        _wait_for_slot_status(runtime, slot, "READY")
        dist.barrier(group=world_group)
        runtime._pump_target_planner_publications()  # noqa: SLF001
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="target_dispatch_ready")  # noqa: SLF001
        dist.barrier(group=world_group)
        state = runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        return {"scenario": "subgroup", "rank": rank, "trace": trace, "state": None if state is None else state.to_dict()}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_cleanup_before_generation(rank: int, process_group) -> dict[str, object]:
    runtime, _trace = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group)
    try:
        runtime.begin_forward(forward_epoch=1)
        _submit_request(runtime)
        runtime.begin_forward(forward_epoch=2)
        _submit_request(runtime)
        runtime.begin_forward(forward_epoch=3)
        state1 = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        submit1 = runtime.target_planner_service.submit(  # type: ignore[union-attr]
            TargetLayerPlanningRequest(
                run_id=str(runtime.run_id),
                forward_epoch=1,
                microbatch_id=str(runtime.microbatch_id),
                source_layer_id="0",
                target_layer_id="1",
                current_p0_rows=((0, 1), (1, 0)),
                previous_p0_rows=((0, 1), (1, 0)),
                predictor_name="copy_current_dispatch",
                policy_id="U_barrier_criticality_global_matching",
                raw_u_policy_id="U_barrier_criticality_global_matching",
                paired_b_policy_id="",
                safe_projection_mode="disabled",
                group_size=2,
                bucket_rows=0,
                policy_options=PlannerPolicyConfig(),
                topology_digest="topo",
                bucket_contract_digest="dynamic_current",
            )
        )
        return {
            "scenario": "cleanup_before_generation",
            "old_terminal_present": state1 is not None,
            "old_submit_status": str(submit1.status.value),
        }
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_stale_replacement(rank: int, process_group) -> dict[str, object]:
    runtime, trace = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group, predictor_delay_seconds=0.15)
    try:
        slot = _submit_request(runtime)
        replacement_request = TargetLayerPlanningRequest(
            run_id=str(runtime.run_id),
            forward_epoch=int(runtime._forward_epoch),  # noqa: SLF001
            microbatch_id=str(runtime.microbatch_id),
            source_layer_id="0",
            target_layer_id="1",
            current_p0_rows=_matrix_for_world_size(4),
            previous_p0_rows=_matrix_for_world_size(4),
            predictor_name="copy_current_dispatch",
            policy_id="U_barrier_criticality_global_matching",
            raw_u_policy_id="U_barrier_criticality_global_matching",
            paired_b_policy_id="",
            safe_projection_mode="disabled",
            group_size=4,
            bucket_rows=0,
            policy_options=PlannerPolicyConfig(),
            topology_digest="topo",
            bucket_contract_digest="dynamic_current",
        )
        submit_result = runtime.target_planner_service.submit(replacement_request)  # type: ignore[union-attr]
        assert str(submit_result.status.value) == "replaced_stale"
        _wait_for_slot_status(runtime, slot, "READY", timeout_seconds=10.0)
        dist.barrier(group=process_group)
        runtime._pump_target_planner_publications()  # noqa: SLF001
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="target_dispatch_ready")  # noqa: SLF001
        state = runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        publication_state = runtime.target_planner_service.publication_state_for_slot(slot)  # type: ignore[union-attr]
        return {
            "scenario": "stale_replacement",
            "trace": trace,
            "state": None if state is None else state.to_dict(),
            "publication_state": None if publication_state is None else publication_state.to_dict(),
            "submit_status": str(submit_result.status.value),
        }
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _worker(rank: int, world_size: int, master_port: int, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
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
            "late_suffix_call_count": 0,
            "late_suffix_provider_present": False,
            "formal_target_commit_after_miss": False,
            "scenarios": [
                _scenario_all_ready(rank, dist.group.WORLD),
                _scenario_not_ready_then_ready(rank, dist.group.WORLD),
                _scenario_failed_rank(rank, dist.group.WORLD),
                _scenario_cancelled(rank, dist.group.WORLD),
                _scenario_slot_mismatch(rank, dist.group.WORLD),
                _scenario_plan_digest_mismatch(rank, dist.group.WORLD),
                _scenario_subgroup(rank, dist.group.WORLD, subgroup),
                _scenario_cleanup_before_generation(rank, dist.group.WORLD),
                _scenario_stale_replacement(rank, dist.group.WORLD),
            ],
        }
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
        (out_path / f"rank{rank}_m1_formal_gloo.json").write_text(json.dumps(local, indent=2), encoding="utf-8")
        if rank == 0:
            summary = {
                "status": "completed",
                "world_size": int(world_size),
                "all_ranks": gathered,
                "late_suffix_call_count": 0,
                "late_suffix_provider_present": False,
                "formal_target_commit_after_miss": False,
            }
            (out_path / "m1_formal_lifecycle_gloo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
        dist.barrier()
    except Exception as exc:
        failure = {
            "rank": int(rank),
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (out_path / f"rank{rank}_m1_formal_gloo_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    finally:
        dist.destroy_process_group(subgroup)
        dist.destroy_process_group()


def main() -> None:
    out_dir = "outputs/closure/m1_formal_lifecycle_publication_gloo"
    port = _free_port()
    mp.spawn(_worker, args=(4, port, out_dir), nprocs=4, join=True)


if __name__ == "__main__":
    main()
