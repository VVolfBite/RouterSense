from __future__ import annotations

import json
import os
import socket
import time
import traceback
from dataclasses import replace
from functools import wraps
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp
import torch

from rs.planning import PlannerPolicyConfig
from rs.scheduling.validation import stable_hash
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.host import ControlGroupRegistry
from rs.runtime.online.megatron_ep.control.communication_lane import GlooControlCommunicationLane, slot_from_request
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.public_types import (
    CombineCompleteEvent,
    CombineReadyEvent,
    DispatchCompleteEvent,
    DispatchReadyEvent,
    ForwardBeginEvent,
    ForwardEndEvent,
    LocalPreparationToken,
    LocalPublicationCandidate,
    PublicationPollResult,
    PublicationPollStatus,
    PublicationSlot,
)
from rs.runtime.online.megatron_ep.target_planning.contracts import TwoHorizonPrediction
from rs.runtime.online.megatron_ep.target_planning.predictor import TwoHorizonPredictionBundle
from rs.runtime.online.megatron_ep.target_planning.planner_service import TargetLayerPlannerService, TargetLayerPlanningRequest
from rs.runtime.online.megatron_ep.target_planning.contracts import TargetPlanKey
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
        h1_digest = stable_hash(
            {
                "horizon": 1,
                "source_layer_id": str(source_layer_id),
                "target_layer_id": str(next_layer_id),
                "matrix_rows": [list(row) for row in current],
            }
        )
        h2_digest = stable_hash(
            {
                "horizon": 2,
                "source_layer_id": str(next_layer_id),
                "target_layer_id": str(int(next_layer_id) + 1),
                "matrix_rows": [list(row) for row in current],
            }
        )
        return TwoHorizonPredictionBundle(
            h1=TwoHorizonPrediction(
                forecast_horizon=1,
                source_layer_id=str(source_layer_id),
                target_layer_id=str(next_layer_id),
                matrix_unit="rows",
                matrix_rows=current,
                matrix_digest=str(h1_digest),
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
                matrix_digest=str(h2_digest),
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


class _FakeDispatcher:
    def __init__(self, *, input_splits: tuple[int, ...], output_splits: tuple[int, ...]) -> None:
        self.input_splits = tuple(int(v) for v in input_splits)
        self.output_splits = tuple(int(v) for v in output_splits)
        self.tokens_per_expert = self.input_splits

    def _maybe_dtoh_and_synchronize(self, _stage: str, tokens_per_expert):
        return tokens_per_expert


def _matrix_for_world_size(world_size: int) -> tuple[tuple[int, ...], ...]:
    if int(world_size) == 2:
        return ((0, 4), (3, 0))
    if int(world_size) == 4:
        return ((0, 4, 0, 2), (1, 0, 3, 0), (0, 2, 0, 5), (4, 0, 1, 0))
    raise ValueError(f"unsupported world_size {world_size!r}")


def _dispatcher_for_phase(*, rank: int, group_ranks: tuple[int, ...], phase: str) -> tuple[_FakeDispatcher, torch.Tensor, torch.Tensor | None]:
    matrix = _matrix_for_world_size(len(group_ranks))
    local_group_rank = tuple(int(v) for v in group_ranks).index(int(rank))
    row = tuple(int(value) for value in matrix[local_group_rank])
    col = tuple(int(matrix[src][local_group_rank]) for src in range(len(matrix)))
    if phase == "P0":
        input_splits = row
        output_splits = col
        hidden_rows = int(sum(row))
        probs_rows = int(sum(row))
        return (
            _FakeDispatcher(input_splits=input_splits, output_splits=output_splits),
            torch.zeros((hidden_rows, 4), dtype=torch.float32),
            torch.zeros((probs_rows, 1), dtype=torch.float32),
        )
    if phase == "P1":
        input_splits = row
        output_splits = col
        hidden_rows = int(sum(row))
        return (
            _FakeDispatcher(input_splits=input_splits, output_splits=output_splits),
            torch.zeros((hidden_rows, 4), dtype=torch.float32),
            None,
        )
    raise ValueError(f"unsupported phase {phase!r}")


def _runtime(
    *,
    rank: int,
    world_size: int,
    group_ranks: tuple[int, ...],
    root_rank: int,
    process_group,
    planner_factory=None,
    predictor_delay_seconds: float = 0.0,
) -> tuple[RouterSenseInjectionRuntime, list[dict[str, object]], dict[str, int | bool]]:
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
            selected_layer_ids=("1",),
            schedule_layer_selector="selected",
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
    runtime.configure_hook_scope(available_layer_names=("model.layers.0.mlp", "model.layers.1.mlp"))
    spy: dict[str, int | bool] = {
        "late_suffix_call_count": 0,
        "late_suffix_provider_call_count": 0,
        "late_suffix_consume_count": 0,
        "post_commit_publication_count": 0,
        "target_commit_seen": False,
    }

    original_agree_late_suffix = runtime._agree_late_suffix  # noqa: SLF001

    @wraps(original_agree_late_suffix)
    def _counted_agree_late_suffix(*args, **kwargs):
        spy["late_suffix_call_count"] = int(spy["late_suffix_call_count"]) + 1
        return original_agree_late_suffix(*args, **kwargs)

    runtime._agree_late_suffix = _counted_agree_late_suffix  # type: ignore[method-assign]  # noqa: SLF001

    original_late_suffix_provider = runtime._late_suffix_provider  # noqa: SLF001

    @wraps(original_late_suffix_provider)
    def _counted_late_suffix_provider(*args, **kwargs):
        spy["late_suffix_provider_call_count"] = int(spy["late_suffix_provider_call_count"]) + 1
        return original_late_suffix_provider(*args, **kwargs)

    runtime._late_suffix_provider = _counted_late_suffix_provider  # type: ignore[method-assign]  # noqa: SLF001

    original_consume_once = runtime.target_plan_store.consume_once  # type: ignore[union-attr]

    @wraps(original_consume_once)
    def _counted_consume_once(*args, **kwargs):
        if str(kwargs.get("execution_origin", "")) == "provisional_then_late_suffix":
            spy["late_suffix_consume_count"] = int(spy["late_suffix_consume_count"]) + 1
        return original_consume_once(*args, **kwargs)

    runtime.target_plan_store.consume_once = _counted_consume_once  # type: ignore[method-assign,union-attr]

    original_publish_if_current = runtime.target_plan_store.publish_if_current  # type: ignore[union-attr]

    @wraps(original_publish_if_current)
    def _counted_publish_if_current(*args, **kwargs):
        if bool(spy["target_commit_seen"]):
            spy["post_commit_publication_count"] = int(spy["post_commit_publication_count"]) + 1
        return original_publish_if_current(*args, **kwargs)

    runtime.target_plan_store.publish_if_current = _counted_publish_if_current  # type: ignore[method-assign,union-attr]
    return runtime, trace, spy


def _begin_forward(runtime: RouterSenseInjectionRuntime) -> None:
    runtime.handle(ForwardBeginEvent(forward_epoch=1))


def _emit_source_events(runtime: RouterSenseInjectionRuntime, *, rank: int, group_ranks: tuple[int, ...]) -> PublicationSlot:
    source_dispatcher, source_hidden, source_probs = _dispatcher_for_phase(rank=rank, group_ranks=group_ranks, phase="P0")
    source_combine_dispatcher, source_combine_hidden, _ = _dispatcher_for_phase(rank=rank, group_ranks=group_ranks, phase="P1")
    runtime.handle(
        DispatchReadyEvent(
            layer_name="model.layers.0.mlp",
            dispatcher=source_dispatcher,
            packed_hidden_states=source_hidden,
            packed_probs=source_probs,
            layer_role="prediction_source",
        )
    )
    runtime.handle(
        DispatchCompleteEvent(
            layer_name="model.layers.0.mlp",
            dispatcher=source_dispatcher,
            packed_hidden_states=source_hidden,
            result=(source_hidden.clone(), source_probs.clone() if source_probs is not None else None),
            layer_role="prediction_source",
        )
    )
    runtime.handle(
        CombineReadyEvent(
            layer_name="model.layers.0.mlp",
            dispatcher=source_combine_dispatcher,
            packed_hidden_states=source_combine_hidden,
        )
    )
    runtime.handle(
        CombineCompleteEvent(
            layer_name="model.layers.0.mlp",
            dispatcher=source_combine_dispatcher,
            packed_hidden_states=source_combine_hidden,
            result=source_combine_hidden.clone(),
        )
    )
    slot = slot_from_request(
        run_id=str(runtime.run_id),
        forward_generation=int(runtime._forward_epoch),  # noqa: SLF001
        microbatch_id=str(runtime.microbatch_id),
        source_layer_id="0",
        target_layer_id="1",
    )
    return slot


def _emit_target_dispatch_ready(runtime: RouterSenseInjectionRuntime, *, rank: int, group_ranks: tuple[int, ...]) -> None:
    target_dispatcher, target_hidden, target_probs = _dispatcher_for_phase(rank=rank, group_ranks=group_ranks, phase="P0")
    runtime.handle(
        DispatchReadyEvent(
            layer_name="model.layers.1.mlp",
            dispatcher=target_dispatcher,
            packed_hidden_states=target_hidden,
            packed_probs=target_probs,
            layer_role="selected",
        )
    )


def _emit_target_dispatch_complete(runtime: RouterSenseInjectionRuntime, *, rank: int, group_ranks: tuple[int, ...], spy: dict[str, int | bool]) -> None:
    target_dispatcher, target_hidden, target_probs = _dispatcher_for_phase(rank=rank, group_ranks=group_ranks, phase="P0")
    runtime.handle(
        DispatchCompleteEvent(
            layer_name="model.layers.1.mlp",
            dispatcher=target_dispatcher,
            packed_hidden_states=target_hidden,
            result=(target_hidden.clone(), target_probs.clone() if target_probs is not None else None),
            layer_role="selected",
        )
    )
    spy["target_commit_seen"] = True


def _emit_target_combine_complete(runtime: RouterSenseInjectionRuntime, *, rank: int, group_ranks: tuple[int, ...]) -> None:
    target_dispatcher, target_hidden, _ = _dispatcher_for_phase(rank=rank, group_ranks=group_ranks, phase="P1")
    runtime.handle(
        CombineReadyEvent(
            layer_name="model.layers.1.mlp",
            dispatcher=target_dispatcher,
            packed_hidden_states=target_hidden,
        )
    )
    runtime.handle(
        CombineCompleteEvent(
            layer_name="model.layers.1.mlp",
            dispatcher=target_dispatcher,
            packed_hidden_states=target_hidden,
            result=target_hidden.clone(),
        )
    )


def _end_forward(runtime: RouterSenseInjectionRuntime) -> None:
    runtime.handle(ForwardEndEvent())


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
    runtime, trace, spy = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group)
    try:
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        _wait_for_slot_status(runtime, slot, "READY")
        _emit_target_dispatch_ready(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        assert _wait_until(lambda: any(str(item.get("safe_status")) == "ready" for item in trace), timeout_seconds=10.0)
        state = runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        assert state is not None
        _emit_target_dispatch_complete(runtime, rank=rank, group_ranks=(0, 1, 2, 3), spy=spy)
        _emit_target_combine_complete(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        _end_forward(runtime)
        terminal = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        assert terminal is not None
        return {
            "scenario": "all_ready",
            "trace": trace,
            "state_before_end": state.to_dict(),
            "terminal_after_end": terminal.to_dict(),
            "spy": dict(spy),
        }
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_not_ready_then_ready(rank: int, process_group) -> dict[str, object]:
    delay = 0.4 if rank == 3 else 0.0
    runtime, trace, spy = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group, predictor_delay_seconds=delay)
    try:
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        assert _wait_until(lambda: any(str(item.get("safe_status")) == "not_ready" for item in trace))
        _wait_for_slot_status(runtime, slot, "READY")
        _emit_target_dispatch_ready(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        assert _wait_until(lambda: any(str(item.get("safe_status")) == "ready" for item in trace), timeout_seconds=10.0)
        state = runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        assert state is not None
        _emit_target_dispatch_complete(runtime, rank=rank, group_ranks=(0, 1, 2, 3), spy=spy)
        _emit_target_combine_complete(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        _end_forward(runtime)
        return {
            "scenario": "not_ready_then_ready",
            "trace": trace,
            "state_before_end": state.to_dict(),
            "spy": dict(spy),
        }
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_failed_rank(rank: int, process_group) -> dict[str, object]:
    planner_factory = (lambda _planner_id, _config: _FailingPlanner()) if rank == 2 else None
    runtime, trace, spy = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group, planner_factory=planner_factory)
    try:
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        if rank == 2:
            _wait_for_slot_status(runtime, slot, "FAILED")
        else:
            assert _wait_until(lambda: runtime.target_planner_service.publication_state_for_slot(slot) is not None, timeout_seconds=10.0)  # type: ignore[union-attr]
        _emit_target_dispatch_ready(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        dist.barrier(group=process_group)
        assert _wait_until(
            lambda: runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")) is not None,  # noqa: SLF001
            timeout_seconds=10.0,
        )
        terminal = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        _end_forward(runtime)
        return {"scenario": "failed_rank", "trace": trace, "terminal": terminal.to_dict(), "spy": dict(spy)}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_cancelled(rank: int, process_group) -> dict[str, object]:
    runtime, trace, spy = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group, predictor_delay_seconds=0.4)
    try:
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        runtime.target_planner_service.cancel_generation(run_id=str(runtime.run_id), forward_epoch=1, microbatch_id=str(runtime.microbatch_id))  # type: ignore[union-attr]
        _wait_for_slot_status(runtime, slot, "CANCELLED")
        _emit_target_dispatch_ready(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        dist.barrier(group=process_group)
        assert _wait_until(
            lambda: runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")) is not None,  # noqa: SLF001
            timeout_seconds=10.0,
        )
        terminal = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        _end_forward(runtime)
        return {"scenario": "cancelled", "trace": trace, "terminal": terminal.to_dict(), "spy": dict(spy)}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_slot_mismatch(rank: int, process_group) -> dict[str, object]:
    runtime, trace, spy = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group)
    try:
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        candidate = _wait_for_slot_status(runtime, slot, "READY")
        runtime._pump_target_planner_publications()  # noqa: SLF001
        original_pump = runtime._pump_target_planner_publications  # noqa: SLF001
        runtime._pump_target_planner_publications = lambda: None  # type: ignore[method-assign]  # noqa: SLF001
        if rank == 1:
            tampered = replace(candidate, token=LocalPreparationToken(
                service_session_id=int(candidate.token.service_session_id),
                forward_generation=int(candidate.token.forward_generation),
                target_layer_id=str(candidate.token.target_layer_id),
                task_version=int(candidate.token.task_version),
                publication_slot_digest="tampered-slot-digest",
            ))
            runtime.target_planner_service._publication_state_by_slot[str(slot.semantic_digest())] = tampered  # type: ignore[attr-defined]  # noqa: SLF001
            ready_pair = runtime._ready_target_plan_candidates.get(str(slot.semantic_digest()))  # type: ignore[attr-defined]  # noqa: SLF001
            if ready_pair is not None:
                runtime._ready_target_plan_candidates[str(slot.semantic_digest())] = (ready_pair[0], tampered)  # type: ignore[attr-defined]  # noqa: SLF001
        _emit_target_dispatch_ready(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        runtime._pump_target_planner_publications = original_pump  # type: ignore[method-assign]  # noqa: SLF001
        dist.barrier(group=process_group)
        if not _wait_until(
            lambda: runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")) is not None,  # noqa: SLF001
            timeout_seconds=10.0,
        ):
            raise AssertionError(
                {
                    "scenario": "slot_mismatch",
                    "rank": int(rank),
                    "state": None
                    if runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")) is None  # noqa: SLF001
                    else runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")).to_dict(),  # noqa: SLF001
                    "terminal": None
                    if runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")) is None  # noqa: SLF001
                    else runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")).to_dict(),  # noqa: SLF001
                    "trace": trace,
                    "publication_state": None
                    if runtime.target_planner_service.publication_state_for_slot(slot) is None  # type: ignore[union-attr]
                    else runtime.target_planner_service.publication_state_for_slot(slot).to_dict(),  # type: ignore[union-attr]
                }
            )
        terminal = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        _end_forward(runtime)
        return {"scenario": "slot_mismatch", "trace": trace, "terminal": terminal.to_dict(), "spy": dict(spy)}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_plan_digest_mismatch(rank: int, process_group) -> dict[str, object]:
    runtime, trace, spy = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group)
    try:
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        candidate = _wait_for_slot_status(runtime, slot, "READY")
        runtime._pump_target_planner_publications()  # noqa: SLF001
        original_pump = runtime._pump_target_planner_publications  # noqa: SLF001
        runtime._pump_target_planner_publications = lambda: None  # type: ignore[method-assign]  # noqa: SLF001
        if rank == 0:
            tampered_metadata = dict(candidate.metadata)
            tampered_plan = dict(tampered_metadata["plan"])
            tampered_plan["logical_plan_digest"] = "tampered-plan-digest"
            tampered_metadata["plan"] = tampered_plan
            runtime.target_planner_service._publication_state_by_slot[str(slot.semantic_digest())] = replace(candidate, metadata=tampered_metadata)  # type: ignore[attr-defined]  # noqa: SLF001
            ready_pair = runtime._ready_target_plan_candidates.get(str(slot.semantic_digest()))  # type: ignore[attr-defined]  # noqa: SLF001
            if ready_pair is not None:
                runtime._ready_target_plan_candidates[str(slot.semantic_digest())] = (  # type: ignore[attr-defined]  # noqa: SLF001
                    ready_pair[0],
                    replace(ready_pair[1], metadata=tampered_metadata),
                )
        _emit_target_dispatch_ready(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        runtime._pump_target_planner_publications = original_pump  # type: ignore[method-assign]  # noqa: SLF001
        dist.barrier(group=process_group)
        assert _wait_until(
            lambda: runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")) is not None,  # noqa: SLF001
            timeout_seconds=10.0,
        )
        terminal = runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        _end_forward(runtime)
        return {"scenario": "plan_digest_mismatch", "trace": trace, "terminal": terminal.to_dict(), "spy": dict(spy)}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_subgroup(rank: int, world_group, subgroup) -> dict[str, object]:
    if rank not in {2, 3}:
        dist.barrier(group=world_group)
        dist.barrier(group=world_group)
        dist.barrier(group=world_group)
        return {"scenario": "subgroup", "rank": rank, "skipped": True}
    runtime, trace, spy = _runtime(rank=rank, world_size=2, group_ranks=(2, 3), root_rank=2, process_group=subgroup)
    try:
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=(2, 3))
        _wait_for_slot_status(runtime, slot, "READY")
        dist.barrier(group=world_group)
        runtime._pump_target_planner_publications()  # noqa: SLF001
        runtime._poll_target_plan_slot(target_layer_id="1", safe_point="target_dispatch_ready")  # noqa: SLF001
        dist.barrier(group=world_group)
        state = runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        assert state is not None
        _end_forward(runtime)
        dist.barrier(group=world_group)
        return {"scenario": "subgroup", "rank": rank, "trace": trace, "state_before_end": state.to_dict(), "spy": dict(spy)}
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_cleanup_before_generation(rank: int, process_group) -> dict[str, object]:
    runtime, _trace, spy = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group)
    try:
        runtime.begin_forward(forward_epoch=1)
        _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        runtime.begin_forward(forward_epoch=2)
        _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        runtime.begin_forward(forward_epoch=3)
        old_key_1 = TargetPlanKey(
            run_id=str(runtime.run_id),
            forward_epoch=1,
            microbatch_id=str(runtime.microbatch_id),
            target_layer_id="1",
        )
        old_key_2 = TargetPlanKey(
            run_id=str(runtime.run_id),
            forward_epoch=2,
            microbatch_id=str(runtime.microbatch_id),
            target_layer_id="1",
        )
        current_key = TargetPlanKey(
            run_id=str(runtime.run_id),
            forward_epoch=3,
            microbatch_id=str(runtime.microbatch_id),
            target_layer_id="1",
        )
        submit1 = runtime.target_planner_service.submit(  # type: ignore[union-attr]
            TargetLayerPlanningRequest(
                run_id=str(runtime.run_id),
                forward_epoch=1,
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
        )
        submit2 = runtime.target_planner_service.submit(  # type: ignore[union-attr]
            TargetLayerPlanningRequest(
                run_id=str(runtime.run_id),
                forward_epoch=2,
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
        )
        return {
            "scenario": "cleanup_before_generation",
            "old_terminal_1": None if runtime.target_plan_store.get_terminal_record(old_key_1) is None else runtime.target_plan_store.get_terminal_record(old_key_1).to_dict(),
            "old_terminal_2": None if runtime.target_plan_store.get_terminal_record(old_key_2) is None else runtime.target_plan_store.get_terminal_record(old_key_2).to_dict(),
            "current_state": None if runtime.target_plan_store.get_state_record(current_key) is None else runtime.target_plan_store.get_state_record(current_key).to_dict(),
            "old_submit_status": str(submit1.status.value),
            "old_submit_status_2": str(submit2.status.value),
            "spy": dict(spy),
        }
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_stale_replacement(rank: int, process_group) -> dict[str, object]:
    runtime, trace, spy = _runtime(rank=rank, world_size=4, group_ranks=(0, 1, 2, 3), root_rank=0, process_group=process_group, predictor_delay_seconds=0.15)
    try:
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
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
        _emit_target_dispatch_ready(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        state = runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp"))  # noqa: SLF001
        publication_state = runtime.target_planner_service.publication_state_for_slot(slot)  # type: ignore[union-attr]
        assert state is not None
        _emit_target_dispatch_complete(runtime, rank=rank, group_ranks=(0, 1, 2, 3), spy=spy)
        _end_forward(runtime)
        return {
            "scenario": "stale_replacement",
            "trace": trace,
            "state": state.to_dict(),
            "publication_state": None if publication_state is None else publication_state.to_dict(),
            "submit_status": str(submit_result.status.value),
            "spy": dict(spy),
        }
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _scenario_target_commit_miss_then_worker_ready(rank: int, process_group) -> dict[str, object]:
    runtime, trace, spy = _runtime(
        rank=rank,
        world_size=4,
        group_ranks=(0, 1, 2, 3),
        root_rank=0,
        process_group=process_group,
        predictor_delay_seconds=0.8,
    )
    try:
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        assert _wait_until(lambda: any(str(item.get("safe_status")) == "not_ready" for item in trace), timeout_seconds=10.0)
        _emit_target_dispatch_ready(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        assert len([item for item in trace if str(item.get("safe_status")) == "not_ready"]) >= 2
        _emit_target_dispatch_complete(runtime, rank=rank, group_ranks=(0, 1, 2, 3), spy=spy)
        _wait_for_slot_status(runtime, slot, "READY", timeout_seconds=10.0)
        _emit_target_combine_complete(runtime, rank=rank, group_ranks=(0, 1, 2, 3))
        _end_forward(runtime)
        return {
            "scenario": "target_commit_miss_then_worker_ready",
            "trace": trace,
            "post_commit_publication_count": int(spy["post_commit_publication_count"]),
            "late_suffix_call_count": int(spy["late_suffix_call_count"]),
            "late_suffix_consume_count": int(spy["late_suffix_consume_count"]),
            "state_after_end": None if runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")) is None else runtime.target_plan_store.get_state_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")).to_dict(),  # noqa: SLF001
            "terminal_after_end": None if runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")) is None else runtime.target_plan_store.get_terminal_record(runtime._target_plan_key(layer_name="model.layers.1.mlp")).to_dict(),  # noqa: SLF001
            "spy": dict(spy),
        }
    finally:
        runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def _validate_global_gate_result(gathered: list[dict[str, object] | None]) -> None:
    ranks = [dict(item or {}) for item in gathered]
    assert all("scenarios" in item for item in ranks), gathered
    total_late_suffix_calls = sum(int(item.get("late_suffix_call_count", 0)) for item in ranks)
    total_late_suffix_provider_calls = sum(int(item.get("late_suffix_provider_call_count", 0)) for item in ranks)
    total_late_suffix_consumes = sum(int(item.get("late_suffix_consume_count", 0)) for item in ranks)
    total_post_commit_publications = sum(int(item.get("post_commit_publication_count", 0)) for item in ranks)
    assert total_late_suffix_calls == 0, gathered
    assert total_late_suffix_provider_calls == 0, gathered
    assert total_late_suffix_consumes == 0, gathered
    assert total_post_commit_publications == 0, gathered
    for item in ranks:
        scenarios = {str(entry.get("scenario")): dict(entry) for entry in list(item.get("scenarios", []))}
        all_ready = scenarios["all_ready"]
        assert str(all_ready["terminal_after_end"]["final_status"]) == "COMPLETED", all_ready
        assert any(str(entry.get("safe_status")) == "ready" for entry in all_ready["trace"]), all_ready
        not_ready_then_ready = scenarios["not_ready_then_ready"]
        statuses = [str(entry.get("safe_status")) for entry in not_ready_then_ready["trace"]]
        assert "not_ready" in statuses and "ready" in statuses, not_ready_then_ready
        failed_rank = scenarios["failed_rank"]
        assert str(failed_rank["terminal"]["final_status"]) == "FAILED", failed_rank
        cancelled = scenarios["cancelled"]
        assert str(cancelled["terminal"]["final_status"]) == "CANCELLED", cancelled
        slot_mismatch = scenarios["slot_mismatch"]
        assert str(slot_mismatch["terminal"]["final_status"]) == "FAILED", slot_mismatch
        plan_mismatch = scenarios["plan_digest_mismatch"]
        assert str(plan_mismatch["terminal"]["final_status"]) == "FAILED", plan_mismatch
        cleanup = scenarios["cleanup_before_generation"]
        assert str(cleanup["old_submit_status"]) == "rejected_expired", cleanup
        assert str(cleanup["old_submit_status_2"]) == "rejected_expired", cleanup
        commit_miss = scenarios["target_commit_miss_then_worker_ready"]
        assert int(commit_miss["post_commit_publication_count"]) == 0, commit_miss
        assert int(commit_miss["late_suffix_call_count"]) == 0, commit_miss
        assert int(commit_miss["late_suffix_consume_count"]) == 0, commit_miss


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
    registry, local_control_handle = ControlGroupRegistry.initialize(
        local_ep_group_ranks=(0, 1) if rank in {0, 1} else (2, 3),
        world_group=dist.group.WORLD,
    )
    subgroup = registry.groups[(2, 3)]
    try:
        scenarios = [
            _scenario_all_ready(rank, dist.group.WORLD),
            _scenario_not_ready_then_ready(rank, dist.group.WORLD),
            _scenario_failed_rank(rank, dist.group.WORLD),
            _scenario_cancelled(rank, dist.group.WORLD),
            _scenario_slot_mismatch(rank, dist.group.WORLD),
            _scenario_plan_digest_mismatch(rank, dist.group.WORLD),
            _scenario_subgroup(rank, dist.group.WORLD, subgroup),
            _scenario_cleanup_before_generation(rank, dist.group.WORLD),
            _scenario_stale_replacement(rank, dist.group.WORLD),
            _scenario_target_commit_miss_then_worker_ready(rank, dist.group.WORLD),
        ]
        local_late_suffix_call_count = sum(
            int((scenario.get("spy") or {}).get("late_suffix_call_count", 0))
            for scenario in scenarios
            if isinstance(scenario, dict)
        )
        local_late_suffix_provider_calls = sum(
            int((scenario.get("spy") or {}).get("late_suffix_provider_call_count", 0))
            for scenario in scenarios
            if isinstance(scenario, dict)
        )
        local_late_suffix_consume_count = sum(
            int((scenario.get("spy") or {}).get("late_suffix_consume_count", 0))
            for scenario in scenarios
            if isinstance(scenario, dict)
        )
        local_post_commit_publication_count = sum(
            int((scenario.get("spy") or {}).get("post_commit_publication_count", 0))
            for scenario in scenarios
            if isinstance(scenario, dict)
        )
        local = {
            "rank": int(rank),
            "late_suffix_call_count": int(local_late_suffix_call_count),
            "late_suffix_provider_call_count": int(local_late_suffix_provider_calls),
            "late_suffix_consume_count": int(local_late_suffix_consume_count),
            "post_commit_publication_count": int(local_post_commit_publication_count),
            "late_suffix_provider_installed": False,
            "supports_late_suffix_splice": False,
            "scenarios": scenarios,
        }
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
        (out_path / f"rank{rank}_m1_formal_gloo.json").write_text(json.dumps(local, indent=2), encoding="utf-8")
        if rank == 0:
            _validate_global_gate_result(gathered)
            total_late_suffix_calls = sum(int((item or {}).get("late_suffix_call_count", 0)) for item in gathered)
            total_late_suffix_provider_calls = sum(int((item or {}).get("late_suffix_provider_call_count", 0)) for item in gathered)
            total_late_suffix_consumes = sum(int((item or {}).get("late_suffix_consume_count", 0)) for item in gathered)
            total_post_commit_publication = sum(int((item or {}).get("post_commit_publication_count", 0)) for item in gathered)
            supports_late_suffix_splice = any(bool((item or {}).get("supports_late_suffix_splice", False)) for item in gathered)
            summary = {
                "status": "passed",
                "world_size": int(world_size),
                "all_ranks": gathered,
                "late_suffix_call_count": int(total_late_suffix_calls),
                "late_suffix_provider_call_count": int(total_late_suffix_provider_calls),
                "late_suffix_consume_count": int(total_late_suffix_consumes),
                "post_commit_publication_count": int(total_post_commit_publication),
                "supports_late_suffix_splice": bool(supports_late_suffix_splice),
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
        registry.close(local_ep_group_ranks=(0, 1) if rank in {0, 1} else (2, 3))
        dist.destroy_process_group()


def main() -> None:
    out_dir = "outputs/closure/m1_formal_lifecycle_publication_gloo"
    port = _free_port()
    mp.spawn(_worker, args=(4, port, out_dir), nprocs=4, join=True)


if __name__ == "__main__":
    main()
