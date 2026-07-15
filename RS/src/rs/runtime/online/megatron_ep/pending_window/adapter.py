"""Pending-window 适配器。

它不是 scheduling policy，而是 runtime 侧编译器：
- 先基于 joint logical window 生成逻辑计划
- 再把“当前 phase 能执行的那一片”编译回 PhaseExecutionPlan
它不绕过 frozen phase-local executor。
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from rs.runtime.online.megatron_ep.pending_window.policy_adapter import (
    build_phase_policy_fast_path,
    compile_prepared_window_phase_plan,
    get_or_build_prepared_priority_cache,
)
from rs.scheduling.contracts import (
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    PreparedWindowPlan,
    ReleaseConstraint,
)
from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext
from rs.scheduling.validation import stable_hash


class UnsupportedPendingWindowAdapter(ValueError):
    pass


class MultiphasePendingWindowAdapter:
    """Compile a current phase plan from a joint logical window schedule."""

    def __init__(
        self,
        *,
        shared_state: dict[str, Any],
        phase_policy_name: str,
        bucket_rows: int,
        max_waves: int = 256,
        p0_weight: float,
        p1_reservation_weight: float,
        p2_hint_weight: float,
        fast_path_enabled: bool = False,
    ) -> None:
        if phase_policy_name not in {"routersense_p0p1p2_hint"}:
            raise UnsupportedPendingWindowAdapter(
                "multiphase_pending_window currently supports only "
                f"'routersense_p0p1p2_hint'; got {phase_policy_name!r}"
            )
        self._shared_state = shared_state
        self._phase_policy_name = phase_policy_name
        self._bucket_rows = int(bucket_rows)
        self._max_waves = int(max_waves)
        self._p0_weight = float(p0_weight)
        self._p1_reservation_weight = float(p1_reservation_weight)
        self._p2_hint_weight = float(p2_hint_weight)
        self._fast_path_enabled = bool(fast_path_enabled)
        self._phase_policy = build_phase_policy_fast_path(
            bucket_rows=self._bucket_rows,
            p0_weight=self._p0_weight,
            p1_reservation_weight=self._p1_reservation_weight,
            p2_hint_weight=self._p2_hint_weight,
        )

    def build_plan(
        self,
        *,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> PhaseExecutionPlan:
        prepared_plan = self._shared_state.get("prepared_plan")
        if self._fast_path_enabled:
            fast_path_start_ns = time.monotonic_ns()
            fallback_reason = ""
            cache_hit = False
            cache_build_time_us = 0.0
            lookup_time_us = 0.0
            if prepared_plan is not None:
                prepared_priority_cache, cache_hit, cache_build_time_us = get_or_build_prepared_priority_cache(
                    shared_state=self._shared_state,
                    prepared_plan=prepared_plan,
                )
                lookup_start_ns = time.monotonic_ns()
                phase_priority = dict(prepared_priority_cache.get("priority_by_phase", {}).get(str(local_context.phase), {}))
                lookup_end_ns = time.monotonic_ns()
                lookup_time_us = (lookup_end_ns - lookup_start_ns) / 1000.0
                if phase_priority:
                    compile_start_ns = time.monotonic_ns()
                    phase_plan = compile_prepared_window_phase_plan(
                        prepared_plan=prepared_plan,
                        local_context=local_context,
                        global_contexts=global_contexts,
                        bucket_rows=self._bucket_rows,
                        p0_weight=self._p0_weight,
                        p1_reservation_weight=self._p1_reservation_weight,
                        p2_hint_weight=self._p2_hint_weight,
                        policy_name=self._phase_policy_name,
                        prepared_priority_cache=prepared_priority_cache,
                        phase_policy=self._phase_policy,
                    )
                    compile_end_ns = time.monotonic_ns()
                    wave_sizes = [len(wave.bucket_tasks) for wave in phase_plan.waves]
                    wave_conflict_count = sum(
                        1
                        for wave in phase_plan.waves
                        if len({int(task.src_rank) for task in wave.bucket_tasks}) != len(wave.bucket_tasks)
                        or len({int(task.dst_rank) for task in wave.bucket_tasks}) != len(wave.bucket_tasks)
                    )
                    fast_path_end_ns = time.monotonic_ns()
                    return replace(
                        phase_plan,
                        metrics={
                            **phase_plan.metrics,
                            "compiled_from_pending_window": True,
                            "pending_window_logical_policy_name": "routersense_fast_path",
                            "pending_window_plan_hash": str(prepared_priority_cache.get("source_logical_plan_hash", "")),
                            "pending_window_information_mode": "p0_p1_p2",
                            "pending_window_forecast_available": True,
                            "pending_window_phase": local_context.phase,
                            "pending_window_p0_total_bytes": int(sum(int(v) for v in local_context.per_peer_bytes)),
                            "pending_window_p1_total_bytes": 0,
                            "pending_window_p2_total_bytes": int(self._shared_state.get("p2_matrix_total_bytes", 0) or 0),
                            "pending_window_p1_matrix_source": "current_phase_context",
                            "pending_window_p2_matrix_source": "prepared_window_plan",
                            "pending_window_logical_build_time_us": 0.0,
                            "pending_window_compile_time_us": (compile_end_ns - compile_start_ns) / 1000.0,
                            "routersense_fast_path_enabled": True,
                            "routersense_heavy_path_used": False,
                            "routersense_fast_path_fallback_reason": "",
                            "fast_path_mode": "edge_priority_wave_pack",
                            "fast_path_wave_plan_valid": wave_conflict_count == 0,
                            "wave_conflict_count": int(wave_conflict_count),
                            "planned_wave_count": int(len(phase_plan.waves)),
                            "max_tasks_per_wave": int(max(wave_sizes, default=0)),
                            "min_tasks_per_wave": int(min(wave_sizes, default=0)) if wave_sizes else 0,
                            "zero_participation_rank_count": max(0, len(local_context.ep_group_ranks) - len({int(task.src_rank) for wave in phase_plan.waves for task in wave.bucket_tasks} | {int(task.dst_rank) for wave in phase_plan.waves for task in wave.bucket_tasks})),
                            "fallback_policy": "",
                            "fast_path_fallback_reason": "",
                            "prepared_priority_cache_hit": bool(cache_hit),
                            "prepared_priority_cache_build_time_us": float(cache_build_time_us),
                            "prepared_priority_lookup_time_us": float(lookup_time_us),
                            "source_logical_plan_hash": str(prepared_priority_cache.get("source_logical_plan_hash", "")),
                            "fast_path_total_time_us": (fast_path_end_ns - fast_path_start_ns) / 1000.0,
                        },
                    )
                fallback_reason = "no_prepared_priority_for_phase"
            else:
                fallback_reason = "no_prepared_plan"
            fallback_start_ns = time.monotonic_ns()
            fallback_plan = self._phase_policy.build_plan(local_context=local_context, global_contexts=global_contexts)
            fallback_end_ns = time.monotonic_ns()
            return replace(
                fallback_plan,
                metrics={
                    **fallback_plan.metrics,
                    "compiled_from_pending_window": False,
                    "pending_window_logical_build_time_us": 0.0,
                    "pending_window_compile_time_us": 0.0,
                    "routersense_fast_path_enabled": True,
                    "routersense_heavy_path_used": False,
                    "routersense_fast_path_fallback_reason": str(fallback_reason),
                    "fast_path_mode": "edge_priority_wave_pack",
                    "fast_path_wave_plan_valid": True,
                    "wave_conflict_count": 0,
                    "planned_wave_count": int(len(fallback_plan.waves)),
                    "fallback_policy": self._phase_policy_name,
                    "fast_path_fallback_reason": str(fallback_reason),
                    "prepared_priority_cache_hit": bool(cache_hit),
                    "prepared_priority_cache_build_time_us": float(cache_build_time_us),
                    "prepared_priority_lookup_time_us": float(lookup_time_us),
                    "fast_path_total_time_us": (fallback_end_ns - fallback_start_ns) / 1000.0,
                },
            )

        problem = _build_problem_from_contexts(
            local_context=local_context,
            global_contexts=global_contexts,
            prepared_plan=prepared_plan,
            p0_weight=self._p0_weight,
            p1_reservation_weight=self._p1_reservation_weight,
            p2_hint_weight=self._p2_hint_weight,
            max_waves=self._max_waves,
        )
        logical_policy = RouterSenseMultiphaseLookaheadPolicy(
            information_mode=problem.options.information_mode,
            p0_weight=self._p0_weight,
            p1_reservation_weight=self._p1_reservation_weight,
            p2_hint_weight=self._p2_hint_weight,
        )
        logical_build_start_ns = time.monotonic_ns()
        logical_plan = logical_policy.build_logical_plan(problem)
        logical_build_end_ns = time.monotonic_ns()
        prepared = PreparedWindowPlan(
            window_key=stable_hash(
                {
                    "layer_id": local_context.layer_id,
                    "phase": local_context.phase,
                    "logical_plan": logical_plan.to_dict(),
                }
            ),
            forecast_digest="" if problem.forecast is None else str(problem.forecast.digest),
            logical_plan=logical_plan,
            created_at_layer_id=str(local_context.layer_id),
            applies_from_layer_id=str(local_context.layer_id),
            execution_capability_required="multiphase_pending_window",
        )
        compile_start_ns = time.monotonic_ns()
        phase_plan = compile_prepared_window_phase_plan(
            prepared_plan=prepared,
            local_context=local_context,
            global_contexts=global_contexts,
            bucket_rows=self._bucket_rows,
            p0_weight=self._p0_weight,
            p1_reservation_weight=self._p1_reservation_weight,
            p2_hint_weight=self._p2_hint_weight,
            policy_name=self._phase_policy_name,
            phase_policy=self._phase_policy,
        )
        compile_end_ns = time.monotonic_ns()
        return replace(
            phase_plan,
            metrics={
                **phase_plan.metrics,
                "compiled_from_pending_window": True,
                "pending_window_logical_policy_name": logical_plan.policy_name,
                "pending_window_plan_hash": stable_hash(logical_plan.to_dict()),
                "pending_window_information_mode": problem.options.information_mode,
                "pending_window_forecast_available": problem.forecast is not None,
                "pending_window_phase": local_context.phase,
                "pending_window_p0_total_bytes": _matrix_total(problem.p0_dispatch_matrix),
                "pending_window_p1_total_bytes": _matrix_total(problem.p1_return_matrix),
                "pending_window_p2_total_bytes": _matrix_total(problem.p2_next_dispatch_forecast_matrix),
                "pending_window_p1_matrix_source": "prepared_window_plan" if local_context.phase == "P0" else "current_phase_context",
                "pending_window_p2_matrix_source": "prepared_window_plan",
                "pending_window_logical_build_time_us": (logical_build_end_ns - logical_build_start_ns) / 1000.0,
                "pending_window_compile_time_us": (compile_end_ns - compile_start_ns) / 1000.0,
                "routersense_fast_path_enabled": False,
                "routersense_heavy_path_used": True,
                "routersense_fast_path_fallback_reason": "disabled",
            },
        )


def _build_problem_from_contexts(
    *,
    local_context: PhaseReadyContext,
    global_contexts: tuple[PhaseReadyContext, ...],
    prepared_plan: PreparedWindowPlan | None,
    p0_weight: float,
    p1_reservation_weight: float,
    p2_hint_weight: float,
    max_waves: int,
) -> MultiPhaseSchedulingProblem:
    num_gpus = len(local_context.ep_group_ranks)
    current_matrix = _matrix_from_contexts(global_contexts, num_gpus=num_gpus)
    if local_context.phase == "P0":
        p0_dispatch_matrix = current_matrix
        p1_return_matrix = _matrix_from_prepared_plan(prepared_plan, phase="p1_return", num_gpus=num_gpus)
    else:
        p0_dispatch_matrix = _zero_matrix(num_gpus)
        p1_return_matrix = current_matrix
    p2_forecast_matrix = _matrix_from_prepared_plan(prepared_plan, phase="p2_next_dispatch_forecast", num_gpus=num_gpus)
    has_p1_dependency = local_context.phase == "P0" and _has_nonzero(p1_return_matrix)
    has_p2_forecast = _has_nonzero(p2_forecast_matrix)
    information_mode = "p0_p1_p2" if has_p2_forecast else "p0_p1" if has_p1_dependency else "p0_only"
    forecast = None
    if has_p2_forecast:
        forecast = ForecastPressure(
            source="prepared_window_plan",
            digest=str(getattr(prepared_plan, "forecast_digest", "")),
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(num_gpus, num_gpus),
            matrix_total_bytes=sum(sum(int(value) for value in row) for row in p2_forecast_matrix),
            matrix=p2_forecast_matrix,
            metadata={"source_layer": str(getattr(prepared_plan, "created_at_layer_id", ""))},
        )
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(),
        topology=LogicalTopology(num_gpus=num_gpus),
        release_model=ReleaseConstraint(
            phase="p1_return",
            rank=int(local_context.global_rank),
            release_after_phase="p0_dispatch",
            expert_compute_delay=0.0,
        ),
        forecast=forecast,
        options=GlobalReadySetOptions(
            scheduling_mode="runtime_lookahead",
            information_mode=information_mode,
            prediction_confidence=1.0 if forecast is not None else 0.0,
            p0_weight=float(p0_weight),
            p1_reservation_weight=float(p1_reservation_weight),
            p2_hint_weight=float(p2_hint_weight),
            max_waves=int(max_waves),
        ),
        p0_dispatch_matrix=p0_dispatch_matrix,
        p1_return_matrix=p1_return_matrix,
        p2_next_dispatch_forecast_matrix=p2_forecast_matrix,
    )


def _matrix_from_contexts(global_contexts: tuple[PhaseReadyContext, ...], *, num_gpus: int) -> tuple[tuple[int, ...], ...]:
    matrix = [[0 for _ in range(num_gpus)] for _ in range(num_gpus)]
    rank_to_index = {int(rank): idx for idx, rank in enumerate(global_contexts[0].ep_group_ranks)}
    for context in global_contexts:
        src_index = rank_to_index.get(int(context.global_rank), int(context.local_rank))
        for dst_index, byte_count in enumerate(tuple(int(v) for v in context.per_peer_bytes)[:num_gpus]):
            if src_index == dst_index or int(byte_count) <= 0:
                continue
            matrix[src_index][dst_index] += int(byte_count)
    return tuple(tuple(row) for row in matrix)


def _matrix_from_prepared_plan(prepared_plan: PreparedWindowPlan | None, *, phase: str, num_gpus: int) -> tuple[tuple[int, ...], ...]:
    if prepared_plan is None:
        return _zero_matrix(num_gpus)
    if phase == "p2_next_dispatch_forecast":
        forecast_matrix = tuple(
            tuple(int(value) for value in row)
            for row in getattr(prepared_plan, "forecast_matrix", ())
        )
        if forecast_matrix:
            return forecast_matrix
    matrix = [[0 for _ in range(num_gpus)] for _ in range(num_gpus)]
    for wave in prepared_plan.logical_plan.waves:
        for flow in wave.flows:
            if str(flow.phase) != phase:
                continue
            src_rank = int(flow.src_rank)
            dst_rank = int(flow.dst_rank)
            if 0 <= src_rank < num_gpus and 0 <= dst_rank < num_gpus and src_rank != dst_rank:
                matrix[src_rank][dst_rank] += int(flow.byte_count)
    return tuple(tuple(row) for row in matrix)


def _zero_matrix(num_gpus: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in range(num_gpus)) for _ in range(num_gpus))


def _has_nonzero(matrix: tuple[tuple[int, ...], ...]) -> bool:
    return any(any(int(value) > 0 for value in row) for row in matrix)


def _matrix_total(matrix: tuple[tuple[int, ...], ...]) -> int:
    return sum(sum(int(value) for value in row) for row in matrix)


__all__ = ["MultiphasePendingWindowAdapter", "UnsupportedPendingWindowAdapter"]
