"""Prepared window plan 到当前 phase plan 的编译器。

主要函数：
- compile_prepared_window_phase_plan()
它负责把 prepared logical plan 的优先级信息注入当前 phase 的可执行计划。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from typing import Any

from rs.runtime.online.megatron_ep.control.p2_provider import extract_prepared_plan_priority
from rs.scheduling.phase_local.p0p1p2_hint_order import RouterSenseP0P1P2HintPolicy
from rs.scheduling.registry import resolve_phase_policy, supported_phase_policies
from rs.scheduling.contracts import PreparedWindowPlan
from rs.scheduling.phase_execution import FutureDemandHint, PhaseExecutionPlan, PhaseReadyContext


def compile_prepared_window_phase_plan(
    *,
    prepared_plan: PreparedWindowPlan,
    local_context: PhaseReadyContext,
    global_contexts: tuple[PhaseReadyContext, ...],
    bucket_rows: int,
    p0_weight: float = 1.0,
    p1_reservation_weight: float = 1.0,
    p2_hint_weight: float = 1.0,
    policy_name: str = "routersense_p0p1p2_hint",
    prepared_priority_cache: dict[str, Any] | None = None,
    phase_policy: Any | None = None,
) -> PhaseExecutionPlan:
    """Compile the current phase from a prepared logical window plan.

    The prepared plan supplies logical edge priority. Current phase layouts
    supply tensor offsets and payload slices; future P2 forecast edges are not
    executable and are ignored by the phase-local compiler.
    """

    forecast_digest = str(prepared_plan.forecast_digest)
    hint_digest = hashlib.sha256(f"{forecast_digest}:{local_context.layer_id}".encode("utf-8")).hexdigest()[:16]
    priority_lookup_start_ns = time.monotonic_ns()
    priority_payload = prepared_priority_cache if prepared_priority_cache is not None else extract_prepared_plan_priority(prepared_plan)
    priority_lookup_end_ns = time.monotonic_ns()
    source_logical_plan_hash = str(
        (prepared_priority_cache or {}).get("source_logical_plan_hash", _logical_plan_hash(prepared_plan))
    )
    hint = FutureDemandHint(
        hint_mode="calibrated_artifact",
        hint_digest=hint_digest,
        hint_source=f"prepared_window_plan:{prepared_plan.window_key}",
        metadata={
            "window_key": str(prepared_plan.window_key),
            "forecast_digest": forecast_digest,
            "source_logical_plan_hash": source_logical_plan_hash,
            "created_at_layer_id": str(prepared_plan.created_at_layer_id),
            "applies_from_layer_id": str(prepared_plan.applies_from_layer_id),
            "p2_matrix_source": str((prepared_priority_cache or {}).get("p2_matrix_source", "")),
            "p2_matrix_is_replicated_local_row": bool(
                (prepared_priority_cache or {}).get("p2_matrix_is_replicated_local_row", False)
            ),
            **priority_payload,
        },
    )
    context_replace_start_ns = time.monotonic_ns()
    hinted_local = replace(local_context, p2_hint=hint)
    context_replace_end_ns = time.monotonic_ns()
    policy = phase_policy
    if policy is None and str(policy_name) == "routersense_p0p1p2_hint":
        policy = build_phase_policy_fast_path(
            bucket_rows=bucket_rows,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
            p2_hint_weight=p2_hint_weight,
        )
    if policy is None:
        policy = resolve_phase_policy(
            policy_name=policy_name,
            bucket_rows=bucket_rows,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
            p2_hint_weight=p2_hint_weight,
        )
    phase_policy_build_start_ns = time.monotonic_ns()
    plan = policy.build_plan(local_context=hinted_local, global_contexts=global_contexts)
    phase_policy_build_end_ns = time.monotonic_ns()
    return replace(
        plan,
        metrics={
            **plan.metrics,
            "compiled_from_prepared_plan": True,
            "prepared_window_key": str(prepared_plan.window_key),
            "source_logical_plan_hash": source_logical_plan_hash,
            "forecast_digest": forecast_digest,
            "prepared_plan_order_preserved": bool(plan.metrics.get("ordered_by_prepared_plan", False)),
            "prepared_priority_extract_time_us": (priority_lookup_end_ns - priority_lookup_start_ns) / 1000.0,
            "prepared_context_replace_time_us": (context_replace_end_ns - context_replace_start_ns) / 1000.0,
            "prepared_phase_policy_build_time_us": (phase_policy_build_end_ns - phase_policy_build_start_ns) / 1000.0,
        },
    )


def get_or_build_prepared_priority_cache(
    *,
    shared_state: dict[str, Any],
    prepared_plan: PreparedWindowPlan,
) -> tuple[dict[str, Any], bool, float]:
    window_key = str(prepared_plan.window_key)
    cached = shared_state.get("prepared_priority_cache")
    if isinstance(cached, dict) and str(cached.get("window_key", "")) == window_key:
        return cached, True, 0.0
    build_start_ns = time.monotonic_ns()
    priority_payload = extract_prepared_plan_priority(prepared_plan)
    cache = {
        **priority_payload,
        "window_key": window_key,
        "forecast_digest": str(prepared_plan.forecast_digest),
        "source_logical_plan_hash": _logical_plan_hash(prepared_plan),
        "p2_matrix_source": str(shared_state.get("p2_matrix_source", "")),
        "p2_matrix_is_replicated_local_row": bool(shared_state.get("p2_matrix_is_replicated_local_row", False)),
        "priority_by_phase": {
            "P0": {
                (int(item.get("src_rank", -1)), int(item.get("dst_rank", -1))): int(item.get("priority", 0))
                for item in priority_payload.get("preferred_edges", [])
                if str(item.get("phase", "")) == "P0"
            },
            "P1": {
                (int(item.get("src_rank", -1)), int(item.get("dst_rank", -1))): int(item.get("priority", 0))
                for item in priority_payload.get("preferred_edges", [])
                if str(item.get("phase", "")) == "P1"
            },
        },
    }
    shared_state["prepared_priority_cache"] = cache
    build_end_ns = time.monotonic_ns()
    return cache, False, (build_end_ns - build_start_ns) / 1000.0


def build_phase_policy_fast_path(
    *,
    bucket_rows: int,
    p0_weight: float,
    p1_reservation_weight: float,
    p2_hint_weight: float,
) -> RouterSenseP0P1P2HintPolicy:
    return RouterSenseP0P1P2HintPolicy(
        bucket_rows=bucket_rows,
        p0_weight=p0_weight,
        p1_reservation_weight=p1_reservation_weight,
        p2_hint_weight=p2_hint_weight,
    )


def _logical_plan_hash(prepared_plan: PreparedWindowPlan) -> str:
    blob = json.dumps(prepared_plan.logical_plan.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


__all__ = [
    "build_phase_policy_fast_path",
    "compile_prepared_window_phase_plan",
    "get_or_build_prepared_priority_cache",
    "resolve_phase_policy",
    "supported_phase_policies",
]
