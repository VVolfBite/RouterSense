"""Policy adapters for the formal online runtime path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from rs.runtime.online.megatron_ep.p2_provider import extract_prepared_plan_priority
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
) -> PhaseExecutionPlan:
    """Compile the current phase from a prepared logical window plan.

    The prepared plan supplies logical edge priority. Current phase layouts
    supply tensor offsets and payload slices; future P2 forecast edges are not
    executable and are ignored by the phase-local compiler.
    """

    forecast_digest = str(prepared_plan.forecast_digest)
    hint_digest = hashlib.sha256(f"{forecast_digest}:{local_context.layer_id}".encode("utf-8")).hexdigest()[:16]
    priority_payload = extract_prepared_plan_priority(prepared_plan)
    hint = FutureDemandHint(
        hint_mode="calibrated_artifact",
        hint_digest=hint_digest,
        hint_source=f"prepared_window_plan:{prepared_plan.window_key}",
        metadata={
            "window_key": str(prepared_plan.window_key),
            "forecast_digest": forecast_digest,
            "source_logical_plan_hash": _logical_plan_hash(prepared_plan),
            "created_at_layer_id": str(prepared_plan.created_at_layer_id),
            "applies_from_layer_id": str(prepared_plan.applies_from_layer_id),
            **priority_payload,
        },
    )
    hinted_contexts = tuple(
        replace(context, p2_hint=hint) if context.phase == local_context.phase else context
        for context in global_contexts
    )
    hinted_local = replace(local_context, p2_hint=hint)
    policy = resolve_phase_policy(
        policy_name=policy_name,
        bucket_rows=bucket_rows,
        p0_weight=p0_weight,
        p1_reservation_weight=p1_reservation_weight,
        p2_hint_weight=p2_hint_weight,
    )
    plan = policy.build_plan(local_context=hinted_local, global_contexts=hinted_contexts)
    return replace(
        plan,
        metrics={
            **plan.metrics,
            "compiled_from_prepared_plan": True,
            "prepared_window_key": str(prepared_plan.window_key),
            "source_logical_plan_hash": _logical_plan_hash(prepared_plan),
            "forecast_digest": forecast_digest,
            "prepared_plan_order_preserved": bool(plan.metrics.get("ordered_by_prepared_plan", False)),
        },
    )


def _logical_plan_hash(prepared_plan: PreparedWindowPlan) -> str:
    blob = json.dumps(prepared_plan.logical_plan.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


__all__ = ["compile_prepared_window_phase_plan", "resolve_phase_policy", "supported_phase_policies"]
