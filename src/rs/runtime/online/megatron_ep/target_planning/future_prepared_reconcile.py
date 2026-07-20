from __future__ import annotations

"""Thin target-side wrapper for Future-P012 prepared-order plans.

The only caller is the existing ``reconcile_once`` authority.  Store, token,
deadline, fallback, materialization, and execution ownership remain unchanged.
"""

from typing import Mapping

from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave
from rs.scheduling.validation import stable_hash
from rs.scheduling.p012_future.prepared_binding import bind_prepared_order_payload
from .contracts import MatrixRows, ReconciledExecutionPlan, TargetLayerPreparedJointPlan


def is_future_prepared_plan(prepared_plan: TargetLayerPreparedJointPlan) -> bool:
    window = prepared_plan.window_plan
    return bool(
        window is not None
        and isinstance(getattr(window, "metadata", None), Mapping)
        and isinstance(window.metadata.get("future_prepared_order"), Mapping)
    )


def reconcile_future_prepared_order(
    *,
    prepared_plan: TargetLayerPreparedJointPlan,
    actual_p0_rows: MatrixRows,
    frozen_frontier: set[str] | None = None,
) -> ReconciledExecutionPlan:
    """Bind actual P0/P1 or reject to the existing fallback owner."""
    try:
        if not is_future_prepared_plan(prepared_plan):
            raise ValueError("plan does not carry a Future prepared-order payload")
        payload = dict(prepared_plan.window_plan.metadata["future_prepared_order"])
        decision = bind_prepared_order_payload(
            payload=payload,
            predicted_p0_rows=prepared_plan.h1_rows,
            actual_p0_rows=actual_p0_rows,
            planner_id=str(prepared_plan.policy),
            request_digest=str(prepared_plan.target_problem_digest),
        )
        details = {
            **dict(decision.metrics),
            "future_prepared_order": True,
            "frozen_frontier": sorted(frozen_frontier or ()),
        }
        if not decision.accepted:
            return ReconciledExecutionPlan(
                status="rejected",
                logical_plan=None,
                logical_plan_digest=None,
                preserved_edge_ratio=float(
                    dict(decision.metrics.get("future_gate_metrics", {}))
                    .get("p0", {})
                    .get("support_recall", 0.0)
                ),
                inserted_edge_count=0,
                removed_edge_count=0,
                resized_edge_count=0,
                repair_us=float(decision.elapsed_us),
                details=details,
            )
        if decision.exact:
            return ReconciledExecutionPlan(
                status="exact",
                logical_plan=prepared_plan.logical_plan,
                logical_plan_digest=str(prepared_plan.logical_plan_digest),
                preserved_edge_ratio=1.0,
                inserted_edge_count=0,
                removed_edge_count=0,
                resized_edge_count=0,
                repair_us=float(decision.elapsed_us),
                details=details,
            )
        logical = LogicalSchedulePlan(
            policy_name=str(prepared_plan.policy),
            waves=tuple(
                LogicalWave(
                    wave_id=int(wave.wave_id),
                    flows=tuple(
                        FlowDemand(
                            flow_id=str(flow.flow_id),
                            phase=str(flow.phase),
                            src_rank=int(flow.src_rank),
                            dst_rank=int(flow.dst_rank),
                            byte_count=int(flow.row_count),
                            release_state=str(flow.release_state),
                            is_executable=bool(flow.executable),
                            dependency_metadata={"future_prepared_bind": True},
                        )
                        for flow in wave.flows
                    ),
                    duration=float(wave.estimated_duration),
                )
                for wave in decision.waves
            ),
            diagnostics=details,
        )
        digest = str(stable_hash(logical.to_dict()))
        return ReconciledExecutionPlan(
            status="repaired",
            logical_plan=logical,
            logical_plan_digest=digest,
            preserved_edge_ratio=float(decision.metrics.get("template_support_coverage", 1.0)),
            inserted_edge_count=0,
            removed_edge_count=0,
            resized_edge_count=int(decision.metrics.get("resized_edges", 0)),
            repair_us=float(decision.elapsed_us),
            details=details,
        )
    except Exception as exc:
        return ReconciledExecutionPlan(
            status="rejected",
            logical_plan=None,
            logical_plan_digest=None,
            preserved_edge_ratio=0.0,
            inserted_edge_count=0,
            removed_edge_count=0,
            resized_edge_count=0,
            repair_us=0.0,
            details={
                "reason": "future_prepared_payload_invalid",
                "error": f"{type(exc).__name__}: {exc}",
                "frozen_frontier": sorted(frozen_frontier or ()),
            },
        )


__all__ = ["is_future_prepared_plan", "reconcile_future_prepared_order"]
