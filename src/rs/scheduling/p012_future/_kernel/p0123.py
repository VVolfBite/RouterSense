from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .artifacts import ForecastArtifact
from .contracts import ForecastPlanningRequest, P2RevealRequest
from .event_core import bind_template, plan_event_p0123
from .families import FAMILY_SPECS, GlobalOrderingUPlanner, build_scoped_planner
from .p01_aware import release_delay
from .plan import tuple_to_compact_plan


def _resources(request: ForecastPlanningRequest) -> dict:
    slope, intercept = request.cost_matrices()
    return {
        "edge_slope": slope,
        "edge_intercept": intercept,
        "expert_compute_delay": request.constraints.expert_compute_delay,
        "wave_launch_b": request.cost_model.wave_launch_b,
        "max_waves": request.constraints.max_waves,
    }


def _p3_rows(request: ForecastPlanningRequest) -> np.ndarray:
    """Derive the next-layer return matrix without introducing new truth.

    The derivation is deterministic from the P2 hint and therefore preserves
    the forecast/truth isolation contract.
    """
    p3 = np.ascontiguousarray(request.prediction_hint.matrix().T)
    np.fill_diagonal(p3, 0)
    p3.setflags(write=False)
    return p3


@dataclass
class EventP0123Planner:
    """P012 executable planning with advisory P3 return lookahead.

    The emitted template and bound plan still contain exactly P0/P1/P2.  P3 is
    used only inside the critical-tail geometry that ranks executable edges.
    """

    family: str

    def __post_init__(self) -> None:
        self.family = str(self.family).lower()
        self.spec = FAMILY_SPECS[self.family]
        self.planner_id = f"U_{self.family}_event_p0123"
        self.planner_family = "joint_u_event_p0123_advisory"

    def plan_forecast(self, request: ForecastPlanningRequest) -> ForecastArtifact:
        request.validate()
        p0, p1, hint = request.matrices()
        p3 = _p3_rows(request)
        p3_weight = float(request.prediction_hint.confidence)
        start = time.perf_counter_ns()
        template = plan_event_p0123(
            [p0, p1, hint],
            hint=hint,
            p3_hint=p3,
            p3_weight=p3_weight,
            scope="joint",
            full_truth_geometry=True,
            weights=self.spec.p012_runtime_weights(),
            **_resources(request),
        )
        planning_ms = (time.perf_counter_ns() - start) / 1e6
        metadata = {
            "planning_horizon": "p0123",
            "execution_horizon": "p012",
            "planning_timing": "on_demand",
            "p3_mode": "transpose_p2_hint",
            "p3_advisory_only": True,
            "p3_weight": p3_weight,
            "family_id": self.family,
            "weights": list(self.spec.p012_runtime_weights()),
            "planning_ms": planning_ms,
            "forecast_only": True,
            "predictor_id": request.prediction_hint.predictor_id,
            "prediction_confidence": float(request.prediction_hint.confidence),
            "topology": request.topology.to_dict(),
            "cost_model": request.cost_model.to_dict(),
            "constraints": request.constraints.to_dict(),
        }
        plan = tuple_to_compact_plan(
            template,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            branch="event_p0123",
            request_digest=request.semantic_digest(),
            forecast=True,
            metadata=metadata,
        )
        return ForecastArtifact(
            request.semantic_digest(),
            self.planner_id,
            self.planner_family,
            "event_p0123",
            template,
            plan,
            hint,
            metadata,
        )

    def bind(self, artifact: ForecastArtifact, request: ForecastPlanningRequest, reveal: P2RevealRequest):
        request.validate(); reveal.validate(world_size=request.topology.world_size)
        if reveal.request_id != request.request_id:
            raise ValueError("P2 reveal request_id mismatch")
        if artifact.request_digest != request.semantic_digest():
            raise ValueError("forecast artifact does not belong to this request")
        if reveal.forecast_request_digest != artifact.semantic_digest():
            raise ValueError("P2 reveal references another forecast artifact")
        p0, p1, _ = request.matrices()
        p2 = reveal.matrix(world_size=request.topology.world_size)
        start = time.perf_counter_ns()
        bound = bind_template(
            [p0, p1, p2],
            np.asarray(artifact.hint_rows, dtype=np.float64),
            artifact.raw_template,
            **_resources(request),
        )
        bind_ms = (time.perf_counter_ns() - start) / 1e6
        metadata = dict(artifact.metadata)
        metadata.update({
            "bound_from_forecast_digest": artifact.semantic_digest(),
            "truth_binding_only": True,
            "bind_ms_internal": bind_ms,
        })
        return tuple_to_compact_plan(
            bound,
            planner_id=artifact.planner_id,
            planner_family=artifact.planner_family,
            branch=artifact.branch,
            request_digest=request.semantic_digest(),
            forecast=False,
            metadata=metadata,
        )


@dataclass
class GlobalP0123Planner(EventP0123Planner):
    """Forecast-only selector between frozen P012 and P3-aware P012 templates."""

    margin: float = 0.0
    max_release_slack: float = 0.08

    def __post_init__(self) -> None:
        super().__post_init__()
        self.planner_id = f"U_{self.family}_global_ordering_p0123"
        self.planner_family = "joint_u_global_p0123_advisory"
        self._base = GlobalOrderingUPlanner(self.family)

    def plan_forecast(self, request: ForecastPlanningRequest) -> ForecastArtifact:
        request.validate()
        started = time.perf_counter_ns()
        base = self._base.plan_forecast(request)
        p3_event = super().plan_forecast(request)
        confidence = float(request.prediction_hint.confidence)
        dmax, dmean = release_delay(p3_event.raw_template, base.raw_template)
        improvement = (float(base.plan.makespan) - float(p3_event.plan.makespan)) / max(float(base.plan.makespan), 1e-12)
        release_slack = self.max_release_slack * confidence
        p3_eligible = bool(improvement >= self.margin and dmax <= release_slack)
        selected = p3_event if p3_eligible else base
        selected_name = "p0123_return_aware" if p3_eligible else "frozen_p012_guard"
        planning_ms = (time.perf_counter_ns() - started) / 1e6
        metadata = dict(selected.metadata)
        metadata.update({
            "planning_horizon": "p0123",
            "execution_horizon": "p012",
            "planning_timing": "on_demand",
            "p3_mode": "transpose_p2_hint",
            "p3_advisory_only": True,
            "selected_candidate": selected_name,
            "forecast_only_selection": True,
            "p0123_candidate_eligible": p3_eligible,
            "p0123_proxy_improvement": float(improvement),
            "p0123_release_delay_max": float(dmax),
            "p0123_release_delay_mean": float(dmean),
            "p0123_release_slack": float(release_slack),
            "candidate_proxy_makespans": {
                "frozen_p012": float(base.plan.makespan),
                "p0123_return_aware": float(p3_event.plan.makespan),
            },
            "planning_ms": float(planning_ms),
        })
        plan = tuple_to_compact_plan(
            selected.raw_template,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            branch="global_p0123",
            request_digest=request.semantic_digest(),
            forecast=True,
            metadata=metadata,
        )
        return ForecastArtifact(
            request.semantic_digest(), self.planner_id, self.planner_family,
            "global_p0123", selected.raw_template, plan,
            np.asarray(selected.hint_rows, dtype=np.int32), metadata,
        )



def warmup_p0123_kernel() -> None:
    matrix = np.array([[0, 3, 1], [2, 0, 1], [1, 2, 0]], dtype=np.int32)
    plan_event_p0123(
        [matrix, matrix.T.copy(), matrix], hint=matrix, p3_hint=matrix.T.copy(),
        p3_weight=0.75, scope="joint", full_truth_geometry=True,
    )


def build_p0123_planner(
    branch: str | None = None,
    family: str | None = None,
    *,
    scope: str | None = None,
    engine: str | None = None,
):
    """Build the P0123 wrapper with explicit scope/engine axes.

    Local P0123 is the strict paired baseline: P3 remains invisible because a
    phase-local planner cannot consume cross-phase advisory information.  It is
    therefore implemented by the same local P012 core/engine wrapper and
    labeled by the formal adapter as a P0123 local ablation.
    """
    if family is None:
        raise ValueError("family is required")
    if scope is None or engine is None:
        normalized = str(branch).lower()
        if normalized in {"event", "event_drive", "u_event"}:
            scope, engine = "joint", "event"
        elif normalized in {"global", "global_ordering", "global_selector", "u_global"}:
            scope, engine = "joint", "global"
        else:
            raise KeyError(f"unknown P0123 branch {branch!r}")
    scope = str(scope).lower()
    engine = str(engine).lower()
    if scope == "local":
        return build_scoped_planner(scope="local", engine=engine, family=family)
    if scope != "joint":
        raise ValueError(f"unsupported P0123 scope {scope!r}")
    if engine == "event":
        return EventP0123Planner(family)
    if engine == "global":
        return GlobalP0123Planner(family)
    raise ValueError(f"unsupported P0123 engine {engine!r}")


__all__ = ["EventP0123Planner", "GlobalP0123Planner", "build_p0123_planner", "warmup_p0123_kernel"]
