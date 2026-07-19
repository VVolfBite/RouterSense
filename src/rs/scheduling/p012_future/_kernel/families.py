from __future__ import annotations

"""Orthogonal scope/engine wrappers over the shared scheduling cores.

The authoritative core definitions live in :mod:`rs.scheduling.families.core`.
This module only applies the Local/Joint and Event/Global wrappers required by
P012/P0123/Future-P012.  It deliberately contains no second family-parameter
registry.
"""

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from rs.scheduling.families.core import FAMILY_KERNEL_SPECS, FamilyKernelSpec

from .artifacts import ForecastArtifact
from .axes import CORES, ENGINES, SCOPES, PlannerAxes
from .contracts import (
    ForecastPlanningRequest,
    P2RevealRequest,
    PlannerConstraints,
    TrafficHint,
)
from .event_core import bind_template, plan_event
from .global_planner import GlobalPlanSelector
from .plan import tuple_to_compact_plan


FAMILY_SPECS: dict[str, FamilyKernelSpec] = {
    family: FAMILY_KERNEL_SPECS[family] for family in CORES
}


def _weights(spec: FamilyKernelSpec) -> tuple[float, float, float, float]:
    return spec.p012_runtime_weights()


def _resource(request: ForecastPlanningRequest) -> dict[str, Any]:
    slope, intercept = request.cost_matrices()
    c = request.constraints
    return {
        "edge_slope": slope,
        "edge_intercept": intercept,
        "expert_compute_delay": c.expert_compute_delay,
        "wave_launch_b": request.cost_model.wave_launch_b,
        "max_waves": c.max_waves,
    }


def _merge_templates(parts: list[tuple], phase_ids: list[int], gaps: list[float]) -> tuple:
    if not parts:
        raise ValueError("parts must be non-empty")
    n = parts[0][2].shape[1]
    total_waves = sum(int(part[1]) for part in parts)
    phase = np.full((total_waves, n), -1, dtype=np.int8)
    dst = np.full((total_waves, n), -1, dtype=np.int16)
    size = np.zeros((total_waves, n), dtype=np.int32)
    quantum = np.zeros(total_waves, dtype=np.int32)
    duration = np.zeros(total_waves, dtype=np.float64)
    starts = np.zeros(total_waves, dtype=np.float64)
    ends = np.zeros(total_waves, dtype=np.float64)
    phase_done = np.zeros(3, dtype=np.float64)
    rel1 = np.full(n, -1.0, dtype=np.float64)
    rel2 = np.full(n, -1.0, dtype=np.float64)
    cursor = 0
    offset = 0.0
    for index, (part, phase_id) in enumerate(zip(parts, phase_ids, strict=True)):
        if index > 0:
            offset += float(gaps[index - 1])
        waves = int(part[1])
        for wave in range(waves):
            for source in range(n):
                if int(part[2][wave, source]) >= 0:
                    phase[cursor + wave, source] = int(phase_id)
                    dst[cursor + wave, source] = int(part[3][wave, source])
                    size[cursor + wave, source] = int(part[4][wave, source])
            quantum[cursor + wave] = int(part[5][wave])
            duration[cursor + wave] = float(part[6][wave])
            starts[cursor + wave] = offset + float(part[7][wave])
            ends[cursor + wave] = offset + float(part[8][wave])
        if waves:
            offset = float(ends[cursor + waves - 1])
            phase_done[phase_id] = offset
        cursor += waves
    makespan = offset
    # Local scope exposes the next phase only after the preceding phase has
    # completed in full.  This is the defining scope difference; the engine and
    # core remain unchanged.
    if len(parts) >= 1:
        rel1[:] = phase_done[0] + (gaps[0] if gaps else 0.0)
    if len(parts) >= 2:
        rel2[:] = phase_done[1]
    return (
        makespan,
        total_waves,
        phase,
        dst,
        size,
        quantum,
        duration,
        starts,
        ends,
        phase_done,
        rel1,
        rel2,
        1,
    )


def _zero_hint(request: ForecastPlanningRequest, *, suffix: str) -> TrafficHint:
    zero = np.zeros((request.topology.world_size, request.topology.world_size), dtype=np.int32)
    return TrafficHint(
        predictor_id=f"local_scope_zero:{suffix}",
        target_dispatch_rows=zero,
        confidence=0.0,
        hint_kind="zero_hint",
        oracle=False,
        matrix_kind="remote_rows",
        metadata={"scope": "local", "engine_adapter": suffix},
    )


class LocalScopedPlanner:
    """Phase-local planner with an explicit Event or Global engine wrapper."""

    def __init__(self, family: str, *, engine: str = "event", **global_kwargs: Any) -> None:
        family = str(family).lower()
        engine = str(engine).lower()
        if family not in FAMILY_SPECS:
            raise KeyError(f"unknown family {family!r}")
        if engine not in ENGINES:
            raise ValueError(f"unsupported local engine {engine!r}")
        self.spec = FAMILY_SPECS[family]
        self.engine = engine
        self.axes = PlannerAxes("current", "p012", "local", engine, family)
        self.planner_id = f"B_{family}_local_{engine}"
        self.planner_family = f"local_{engine}"
        self._global_selector = GlobalPlanSelector(
            planner_id=f"B_{family}_local_global_phase_selector",
            planner_family="local_global_phase_selector",
            weights=_weights(self.spec),
            **global_kwargs,
        )

    def _single_event(self, matrix: np.ndarray, request: ForecastPlanningRequest) -> tuple:
        zero = np.zeros_like(matrix)
        kwargs = _resource(request)
        kwargs["expert_compute_delay"] = 0.0
        return plan_event(
            [matrix, zero, zero],
            scope="local",
            weights=_weights(self.spec),
            **kwargs,
        )

    def _single_global(self, matrix: np.ndarray, request: ForecastPlanningRequest, *, phase: str) -> tuple:
        zero = np.zeros_like(matrix)
        phase_request = ForecastPlanningRequest(
            p0_dispatch_rows=matrix,
            p1_return_rows=zero,
            prediction_hint=_zero_hint(request, suffix=phase),
            topology=request.topology,
            cost_model=request.cost_model,
            constraints=PlannerConstraints(
                expert_compute_delay=0.0,
                max_waves=request.constraints.max_waves,
            ),
            request_id=f"{request.request_id}:local_global:{phase}",
        )
        # This invokes the same one-shot complete-plan selector used by the
        # Joint-Global branch.  With local visibility and a zero future hint it
        # intentionally selects only among phase-local candidates.
        return self._global_selector.plan_forecast(phase_request).raw_template

    def _single(self, matrix: np.ndarray, request: ForecastPlanningRequest, *, phase: str) -> tuple:
        if self.engine == "global":
            return self._single_global(matrix, request, phase=phase)
        return self._single_event(matrix, request)

    def plan_forecast(self, request: ForecastPlanningRequest) -> ForecastArtifact:
        request.validate()
        p0, p1, hint = request.matrices()
        start = time.perf_counter_ns()
        p0_plan = self._single(p0, request, phase="p0")
        p1_plan = self._single(p1, request, phase="p1")
        template = _merge_templates(
            [p0_plan, p1_plan],
            [0, 1],
            [request.constraints.expert_compute_delay],
        )
        elapsed = (time.perf_counter_ns() - start) / 1e6
        metadata = {
            "family_id": self.spec.family_id,
            "matching_core_id": self.spec.matching_core_id,
            "core_kernel_version": self.spec.kernel_version,
            "weights": list(_weights(self.spec)),
            "scope": "local",
            "engine": self.engine,
            "axes": self.axes.to_dict(),
            "planning_ms": elapsed,
            "p2_policy": "local_after_reveal",
            "prediction_consumed": False,
            "topology": request.topology.to_dict(),
            "cost_model": request.cost_model.to_dict(),
            "constraints": request.constraints.to_dict(),
        }
        branch = f"local_{self.engine}"
        plan = tuple_to_compact_plan(
            template,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            branch=branch,
            request_digest=request.semantic_digest(),
            forecast=True,
            metadata=metadata,
        )
        return ForecastArtifact(
            request.semantic_digest(),
            self.planner_id,
            self.planner_family,
            branch,
            template,
            plan,
            np.zeros_like(hint).tolist(),
            metadata,
        )

    def bind(self, artifact: ForecastArtifact, request: ForecastPlanningRequest, reveal: P2RevealRequest):
        request.validate()
        reveal.validate(world_size=request.topology.world_size)
        if reveal.request_id != request.request_id:
            raise ValueError("P2 reveal request_id mismatch")
        if artifact.request_digest != request.semantic_digest() or reveal.forecast_request_digest != artifact.semantic_digest():
            raise ValueError("forecast/reveal digest mismatch")
        p0, p1, _ = request.matrices()
        p2 = reveal.matrix(world_size=request.topology.world_size)
        p0_plan = self._single(p0, request, phase="p0")
        p1_plan = self._single(p1, request, phase="p1")
        p2_plan = self._single(p2, request, phase="p2")
        bound = _merge_templates(
            [p0_plan, p1_plan, p2_plan],
            [0, 1, 2],
            [request.constraints.expert_compute_delay, 0.0],
        )
        metadata = dict(artifact.metadata)
        metadata.update(
            {
                "truth_binding_only": True,
                "bound_from_forecast_digest": artifact.semantic_digest(),
            }
        )
        return tuple_to_compact_plan(
            bound,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            branch=f"local_{self.engine}",
            request_digest=request.semantic_digest(),
            forecast=False,
            metadata=metadata,
        )


class EventJointPlanner:
    def __init__(self, family: str, *, mode: str = "forecast") -> None:
        if mode not in {"forecast", "rank_hint"}:
            raise ValueError("mode must be forecast or rank_hint")
        family = str(family).lower()
        if family not in FAMILY_SPECS:
            raise KeyError(f"unknown family {family!r}")
        self.spec = FAMILY_SPECS[family]
        self.mode = mode
        self.axes = PlannerAxes("current", "p012", "joint", "event", family)
        self.planner_id = f"U_{family}_joint_event"
        self.planner_family = "joint_event"

    def plan_forecast(self, request: ForecastPlanningRequest) -> ForecastArtifact:
        request.validate()
        p0, p1, hint = request.matrices()
        zero = np.zeros_like(p0)
        start = time.perf_counter_ns()
        kwargs = _resource(request)
        if self.mode == "forecast":
            template = plan_event(
                [p0, p1, hint],
                hint=hint,
                scope="joint",
                full_truth_geometry=True,
                weights=_weights(self.spec),
                **kwargs,
            )
            binding_hint = hint
        else:
            template = plan_event(
                [p0, p1, zero],
                hint=hint,
                scope="joint",
                full_truth_geometry=False,
                weights=_weights(self.spec),
                **kwargs,
            )
            binding_hint = hint
        elapsed = (time.perf_counter_ns() - start) / 1e6
        metadata = {
            "family_id": self.spec.family_id,
            "matching_core_id": self.spec.matching_core_id,
            "core_kernel_version": self.spec.kernel_version,
            "weights": list(_weights(self.spec)),
            "scope": "joint",
            "engine": "event",
            "axes": self.axes.to_dict(),
            "event_mode": self.mode,
            "planning_ms": elapsed,
            "forecast_only": True,
            "topology": request.topology.to_dict(),
            "cost_model": request.cost_model.to_dict(),
            "constraints": request.constraints.to_dict(),
        }
        plan = tuple_to_compact_plan(
            template,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            branch="joint_event",
            request_digest=request.semantic_digest(),
            forecast=True,
            metadata=metadata,
        )
        return ForecastArtifact(
            request.semantic_digest(),
            self.planner_id,
            self.planner_family,
            "joint_event",
            template,
            plan,
            np.asarray(binding_hint, dtype=np.int32).tolist(),
            metadata,
        )

    def bind(self, artifact: ForecastArtifact, request: ForecastPlanningRequest, reveal: P2RevealRequest):
        request.validate()
        reveal.validate(world_size=request.topology.world_size)
        if reveal.request_id != request.request_id:
            raise ValueError("P2 reveal request_id mismatch")
        if artifact.request_digest != request.semantic_digest() or reveal.forecast_request_digest != artifact.semantic_digest():
            raise ValueError("forecast/reveal digest mismatch")
        p0, p1, _ = request.matrices()
        p2 = reveal.matrix(world_size=request.topology.world_size)
        kwargs = _resource(request)
        bound = bind_template(
            [p0, p1, p2],
            np.asarray(artifact.hint_rows, dtype=np.float64),
            artifact.raw_template,
            **kwargs,
        )
        metadata = dict(artifact.metadata)
        metadata.update(
            {
                "truth_binding_only": True,
                "bound_from_forecast_digest": artifact.semantic_digest(),
            }
        )
        return tuple_to_compact_plan(
            bound,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            branch="joint_event",
            request_digest=request.semantic_digest(),
            forecast=False,
            metadata=metadata,
        )


class GlobalJointPlanner(GlobalPlanSelector):
    def __init__(self, family: str, **kwargs: Any) -> None:
        family = str(family).lower()
        if family not in FAMILY_SPECS:
            raise KeyError(f"unknown family {family!r}")
        spec = FAMILY_SPECS[family]
        super().__init__(
            planner_id=f"U_{family}_joint_global",
            planner_family="joint_global",
            weights=_weights(spec),
            **kwargs,
        )
        self.spec = spec
        self.axes = PlannerAxes("current", "p012", "joint", "global", family)

    def plan_forecast(self, request: ForecastPlanningRequest) -> ForecastArtifact:
        artifact = super().plan_forecast(request)
        metadata = dict(artifact.metadata)
        metadata.update(
            {
                "family_id": self.spec.family_id,
                "matching_core_id": self.spec.matching_core_id,
                "core_kernel_version": self.spec.kernel_version,
                "scope": "joint",
                "engine": "global",
                "axes": self.axes.to_dict(),
            }
        )
        plan = tuple_to_compact_plan(
            artifact.raw_template,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            branch="joint_global",
            request_digest=request.semantic_digest(),
            forecast=True,
            metadata=metadata,
        )
        return ForecastArtifact(
            artifact.request_digest,
            self.planner_id,
            self.planner_family,
            "joint_global",
            artifact.raw_template,
            plan,
            artifact.hint_rows,
            metadata,
        )


# Compatibility names used by the recovered runtime and old reports.
LocalBPlanner = LocalScopedPlanner
EventUPlanner = EventJointPlanner
GlobalOrderingUPlanner = GlobalJointPlanner


def build_scoped_planner(*, scope: str, engine: str, family: str, **kwargs: Any):
    normalized_scope = str(scope).lower()
    normalized_engine = str(engine).lower()
    if normalized_scope not in SCOPES:
        raise ValueError(f"unsupported planner scope {scope!r}")
    if normalized_engine not in ENGINES:
        raise ValueError(f"unsupported planner engine {engine!r}")
    if normalized_scope == "local":
        return LocalScopedPlanner(family, engine=normalized_engine, **kwargs)
    if normalized_engine == "event":
        return EventJointPlanner(family, **kwargs)
    return GlobalJointPlanner(family, **kwargs)


__all__ = [
    "EventJointPlanner",
    "EventUPlanner",
    "FAMILY_SPECS",
    "GlobalJointPlanner",
    "GlobalOrderingUPlanner",
    "LocalBPlanner",
    "LocalScopedPlanner",
    "build_scoped_planner",
]
