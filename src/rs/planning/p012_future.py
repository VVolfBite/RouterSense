from __future__ import annotations

"""Formal adapters for the migrated P012 and Future-P012 algorithm assets.

The migrated algorithm kernel is private to ``rs.scheduling.p012_future``.
This module is the only public planning boundary: both offline and online
callers submit the canonical :class:`rs.core.contracts.PlanningRequest` and
receive the canonical :class:`rs.core.contracts.WindowPlan`.
"""

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np

from rs.core.contracts import (
    PlanWave,
    PlannedFlow,
    PlanningRequest,
    PredictionHint,
    WindowPlan,
)
from rs.planning.api import PlannerSpec
from rs.scheduling.p012_future._kernel.axes import (
    CORES as P012_CORES,
    ENGINES as P012_ENGINES,
    HORIZONS as P012_HORIZONS,
    SCOPES as P012_SCOPES,
    TIMINGS as P012_TIMINGS,
    PlannerAxes,
    parse_planner_axes,
    planner_axis_matrix,
)
from rs.scheduling.p012_future._kernel.contracts import (
    AffineLinkCost,
    ForecastPlanningRequest as KernelPlanningRequest,
    HomogeneousTopology,
    PlannerConstraints as KernelConstraints,
    TrafficHint as KernelTrafficHint,
)
from rs.scheduling.p012_future._kernel.future import (
    _compile_prepared_order as compile_kernel_prepared_order,
)
from rs.scheduling.p012_future._kernel.p0123 import build_p0123_planner
from rs.scheduling.p012_future._kernel.plan import (
    CompactWindowPlan as KernelCompactWindowPlan,
    WindowPlan as KernelWindowPlan,
    tuple_to_compact_plan,
)
from rs.scheduling.p012_future._kernel.registry import build_planner


_DEPLOYABLE_BRANCHES = {"local", "event", "global"}  # legacy three-part IDs
_DEPLOYABLE_FAMILIES = set(P012_CORES)

_KERNEL_TO_FORMAL_RELEASE_STATE = {
    "barrier_released": "blocked",
}


def _formal_release_state(value: object) -> str:
    """Normalize private-kernel release labels to the canonical planning contract.

    The P012 kernel labels P1 traffic as ``barrier_released`` to describe the
    event that eventually unlocks it.  At publication time the canonical
    runtime contract represents the same traffic as ``blocked``; the existing
    release ledger later transitions it when expert compute completes.
    """
    normalized = str(value)
    return _KERNEL_TO_FORMAL_RELEASE_STATE.get(normalized, normalized)


def _matrix(value: object, *, name: str, world_size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (int(world_size), int(world_size)):
        raise ValueError(f"{name} shape {array.shape} != ({world_size}, {world_size})")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=0.0, rtol=0.0) or (rounded < 0).any():
        raise ValueError(f"{name} must contain non-negative integral row counts")
    result = np.ascontiguousarray(rounded.astype(np.int32, copy=False)).copy()
    np.fill_diagonal(result, 0)
    return result



def _layer_number(value: str | None, *, fallback: int = 0) -> int:
    if value is None:
        return int(fallback)
    matches = re.findall(r"\d+", str(value))
    return int(matches[-1]) if matches else int(fallback)


def _kernel_hint(hint: PredictionHint, *, world_size: int, runtime_usage: bool) -> KernelTrafficHint:
    hint.validate(world_size=world_size)
    if runtime_usage and bool(hint.oracle):
        raise ValueError("oracle/perfect-trace hints are forbidden in deployable runtime planning")
    rows = _matrix(hint.target_dispatch_rows, name="prediction_hint.target_dispatch_rows", world_size=world_size)
    normalized = str(hint.hint_type).lower()
    if bool(hint.oracle):
        kind = "perfect_trace_hint"
    elif not rows.any():
        kind = "zero_hint"
    elif "copy" in normalized:
        kind = "copy_current_dispatch"
    elif "expert" in normalized or "fate" in normalized:
        kind = "expert_route"
    else:
        kind = "learned_prediction"
    confidence = 1.0 if hint.oracle else (0.0 if kind == "zero_hint" else float(hint.confidence or 0.0))
    return KernelTrafficHint(
        predictor_id=str(hint.predictor_id),
        target_dispatch_rows=rows,
        confidence=confidence,
        hint_kind=kind,
        oracle=bool(hint.oracle),
        matrix_kind="remote_rows",
        metadata={
            "formal_hint_type": str(hint.hint_type),
            "source_layer_id": hint.source_layer_id,
            "target_layer_id": hint.target_layer_id,
        },
    )



def _cost_matrix(value: object, *, name: str, world_size: int, strictly_positive: bool) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (int(world_size), int(world_size)):
        raise ValueError(f"{name} shape {array.shape} != ({world_size}, {world_size})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    if strictly_positive and (array <= 0.0).any():
        raise ValueError(f"{name} must be strictly positive")
    if not strictly_positive and (array < 0.0).any():
        raise ValueError(f"{name} must be non-negative")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _PairwiseAffineCost:
    slope: np.ndarray
    intercept: np.ndarray
    wave_launch_b: float
    profile_id: str

    def validate(self) -> None:
        if self.slope.shape != self.intercept.shape or self.slope.ndim != 2:
            raise ValueError("pairwise cost matrices must be matching squares")
        if not np.isfinite(self.slope).all() or not np.isfinite(self.intercept).all():
            raise ValueError("pairwise cost matrices must be finite")
        if (self.slope <= 0.0).any() or (self.intercept < 0.0).any():
            raise ValueError("pairwise slopes must be positive and intercepts non-negative")
        if not np.isfinite(float(self.wave_launch_b)) or float(self.wave_launch_b) < 0.0:
            raise ValueError("wave_launch_b must be finite and non-negative")

    def matrices(self, topology: HomogeneousTopology) -> tuple[np.ndarray, np.ndarray]:
        topology.validate(); self.validate()
        if self.slope.shape != (int(topology.world_size), int(topology.world_size)):
            raise ValueError("pairwise profile world size does not match topology")
        return self.slope, self.intercept

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "cost_semantic_version": "measured_pairwise_affine_v1",
            "profile_id": str(self.profile_id),
            "wave_launch_b": float(self.wave_launch_b),
            "slope_ms_per_row": self.slope.tolist(),
            "intercept_ms": self.intercept.tolist(),
        }

def stable_formal_weights_digest(request: PlanningRequest) -> str:
    """Record, but do not reinterpret, the canonical formal weight contract.

    The migrated algorithm families have frozen family-specific coefficients.
    Silently mapping unrelated formal knobs would change their validated math, so
    the adapter records the formal weight payload and labels the fixed mode.
    """
    import hashlib
    import json
    payload = request.weights.to_dict() if hasattr(request.weights, "to_dict") else vars(request.weights)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class P012AdapterConfig:
    # Explicit orthogonal axes. ``branch`` is retained only for old three-part
    # configuration payloads and overrides scope/engine when provided.
    timing: str = "current"
    horizon: str = "p012"
    scope: str = "joint"
    engine: str = "global"
    family: str = "rscf"
    branch: str | None = None
    public_planner_id: str = ""
    runtime_usage: bool = True
    ranks_per_node: int | None = None
    rank_to_node: tuple[int, ...] | None = None
    intra_k: float = 1.0
    intra_b: float = 0.0
    inter_k: float = 1.0
    inter_b: float = 0.0
    wave_launch_b: float = 0.0
    edge_slope: object | None = None
    edge_intercept: object | None = None
    cost_profile_id: str = "formal_adapter"
    max_p0_relative_l1: float = 0.12
    max_p1_relative_l1: float = 0.12
    min_p0_support_recall: float = 0.98
    min_p0_support_precision: float = 0.80

    def planner_axes(self) -> PlannerAxes:
        if self.branch is not None:
            normalized = str(self.branch).lower()
            if normalized == "local":
                scope, engine = "local", "event"
            elif normalized in {"event", "global"}:
                scope, engine = "joint", normalized
            else:
                raise ValueError(f"unsupported legacy P012 branch {self.branch!r}")
        else:
            scope, engine = str(self.scope).lower(), str(self.engine).lower()
        return PlannerAxes(
            timing=str(self.timing),
            horizon=str(self.horizon),
            scope=scope,
            engine=engine,
            core=str(self.family),
        )

    def validate(self) -> None:
        axes = self.planner_axes()
        if axes.timing not in P012_TIMINGS or axes.horizon not in P012_HORIZONS:
            raise ValueError("invalid P012 planner timing/horizon")
        if axes.scope not in P012_SCOPES or axes.engine not in P012_ENGINES:
            raise ValueError("invalid P012 planner scope/engine")
        if axes.core not in _DEPLOYABLE_FAMILIES:
            raise ValueError(f"unsupported P012 family {self.family!r}")
        if self.public_planner_id and parse_planner_axes(self.public_planner_id) != axes:
            raise ValueError("public_planner_id does not match configured planner axes")
        if self.ranks_per_node is not None and int(self.ranks_per_node) <= 0:
            raise ValueError("ranks_per_node must be positive when provided")
        if self.rank_to_node is not None:
            mapping = tuple(int(value) for value in self.rank_to_node)
            if not mapping or min(mapping) < 0:
                raise ValueError("rank_to_node must contain non-negative node ids")
        for name, value in (
            ("intra_k", self.intra_k), ("intra_b", self.intra_b),
            ("inter_k", self.inter_k), ("inter_b", self.inter_b),
            ("wave_launch_b", self.wave_launch_b),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(self.intra_k) <= 0.0 or float(self.inter_k) <= 0.0:
            raise ValueError("intra_k and inter_k must be positive")
        if (self.edge_slope is None) != (self.edge_intercept is None):
            raise ValueError("edge_slope and edge_intercept must be provided together")
        if not str(self.cost_profile_id):
            raise ValueError("cost_profile_id must be non-empty")
        if float(self.max_p0_relative_l1) < 0.0 or float(self.max_p1_relative_l1) < 0.0:
            raise ValueError("Future-P012 relative-L1 thresholds must be non-negative")
        for name, value in (
            ("min_p0_support_recall", self.min_p0_support_recall),
            ("min_p0_support_precision", self.min_p0_support_precision),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")



class P012FormalPlanner:
    """Canonical Planner implementation shared by offline and online callers."""

    def __init__(self, config: P012AdapterConfig | None = None) -> None:
        self.config = config or P012AdapterConfig()
        self.config.validate()
        self.axes = self.config.planner_axes()
        if self.axes.timing != "current" or self.axes.horizon != "p012":
            raise ValueError("P012FormalPlanner requires current/P012 axes")
        self._kernel_planner = build_planner(axes=self.axes)

    @property
    def planner_id(self) -> str:
        return str(self.config.public_planner_id or self.axes.canonical_id)

    @property
    def planner_family(self) -> str:
        return self.axes.scope

    def _kernel_request(self, request: PlanningRequest) -> KernelPlanningRequest:
        request.validate()
        world = int(request.topology.world_size)
        if not bool(getattr(request.topology, "full_duplex", False)):
            raise ValueError("P012 runtime requires full-duplex send/receive semantics")
        if int(getattr(request.topology, "max_outgoing_per_rank_per_wave", 0)) != 1:
            raise ValueError("P012 runtime requires exactly one outgoing transfer per rank per wave")
        if int(getattr(request.topology, "max_incoming_per_rank_per_wave", 0)) != 1:
            raise ValueError("P012 runtime requires exactly one incoming transfer per rank per wave")
        if self.config.rank_to_node is not None and len(self.config.rank_to_node) != world:
            raise ValueError("rank_to_node length must equal PlanningTopology.world_size")
        p0 = _matrix(request.traffic.p0_dispatch_rows, name="p0_dispatch_rows", world_size=world)
        p1 = _matrix(request.traffic.p1_return_rows, name="p1_return_rows", world_size=world)
        hint = _kernel_hint(request.prediction_hint, world_size=world, runtime_usage=self.config.runtime_usage)
        topology = HomogeneousTopology(
            world_size=world,
            ranks_per_node=int(self.config.ranks_per_node or world),
            rank_to_node=(
                None if self.config.rank_to_node is None
                else tuple(int(value) for value in self.config.rank_to_node)
            ),
        )
        topology.validate()
        if self.config.edge_slope is not None:
            cost = _PairwiseAffineCost(
                slope=_cost_matrix(
                    self.config.edge_slope, name="edge_slope", world_size=world, strictly_positive=True
                ),
                intercept=_cost_matrix(
                    self.config.edge_intercept, name="edge_intercept", world_size=world, strictly_positive=False
                ),
                wave_launch_b=float(self.config.wave_launch_b),
                profile_id=str(self.config.cost_profile_id),
            )
        else:
            cost = AffineLinkCost(
                intra_k=float(self.config.intra_k), intra_b=float(self.config.intra_b),
                inter_k=float(self.config.inter_k), inter_b=float(self.config.inter_b),
                wave_launch_b=float(self.config.wave_launch_b),
            )
        constraints = KernelConstraints(
            expert_compute_delay=float(request.constraints.expert_compute_delay),
            max_waves=int(request.constraints.max_waves),
        )
        return KernelPlanningRequest(
            p0_dispatch_rows=p0,
            p1_return_rows=p1,
            prediction_hint=hint,
            topology=topology,
            cost_model=cost,
            constraints=constraints,
            request_id=str(request.identity.request_id),
        )

    def plan(self, request: PlanningRequest) -> WindowPlan:
        kernel_request = self._kernel_request(request)
        artifact = self._kernel_planner.plan_forecast(kernel_request)
        return _to_formal_plan(
            artifact.plan,
            request=request,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            extra_metadata={
                "p012_adapter_version": "formal_p012_adapter_v1",
                "scope": self.axes.scope,
                "engine": self.axes.engine,
                "family": self.axes.core,
                "planner_axes": self.axes.to_dict(),
                "kernel_artifact_digest": artifact.semantic_digest(),
                "runtime_usage": bool(self.config.runtime_usage),
                "topology_mode": (
                    "explicit_rank_to_node" if self.config.rank_to_node is not None
                    else "contiguous_homogeneous"
                ),
                "policy_weight_mode": "family_fixed_v1",
                "formal_weights_digest": stable_formal_weights_digest(request),
                "cost_profile_mode": (
                    "measured_pairwise" if self.config.edge_slope is not None
                    else "homogeneous_affine"
                ),
                "cost_profile_id": str(self.config.cost_profile_id),
            },
        )


def _encode_prepared_order_payload(prepared, kernel_request: KernelPlanningRequest, config: P012AdapterConfig) -> dict[str, object]:
    """Serialize the solver-free P01 order into canonical plan metadata.

    This payload is advisory state carried by the existing ``WindowPlan`` and
    ``TargetLayerPreparedJointPlan``.  It is not a second mailbox and contains
    no executable predicted bytes.
    """
    phase = np.asarray(prepared.p01_raw_template[2], dtype=np.int8)
    destination = np.asarray(prepared.p01_raw_template[3], dtype=np.int16)
    size = np.asarray(prepared.p01_raw_template[4], dtype=np.int32)
    if phase.shape != destination.shape or phase.shape != size.shape:
        raise ValueError("prepared P01 template arrays must share one shape")
    slope, intercept = kernel_request.cost_matrices()
    return {
        "semantic_version": "future_prepared_order_payload_v1",
        "world_size": int(kernel_request.topology.world_size),
        "phase": phase.tolist(),
        "destination": destination.tolist(),
        "predicted_size": size.tolist(),
        "edge_slope": np.asarray(slope, dtype=np.float64).tolist(),
        "edge_intercept": np.asarray(intercept, dtype=np.float64).tolist(),
        "expert_compute_delay": float(kernel_request.constraints.expert_compute_delay),
        "wave_launch_b": float(kernel_request.cost_model.wave_launch_b),
        "max_waves": int(kernel_request.constraints.max_waves),
        "max_p0_relative_l1": float(config.max_p0_relative_l1),
        "max_p1_relative_l1": float(config.max_p1_relative_l1),
        "min_p0_support_recall": float(config.min_p0_support_recall),
        "min_p0_support_precision": float(config.min_p0_support_precision),
        "prepared_order_digest": prepared.semantic_digest(),
        "online_matching_solver": False,
        "online_candidate_selection": False,
    }


class FuturePreparedFormalPlanner(P012FormalPlanner):
    """Production Future-P012 planner for the existing target-planning service.

    The existing asynchronous ``TargetLayerPlannerService`` supplies predicted
    target P0/P1 plus advisory P2 through the canonical ``PlanningRequest``.
    This planner runs one P012 search before the target frontier, stores a
    compact prepared-order payload in the returned canonical ``WindowPlan``,
    and relies on the existing ``reconcile_once`` entry for target binding.
    """

    def __init__(self, config: P012AdapterConfig | None = None) -> None:
        self.config = config or P012AdapterConfig(timing="future", horizon="p012")
        self.config.validate()
        self.axes = self.config.planner_axes()
        if self.axes.timing != "future" or self.axes.horizon != "p012":
            raise ValueError("FuturePreparedFormalPlanner requires future/P012 axes")
        if not self.config.runtime_usage:
            raise ValueError("Future-P012 is a runtime-safe formal planner")
        # Timing is an outer wrapper. The exact same scope/engine/core kernel is
        # executed early and its prepared order is reconciled at target entry.
        self._kernel_planner = build_planner(
            scope=self.axes.scope,
            engine=self.axes.engine,
            family=self.axes.core,
        )

    @property
    def planner_id(self) -> str:
        return str(self.config.public_planner_id or self.axes.canonical_id)

    @property
    def planner_family(self) -> str:
        return self.axes.scope

    def plan(self, request: PlanningRequest) -> WindowPlan:
        kernel_request = self._kernel_request(request)
        artifact = self._kernel_planner.plan_forecast(kernel_request)
        prepared = compile_kernel_prepared_order(artifact, request.topology.world_size)
        payload = _encode_prepared_order_payload(prepared, kernel_request, self.config)
        return _to_formal_plan(
            artifact.plan,
            request=request,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            extra_metadata={
                "future_prepared_adapter_version": "formal_future_prepared_v1",
                "planning_horizon": "p012",
                "execution_horizon": "p01_until_truth_reveal",
                "planning_timing": "previous_layer",
                "future_prepared_order": payload,
                "future_prepared_order_digest": prepared.semantic_digest(),
                "target_bind_entry": "target_planning.reconcile_once",
                "uses_existing_target_plan_store": True,
                "scope": self.axes.scope,
                "engine": self.axes.engine,
                "family": self.axes.core,
                "planner_axes": self.axes.to_dict(),
                "kernel_artifact_digest": artifact.semantic_digest(),
                "topology_mode": (
                    "explicit_rank_to_node" if self.config.rank_to_node is not None
                    else "contiguous_homogeneous"
                ),
                "cost_profile_mode": (
                    "measured_pairwise" if self.config.edge_slope is not None
                    else "homogeneous_affine"
                ),
                "cost_profile_id": str(self.config.cost_profile_id),
            },
        )


class P0123FormalPlanner(P012FormalPlanner):
    """Online-compatible advisory P0123 ablation.

    P3 is derived only from the causal P2 prediction and is advisory.  The
    returned executable contract still contains P0/P1 plus non-executable
    forecast P2; no P3 flow is ever published or materialized.
    """

    def __init__(self, config: P012AdapterConfig | None = None) -> None:
        self.config = config or P012AdapterConfig(timing="current", horizon="p0123")
        self.config.validate()
        self.axes = self.config.planner_axes()
        if self.axes.timing != "current" or self.axes.horizon != "p0123":
            raise ValueError("P0123FormalPlanner requires current/P0123 axes")
        self._kernel_planner = build_p0123_planner(
            scope=self.axes.scope,
            engine=self.axes.engine,
            family=self.axes.core,
        )

    @property
    def planner_id(self) -> str:
        return str(self.config.public_planner_id or self.axes.canonical_id)

    @property
    def planner_family(self) -> str:
        return self.axes.scope

    def plan(self, request: PlanningRequest) -> WindowPlan:
        kernel_request = self._kernel_request(request)
        artifact = self._kernel_planner.plan_forecast(kernel_request)
        return _to_formal_plan(
            artifact.plan,
            request=request,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            extra_metadata={
                "p0123_adapter_version": "formal_p0123_advisory_adapter_v1",
                "planning_horizon": "p0123",
                "execution_horizon": "p012",
                "p3_advisory_only": True,
                "ablation_only": True,
                "scope": self.axes.scope,
                "engine": self.axes.engine,
                "family": self.axes.core,
                "planner_axes": self.axes.to_dict(),
                "p3_effective": bool(self.axes.scope == "joint"),
                "kernel_artifact_digest": artifact.semantic_digest(),
                "runtime_usage": bool(self.config.runtime_usage),
                "topology_mode": (
                    "explicit_rank_to_node" if self.config.rank_to_node is not None
                    else "contiguous_homogeneous"
                ),
                "cost_profile_mode": (
                    "measured_pairwise" if self.config.edge_slope is not None
                    else "homogeneous_affine"
                ),
                "cost_profile_id": str(self.config.cost_profile_id),
            },
        )


def _to_formal_plan(
    kernel_plan: KernelCompactWindowPlan | KernelWindowPlan,
    *,
    request: PlanningRequest,
    planner_id: str,
    planner_family: str,
    extra_metadata: Mapping[str, object] | None = None,
) -> WindowPlan:
    materialized = kernel_plan.materialize() if isinstance(kernel_plan, KernelCompactWindowPlan) else kernel_plan
    if not bool(materialized.valid):
        raise ValueError("kernel planner returned an invalid plan")
    waves: list[PlanWave] = []
    for wave in materialized.waves:
        flows = tuple(
            PlannedFlow(
                flow_id=str(flow.segment_id),
                phase=str(flow.phase),
                src_rank=int(flow.src_rank),
                dst_rank=int(flow.dst_rank),
                row_count=int(flow.row_count),
                release_state=_formal_release_state(flow.release_state),
                executable=bool(flow.executable),
            )
            for flow in wave.flows
        )
        waves.append(
            PlanWave(
                wave_id=int(wave.wave_id),
                flows=flows,
                estimated_duration=float(wave.estimated_duration),
            )
        )
    metadata = {
        "formal_information_mode": str(getattr(request, "information_mode", "")),
        "formal_weights_digest": stable_formal_weights_digest(request),
        "policy_weight_mode": "family_fixed_v1",
        "kernel_plan_digest": materialized.semantic_digest(),
        "kernel_planner_id": str(materialized.planner_id),
        "kernel_planner_family": str(materialized.planner_family),
        "kernel_branch": str(materialized.branch),
        "kernel_forecast": bool(materialized.forecast),
        "kernel_makespan": float(materialized.makespan),
        "kernel_phase_completion": list(materialized.phase_completion),
        "kernel_release1": list(materialized.release1),
        "kernel_release2": list(materialized.release2),
        **dict(extra_metadata or {}),
    }
    formal = WindowPlan(
        planner_id=str(planner_id),
        planner_family=str(planner_family),
        request_digest=request.semantic_digest(),
        waves=tuple(waves),
        metadata=metadata,
    )
    formal.validate()
    # Formal execution must never receive predicted P2 as executable payload.
    for wave in formal.waves:
        for flow in wave.flows:
            if flow.phase == "p2_next_dispatch_forecast" and flow.executable:
                raise ValueError("forecast P2 must be advisory/non-executable")
    return formal


# Legacy three-part IDs remain first-class compatibility entries so existing
# configs and evidence bundles remain reproducible. New experiments should use
# the explicit five-axis IDs below.
_LEGACY_P012_SPECS = tuple(
    PlannerSpec(
        planner_id=f"p012:{branch}:{family}",
        planner_family="local" if branch == "local" else "joint",
        deployable=True,
        reference_only=False,
        requires_prediction=branch != "local",
        exact=False,
        historical_aliases=(
            (f"B_{family}_local", f"B_{family}_local_event")
            if branch == "local"
            else (
                (f"U_{family}_event", f"U_{family}_joint_event")
                if branch == "event"
                else (f"U_{family}_global_ordering", f"U_{family}_joint_global")
            )
        ),
    )
    for branch in ("local", "event", "global")
    for family in P012_CORES
)

_LEGACY_P0123_SPECS = tuple(
    PlannerSpec(
        planner_id=f"p0123:{branch}:{family}",
        planner_family="joint",
        deployable=True,
        reference_only=False,
        requires_prediction=True,
        exact=False,
        historical_aliases=(
            f"U_{family}_{'event' if branch == 'event' else 'global_ordering'}_p0123",
        ),
    )
    for branch in ("event", "global")
    for family in P012_CORES
)

_LEGACY_FUTURE_SPECS = tuple(
    PlannerSpec(
        planner_id=f"future_prepared:{branch}:{family}",
        planner_family="joint",
        deployable=True,
        reference_only=False,
        requires_prediction=True,
        exact=False,
        historical_aliases=(f"future_p01_{branch}_{family}",),
    )
    for branch in ("event", "global")
    for family in P012_CORES
)

_EXPLICIT_AXIS_SPECS = tuple(
    PlannerSpec(
        planner_id=axes.canonical_id,
        planner_family=axes.scope,
        deployable=True,
        reference_only=False,
        requires_prediction=axes.requires_prediction,
        exact=False,
        historical_aliases=(),
    )
    for axes in planner_axis_matrix(include_future=True, include_p0123=True)
)


def merge_p012_registry_specs(existing_specs: tuple[PlannerSpec, ...]) -> tuple[PlannerSpec, ...]:
    """Merge P012 assets without creating ID/alias collisions."""
    existing = tuple(existing_specs)
    occupied: set[str] = set()
    for spec in existing:
        occupied.add(str(spec.planner_id))
        occupied.update(str(alias) for alias in tuple(spec.historical_aliases))
    additions: list[PlannerSpec] = []
    for spec in p012_planner_specs():
        names = {str(spec.planner_id), *(str(alias) for alias in spec.historical_aliases)}
        if names & occupied:
            continue
        additions.append(spec)
        occupied.update(names)
    return existing + tuple(additions)


def p012_planner_specs() -> tuple[PlannerSpec, ...]:
    """Return legacy compatibility IDs plus the complete orthogonal matrix.

    The explicit matrix covers:
      Current/Future x P012/P0123 x Local/Joint x Event/Global x three cores,
    with the currently unsupported Future-P0123 cells excluded by
    :class:`PlannerAxes` validation.
    """
    return (
        _LEGACY_P012_SPECS
        + _LEGACY_P0123_SPECS
        + _LEGACY_FUTURE_SPECS
        + _EXPLICIT_AXIS_SPECS
    )


def create_p012_planner(planner_id: str, config: Mapping[str, object] | None = None) -> P012FormalPlanner:
    normalized = str(planner_id)
    specs = p012_planner_specs()
    alias_map = {
        str(alias): str(spec.planner_id)
        for spec in specs
        for alias in tuple(spec.historical_aliases)
    }
    normalized = alias_map.get(normalized, normalized)
    try:
        axes = parse_planner_axes(normalized)
    except ValueError as exc:
        raise ValueError(f"unknown formal P012-family planner {planner_id!r}") from exc

    accepted = set(P012AdapterConfig.__dataclass_fields__)
    values = {key: value for key, value in dict(config or {}).items() if key in accepted}
    values.update(
        {
            "timing": axes.timing,
            "horizon": axes.horizon,
            "scope": axes.scope,
            "engine": axes.engine,
            "family": axes.core,
            "branch": None,
            "public_planner_id": normalized,
        }
    )
    adapter_config = P012AdapterConfig(**values)
    if axes.timing == "future":
        return FuturePreparedFormalPlanner(adapter_config)
    if axes.horizon == "p0123":
        return P0123FormalPlanner(adapter_config)
    return P012FormalPlanner(adapter_config)


__all__ = [
    "FuturePreparedFormalPlanner",
    "P012AdapterConfig",
    "P012FormalPlanner",
    "P0123FormalPlanner",
    "create_p012_planner",
    "p012_planner_specs",
    "merge_p012_registry_specs",
]
