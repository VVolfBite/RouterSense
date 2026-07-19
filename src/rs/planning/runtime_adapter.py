from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import PlanningRequest
from rs.scheduling.contracts import (
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    ReleaseConstraint,
)
from rs.scheduling.families import is_scoped_family_policy
from rs.scheduling.registry import resolve_policy
from rs.scheduling.traffic_matrix import matrix_digest_remote, matrix_remote_bytes

from ._legacy_runtime import _effective_phase_rows, _legacy_scheduling_mode
from .api import Planner


def _window_plan_from_logical_plan(*, planner_id: str, planner_family: str, request: PlanningRequest, logical_plan) -> "WindowPlan":
    from rs.core.contracts import PlanWave, PlannedFlow, WindowPlan

    waves = []
    for wave in logical_plan.waves:
        flows = tuple(
            PlannedFlow(
                flow_id=str(flow.flow_id),
                phase=str(flow.phase),
                src_rank=int(flow.src_rank),
                dst_rank=int(flow.dst_rank),
                row_count=int(flow.byte_count),
                release_state=str(flow.release_state),
                executable=bool(flow.is_executable),
            )
            for flow in wave.flows
        )
        waves.append(
            PlanWave(
                wave_id=int(wave.wave_id),
                flows=flows,
                estimated_duration=float(wave.duration),
            )
        )
    metadata = dict(getattr(logical_plan, "diagnostics", {}) or {})
    metadata.setdefault("legacy_policy_name", str(logical_plan.policy_name))
    metadata.setdefault("legacy_makespan", float(metadata.get("makespan", sum(float(item.duration) for item in logical_plan.waves)) or 0.0))
    return WindowPlan(
        planner_id=str(planner_id),
        planner_family=str(planner_family),
        request_digest=request.semantic_digest(),
        waves=tuple(waves),
        metadata=metadata,
    )


def _flows_from_matrix(matrix, *, phase: str, release_state: str, executable: bool):
    flows = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, row_count in enumerate(row):
            if src_rank == dst_rank or int(row_count) <= 0:
                continue
            flows.append(
                {
                    "flow_id": f"{phase}:{src_rank}->{dst_rank}",
                    "phase": str(phase),
                    "src_rank": int(src_rank),
                    "dst_rank": int(dst_rank),
                    "byte_count": int(row_count),
                    "release_state": str(release_state),
                    "is_executable": bool(executable),
                }
            )
    from rs.scheduling.contracts import FlowDemand

    return tuple(FlowDemand(**flow) for flow in flows)


def _problem_from_planning_request(request: PlanningRequest) -> MultiPhaseSchedulingProblem:
    request.validate()
    p0_rows, p1_rows, p2_rows = _effective_phase_rows(request)
    p3_rows = request.p3_return_rows or ()
    prediction_kind = str(request.prediction_hint.to_dict().get("prediction_kind", request.prediction_hint.hint_type))
    confidence = 0.0 if prediction_kind == "zero_hint" else float(request.prediction_hint.confidence or 0.0)
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows_from_matrix(p0_rows, phase="p0_dispatch", release_state="ready", executable=True),
            blocked_flows=_flows_from_matrix(p1_rows, phase="p1_return", release_state="blocked", executable=True),
            forecast_pressure=_flows_from_matrix(
                p2_rows,
                phase="p2_next_dispatch_forecast" if str(request.p2_semantics) == "advisory_hint" else "p2_next_dispatch",
                release_state="advisory_only" if str(request.p2_semantics) == "advisory_hint" else "ready",
                executable=str(request.p2_semantics) == "executable_actual",
            ),
        ),
        topology=LogicalTopology(num_gpus=int(request.topology.world_size)),
        release_model=ReleaseConstraint(
            phase=str(request.constraints.phase_release_model),
            rank=0,
            release_after_phase="p0_dispatch",
            expert_compute_delay=float(request.constraints.expert_compute_delay),
        ),
        forecast=ForecastPressure(
            source=prediction_kind,
            digest=matrix_digest_remote(p2_rows),
            oracle=bool(request.prediction_hint.oracle),
            evaluation_eligible=not bool(request.prediction_hint.oracle),
            matrix_shape=(len(p2_rows), len(p2_rows[0]) if p2_rows else 0),
            matrix_total_bytes=int(matrix_remote_bytes(p2_rows)),
            matrix=p2_rows,
            metadata={
                "request_id": str(request.identity.request_id),
                "planning_track": str(request.planning_track),
                "p2_semantics": str(request.p2_semantics),
                "predictor_id": str(request.prediction_hint.predictor_id),
            },
        ),
        options=GlobalReadySetOptions(
            scheduling_mode=_legacy_scheduling_mode(request),
            information_mode=str(request.information_mode),
            prediction_confidence=float(confidence),
            p0_weight=float(request.weights.p0_weight),
            p1_reservation_weight=float(request.weights.p1_weight),
            p2_hint_weight=float(request.weights.p2_weight),
            p3_return_weight=float(request.weights.p3_return_weight),
            max_waves=int(request.constraints.max_waves),
            bucket_rows=int(request.constraints.bucket_rows),
        ),
        p0_dispatch_matrix=p0_rows,
        p1_return_matrix=p1_rows,
        p2_next_dispatch_forecast_matrix=p2_rows,
        p3_next_return_advisory_matrix=tuple(tuple(int(v) for v in row) for row in p3_rows),
    )


@dataclass(frozen=True)
class FormalRuntimePlanner(Planner):
    _planner_id: str
    _planner_family: str

    @property
    def planner_id(self) -> str:
        return self._planner_id

    @property
    def planner_family(self) -> str:
        return self._planner_family

    def plan(self, request: PlanningRequest):
        problem = _problem_from_planning_request(request)
        # Scoped scheduling families own an immutable kernel specification.
        # PlanningWeights carries historical non-null defaults, so forwarding
        # those values would silently replace every family's intended kernel
        # parameters.  Family-specific tuning must be explicit at policy
        # construction time; the formal adapter therefore preserves the
        # registered family defaults.
        scoped_family = is_scoped_family_policy(self._planner_id)
        policy = resolve_policy(
            policy_name=self._planner_id,
            bucket_rows=int(request.constraints.bucket_rows),
            p0_weight=float(request.weights.p0_weight),
            p1_reservation_weight=float(request.weights.p1_weight),
            p2_hint_weight=float(request.weights.p2_weight),
            residual_weight=None if scoped_family else float(request.weights.residual_weight),
            barrier_weight=None if scoped_family else float(request.weights.barrier_weight),
            age_weight=None if scoped_family else float(request.weights.age_weight),
            prediction_weight=None if scoped_family else float(request.weights.prediction_weight),
        )
        logical_plan = policy.build_logical_plan(problem)
        return _window_plan_from_logical_plan(
            planner_id=self._planner_id,
            planner_family=self._planner_family,
            request=request,
            logical_plan=logical_plan,
        )


__all__ = ["FormalRuntimePlanner"]
