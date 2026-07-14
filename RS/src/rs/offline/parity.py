from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import (
    ActualPhaseContext,
    EvaluationSpec,
    MaterializedPlan,
    OfflineWindow,
    PlanningRequest,
    PredictionResult,
    PublishedPlan,
    WindowPlan,
)
from rs.planning import PlannerRegistry
from rs.planning.request_builder import build_window_planning_request
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.execution.pipeline import PreparedExecution, RuntimeExecutionPipeline
from rs.runtime.online.megatron_ep.materialization import CommonPlanMaterializer

from .builder import OfflinePlanningRequestBuilder


@dataclass(frozen=True)
class PlanningParityCase:
    offline_request: PlanningRequest
    online_request: PlanningRequest
    offline_plan: WindowPlan
    online_plan: WindowPlan


@dataclass(frozen=True)
class MaterializationParityCase:
    planning: PlanningParityCase
    published_plan: PublishedPlan
    offline_materialized_plan: MaterializedPlan
    online_prepared_execution: PreparedExecution


def build_planning_parity_case(
    *,
    window: OfflineWindow,
    prediction: PredictionResult,
    spec: EvaluationSpec,
    planner_id: str,
    bucket_rows: int = 0,
    max_waves: int = 256,
    information_mode: str = "p0_p1_p2",
) -> PlanningParityCase:
    builder = OfflinePlanningRequestBuilder(
        bucket_rows=int(bucket_rows),
        max_waves=int(max_waves),
        information_mode=str(information_mode),
    )
    offline_request = builder.build(window, prediction, spec)
    online_request = build_window_planning_request(
        identity=offline_request.identity,
        p0_dispatch_rows=window.p0_actual,
        p1_return_rows=window.p1_actual,
        p2_hint_rows=prediction.hint.target_dispatch_rows,
        predictor_id=str(prediction.hint.predictor_id),
        confidence=float(prediction.hint.confidence),
        topology=offline_request.topology,
        constraints=offline_request.constraints,
        weights=offline_request.weights,
        information_mode=str(information_mode),
        hint_type=str(prediction.hint.hint_type),
        oracle=bool(prediction.hint.oracle),
    )
    planner = PlannerRegistry.create(str(planner_id), None)
    offline_plan = planner.plan(offline_request)
    online_plan = planner.plan(online_request)
    return PlanningParityCase(
        offline_request=offline_request,
        online_request=online_request,
        offline_plan=offline_plan,
        online_plan=online_plan,
    )


def build_materialization_parity_case(
    *,
    window: OfflineWindow,
    prediction: PredictionResult,
    spec: EvaluationSpec,
    planner_id: str,
    publication_slot: dict[str, object],
    rank_map: RankMap,
    actual_phase_context: ActualPhaseContext,
    bucket_rows: int = 0,
    max_waves: int = 256,
    information_mode: str = "p0_p1_p2",
) -> MaterializationParityCase:
    planning = build_planning_parity_case(
        window=window,
        prediction=prediction,
        spec=spec,
        planner_id=str(planner_id),
        bucket_rows=int(bucket_rows),
        max_waves=int(max_waves),
        information_mode=str(information_mode),
    )
    if planning.offline_plan.semantic_digest() != planning.online_plan.semantic_digest():
        raise ValueError("plan parity failed before materialization")
    publisher = CanonicalPlanPublisher(rank_map=rank_map)
    published_plan = publisher.build(
        publication_slot=dict(publication_slot),
        window_plan=planning.online_plan,
    )
    offline_materialized_plan = CommonPlanMaterializer().materialize(published_plan, actual_phase_context)
    online_prepared_execution = RuntimeExecutionPipeline().prepare(published_plan, actual_phase_context)
    return MaterializationParityCase(
        planning=planning,
        published_plan=published_plan,
        offline_materialized_plan=offline_materialized_plan,
        online_prepared_execution=online_prepared_execution,
    )


def expected_completed_task_ids(materialized_plan: MaterializedPlan, *, payload_role: str) -> tuple[str, ...]:
    materialized_plan.validate()
    task_ids: list[str] = []
    for batch in materialized_plan.batches:
        for item in batch.slices:
            if str(item.payload_role) == str(payload_role):
                task_ids.append(str(item.task_id))
    return tuple(dict.fromkeys(task_ids))


__all__ = [
    "MaterializationParityCase",
    "PlanningParityCase",
    "build_materialization_parity_case",
    "build_planning_parity_case",
    "expected_completed_task_ids",
]
