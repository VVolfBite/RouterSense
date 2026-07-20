from __future__ import annotations

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
)


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    if not normalized:
        raise ValueError("optional identity field must not be empty")
    return normalized


def build_window_planning_request(
    *,
    identity: PlanningIdentity,
    p0_dispatch_rows,
    p1_return_rows,
    p2_hint_rows,
    predictor_id: str,
    confidence: float,
    topology: PlanningTopology,
    constraints: PlanningConstraints,
    weights: PlanningWeights,
    information_mode: str,
    hint_type: str = "traffic_matrix",
    oracle: bool = False,
    planning_track: str = "runtime_lookahead",
    p2_semantics: str = "advisory_hint",
    p3_return_rows=None,
) -> PlanningRequest:
    normalized_p2 = tuple(tuple(int(v) for v in row) for row in p2_hint_rows)
    normalized_p3 = p3_return_rows
    if str(information_mode) == "p0_p1_p2_p3" and normalized_p3 is None:
        world_size = len(normalized_p2)
        normalized_p3 = tuple(
            tuple(int(normalized_p2[col][row]) for col in range(world_size))
            for row in range(world_size)
        )
    if normalized_p3 is not None:
        normalized_p3 = tuple(tuple(int(v) for v in row) for row in normalized_p3)
    request = PlanningRequest(
        identity=PlanningIdentity(
            request_id=str(identity.request_id),
            run_id=_optional_str(identity.run_id),
            forward_id=_optional_str(identity.forward_id),
            window_id=_optional_str(identity.window_id),
            source_layer_id=_optional_str(identity.source_layer_id),
            target_layer_id=_optional_str(identity.target_layer_id),
        ),
        traffic=PlanningTraffic(
            p0_dispatch_rows=tuple(tuple(int(v) for v in row) for row in p0_dispatch_rows),
            p1_return_rows=tuple(tuple(int(v) for v in row) for row in p1_return_rows),
        ),
        prediction_hint=PredictionHint(
            predictor_id=str(predictor_id),
            hint_type=str(hint_type),
            target_dispatch_rows=normalized_p2,
            confidence=float(confidence),
            oracle=bool(oracle),
            source_layer_id=_optional_str(identity.source_layer_id),
            target_layer_id=_optional_str(identity.target_layer_id),
        ),
        topology=PlanningTopology(
            world_size=int(topology.world_size),
            full_duplex=bool(topology.full_duplex),
        ),
        constraints=PlanningConstraints(
            bucket_rows=int(constraints.bucket_rows),
            max_waves=int(constraints.max_waves),
            expert_compute_delay=float(constraints.expert_compute_delay),
            phase_release_model=str(constraints.phase_release_model),
        ),
        weights=PlanningWeights(
            p0_weight=float(weights.p0_weight),
            p1_weight=float(weights.p1_weight),
            p2_weight=float(weights.p2_weight),
            p3_return_weight=float(weights.p3_return_weight),
            residual_weight=float(weights.residual_weight),
            barrier_weight=float(weights.barrier_weight),
            age_weight=float(weights.age_weight),
            prediction_weight=float(weights.prediction_weight),
            criticality_weight=float(weights.criticality_weight),
            iteration_budget=weights.iteration_budget,
        ),
        information_mode=str(information_mode),
        planning_track=str(planning_track),
        p2_semantics=str(p2_semantics),
        p3_return_rows=normalized_p3,
    )
    request.validate()
    return request


__all__ = ["build_window_planning_request"]
