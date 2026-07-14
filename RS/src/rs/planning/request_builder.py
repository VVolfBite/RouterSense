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
) -> PlanningRequest:
    request = PlanningRequest(
        identity=PlanningIdentity(
            request_id=str(identity.request_id),
            run_id=str(identity.run_id),
            forward_id=str(identity.forward_id),
            window_id=str(identity.window_id),
            source_layer_id=str(identity.source_layer_id),
            target_layer_id=str(identity.target_layer_id),
        ),
        traffic=PlanningTraffic(
            p0_dispatch_rows=tuple(tuple(int(v) for v in row) for row in p0_dispatch_rows),
            p1_return_rows=tuple(tuple(int(v) for v in row) for row in p1_return_rows),
        ),
        prediction_hint=PredictionHint(
            predictor_id=str(predictor_id),
            hint_type=str(hint_type),
            target_dispatch_rows=tuple(tuple(int(v) for v in row) for row in p2_hint_rows),
            confidence=float(confidence),
            oracle=bool(oracle),
            source_layer_id=str(identity.source_layer_id),
            target_layer_id=str(identity.target_layer_id),
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
            residual_weight=float(weights.residual_weight),
            barrier_weight=float(weights.barrier_weight),
            age_weight=float(weights.age_weight),
            prediction_weight=float(weights.prediction_weight),
            criticality_weight=float(weights.criticality_weight),
            iteration_budget=weights.iteration_budget,
        ),
        information_mode=str(information_mode),
    )
    request.validate()
    return request


__all__ = ["build_window_planning_request"]
