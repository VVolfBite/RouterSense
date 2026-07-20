from __future__ import annotations

from rs.core.contracts import WindowPlan
from rs.planning.api import to_logical_plan
from rs.runtime.online.megatron_ep.target_planning import TargetLayerPreparedJointPlan, reconcile_target_plan
from rs.scheduling.validation import stable_hash


def _prepared(h1=((0, 2), (1, 0))) -> TargetLayerPreparedJointPlan:
    window_plan = WindowPlan(
        planner_id="u",
        planner_family="joint",
        request_digest="req",
        waves=(),
        metadata={"policy_name": "u"},
    )
    logical_plan = to_logical_plan(window_plan)
    return TargetLayerPreparedJointPlan(
        source_layer_id="0",
        target_layer_id="1",
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        h1_prediction_digest="h1",
        h2_prediction_digest="h2",
        target_problem_digest="tp",
        window_plan=window_plan,
        logical_plan=logical_plan,
        logical_plan_digest=window_plan.semantic_digest(),
        logical_payload_digest=stable_hash(logical_plan.to_dict()),
        policy="u",
        weights={},
        bucket_contract_digest="bucket",
        topology_digest="topo",
        h1_rows=h1,
        derived_p1_rows=((0, 1), (2, 0)),
        h2_rows=((0, 1), (1, 0)),
        created_at_ns=1,
        ready_at_ns=2,
    )


def test_reconcile_exact_match() -> None:
    outcome = reconcile_target_plan(prepared_plan=_prepared(), actual_p0_rows=((0, 2), (1, 0)))
    assert outcome.status == "exact"


def test_reconcile_repairable() -> None:
    outcome = reconcile_target_plan(prepared_plan=_prepared(), actual_p0_rows=((0, 4), (1, 0)))
    assert outcome.status == "repaired"
    assert outcome.resized_edges == 1


def test_reconcile_reject() -> None:
    outcome = reconcile_target_plan(prepared_plan=_prepared(h1=((0, 2), (0, 0))), actual_p0_rows=((0, 0), (5, 0)))
    assert outcome.status == "rejected"


def test_repaired_plan_exactly_covers_actual_dispatch_and_transposed_return() -> None:
    from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningTopology, PlanningWeights
    from rs.planning import PlannerRegistry
    from rs.planning.api import to_logical_plan
    from rs.planning.request_builder import build_window_planning_request
    from rs.runtime.online.megatron_ep.target_planning import reconcile_once

    predicted = (
        (0, 4, 0),
        (0, 0, 3),
        (2, 0, 0),
    )
    actual = (
        (0, 7, 1),
        (0, 0, 0),
        (2, 5, 0),
    )
    transposed = tuple(tuple(actual[col][row] for col in range(3)) for row in range(3))
    request = build_window_planning_request(
        identity=PlanningIdentity(request_id="repair-coverage"),
        p0_dispatch_rows=predicted,
        p1_return_rows=tuple(tuple(predicted[col][row] for col in range(3)) for row in range(3)),
        p2_hint_rows=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        predictor_id="fixture",
        confidence=1.0,
        topology=PlanningTopology(world_size=3),
        constraints=PlanningConstraints(
            bucket_rows=1,
            max_waves=128,
            expert_compute_delay=0.0,
            phase_release_model="p1_return",
        ),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    window_plan = PlannerRegistry.create("current:p012:joint:global:rscf", usage="runtime").plan(request)
    logical = to_logical_plan(window_plan)
    prepared = TargetLayerPreparedJointPlan(
        source_layer_id="0",
        target_layer_id="1",
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        h1_prediction_digest="h1",
        h2_prediction_digest="h2",
        target_problem_digest="tp",
        window_plan=window_plan,
        logical_plan=logical,
        logical_plan_digest=window_plan.semantic_digest(),
        logical_payload_digest=stable_hash(logical.to_dict()),
        policy=str(window_plan.planner_id),
        weights={},
        bucket_contract_digest="bucket",
        topology_digest="topo",
        h1_rows=predicted,
        derived_p1_rows=tuple(tuple(predicted[col][row] for col in range(3)) for row in range(3)),
        h2_rows=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        created_at_ns=1,
        ready_at_ns=2,
    )
    outcome = reconcile_once(prepared_plan=prepared, actual_p0_rows=actual)
    assert outcome.status == "repaired"
    assert outcome.logical_plan is not None
    observed: dict[str, dict[tuple[int, int], int]] = {"p0_dispatch": {}, "p1_return": {}}
    for wave in outcome.logical_plan.waves:
        for flow in wave.flows:
            if flow.phase not in observed:
                continue
            edge = (int(flow.src_rank), int(flow.dst_rank))
            assert edge not in observed[flow.phase]
            observed[flow.phase][edge] = int(flow.byte_count)
    expected_p0 = {(src, dst): int(value) for src, row in enumerate(actual) for dst, value in enumerate(row) if src != dst and value > 0}
    expected_p1 = {(src, dst): int(value) for src, row in enumerate(transposed) for dst, value in enumerate(row) if src != dst and value > 0}
    assert observed["p0_dispatch"] == expected_p0
    assert observed["p1_return"] == expected_p1
