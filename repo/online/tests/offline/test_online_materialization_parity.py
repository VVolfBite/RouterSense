from __future__ import annotations

import json
from pathlib import Path

from rs.core.contracts import (
    ActualPhaseContext,
    EvaluationSpec,
    OfflineWindow,
    PredictionHint,
    PredictionIdentity,
    PredictionResult,
    TrafficProvenance,
)
from rs.offline.parity import (
    build_materialization_parity_case,
    build_planning_parity_case,
    expected_completed_task_ids,
)
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "offline_replay_smoke"


def _fixture_payload(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _fixture_window() -> OfflineWindow:
    payload = _fixture_payload("replay_layer_1.json")
    metadata = dict(payload["metadata"])
    return OfflineWindow(
        window_identity="fixture:1->2",
        source_layer=str(metadata["layer_id"]),
        target_layer=str(metadata["next_layer_id"]),
        p0_actual=tuple(tuple(int(v) for v in row) for row in payload["p0_dispatch_matrix"]),
        p1_actual=tuple(tuple(int(v) for v in row) for row in payload["p1_return_matrix"]),
        p2_actual=tuple(tuple(int(v) for v in row) for row in payload["p2_next_dispatch_matrix"]),
        placement_snapshot={"group_size": 4, "fixture_type": str(metadata["fixture_type"])},
        traffic_provenance=TrafficProvenance.ROUTE_RECONSTRUCTED,
        matrix_unit="rows",
        return_model="transpose_dispatch",
        raw_token_count=13,
        used_token_count=13,
        dropped_token_count=0,
        drop_reason=None,
        trace_digest="fixture-trace-layer1",
    )


def _fixture_prediction(window: OfflineWindow) -> PredictionResult:
    return PredictionResult(
        identity=PredictionIdentity(
            request_id=str(window.window_identity),
            source_layer_id=str(window.source_layer),
            target_layer_id=str(window.target_layer),
        ),
        hint=PredictionHint(
            predictor_id="copy_current",
            hint_type="traffic_matrix",
            target_dispatch_rows=window.p2_actual,
            confidence=1.0,
            oracle=False,
            source_layer_id=str(window.source_layer),
            target_layer_id=str(window.target_layer),
        ),
    )


def _fixture_spec(*, track: str = "runtime_lookahead") -> EvaluationSpec:
    return EvaluationSpec(
        track=track,
        world_size=4,
        task_granularity="matrix_cell",
        matrix_unit="rows",
        time_unit="row_cost",
        cost_model_id="offline_common_v1",
        release_model="p1_return",
        return_model="transpose_dispatch",
        full_duplex=True,
        launch_cost=0.0,
        bytes_per_row=1,
        bandwidth=1.0,
        compute_delay=0.0,
        p2_semantics="lookahead",
        residual_policy="reject",
    )


def test_planning_request_and_window_plan_parity_for_offline_fixture() -> None:
    window = _fixture_window()
    prediction = _fixture_prediction(window)
    spec = _fixture_spec()
    case = build_planning_parity_case(
        window=window,
        prediction=prediction,
        spec=spec,
        planner_id="fifo_bucket",
        bucket_rows=4,
        max_waves=64,
    )
    assert case.offline_request.semantic_digest() == case.online_request.semantic_digest()
    assert case.offline_plan.semantic_digest() == case.online_plan.semantic_digest()


def test_materialized_plan_digest_matches_runtime_pipeline_for_p0_fixture() -> None:
    window = _fixture_window()
    prediction = _fixture_prediction(window)
    spec = _fixture_spec()
    contexts = make_contexts_from_matrix(phase="P0", matrix=window.p0_actual, p2_hint_mode="deterministic_stub")
    actual_context = ActualPhaseContext(
        layer_id=str(contexts[0].layer_id),
        phase="P0",
        world_size=4,
        rank_space="global",
        layout_digest=str(contexts[0].canonical_receive_layout_id),
        metadata={"phase_ready_context": contexts[0].to_dict()},
    )
    case = build_materialization_parity_case(
        window=window,
        prediction=prediction,
        spec=spec,
        planner_id="fifo_bucket",
        publication_slot={
            "run_id": str(window.trace_digest),
            "forward_generation": 0,
            "microbatch_id": "mb0",
            "source_layer_id": str(window.source_layer),
            "target_layer_id": str(window.target_layer),
            "planning_slot": f"{window.source_layer}->{window.target_layer}",
        },
        rank_map=RankMap(group_ranks=(0, 1, 2, 3), root_rank=0),
        actual_phase_context=actual_context,
        bucket_rows=4,
        max_waves=64,
    )
    assert (
        case.offline_materialized_plan.materialized_plan_digest
        == case.online_prepared_execution.materialized_plan.materialized_plan_digest
    )
    assert case.online_prepared_execution.validation.valid is True


def test_expected_completed_tasks_come_from_materialized_plan_not_truth() -> None:
    window = _fixture_window()
    prediction = _fixture_prediction(window)
    spec = _fixture_spec()
    contexts = make_contexts_from_matrix(phase="P0", matrix=window.p0_actual, p2_hint_mode="deterministic_stub")
    actual_context = ActualPhaseContext(
        layer_id=str(contexts[1].layer_id),
        phase="P0",
        world_size=4,
        rank_space="global",
        layout_digest=str(contexts[1].canonical_receive_layout_id),
        metadata={"phase_ready_context": contexts[1].to_dict()},
    )
    case = build_materialization_parity_case(
        window=window,
        prediction=prediction,
        spec=spec,
        planner_id="fifo_bucket",
        publication_slot={
            "run_id": str(window.trace_digest),
            "forward_generation": 0,
            "microbatch_id": "mb0",
            "source_layer_id": str(window.source_layer),
            "target_layer_id": str(window.target_layer),
            "planning_slot": f"{window.source_layer}->{window.target_layer}",
        },
        rank_map=RankMap(group_ranks=(0, 1, 2, 3), root_rank=0),
        actual_phase_context=actual_context,
        bucket_rows=4,
        max_waves=64,
    )
    for spec_item in case.online_prepared_execution.materialized_plan.payload_specs:
        expected_ids = expected_completed_task_ids(
            case.online_prepared_execution.materialized_plan,
            payload_role=str(spec_item.payload_role),
        )
        assert expected_ids
        assert all(isinstance(item, str) and item for item in expected_ids)
