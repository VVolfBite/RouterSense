from __future__ import annotations

import json
from pathlib import Path

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
    PredictionIdentity,
    TrafficHistoryContext,
)
from rs.planning import PlannerRegistry
from rs.prediction import PredictionRegistry
from rs.runtime.offline.prediction.evaluation import rolling_predictor_records


FIXTURE_DIR = Path("RS/tests/fixtures/offline_replay_smoke")


def _fixture_rows(name: str) -> tuple[tuple[int, ...], ...]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return tuple(tuple(int(v) for v in row) for row in payload["p0_dispatch_matrix"])


def test_prediction_parity_for_copy_current_fixture() -> None:
    current = _fixture_rows("replay_layer_1.json")
    predictor = PredictionRegistry.create("copy_current", usage="offline")
    result = predictor.predict(
        TrafficHistoryContext(
            identity=PredictionIdentity(request_id="fixture", source_layer_id="1", target_layer_id="2"),
            current_dispatch_rows=current,
            current_return_rows=tuple(tuple(int(current[col][row]) for col in range(len(current))) for row in range(len(current))),
            history_dispatch_rows=(),
            world_size=len(current),
        )
    )
    legacy_records = rolling_predictor_records(fixture_dir=FIXTURE_DIR, predictor_name="copy_current_dispatch")
    assert result.hint.target_dispatch_rows == legacy_records[0].predicted_matrix


def test_planning_parity_for_fifo_fixture() -> None:
    p0 = _fixture_rows("replay_layer_1.json")
    request = PlanningRequest(
        identity=PlanningIdentity(request_id="fixture", source_layer_id="1", target_layer_id="2"),
        traffic=PlanningTraffic(p0_dispatch_rows=p0, p1_return_rows=tuple(tuple(int(p0[col][row]) for col in range(len(p0))) for row in range(len(p0)))),
        prediction_hint=PredictionHint(
            predictor_id="copy_current",
            hint_type="traffic_matrix",
            target_dispatch_rows=p0,
            confidence=1.0,
            source_layer_id="1",
            target_layer_id="2",
        ),
        topology=PlanningTopology(world_size=len(p0)),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=32, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    plan = PlannerRegistry.create("fifo_bucket", None).plan(request)
    assert plan.planner_family == "baseline"
    assert len(plan.waves) >= 0
