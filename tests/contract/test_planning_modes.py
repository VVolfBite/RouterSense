from __future__ import annotations

import pytest

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
)
from rs.planning._legacy_runtime import to_legacy_request


def _request(*, information_mode: str, planning_track: str, p2_semantics: str) -> PlanningRequest:
    return PlanningRequest(
        identity=PlanningIdentity(request_id="req", source_layer_id="1", target_layer_id="2"),
        traffic=PlanningTraffic(
            p0_dispatch_rows=((0, 4), (3, 0)),
            p1_return_rows=((0, 3), (4, 0)),
        ),
        prediction_hint=PredictionHint(
            predictor_id="copy_current_dispatch",
            hint_type="copy_current_dispatch",
            target_dispatch_rows=((0, 5), (2, 0)),
            confidence=1.0,
            source_layer_id="1",
            target_layer_id="2",
        ),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode=information_mode,
        planning_track=planning_track,
        p2_semantics=p2_semantics,
    )


@pytest.mark.parametrize(
    ("mode", "track", "p2_semantics", "expected_p2"),
    [
        ("p0_only", "runtime_lookahead", "absent", ((0, 0), (0, 0))),
        ("p0_p1", "runtime_lookahead", "absent", ((0, 0), (0, 0))),
        ("p0_p1_p2", "runtime_lookahead", "advisory_hint", ((0, 5), (2, 0))),
    ],
)
def test_information_mode_filters_legacy_phase_rows(
    mode: str,
    track: str,
    p2_semantics: str,
    expected_p2: tuple[tuple[int, ...], ...],
) -> None:
    request = _request(information_mode=mode, planning_track=track, p2_semantics=p2_semantics)
    legacy = to_legacy_request(request)
    assert legacy.p2_hint_rows == expected_p2
    assert legacy.scheduling_mode == ("runtime_lookahead" if track == "runtime_lookahead" else "execution_window")


def test_execution_window_rejects_advisory_p2() -> None:
    with pytest.raises(ValueError, match="execution_window"):
        _request(
            information_mode="p0_p1_p2",
            planning_track="execution_window",
            p2_semantics="advisory_hint",
        ).validate()
