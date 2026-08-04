from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from rs_sim.scheduler.core.rscf_core import RSCFTask, RSCFWireCostModel, order_rscf
from rs_sim.scheduler.prediction.timing import (
    build_rank_timing_profile,
    causal_last_observed_timing_estimate,
    load_rank_timing_profile,
    save_rank_timing_profile,
)


@dataclass(frozen=True)
class _Window:
    layer_id: int
    matrix: tuple[tuple[int, ...], ...]
    p1_source: tuple[int, ...]
    middle: tuple[int, ...]
    postprocess: tuple[int, ...]

    @property
    def mapping(self):
        return SimpleNamespace(world_size=len(self.matrix))

    @property
    def local_compute(self):
        return SimpleNamespace(
            dispatch_release_to_combine_source_ready_ns=self.p1_source,
            combine_release_to_router_ready_ns=tuple(value // 4 for value in self.middle),
            router_and_pack_ns=tuple(value - value // 4 for value in self.middle),
            dispatch_local_postprocess_ns=self.postprocess,
            bootstrap_router_and_pack_ns=self.middle,
        )

    @property
    def window_id(self) -> str:
        return f"window-{self.layer_id}"

    def payload_matrix(self, phase: str):
        assert phase == "DISPATCH"
        return self.matrix


@dataclass(frozen=True)
class _Fixture:
    fixture_id: str
    windows: tuple[_Window, ...]

    @property
    def world_size(self) -> int:
        return len(self.windows[0].matrix)


def _fixture(fixture_id: str, delta: int) -> _Fixture:
    return _Fixture(
        fixture_id=fixture_id,
        windows=(
            _Window(
                layer_id=1,
                matrix=((0, 100 + delta), (300 + delta, 0)),
                p1_source=(10 + delta, 30 + delta),
                middle=(100, 200),
                postprocess=(3, 5),
            ),
            _Window(
                layer_id=2,
                matrix=((0, 200 + delta), (500 + delta, 0)),
                p1_source=(20 + delta, 60 + delta),
                middle=(300, 400),
                postprocess=(7, 11),
            ),
        ),
    )


def test_rank_timing_profile_round_trip(tmp_path: Path) -> None:
    profile = build_rank_timing_profile(
        (_fixture("cal-a", 0), _fixture("cal-b", 2)),
        profile_id="test-profile",
    )
    path = tmp_path / "profile.json"
    save_rank_timing_profile(profile, path)
    loaded = load_rank_timing_profile(path)
    assert loaded == profile
    assert loaded.profile_digest == profile.profile_digest
    assert loaded.layer(1) is not None
    assert loaded.layer(2) is not None


def test_profile_estimate_uses_calibration_not_evaluated_compute_truth() -> None:
    profile = build_rank_timing_profile(
        (_fixture("cal-a", 0), _fixture("cal-b", 2)),
        profile_id="test-profile",
    )
    evaluated = _fixture("evaluation", 7)
    current = evaluated.windows[0]
    altered = _Window(
        layer_id=current.layer_id,
        matrix=current.matrix,
        p1_source=(999_999, 888_888),
        middle=(777_777, 666_666),
        postprocess=(555_555, 444_444),
    )
    predicted_p2 = evaluated.windows[1].matrix
    first = causal_last_observed_timing_estimate(
        current_window=current,
        previous_window=None,
        predicted_p2_matrix=predicted_p2,
        timing_profile=profile,
        following_layer_id=2,
        p2_load_confidence_ppm=1_000_000,
    )
    second = causal_last_observed_timing_estimate(
        current_window=altered,
        previous_window=None,
        predicted_p2_matrix=predicted_p2,
        timing_profile=profile,
        following_layer_id=2,
        p2_load_confidence_ppm=1_000_000,
    )
    assert first == second
    assert first.p1_source_ready_ns[1] > first.p1_source_ready_ns[0]
    assert first.p1_to_p2_delay_ns == profile.layer(1).p1_to_p2_delay_ns


def test_rscf_respects_rank_source_ready_estimate() -> None:
    tasks = (
        RSCFTask("delayed", 1, 0, 1, 1),
        RSCFTask("ready", 1, 1, 0, 1),
    )
    model = RSCFWireCostModel(
        source_ready_by_phase_rank=((1, 0, 100.0), (1, 1, 0.0)),
    )
    plan = order_rscf(tasks, rank_count=2, wire_cost_model=model)
    assert plan.ordered_task_ids[0] == "ready"
    assert plan.waves[0].start_time == 0.0
    assert plan.waves[-1].start_time >= 100.0


def test_rank_timing_profile_rejects_different_capture_family() -> None:
    calibration = _fixture("cal-a", 0)
    object.__setattr__(
        calibration,
        "provenance",
        SimpleNamespace(dataset_id="dataset-a", capture_id="model-a_ep8_s128"),
    )
    profile = build_rank_timing_profile((calibration,), profile_id="test-profile")

    matching = _fixture("eval-a", 1)
    object.__setattr__(
        matching,
        "provenance",
        SimpleNamespace(dataset_id="dataset-a", capture_id="model-a_ep8_s128"),
    )
    profile.assert_compatible(matching)

    mismatched = _fixture("eval-b", 1)
    object.__setattr__(
        mismatched,
        "provenance",
        SimpleNamespace(dataset_id="dataset-b", capture_id="model-b_ep8_s128"),
    )
    import pytest

    with pytest.raises(ValueError, match="dataset mismatch"):
        profile.assert_compatible(mismatched)
