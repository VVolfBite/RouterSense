from __future__ import annotations

"""Causal rank timing estimates for the Current-P12 planning window.

The formal window runs from the start of the current P1/Combine communication
until the start of the following P1/Combine communication.  A rank can determine
that window through three independent local timing terms:

* current P1 source-ready time;
* current P1 completion -> current P2 source-ready time;
* current P2 destination completion -> following P1 source-ready time.

The planner may consume a calibration profile built from separate trace repeats.
The evaluated fixture never contributes realized timing to its own profile.
Current observed Dispatch load and predicted next-Dispatch load only adjust a
frozen model/layer/rank baseline; execution still consumes trace truth.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from rs_sim.scheduler.stable import stable_digest

_GIB_BYTES = 1_000_000_000


def _nonnegative_int_tuple(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must be non-negative")
    return result


def _inbound_totals(matrix: Iterable[Iterable[int]], rank_count: int) -> tuple[int, ...]:
    rows = tuple(tuple(int(value) for value in row) for row in matrix)
    if len(rows) != int(rank_count) or any(len(row) != int(rank_count) for row in rows):
        raise ValueError("Dispatch matrix must be rank_count square")
    if any(value < 0 for row in rows for value in row):
        raise ValueError("Dispatch matrix must be non-negative")
    return tuple(sum(rows[src][dst] for src in range(rank_count)) for dst in range(rank_count))


def _rounded_median(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("median input must not be empty")
    return max(0, int(round(float(median(tuple(int(value) for value in values))))))


def _nonnegative_slope_ns_per_gib(
    loads: Sequence[int], durations_ns: Sequence[int]
) -> int:
    """Robust fixed-point median pairwise slope.

    The stored unit is nanoseconds per 1e9 payload bytes.  Keeping the profile
    integer-only preserves deterministic JSON and avoids machine-specific float
    serialization in formal evidence.
    """

    if len(loads) != len(durations_ns) or not loads:
        raise ValueError("load and duration vectors must be non-empty and aligned")
    slopes: list[int] = []
    for left in range(len(loads)):
        for right in range(left + 1, len(loads)):
            delta_load = int(loads[right]) - int(loads[left])
            if delta_load == 0:
                continue
            delta_duration = int(durations_ns[right]) - int(durations_ns[left])
            slopes.append(int(round(delta_duration * _GIB_BYTES / delta_load)))
    return max(0, _rounded_median(slopes)) if slopes else 0


def _adjust_rank_baseline(
    *,
    baseline_ns: tuple[int, ...],
    baseline_load_bytes: tuple[int, ...],
    observed_load_bytes: tuple[int, ...],
    slope_ns_per_gib: int,
) -> tuple[int, ...]:
    if not (
        len(baseline_ns)
        == len(baseline_load_bytes)
        == len(observed_load_bytes)
    ):
        raise ValueError("rank baseline vectors must have equal length")
    return tuple(
        max(
            0,
            int(baseline_ns[rank])
            + int(
                round(
                    (int(observed_load_bytes[rank]) - int(baseline_load_bytes[rank]))
                    * int(slope_ns_per_gib)
                    / _GIB_BYTES
                )
            ),
        )
        for rank in range(len(baseline_ns))
    )


@dataclass(frozen=True, slots=True)
class P12LayerRankTimingCalibration:
    layer_id: int
    dispatch_inbound_baseline_bytes: tuple[int, ...]
    p1_source_ready_baseline_ns: tuple[int, ...]
    p1_source_ready_slope_ns_per_gib: int
    p1_to_p2_delay_ns: tuple[int, ...]
    dispatch_to_following_p1_tail_baseline_ns: tuple[int, ...]
    dispatch_tail_slope_ns_per_gib: int

    def __post_init__(self) -> None:
        if int(self.layer_id) < 0:
            raise ValueError("layer_id must be non-negative")
        vectors = (
            self.dispatch_inbound_baseline_bytes,
            self.p1_source_ready_baseline_ns,
            self.p1_to_p2_delay_ns,
            self.dispatch_to_following_p1_tail_baseline_ns,
        )
        if not vectors[0] or len({len(vector) for vector in vectors}) != 1:
            raise ValueError("timing calibration vectors must be non-empty and aligned")
        if any(int(value) < 0 for vector in vectors for value in vector):
            raise ValueError("timing calibration values must be non-negative")
        if int(self.p1_source_ready_slope_ns_per_gib) < 0:
            raise ValueError("P1 source-ready slope must be non-negative")
        if int(self.dispatch_tail_slope_ns_per_gib) < 0:
            raise ValueError("Dispatch tail slope must be non-negative")

    @property
    def rank_count(self) -> int:
        return len(self.dispatch_inbound_baseline_bytes)

    def predict_p1_source_ready(
        self, dispatch_inbound_bytes: tuple[int, ...]
    ) -> tuple[int, ...]:
        return _adjust_rank_baseline(
            baseline_ns=self.p1_source_ready_baseline_ns,
            baseline_load_bytes=self.dispatch_inbound_baseline_bytes,
            observed_load_bytes=dispatch_inbound_bytes,
            slope_ns_per_gib=self.p1_source_ready_slope_ns_per_gib,
        )

    def predict_dispatch_tail(
        self,
        dispatch_inbound_bytes: tuple[int, ...],
        *,
        load_confidence_ppm: int,
    ) -> tuple[int, ...]:
        confidence = min(1_000_000, max(0, int(load_confidence_ppm)))
        adjusted = _adjust_rank_baseline(
            baseline_ns=self.dispatch_to_following_p1_tail_baseline_ns,
            baseline_load_bytes=self.dispatch_inbound_baseline_bytes,
            observed_load_bytes=dispatch_inbound_bytes,
            slope_ns_per_gib=self.dispatch_tail_slope_ns_per_gib,
        )
        return tuple(
            max(
                0,
                int(self.dispatch_to_following_p1_tail_baseline_ns[rank])
                + int(
                    round(
                        (
                            int(adjusted[rank])
                            - int(self.dispatch_to_following_p1_tail_baseline_ns[rank])
                        )
                        * confidence
                        / 1_000_000
                    )
                ),
            )
            for rank in range(self.rank_count)
        )

    def stable_payload(self) -> dict[str, Any]:
        return {
            "layer_id": int(self.layer_id),
            "dispatch_inbound_baseline_bytes": self.dispatch_inbound_baseline_bytes,
            "p1_source_ready_baseline_ns": self.p1_source_ready_baseline_ns,
            "p1_source_ready_slope_ns_per_gib": int(
                self.p1_source_ready_slope_ns_per_gib
            ),
            "p1_to_p2_delay_ns": self.p1_to_p2_delay_ns,
            "dispatch_to_following_p1_tail_baseline_ns": (
                self.dispatch_to_following_p1_tail_baseline_ns
            ),
            "dispatch_tail_slope_ns_per_gib": int(
                self.dispatch_tail_slope_ns_per_gib
            ),
        }


@dataclass(frozen=True, slots=True)
class P12RankTimingProfile:
    profile_id: str
    rank_count: int
    layers: tuple[P12LayerRankTimingCalibration, ...]
    calibration_fixture_ids: tuple[str, ...]
    calibration_dataset_ids: tuple[str, ...] = ()
    calibration_capture_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if int(self.rank_count) <= 0:
            raise ValueError("rank_count must be positive")
        if not self.layers:
            raise ValueError("rank timing profile must contain layers")
        layer_ids = tuple(int(layer.layer_id) for layer in self.layers)
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("rank timing profile layer IDs must be unique")
        if any(layer.rank_count != int(self.rank_count) for layer in self.layers):
            raise ValueError("rank timing profile rank counts must match")
        if not self.calibration_fixture_ids:
            raise ValueError("rank timing profile requires calibration fixture IDs")
        if any(not str(value).strip() for value in self.calibration_dataset_ids):
            raise ValueError("calibration dataset IDs must be non-empty")
        if any(not str(value).strip() for value in self.calibration_capture_ids):
            raise ValueError("calibration capture IDs must be non-empty")

    def assert_compatible(self, fixture: Any) -> None:
        if int(fixture.world_size) != int(self.rank_count):
            raise ValueError(
                f"rank timing profile rank_count={self.rank_count} does not match "
                f"fixture world_size={fixture.world_size}"
            )
        provenance = getattr(fixture, "provenance", None)
        if provenance is None:
            return
        dataset_id = str(getattr(provenance, "dataset_id", "")).strip()
        capture_id = str(getattr(provenance, "capture_id", "")).strip()
        if self.calibration_dataset_ids and dataset_id not in self.calibration_dataset_ids:
            raise ValueError(
                f"rank timing profile dataset mismatch: {dataset_id!r} not in "
                f"{self.calibration_dataset_ids!r}"
            )
        if self.calibration_capture_ids and capture_id not in self.calibration_capture_ids:
            raise ValueError(
                f"rank timing profile capture mismatch: {capture_id!r} not in "
                f"{self.calibration_capture_ids!r}"
            )

    def layer(self, layer_id: int) -> P12LayerRankTimingCalibration | None:
        for layer in self.layers:
            if int(layer.layer_id) == int(layer_id):
                return layer
        return None

    def stable_payload(self) -> dict[str, Any]:
        return {
            "schema": "P12_RANK_TIMING_PROFILE",
            "profile_id": self.profile_id,
            "rank_count": int(self.rank_count),
            "calibration_fixture_ids": self.calibration_fixture_ids,
            "calibration_dataset_ids": self.calibration_dataset_ids,
            "calibration_capture_ids": self.calibration_capture_ids,
            "layers": tuple(layer.stable_payload() for layer in self.layers),
        }

    @property
    def profile_digest(self) -> str:
        return stable_digest(self.stable_payload())

    def to_json_dict(self) -> dict[str, Any]:
        payload = self.stable_payload()
        payload["profile_digest"] = self.profile_digest
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "P12RankTimingProfile":
        if str(payload.get("schema")) != "P12_RANK_TIMING_PROFILE":
            raise ValueError("unsupported rank timing profile schema")
        layers = tuple(
            P12LayerRankTimingCalibration(
                layer_id=int(item["layer_id"]),
                dispatch_inbound_baseline_bytes=_nonnegative_int_tuple(
                    item["dispatch_inbound_baseline_bytes"],
                    name="dispatch inbound baseline",
                ),
                p1_source_ready_baseline_ns=_nonnegative_int_tuple(
                    item["p1_source_ready_baseline_ns"],
                    name="P1 source-ready baseline",
                ),
                p1_source_ready_slope_ns_per_gib=int(
                    item["p1_source_ready_slope_ns_per_gib"]
                ),
                p1_to_p2_delay_ns=_nonnegative_int_tuple(
                    item["p1_to_p2_delay_ns"], name="P1-to-P2 delay"
                ),
                dispatch_to_following_p1_tail_baseline_ns=_nonnegative_int_tuple(
                    item["dispatch_to_following_p1_tail_baseline_ns"],
                    name="Dispatch tail baseline",
                ),
                dispatch_tail_slope_ns_per_gib=int(
                    item["dispatch_tail_slope_ns_per_gib"]
                ),
            )
            for item in payload["layers"]
        )
        result = cls(
            profile_id=str(payload["profile_id"]),
            rank_count=int(payload["rank_count"]),
            layers=layers,
            calibration_fixture_ids=tuple(
                str(value) for value in payload["calibration_fixture_ids"]
            ),
            calibration_dataset_ids=tuple(
                str(value) for value in payload.get("calibration_dataset_ids", ())
            ),
            calibration_capture_ids=tuple(
                str(value) for value in payload.get("calibration_capture_ids", ())
            ),
        )
        expected = payload.get("profile_digest")
        if expected is not None and str(expected) != result.profile_digest:
            raise ValueError("rank timing profile digest mismatch")
        return result


def save_rank_timing_profile(profile: P12RankTimingProfile, path: Path) -> None:
    Path(path).write_text(json.dumps(profile.to_json_dict(), indent=2) + "\n")


def load_rank_timing_profile(path: Path) -> P12RankTimingProfile:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("rank timing profile JSON must be an object")
    return P12RankTimingProfile.from_json_dict(payload)


def build_rank_timing_profile(
    calibration_fixtures: Sequence[Any], *, profile_id: str
) -> P12RankTimingProfile:
    fixtures = tuple(calibration_fixtures)
    if not fixtures:
        raise ValueError("at least one calibration fixture is required")
    rank_count = int(fixtures[0].world_size)
    if any(int(fixture.world_size) != rank_count for fixture in fixtures):
        raise ValueError("calibration fixtures must have equal world size")
    windows_by_fixture = [
        {int(window.layer_id): window for window in fixture.windows}
        for fixture in fixtures
    ]
    common_layers = set(windows_by_fixture[0])
    for mapping in windows_by_fixture[1:]:
        common_layers.intersection_update(mapping)
    if not common_layers:
        raise ValueError("calibration fixtures have no common layers")

    layers: list[P12LayerRankTimingCalibration] = []
    for layer_id in sorted(common_layers):
        windows = tuple(mapping[layer_id] for mapping in windows_by_fixture)
        inbound_vectors = tuple(
            _inbound_totals(window.payload_matrix("DISPATCH"), rank_count)
            for window in windows
        )
        p1_vectors = tuple(
            tuple(
                int(window.local_compute.dispatch_release_to_combine_source_ready_ns[rank])
                for rank in range(rank_count)
            )
            for window in windows
        )
        middle_vectors = tuple(
            tuple(
                int(window.local_compute.combine_release_to_router_ready_ns[rank])
                + int(window.local_compute.router_and_pack_ns[rank])
                for rank in range(rank_count)
            )
            for window in windows
        )
        tail_vectors = tuple(
            tuple(
                int(window.local_compute.dispatch_local_postprocess_ns[rank])
                + int(
                    window.local_compute.dispatch_release_to_combine_source_ready_ns[
                        rank
                    ]
                )
                for rank in range(rank_count)
            )
            for window in windows
        )
        baseline_load = tuple(
            _rounded_median(tuple(vector[rank] for vector in inbound_vectors))
            for rank in range(rank_count)
        )
        p1_baseline = tuple(
            _rounded_median(tuple(vector[rank] for vector in p1_vectors))
            for rank in range(rank_count)
        )
        middle_baseline = tuple(
            _rounded_median(tuple(vector[rank] for vector in middle_vectors))
            for rank in range(rank_count)
        )
        tail_baseline = tuple(
            _rounded_median(tuple(vector[rank] for vector in tail_vectors))
            for rank in range(rank_count)
        )
        flattened_load = tuple(value for vector in inbound_vectors for value in vector)
        flattened_p1 = tuple(value for vector in p1_vectors for value in vector)
        flattened_tail = tuple(value for vector in tail_vectors for value in vector)
        layers.append(
            P12LayerRankTimingCalibration(
                layer_id=int(layer_id),
                dispatch_inbound_baseline_bytes=baseline_load,
                p1_source_ready_baseline_ns=p1_baseline,
                p1_source_ready_slope_ns_per_gib=_nonnegative_slope_ns_per_gib(
                    flattened_load, flattened_p1
                ),
                p1_to_p2_delay_ns=middle_baseline,
                dispatch_to_following_p1_tail_baseline_ns=tail_baseline,
                dispatch_tail_slope_ns_per_gib=_nonnegative_slope_ns_per_gib(
                    flattened_load, flattened_tail
                ),
            )
        )
    return P12RankTimingProfile(
        profile_id=str(profile_id),
        rank_count=rank_count,
        layers=tuple(layers),
        calibration_fixture_ids=tuple(str(fixture.fixture_id) for fixture in fixtures),
        calibration_dataset_ids=tuple(
            sorted(
                {
                    str(fixture.provenance.dataset_id)
                    for fixture in fixtures
                    if getattr(fixture, "provenance", None) is not None
                }
            )
        ),
        calibration_capture_ids=tuple(
            sorted(
                {
                    str(fixture.provenance.capture_id)
                    for fixture in fixtures
                    if getattr(fixture, "provenance", None) is not None
                }
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class P12PlanningTimingEstimate:
    estimate_id: str
    p1_source_ready_ns: tuple[int, ...]
    p1_to_p2_delay_ns: tuple[int, ...]
    p2_completion_tail_ns: tuple[int, ...]
    source_window_id: str
    timing_profile_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.estimate_id or not self.source_window_id:
            raise ValueError("timing estimate identifiers must be non-empty")
        vectors = (
            self.p1_source_ready_ns,
            self.p1_to_p2_delay_ns,
            self.p2_completion_tail_ns,
        )
        if not vectors[0] or len({len(vector) for vector in vectors}) != 1:
            raise ValueError("timing estimate vectors must be non-empty and aligned")
        for vector in vectors:
            if any(int(value) < 0 for value in vector):
                raise ValueError("timing estimates must be non-negative")

    @property
    def rank_count(self) -> int:
        return len(self.p1_to_p2_delay_ns)

    @property
    def estimate_digest(self) -> str:
        return stable_digest(
            {
                "schema": "P12_RANK_RELEASE_TIMING_ESTIMATE",
                "estimate_id": self.estimate_id,
                "p1_source_ready_ns": self.p1_source_ready_ns,
                "p1_to_p2_delay_ns": self.p1_to_p2_delay_ns,
                "p2_completion_tail_ns": self.p2_completion_tail_ns,
                "source_window_id": self.source_window_id,
                "timing_profile_digest": self.timing_profile_digest,
            }
        )


def uninformed_timing_estimate(rank_count: int) -> P12PlanningTimingEstimate:
    if int(rank_count) <= 0:
        raise ValueError("rank_count must be positive")
    zeros = (0,) * int(rank_count)
    return P12PlanningTimingEstimate(
        estimate_id="UNINFORMED_ZERO",
        p1_source_ready_ns=zeros,
        p1_to_p2_delay_ns=zeros,
        p2_completion_tail_ns=zeros,
        source_window_id="NO_CAUSAL_HISTORY",
    )


def _monotone_affine_fit(
    loads: tuple[int, ...], durations_ns: tuple[int, ...]
) -> tuple[int, int]:
    slope = _nonnegative_slope_ns_per_gib(loads, durations_ns)
    intercepts = tuple(
        int(duration) - int(round(int(load) * slope / _GIB_BYTES))
        for load, duration in zip(loads, durations_ns, strict=True)
    )
    return _rounded_median(intercepts), slope


def _predict_from_single_completed_window(
    *,
    previous_window: Any,
    target_matrix: tuple[tuple[int, ...], ...],
    duration_vector: tuple[int, ...],
    rank_count: int,
) -> tuple[int, ...]:
    previous_matrix = tuple(
        tuple(int(value) for value in row)
        for row in previous_window.payload_matrix("DISPATCH")
    )
    previous_inbound = _inbound_totals(previous_matrix, rank_count)
    target_inbound = _inbound_totals(target_matrix, rank_count)
    intercept, slope = _monotone_affine_fit(previous_inbound, duration_vector)
    return tuple(
        max(0, int(intercept) + int(round(int(load) * slope / _GIB_BYTES)))
        for load in target_inbound
    )


def causal_last_observed_timing_estimate(
    *,
    current_window: Any,
    previous_window: Any | None,
    predicted_p2_matrix: tuple[tuple[int, ...], ...] | None = None,
    timing_profile: P12RankTimingProfile | None = None,
    following_layer_id: int | None = None,
    p2_load_confidence_ppm: int = 0,
) -> P12PlanningTimingEstimate:
    rank_count = int(current_window.mapping.world_size)
    if timing_profile is not None:
        if timing_profile.rank_count != rank_count:
            raise ValueError("timing profile/current window rank count mismatch")
        current_calibration = timing_profile.layer(int(current_window.layer_id))
        target_layer_id = (
            int(following_layer_id)
            if following_layer_id is not None
            else int(current_window.layer_id) + 1
        )
        following_calibration = timing_profile.layer(target_layer_id)
        if current_calibration is not None and following_calibration is not None:
            current_matrix = tuple(
                tuple(int(value) for value in row)
                for row in current_window.payload_matrix("DISPATCH")
            )
            current_inbound = _inbound_totals(current_matrix, rank_count)
            target_matrix = predicted_p2_matrix or tuple(
                tuple(0 for _ in range(rank_count)) for _ in range(rank_count)
            )
            target_inbound = _inbound_totals(target_matrix, rank_count)
            return P12PlanningTimingEstimate(
                estimate_id="CALIBRATED_LAYER_RANK_RELEASE",
                p1_source_ready_ns=current_calibration.predict_p1_source_ready(
                    current_inbound
                ),
                p1_to_p2_delay_ns=current_calibration.p1_to_p2_delay_ns,
                p2_completion_tail_ns=following_calibration.predict_dispatch_tail(
                    target_inbound,
                    load_confidence_ppm=p2_load_confidence_ppm,
                ),
                source_window_id=timing_profile.profile_id,
                timing_profile_digest=timing_profile.profile_digest,
            )

    current_matrix = tuple(
        tuple(int(value) for value in row)
        for row in current_window.payload_matrix("DISPATCH")
    )
    if previous_window is None:
        p1_source = tuple(
            int(current_window.local_compute.dispatch_local_postprocess_ns[rank])
            for rank in range(rank_count)
        )
        p1_to_p2 = tuple(
            int(current_window.local_compute.bootstrap_router_and_pack_ns[rank])
            for rank in range(rank_count)
        )
        p2_tail = tuple(
            int(current_window.local_compute.dispatch_local_postprocess_ns[rank])
            for rank in range(rank_count)
        )
        source_window_id = f"{current_window.window_id}:bootstrap"
        estimate_id = "CAUSAL_BOOTSTRAP_RANK_RELEASE"
    else:
        if int(previous_window.mapping.world_size) != rank_count:
            raise ValueError("previous/current timing estimate rank count mismatch")
        previous_dispatch_tail = tuple(
            int(previous_window.local_compute.dispatch_local_postprocess_ns[rank])
            + int(
                previous_window.local_compute.dispatch_release_to_combine_source_ready_ns[
                    rank
                ]
            )
            for rank in range(rank_count)
        )
        previous_p1_source = tuple(
            int(previous_window.local_compute.dispatch_release_to_combine_source_ready_ns[rank])
            for rank in range(rank_count)
        )
        p1_source = _predict_from_single_completed_window(
            previous_window=previous_window,
            target_matrix=current_matrix,
            duration_vector=previous_p1_source,
            rank_count=rank_count,
        )
        p1_to_p2 = tuple(
            int(previous_window.local_compute.combine_release_to_router_ready_ns[rank])
            + int(previous_window.local_compute.router_and_pack_ns[rank])
            for rank in range(rank_count)
        )
        if predicted_p2_matrix is None:
            p2_tail = previous_dispatch_tail
        else:
            p2_tail = _predict_from_single_completed_window(
                previous_window=previous_window,
                target_matrix=predicted_p2_matrix,
                duration_vector=previous_dispatch_tail,
                rank_count=rank_count,
            )
        source_window_id = str(previous_window.window_id)
        estimate_id = (
            "CAUSAL_PREVIOUS_DISPATCH_LOAD_MODEL"
            if predicted_p2_matrix is not None
            else "CAUSAL_PREVIOUS_WINDOW_LAST_VALUE"
        )

    return P12PlanningTimingEstimate(
        estimate_id=estimate_id,
        p1_source_ready_ns=p1_source,
        p1_to_p2_delay_ns=p1_to_p2,
        p2_completion_tail_ns=p2_tail,
        source_window_id=source_window_id,
    )


__all__ = [
    "P12LayerRankTimingCalibration",
    "P12PlanningTimingEstimate",
    "P12RankTimingProfile",
    "build_rank_timing_profile",
    "causal_last_observed_timing_estimate",
    "load_rank_timing_profile",
    "save_rank_timing_profile",
    "uninformed_timing_estimate",
]
