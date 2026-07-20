from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from sklearn.linear_model import Ridge

from .artifacts import ForecastArtifact
from .contracts import (
    ForecastPlanningRequest, HomogeneousTopology, PlannerConstraints, TrafficHint,
    _digest, _freeze_metadata, semantic_metadata,
)
from .data import TrafficInstance, max_layer_by_model
from .event_core import bind_prepared_order, bind_template as bind_p2_template
from .plan import CompactWindowPlan, tuple_to_compact_plan
from .predictors import matrix_metrics
from .registry import build_planner


def _round_remote(values: np.ndarray, total: int) -> np.ndarray:
    matrix = np.maximum(np.asarray(values, dtype=np.float64), 0.0).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("bridge prediction must be square")
    np.fill_diagonal(matrix, 0.0)
    if total <= 0:
        return np.zeros(matrix.shape, dtype=np.int32)
    if float(matrix.sum()) <= 0:
        matrix.fill(1.0); np.fill_diagonal(matrix, 0.0)
    matrix *= float(total) / max(float(matrix.sum()), 1e-12)
    floor = np.floor(matrix).astype(np.int32)
    remainder = int(total - int(floor.sum()))
    cells = [
        (float(matrix[s, d] - floor[s, d]), s, d)
        for s in range(matrix.shape[0]) for d in range(matrix.shape[1]) if s != d
    ]
    if remainder > 0:
        for _, source, destination in sorted(cells, reverse=True)[:remainder]:
            floor[source, destination] += 1
    elif remainder < 0:
        for _, source, destination in sorted(cells):
            if remainder == 0:
                break
            if floor[source, destination] > 0:
                floor[source, destination] -= 1; remainder += 1
    np.fill_diagonal(floor, 0)
    return floor


def _instance_index(instances: list[TrafficInstance]) -> dict[tuple[str, str, int, int], TrafficInstance]:
    return {(x.model, x.prompt_id, x.world_size, x.layer): x for x in instances}


def _matrix_digest(matrix: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(matrix, dtype=np.int32))
    h = hashlib.sha256()
    h.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    h.update(values.tobytes())
    return h.hexdigest()


def _raw_template_digest(raw_template: tuple) -> str:
    """Stable binary digest for a bound template without Python list expansion."""
    h = hashlib.sha256()
    h.update(b"routersense-bound-template-v1\0")
    for index, value in enumerate(raw_template):
        h.update(int(index).to_bytes(2, "little", signed=False))
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            dtype = array.dtype.str.encode("ascii")
            h.update(b"A")
            h.update(len(dtype).to_bytes(2, "little", signed=False))
            h.update(dtype)
            h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            h.update(array.tobytes(order="C"))
        elif isinstance(value, (int, np.integer)):
            h.update(b"I")
            h.update(int(value).to_bytes(8, "little", signed=True))
        elif isinstance(value, (float, np.floating)):
            h.update(b"F")
            h.update(np.asarray([float(value)], dtype=np.float64).tobytes())
        else:
            encoded = repr(value).encode("utf-8")
            h.update(b"R")
            h.update(len(encoded).to_bytes(8, "little", signed=False))
            h.update(encoded)
    return h.hexdigest()


def _matrix_gate_metrics(predicted: np.ndarray, actual: np.ndarray, *, name: str) -> dict[str, float]:
    predicted64 = np.asarray(predicted, dtype=np.int64)
    actual64 = np.asarray(actual, dtype=np.int64)
    if predicted64.shape != actual64.shape:
        raise ValueError(f"predicted/actual {name} shape mismatch")
    actual_support = actual64 > 0
    predicted_support = predicted64 > 0
    overlap = int(np.count_nonzero(actual_support & predicted_support))
    actual_edges = int(np.count_nonzero(actual_support))
    predicted_edges = int(np.count_nonzero(predicted_support))
    total = max(int(actual64.sum()), 1)
    # Empty/empty is a perfect structural prediction.  Empty truth with false
    # positives has recall 1 but precision 0; missing all real edges is the
    # converse.  This avoids rejecting genuinely empty target windows.
    support_recall = 1.0 if actual_edges == 0 else float(overlap) / float(actual_edges)
    support_precision = 1.0 if predicted_edges == 0 else float(overlap) / float(predicted_edges)
    return {
        "relative_l1": float(np.abs(predicted64 - actual64).sum()) / float(total),
        "row_relative_l1": float(np.abs(predicted64.sum(axis=1) - actual64.sum(axis=1)).sum()) / float(total),
        "column_relative_l1": float(np.abs(predicted64.sum(axis=0) - actual64.sum(axis=0)).sum()) / float(total),
        "support_recall": support_recall,
        "support_precision": support_precision,
        "actual_edges": float(actual_edges),
        "predicted_edges": float(predicted_edges),
    }


def _p01_prediction_gate_metrics(
    predicted_p0: np.ndarray, actual_p0: np.ndarray, actual_p1: np.ndarray,
) -> dict[str, Any]:
    """Cheap O(N^2) target gate using only already-visible P0/P1 truth.

    P1 is predicted from the P0 transpose under the normal MoE return contract.
    P2 truth is deliberately absent, so this gate cannot leak future traffic.
    """
    predicted0 = np.asarray(predicted_p0, dtype=np.int64)
    predicted1 = np.ascontiguousarray(predicted0.T)
    p0 = _matrix_gate_metrics(predicted0, actual_p0, name="P0")
    p1 = _matrix_gate_metrics(predicted1, actual_p1, name="P1")
    return {**p0, "p0": p0, "p1": p1}


class SecondHopPredictor(Protocol):
    """Minimal Future-P012 second-hop predictor contract."""

    model_name: str
    world_size: int

    def predict_hint(
        self, first_hop_rows: np.ndarray, *, source_layer: int
    ) -> TrafficHint: ...


@dataclass
class BridgeTrafficPredictor:
    """Development-only second-hop predictor for Future-P012.

    It maps an already available prediction of D_(L+1) to a low-confidence
    rank-traffic hint for D_(L+2).  It never predicts or executes model values.
    """

    model_name: str
    world_size: int
    alpha: float
    model: Ridge
    max_layer: int
    confidence: float
    calibration: Mapping[str, Any]
    predictor_id: str = "future_bridge_ridge_v1"

    @staticmethod
    def _feature(matrix: np.ndarray, source_layer: int, max_layer: int) -> np.ndarray:
        rows = np.asarray(matrix, dtype=np.float64).copy(); np.fill_diagonal(rows, 0.0)
        scale = max(float(rows.sum()), 1.0)
        return np.concatenate([
            (rows / scale).ravel(),
            rows.sum(axis=1) / scale,
            rows.sum(axis=0) / scale,
            np.asarray([
                math.log1p(scale) / 10.0,
                float(source_layer) / max(float(max_layer), 1.0),
                1.0,
            ], dtype=np.float64),
        ])

    @classmethod
    def fit(
        cls,
        instances: list[TrafficInstance],
        *,
        model_name: str,
        world_size: int,
        first_hop_provider: Callable[[TrafficInstance], np.ndarray] | None = None,
        alpha: float = 10.0,
    ) -> "BridgeTrafficPredictor":
        index = _instance_index(instances)
        maxima = max_layer_by_model(instances)
        pairs: list[tuple[TrafficInstance, TrafficInstance, np.ndarray]] = []
        for source in instances:
            if source.split != "development" or source.model != model_name or source.world_size != int(world_size):
                continue
            target = index.get((source.model, source.prompt_id, source.world_size, source.layer + 1))
            if target is None or target.is_last_layer:
                continue
            first = source.p2 if first_hop_provider is None else np.asarray(first_hop_provider(source), dtype=np.int32)
            if first.shape != (world_size, world_size):
                raise ValueError("first-hop provider returned wrong matrix shape")
            pairs.append((source, target, first))
        if len(pairs) < 2:
            raise ValueError(f"insufficient development pairs for {model_name}/vEP{world_size}")
        prompts = sorted({source.prompt_id for source, _, _ in pairs})
        calibration_prompts = set(prompts[max(1, int(len(prompts) * 0.75)):])
        train_pairs = [row for row in pairs if row[0].prompt_id not in calibration_prompts] or pairs

        def fit_model(rows: list[tuple[TrafficInstance, TrafficInstance, np.ndarray]]) -> Ridge:
            x: list[np.ndarray] = []; y: list[np.ndarray] = []
            for source, target, first in rows:
                scale = max(float(first.sum()), 1.0)
                x.append(cls._feature(first, source.layer, maxima[model_name]))
                y.append((target.p2.astype(np.float64) / scale).ravel())
            return Ridge(alpha=float(alpha), fit_intercept=True).fit(np.asarray(x), np.asarray(y))

        calibration_model = fit_model(train_pairs)
        calibration_rows = [row for row in pairs if row[0].prompt_id in calibration_prompts]
        metrics: list[dict[str, float]] = []
        for source, target, first in calibration_rows:
            scale = max(float(first.sum()), 1.0)
            raw = calibration_model.predict(
                cls._feature(first, source.layer, maxima[model_name])[None, :]
            ).reshape(world_size, world_size) * scale
            metrics.append(matrix_metrics(_round_remote(raw, int(round(scale))), target.p2))
        if metrics:
            median_cosine = float(np.median([x["cosine"] for x in metrics]))
            median_l1 = float(np.median([x["relative_l1"] for x in metrics]))
            confidence = float(np.clip(0.4 * median_cosine + 0.6 * (1.0 - min(median_l1, 1.0)), 0.05, 0.75))
        else:
            median_cosine, median_l1, confidence = 0.0, 1.0, 0.10
        return cls(
            str(model_name), int(world_size), float(alpha), fit_model(pairs), int(maxima[model_name]),
            confidence,
            {
                "development_pairs": len(pairs),
                "calibration_pairs": len(calibration_rows),
                "median_cosine": median_cosine,
                "median_relative_l1": median_l1,
                "training_split": "development_only",
            },
        )

    def predict_matrix(self, first_hop_rows: np.ndarray, *, source_layer: int) -> np.ndarray:
        rows = np.asarray(first_hop_rows, dtype=np.int32)
        if rows.shape != (self.world_size, self.world_size):
            raise ValueError("bridge input world size mismatch")
        # The target layer is source_layer + 1.  When it is the final MoE layer
        # there is no second-hop dispatch, so Future-P012 must carry a terminal
        # zero hint rather than extrapolating traffic beyond the model.
        if int(source_layer) + 1 >= int(self.max_layer):
            return np.zeros_like(rows)
        total = int(rows.sum())
        raw = self.model.predict(
            self._feature(rows, int(source_layer), self.max_layer)[None, :]
        ).reshape(self.world_size, self.world_size) * max(float(total), 1.0)
        return _round_remote(raw, total)

    def predict_hint(self, first_hop_rows: np.ndarray, *, source_layer: int) -> TrafficHint:
        matrix = self.predict_matrix(first_hop_rows, source_layer=source_layer)
        terminal = int(source_layer) + 1 >= int(self.max_layer)
        return TrafficHint(
            predictor_id=f"{self.predictor_id}:{self.model_name}:vep{self.world_size}",
            target_dispatch_rows=matrix,
            confidence=0.0 if terminal else float(self.confidence),
            hint_kind="zero_hint" if terminal else "learned_prediction",
            metadata={
                "future_hop": 2,
                "source_layer": int(source_layer),
                "terminal_target_layer": bool(terminal),
                "alpha": self.alpha,
                "calibration": dict(self.calibration),
            },
        )


@dataclass(frozen=True)
class FuturePlanKey:
    model: str
    prompt_id: str
    target_layer: int
    world_size: int
    variant_id: str

    def to_string(self) -> str:
        return (
            f"{self.model}:{self.prompt_id}:layer{int(self.target_layer)}:"
            f"vep{int(self.world_size)}:{self.variant_id}"
        )

    def matches(self, item: TrafficInstance) -> bool:
        return (
            self.model, self.prompt_id, int(self.target_layer), int(self.world_size)
        ) == (item.model, item.prompt_id, int(item.layer), int(item.world_size))


@dataclass(frozen=True)
class PreparedOrderTemplate:
    """Ahead-of-time P01 matching skeleton compiled from one P012 search."""

    p01_raw_template: tuple
    raw_template: tuple
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        def freeze_template(template: tuple) -> tuple:
            frozen: list[Any] = []
            for value in template:
                if isinstance(value, np.ndarray):
                    array = np.ascontiguousarray(value)
                    array.setflags(write=False)
                    frozen.append(array)
                else:
                    frozen.append(value)
            return tuple(frozen)

        object.__setattr__(self, "p01_raw_template", freeze_template(self.p01_raw_template))
        object.__setattr__(self, "raw_template", freeze_template(self.raw_template))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "prepared_order_semantic_version": "prepared_p01_order_v2",
            "p01_raw_template_digest": _raw_template_digest(self.p01_raw_template),
            "source_p012_template_digest": _raw_template_digest(self.raw_template),
            "metadata": semantic_metadata(self.metadata),
        }

    def semantic_digest(self) -> str:
        cached = self._digest_cache
        if cached is None:
            cached = _digest(self.semantic_payload())
            object.__setattr__(self, "_digest_cache", cached)
        return cached


def _compile_prepared_order(artifact: ForecastArtifact, world_size: int) -> PreparedOrderTemplate:
    phase = np.asarray(artifact.raw_template[2], dtype=np.int8)
    destination = np.asarray(artifact.raw_template[3], dtype=np.int16)
    size = np.asarray(artifact.raw_template[4], dtype=np.int32)
    n = int(world_size)
    if phase.ndim != 2 or phase.shape[1] != n:
        raise ValueError("forecast template/world-size mismatch")

    keep = np.flatnonzero(np.any((phase >= 0) & (phase < 2), axis=1))
    p01_phase = np.ascontiguousarray(phase[keep].copy())
    p01_destination = np.ascontiguousarray(destination[keep].copy())
    p01_size = np.ascontiguousarray(size[keep].copy())
    valid = (p01_phase >= 0) & (p01_phase < 2)
    p01_phase[~valid] = -1
    p01_destination[~valid] = -1
    p01_size[~valid] = 0

    waves = int(keep.size)
    zeros = np.zeros(waves, dtype=np.float64)
    p01_template = (
        0.0, waves, p01_phase, p01_destination, p01_size,
        np.zeros(waves, dtype=np.int32), zeros.copy(), zeros.copy(), zeros.copy(),
        np.zeros(3, dtype=np.float64),
        np.full(n, -1.0, dtype=np.float64),
        np.full(n, -1.0, dtype=np.float64),
        1,
    )
    return PreparedOrderTemplate(
        p01_template,
        artifact.raw_template,
        {
            "compile_strategy": "forecast_p01_matching_skeleton",
            "source_p012_waves": int(phase.shape[0]),
            "prepared_p01_waves": waves,
            "world_size": n,
            "online_matching_solver": False,
            "online_candidate_selection": False,
        },
    )


@dataclass(frozen=True)
class FutureWindowPlan:
    key: FuturePlanKey
    source_layer: int
    first_hop_hint: TrafficHint
    second_hop_hint: TrafficHint
    forecast_request: ForecastPlanningRequest
    forecast_artifact: ForecastArtifact
    conservative_artifact: ForecastArtifact
    robust_artifact: ForecastArtifact
    prepared_template: PreparedOrderTemplate
    generated_at_ns: int
    prediction_ms: float
    planning_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "future_plan_semantic_version": "future_p012_v4_prepared_order",
            "key": self.key.to_string(),
            "source_layer": int(self.source_layer),
            "first_hop_hint": self.first_hop_hint.semantic_digest(),
            "second_hop_hint": self.second_hop_hint.semantic_digest(),
            "forecast_request": self.forecast_request.semantic_digest(),
            "forecast_artifact": self.forecast_artifact.semantic_digest(),
            "conservative_artifact": self.conservative_artifact.semantic_digest(),
            "robust_artifact": self.robust_artifact.semantic_digest(),
            "prepared_template": self.prepared_template.semantic_digest(),
            "metadata": semantic_metadata(self.metadata),
        }

    def semantic_digest(self) -> str:
        cached = self._digest_cache
        if cached is None:
            cached = _digest(self.semantic_payload()); object.__setattr__(self, "_digest_cache", cached)
        return cached


@dataclass(frozen=True)
class FutureLeadingPlan:
    """Target-frontier result produced without observing target P2 truth."""

    future_plan_digest: str
    key: FuturePlanKey
    target_instance_id: str
    p0_digest: str
    p1_digest: str
    selected_artifact_digest: str
    raw_template: tuple
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)
    _raw_template_digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        frozen = []
        for value in self.raw_template:
            if isinstance(value, np.ndarray):
                arr = np.ascontiguousarray(value); arr.setflags(write=False); frozen.append(arr)
            else:
                frozen.append(value)
        object.__setattr__(self, "raw_template", tuple(frozen))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "future_leading_plan_semantic_version": "future_leading_p012_v1",
            "future_plan_digest": self.future_plan_digest,
            "key": self.key.to_string(),
            "target_instance_id": self.target_instance_id,
            "p0_digest": self.p0_digest,
            "p1_digest": self.p1_digest,
            "selected_artifact_digest": self.selected_artifact_digest,
            "raw_template_digest": self.raw_template_digest(),
            "metadata": semantic_metadata(self.metadata),
        }

    def raw_template_digest(self) -> str:
        cached = self._raw_template_digest_cache
        if cached is None:
            cached = _raw_template_digest(self.raw_template)
            object.__setattr__(self, "_raw_template_digest_cache", cached)
        return cached

    def semantic_digest(self) -> str:
        cached = self._digest_cache
        if cached is None:
            cached = _digest(self.semantic_payload()); object.__setattr__(self, "_digest_cache", cached)
        return cached



class FutureP012Planner:
    """Run the frozen P012 planner on the previous layer and bind at target."""

    def __init__(
        self, *, family: str = "rscf", branch: str = "global",
        max_p0_relative_l1: float = 0.12,
        max_p1_relative_l1: float = 0.12,
        min_p0_support_recall: float = 0.98,
        min_p0_support_precision: float = 0.80,
    ) -> None:
        if branch not in {"event", "global"}:
            raise ValueError("Future-P012 supports event/global branches")
        if max_p0_relative_l1 < 0 or max_p1_relative_l1 < 0:
            raise ValueError("P0/P1 relative-L1 thresholds must be non-negative")
        for name, value in (
            ("min_p0_support_recall", min_p0_support_recall),
            ("min_p0_support_precision", min_p0_support_precision),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.family = str(family); self.branch = str(branch)
        self.max_p0_relative_l1 = float(max_p0_relative_l1)
        self.max_p1_relative_l1 = float(max_p1_relative_l1)
        self.min_p0_support_recall = float(min_p0_support_recall)
        self.min_p0_support_precision = float(min_p0_support_precision)
        self._planner = build_planner(scope="joint", engine=branch, family=family)
        self.planner_id = f"future:p012:joint:{branch}:{family}"

    def preplan(
        self,
        source: TrafficInstance,
        first_hop_hint: TrafficHint,
        bridge: SecondHopPredictor,
        topology: HomogeneousTopology,
        cost_model: Any,
        constraints: PlannerConstraints,
    ) -> FutureWindowPlan:
        if source.is_last_layer:
            raise ValueError("cannot preplan beyond a terminal MoE layer")
        if int(topology.world_size) != int(source.world_size):
            raise ValueError("future topology world size does not match source instance")
        if bridge.model_name != source.model or int(bridge.world_size) != int(source.world_size):
            raise ValueError("bridge predictor does not match source model/world size")
        first_hop_hint.validate(world_size=source.world_size)
        prediction_start = time.perf_counter_ns()
        second_hop = bridge.predict_hint(first_hop_hint.matrix(), source_layer=source.layer)
        prediction_ms = (time.perf_counter_ns() - prediction_start) / 1e6
        predicted_p0 = first_hop_hint.matrix()
        predicted_p1 = np.ascontiguousarray(predicted_p0.T)
        # The store key must distinguish not only policy/predictor names but
        # also the concrete hint, topology, cost profile, and constraints.  Two
        # asynchronous plans for the same layer under different deployment
        # profiles must coexist rather than collide at publication time.
        variant = _digest({
            "branch": self.branch,
            "family": self.family,
            "first_hop_hint": first_hop_hint.semantic_digest(),
            "second_hop_hint": second_hop.semantic_digest(),
            "topology": topology.to_dict(),
            "cost_model": cost_model.to_dict(),
            "constraints": constraints.to_dict(),
        })[:12]
        key = FuturePlanKey(source.model, source.prompt_id, source.layer + 1, source.world_size, variant)
        request = ForecastPlanningRequest(
            predicted_p0, predicted_p1, second_hop, topology, cost_model, constraints,
            request_id=f"future:{key.to_string()}",
        )
        planning_start = time.perf_counter_ns()
        # One ahead-of-time P012 search is enough.  The old implementation built
        # three full candidates and repeated target-side binds to choose among
        # them; traces showed that almost every window selected the same guard.
        full_artifact = self._planner.plan_forecast(request)
        prepared = _compile_prepared_order(full_artifact, source.world_size)
        planning_ms = (time.perf_counter_ns() - planning_start) / 1e6
        future = FutureWindowPlan(
            key, source.layer, first_hop_hint, second_hop, request,
            full_artifact, full_artifact, full_artifact, prepared,
            time.time_ns(), prediction_ms, planning_ms,
            {
                "planning_horizon": "p012",
                "execution_horizon": "p012",
                "planning_timing": "previous_layer",
                "source_layer": source.layer,
                "target_layer": source.layer + 1,
                "target_visible_full_planner": False,
                "target_visible_candidate_selection": False,
                "planner_id": self._planner.planner_id,
                "branch": self.branch,
                "family": self.family,
                "future_contract": "prepared_order_template_v1",
                "preplanned_candidate_count": 1,
                "p0_gate_max_relative_l1": self.max_p0_relative_l1,
                "p1_gate_max_relative_l1": self.max_p1_relative_l1,
                "p0_gate_min_support_recall": self.min_p0_support_recall,
                "p0_gate_min_support_precision": self.min_p0_support_precision,
            },
        )
        # Cache all semantic certificates while the previous layer is running.
        prepared.semantic_digest(); full_artifact.semantic_digest(); future.semantic_digest()
        return future

    @staticmethod
    def _prepared_leading_bind(
        plan: FutureWindowPlan, target: TrafficInstance,
        raw_template: tuple | None = None,
    ) -> tuple[tuple, dict[str, Any], float]:
        slope, intercept = plan.forecast_request.cost_matrices()
        zero_p2 = np.zeros_like(target.p0)
        selected_template = (
            plan.prepared_template.p01_raw_template if raw_template is None else raw_template
        )
        started = time.perf_counter_ns()
        result = bind_prepared_order(
            (target.p0, target.p1, zero_p2), selected_template,
            edge_slope=slope, edge_intercept=intercept,
            expert_compute_delay=float(plan.forecast_request.constraints.expert_compute_delay),
            wave_launch_b=float(plan.forecast_request.cost_model.wave_launch_b),
            max_waves=int(plan.forecast_request.constraints.max_waves),
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        metrics = {
            "fast_bind_total_ms": float(elapsed_ms),
            "bind_strategy": "prepared_order_stable_filter",
            "template_rows_served": int(result[13]),
            "template_waves_used": int(result[14]),
            "projected_template_edges": int(result[15]),
            "topology_tail_rows": int(result[16]),
            "topology_tail_waves": int(result[17]),
            "template_support_coverage": float(result[18]),
            # Compatibility keys: the expensive optimizer-style repair no longer exists.
            "suffix_repair_rows": 0,
            "suffix_repair_waves": 0,
            "suffix_repair_ms": 0.0,
            "online_matching_solver_calls": 0,
        }
        return result[:13], metrics, float(elapsed_ms)

    def bind_leading(self, future: FutureWindowPlan, target: TrafficInstance) -> FutureLeadingPlan:
        """Consume a prepared P01 order behind a cheap P01 quality gate.

        A sufficiently close predicted dispatch/return pair takes the no-search
        prepared path.  A poor prediction triggers exactly one on-demand P012
        forecast plan; it is never repaired or compared with multiple
        candidates.  P2 truth is not observed by either path.
        """
        visible_started = time.perf_counter_ns()
        if not future.key.matches(target):
            raise ValueError("future plan target identity mismatch")
        future_digest = future.semantic_digest()  # cached during preplan/publish

        gate_started = time.perf_counter_ns()
        gate_metrics = _p01_prediction_gate_metrics(
            future.first_hop_hint.matrix(), target.p0, target.p1
        )
        gate_accepted = (
            gate_metrics["p0"]["relative_l1"] <= self.max_p0_relative_l1
            and gate_metrics["p1"]["relative_l1"] <= self.max_p1_relative_l1
            and gate_metrics["p0"]["support_recall"] >= self.min_p0_support_recall
            and gate_metrics["p0"]["support_precision"] >= self.min_p0_support_precision
            and gate_metrics["p1"]["support_recall"] >= self.min_p0_support_recall
            and gate_metrics["p1"]["support_precision"] >= self.min_p0_support_precision
        )
        gate_ms = (time.perf_counter_ns() - gate_started) / 1e6

        fallback_plan_ms = 0.0
        if gate_accepted:
            selected_name = "prepared_p012_order"
            selected_template = future.prepared_template.p01_raw_template
            selected_digest = future.prepared_template.semantic_digest()
            target_visible_full_planner = False
        else:
            selected_name = "on_demand_p012_gate_fallback"
            fallback_request = ForecastPlanningRequest(
                target.p0, target.p1, future.second_hop_hint,
                future.forecast_request.topology, future.forecast_request.cost_model,
                future.forecast_request.constraints,
                request_id=f"future-fallback:{future.key.to_string()}:{target.instance_id}",
            )
            fallback_started = time.perf_counter_ns()
            fallback_artifact = self._planner.plan_forecast(fallback_request)
            # The fallback artifact is already a valid matching skeleton.  Feed it
            # directly to the stable-filter binder; recompiling P2 orders here
            # would only add target-visible Python work.
            selected_template = fallback_artifact.raw_template
            selected_digest = _raw_template_digest(selected_template)
            fallback_plan_ms = (time.perf_counter_ns() - fallback_started) / 1e6
            target_visible_full_planner = True

        leading_template, bind_metrics, leading_bind_ms = self._prepared_leading_bind(
            future, target, selected_template
        )
        bind_metrics = {
            **bind_metrics,
            "p01_gate_accepted": bool(gate_accepted),
            "p01_gate_metrics": gate_metrics,
            # Backward-compatible observability aliases retained for existing
            # result readers; the gate now validates both P0 and P1.
            "p0_gate_accepted": bool(gate_accepted),
            "p0_gate_metrics": gate_metrics,
            "p0_gate_ms": float(gate_ms),
            "fallback_plan_ms": float(fallback_plan_ms),
        }
        target_visible_ms = (time.perf_counter_ns() - visible_started) / 1e6
        metadata = {
            **dict(future.metadata),
            "selected_future_candidate": selected_name,
            "selection_uses_p2_truth": False,
            "selection_uses_actual_p0_p1": True,
            "second_hop_confidence": float(future.second_hop_hint.confidence),
            "future_plan_digest": future_digest,
            "future_template_digest": selected_digest,
            "prediction_ms_hidden": float(future.prediction_ms),
            "planning_ms_hidden": float(future.planning_ms),
            "target_visible_full_planner": bool(target_visible_full_planner),
            "target_visible_selection_ms": 0.0,
            "target_visible_gate_ms": float(gate_ms),
            "target_visible_fallback_plan_ms": float(fallback_plan_ms),
            "target_visible_bind_only_ms": float(leading_bind_ms),
            "target_visible_bind_ms": float(target_visible_ms),
            "forecast_truth_isolation": True,
            "future_bind_strategy": (
                "prepared_order_stable_filter" if gate_accepted
                else "p01_gate_fallback_single_on_demand_plan"
            ),
            "bind_metrics": bind_metrics,
        }
        leading = FutureLeadingPlan(
            future_digest, future.key, target.instance_id,
            _matrix_digest(target.p0),
            _matrix_digest(target.p1),
            selected_digest, leading_template, metadata,
        )
        # Binary template hashing avoids list/JSON expansion on the target path.
        leading.raw_template_digest(); leading.semantic_digest()
        return leading

    def reveal_p2(
        self, future: FutureWindowPlan, leading: FutureLeadingPlan, target: TrafficInstance,
    ) -> CompactWindowPlan:
        """Bind real P2 after the prepared P01 prefix.

        The current P2 truth binder is the legacy correctness-preserving kernel
        and may perform residual P2 matching.  Only the Future-P012 target prefix
        is solver-free in this release.
        """
        future_digest = future.semantic_digest()
        if leading.future_plan_digest != future_digest:
            raise ValueError("leading plan does not belong to future plan")
        if leading.target_instance_id != target.instance_id or not future.key.matches(target):
            raise ValueError("future leading plan target identity mismatch")
        if leading.p0_digest != _matrix_digest(target.p0):
            raise ValueError("target P0 changed after leading bind")
        if leading.p1_digest != _matrix_digest(target.p1):
            raise ValueError("target P1 changed after leading bind")
        slope, intercept = future.forecast_request.cost_matrices()
        p2_bind_started = time.perf_counter_ns()
        generic = bind_p2_template(
            (target.p0, target.p1, target.p2), future.second_hop_hint.matrix(), leading.raw_template,
            edge_slope=slope, edge_intercept=intercept,
            expert_compute_delay=float(future.forecast_request.constraints.expert_compute_delay),
            wave_launch_b=float(future.forecast_request.cost_model.wave_launch_b),
            max_waves=int(future.forecast_request.constraints.max_waves),
        )
        p2_reveal_bind_ms = (time.perf_counter_ns() - p2_bind_started) / 1e6
        request_digest = hashlib.sha256(
            (
                "future-bound-v1:" + future.forecast_request.semantic_digest() + ":" +
                leading.p0_digest + ":" + leading.p1_digest + ":" + target.instance_id
            ).encode()
        ).hexdigest()
        metadata = {
            **dict(leading.metadata),
            "future_leading_plan_digest": leading.semantic_digest(),
            "p2_reveal_bind_ms": float(p2_reveal_bind_ms),
            "p2_bind_strategy": "frozen_p01_prefix_then_legacy_truth_bind",
            "p2_bind_kernel": "legacy_bind_template",
            "p2_solver_free": False,
            "prediction_metrics_deferred": True,
        }
        return tuple_to_compact_plan(
            generic,
            planner_id=self.planner_id,
            planner_family="future_p012",
            branch=f"future_{self.branch}",
            request_digest=request_digest,
            forecast=False,
            metadata=metadata,
            trusted_arrays=True,
        )

    def bind(self, future: FutureWindowPlan, target: TrafficInstance) -> CompactWindowPlan:
        """Offline convenience wrapper preserving the two-stage runtime contract."""
        leading = self.bind_leading(future, target)
        return self.reveal_p2(future, leading, target)


def warmup_future_bind_kernel() -> None:
    """Compile prepared-P01 and later P2 truth binders outside measurements."""
    n = 2
    p0 = np.asarray([[0, 2], [1, 0]], dtype=np.int32)
    p1 = np.ascontiguousarray(p0.T)
    p2 = p0.copy()
    phase = np.asarray([[0, 0], [1, 1]], dtype=np.int8)
    dst = np.asarray([[1, 0], [1, 0]], dtype=np.int16)
    size = np.asarray([[2, 1], [1, 2]], dtype=np.int32)
    zeros = np.zeros(2, dtype=np.float64)
    template = (
        0.0, 2, phase, dst, size, np.ones(2, dtype=np.int32),
        zeros.copy(), zeros.copy(), zeros.copy(), np.zeros(3, dtype=np.float64),
        np.full(n, -1.0, dtype=np.float64), np.full(n, -1.0, dtype=np.float64), 1,
    )
    prepared = bind_prepared_order((p0, p1, np.zeros_like(p0)), template, max_waves=64)
    revealed = bind_p2_template((p0, p1, p2), p2, prepared[:13], max_waves=64)
    if int(prepared[12]) != 1 or int(revealed[12]) != 1:
        raise RuntimeError("Future-P012 binder warmup failed")


__all__ = [
    "BridgeTrafficPredictor",
    "SecondHopPredictor", "FutureP012Planner", "FuturePlanKey",
    "FutureLeadingPlan", "FutureWindowPlan", "PreparedOrderTemplate", "warmup_future_bind_kernel",
]
