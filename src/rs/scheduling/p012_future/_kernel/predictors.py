from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
from sklearn.linear_model import Ridge

from .contracts import TrafficHint
from .data import TrafficInstance, max_layer_by_model


def _round_full_row(values: np.ndarray, total: int) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if int(total) <= 0:
        return np.zeros_like(x, dtype=np.int32)
    if float(x.sum()) <= 0:
        x = np.ones_like(x, dtype=np.float64)
    x *= float(total) / float(x.sum())
    floor = np.floor(x).astype(np.int32)
    remainder = int(total - int(floor.sum()))
    if remainder > 0:
        order = np.argsort(-(x - floor), kind="stable")
        floor[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(x - floor, kind="stable")
        for index in order:
            if remainder == 0:
                break
            if floor[index] > 0:
                floor[index] -= 1
                remainder += 1
    return floor


def preserve_full_assignment_rows(prediction: np.ndarray, full_row_totals: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[0] != pred.shape[1]:
        raise ValueError("prediction must be square")
    totals = np.asarray(full_row_totals, dtype=np.int64).reshape(-1)
    if totals.shape[0] != pred.shape[0] or (totals < 0).any():
        raise ValueError("invalid full row totals")
    out = np.zeros(pred.shape, dtype=np.int32)
    for source in range(pred.shape[0]):
        out[source] = _round_full_row(pred[source], int(totals[source]))
    return out


def matrix_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64).copy()
    target = np.asarray(truth, dtype=np.float64).copy()
    np.fill_diagonal(pred, 0); np.fill_diagonal(target, 0)
    denom = float(np.abs(target).sum()) + 1e-12
    l1 = float(np.abs(pred - target).sum()) / denom
    pn = float(np.linalg.norm(pred)); tn = float(np.linalg.norm(target))
    cosine = float((pred * target).sum() / (pn * tn)) if pn > 0 and tn > 0 else (1.0 if pn == tn else 0.0)
    pr = pred.sum(1) + pred.sum(0); tr = target.sum(1) + target.sum(0)
    k = max(1, min(3, len(pr)))
    hot = len(set(np.argsort(-pr)[:k]) & set(np.argsort(-tr)[:k])) / k
    edge_k = max(1, min(10, pred.size - pred.shape[0]))
    pred_edges = sorted(((pred[s, d], s, d) for s in range(pred.shape[0]) for d in range(pred.shape[1]) if s != d), reverse=True)[:edge_k]
    true_edges = sorted(((target[s, d], s, d) for s in range(target.shape[0]) for d in range(target.shape[1]) if s != d), reverse=True)[:edge_k]
    recall = len({(s, d) for _, s, d in pred_edges} & {(s, d) for _, s, d in true_edges}) / edge_k
    return {
        "relative_l1": l1,
        "cosine": cosine,
        "hot_rank_overlap": float(hot),
        "top_edge_recall": float(recall),
        "remote_total_ratio": float(pred.sum()) / (float(target.sum()) + 1e-12),
    }


class TrafficPredictor(Protocol):
    predictor_id: str
    confidence: float
    def predict_matrix(self, instance: TrafficInstance) -> np.ndarray: ...


@dataclass
class RidgeTrafficPredictorV2:
    """Generic trace-level Ridge baseline with correct assignment conservation.

    The model predicts the *full* next-layer assignment matrix, including local
    assignments. Stable token ownership is enforced on these full row totals;
    only then is the diagonal removed to obtain remote communication work.
    """

    world_size: int
    alpha: float
    model: Ridge
    max_layers: dict[str, int]
    confidence: float
    predictor_id: str = "ridge_full_assignment_v2"

    @staticmethod
    def _feature(instance: TrafficInstance, max_layer: int) -> np.ndarray:
        p0 = instance.p0_full.astype(np.float64)
        scale = max(float(p0.sum()), 1.0)
        layer_fraction = float(instance.layer) / max(float(max_layer), 1.0)
        return np.concatenate([
            (p0 / scale).ravel(),
            p0.sum(axis=1) / scale,
            p0.sum(axis=0) / scale,
            np.array([math.log1p(scale) / 10.0, layer_fraction, 1.0], dtype=np.float64),
        ])

    @classmethod
    def fit(cls, instances: list[TrafficInstance], world_size: int, alpha: float = 10.0) -> "RidgeTrafficPredictorV2":
        rows = sorted((x for x in instances if x.split == "development" and x.world_size == int(world_size)), key=lambda x: x.instance_id)
        if not rows:
            raise ValueError(f"no development rows for world_size={world_size}")
        maxima = max_layer_by_model(instances)
        prompts = sorted({x.prompt_id for x in rows})
        calibration_prompts = set(prompts[max(1, int(len(prompts) * 0.8)):])
        train_rows = [x for x in rows if x.prompt_id not in calibration_prompts] or rows

        def fit_model(source: list[TrafficInstance]) -> Ridge:
            features = []
            targets = []
            for item in source:
                scale = max(float(item.p0_full.sum()), 1.0)
                features.append(cls._feature(item, maxima[item.model]))
                targets.append((item.p2_full.astype(np.float64) / scale).ravel())
            return Ridge(alpha=float(alpha), fit_intercept=True).fit(np.asarray(features), np.asarray(targets))

        calibration_model = fit_model(train_rows)
        calibration = [x for x in rows if x.prompt_id in calibration_prompts]
        metric_rows: list[dict[str, float]] = []
        for item in calibration:
            if item.is_last_layer:
                pred_remote = np.zeros((world_size, world_size), dtype=np.int32)
            else:
                scale = max(float(item.p0_full.sum()), 1.0)
                raw = calibration_model.predict(cls._feature(item, maxima[item.model])[None, :]).reshape(world_size, world_size) * scale
                full = preserve_full_assignment_rows(raw, item.p0_full.sum(axis=1))
                pred_remote = full.copy(); np.fill_diagonal(pred_remote, 0)
            metric_rows.append(matrix_metrics(pred_remote, item.p2))
        if metric_rows:
            median_cos = float(np.median([m["cosine"] for m in metric_rows]))
            median_l1 = float(np.median([m["relative_l1"] for m in metric_rows]))
            confidence = float(np.clip(0.5 * median_cos + 0.5 * (1.0 - min(median_l1, 1.0)), 0.0, 1.0))
        else:
            confidence = 0.5
        final_model = fit_model(rows)
        return cls(int(world_size), float(alpha), final_model, maxima, confidence)

    def predict_full(self, instance: TrafficInstance) -> np.ndarray:
        if instance.world_size != self.world_size:
            raise ValueError("world size mismatch")
        if instance.is_last_layer or instance.layer >= self.max_layers.get(instance.model, instance.layer):
            return np.zeros((self.world_size, self.world_size), dtype=np.int32)
        scale = max(float(instance.p0_full.sum()), 1.0)
        raw = self.model.predict(self._feature(instance, self.max_layers[instance.model])[None, :]).reshape(self.world_size, self.world_size) * scale
        return preserve_full_assignment_rows(raw, instance.p0_full.sum(axis=1))

    def predict_matrix(self, instance: TrafficInstance) -> np.ndarray:
        full = self.predict_full(instance)
        remote = full.copy(); np.fill_diagonal(remote, 0)
        return remote

    def predict_hint(self, instance: TrafficInstance) -> TrafficHint:
        return TrafficHint(
            predictor_id=self.predictor_id,
            target_dispatch_rows=self.predict_matrix(instance),
            confidence=float(self.confidence),
            hint_kind="learned_prediction",
            matrix_kind="remote_rows",
            metadata={"alpha": float(self.alpha), "conservation": "full_assignment_row_totals_then_remove_local"},
        )


@dataclass(frozen=True)
class ZeroPredictor:
    predictor_id: str = "zero_hint"
    confidence: float = 0.0
    def predict_matrix(self, instance: TrafficInstance) -> np.ndarray:
        return np.zeros((instance.world_size, instance.world_size), dtype=np.int32)


@dataclass(frozen=True)
class CopyCurrentPredictor:
    predictor_id: str = "copy_current_full_assignment"
    confidence: float = 0.25
    def predict_matrix(self, instance: TrafficInstance) -> np.ndarray:
        full = instance.p0_full.copy()
        if instance.is_last_layer:
            full.fill(0)
        np.fill_diagonal(full, 0)
        return full


__all__ = [
    "CopyCurrentPredictor", "RidgeTrafficPredictorV2", "TrafficPredictor", "ZeroPredictor",
    "matrix_metrics", "preserve_full_assignment_rows",
]
