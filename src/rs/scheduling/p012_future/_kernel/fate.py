from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .contracts import TrafficHint


@dataclass(frozen=True)
class FateExpertPrediction:
    """FATE cross-layer prediction output before placement aggregation."""

    source_ranks: np.ndarray
    expert_scores: np.ndarray
    prefetch_mask: np.ndarray
    expected_assignment_mass: np.ndarray
    top_k: int
    predictor_id: str = "fate_cross_layer_gate_v1"

    def validate(self) -> None:
        src = np.asarray(self.source_ranks)
        scores = np.asarray(self.expert_scores)
        mask = np.asarray(self.prefetch_mask)
        mass = np.asarray(self.expected_assignment_mass)
        if src.ndim != 1 or scores.ndim != 2 or scores.shape[0] != src.shape[0]:
            raise ValueError("invalid source_ranks/expert_scores shape")
        if mask.shape != scores.shape or mass.shape != scores.shape:
            raise ValueError("prefetch_mask and expected_assignment_mass must match scores")
        if not np.isfinite(scores).all() or not np.isfinite(mass).all() or (mass < 0).any():
            raise ValueError("FATE scores/mass must be finite and non-negative")
        if int(self.top_k) <= 0 or int(self.top_k) > scores.shape[1]:
            raise ValueError("invalid top_k")
        row_mass = mass.sum(axis=1)
        if not np.allclose(row_mass, float(self.top_k), atol=1e-5):
            raise ValueError("expected assignment mass must sum to top_k per token")


class NextLayerGate(Protocol):
    def __call__(self, gate_input: np.ndarray) -> np.ndarray: ...


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(values, -60.0, 60.0))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class FateCrossLayerGatePredictor:
    """Faithful implementation of FATE's cross-layer gate direction.

    The current MoE block's gate input is evaluated by the *next* block's gate.
    Experts above the per-token routing-score percentile form the prefetch set.
    For RouterSense traffic planning, their probabilities are normalized to the
    model's top-k assignment mass, producing expected token-copy counts rather
    than executable bytes.
    """

    percentile: float = 75.0
    min_candidates: int | None = None
    predictor_id: str = "fate_cross_layer_gate_v1"

    def __post_init__(self) -> None:
        percentile = float(self.percentile)
        if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be finite and within [0, 100]")
        if self.min_candidates is not None and int(self.min_candidates) <= 0:
            raise ValueError("min_candidates must be positive when provided")
        if not str(self.predictor_id):
            raise ValueError("predictor_id must be non-empty")

    def predict_from_gate_output(
        self,
        gate_output: np.ndarray,
        source_ranks: np.ndarray,
        *,
        top_k: int,
        gate_output_domain: str = "logits",
    ) -> FateExpertPrediction:
        src = np.asarray(source_ranks, dtype=np.int64).reshape(-1)
        raw = np.asarray(gate_output, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[0] != src.shape[0]:
            raise ValueError("gate output and source_ranks token counts must match")
        if not np.isfinite(raw).all():
            raise ValueError("next-layer gate output must be finite")
        if gate_output_domain == "logits":
            scores = _softmax(raw)
        elif gate_output_domain in {"probabilities", "nonnegative_scores"}:
            if (raw < 0).any():
                raise ValueError("non-logit gate output must be non-negative")
            totals = raw.sum(axis=1, keepdims=True)
            scores = np.empty_like(raw, dtype=np.float64)
            nonzero = totals[:, 0] > 0
            scores[nonzero] = raw[nonzero] / totals[nonzero]
            # A zero score row contains no ranking information.  Treat it as a
            # uniform prior rather than producing zero expected assignment mass.
            if np.any(~nonzero):
                scores[~nonzero] = 1.0 / float(raw.shape[1])
        else:
            raise ValueError(f"unsupported gate_output_domain {gate_output_domain!r}")
        if int(top_k) <= 0 or int(top_k) > scores.shape[1]:
            raise ValueError("invalid top_k")

        threshold = np.percentile(scores, float(self.percentile), axis=1, keepdims=True)
        mask = scores >= threshold
        minimum = max(int(top_k), int(self.min_candidates or top_k))
        if minimum > scores.shape[1]:
            raise ValueError("min_candidates cannot exceed next-layer expert count")
        for token in range(scores.shape[0]):
            if int(mask[token].sum()) < minimum:
                indices = np.argsort(-scores[token], kind="stable")[:minimum]
                mask[token, indices] = True
        selected = np.where(mask, scores, 0.0)
        denom = np.maximum(selected.sum(axis=1, keepdims=True), 1e-12)
        expected_mass = selected / denom * float(top_k)
        result = FateExpertPrediction(
            source_ranks=src,
            expert_scores=scores,
            prefetch_mask=mask,
            expected_assignment_mass=expected_mass,
            top_k=int(top_k),
            predictor_id=self.predictor_id,
        )
        result.validate()
        return result

    def predict_experts(
        self,
        current_gate_input: np.ndarray,
        next_layer_gate: NextLayerGate | Callable[[np.ndarray], np.ndarray],
        source_ranks: np.ndarray,
        *,
        top_k: int,
        gate_output_domain: str = "logits",
    ) -> FateExpertPrediction:
        hidden = np.asarray(current_gate_input)
        src = np.asarray(source_ranks, dtype=np.int64).reshape(-1)
        if hidden.ndim != 2 or hidden.shape[0] != src.shape[0]:
            raise ValueError("current_gate_input and source_ranks token counts must match")
        raw = np.asarray(next_layer_gate(hidden), dtype=np.float64)
        return self.predict_from_gate_output(
            raw, src, top_k=top_k, gate_output_domain=gate_output_domain,
        )


@dataclass(frozen=True)
class FateTrafficAdapter:
    """Shared offline/online expert→rank traffic aggregation."""

    expert_to_rank: np.ndarray
    world_size: int

    def __post_init__(self) -> None:
        world = int(self.world_size)
        if world <= 0:
            raise ValueError("world_size must be positive")
        raw = np.asarray(self.expert_to_rank)
        if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.number):
            raise ValueError("expert_to_rank must be a non-empty numeric vector")
        rounded = np.rint(raw)
        if not np.allclose(raw, rounded, atol=0.0, rtol=0.0):
            raise ValueError("expert_to_rank must contain integral ranks")
        mapping = np.ascontiguousarray(rounded.astype(np.int64, copy=False)).copy()
        if int(mapping.min()) < 0 or int(mapping.max()) >= world:
            raise ValueError("expert_to_rank mapping outside world")
        mapping.setflags(write=False)
        object.__setattr__(self, "expert_to_rank", mapping)
        object.__setattr__(self, "world_size", world)

    def full_assignment_matrix(self, prediction: FateExpertPrediction) -> np.ndarray:
        prediction.validate()
        mapping = np.asarray(self.expert_to_rank, dtype=np.int64).reshape(-1)
        if len(mapping) != prediction.expert_scores.shape[1]:
            raise ValueError("expert_to_rank size must match next gate expert count")
        if mapping.size and (int(mapping.min()) < 0 or int(mapping.max()) >= int(self.world_size)):
            raise ValueError("expert_to_rank mapping outside world")
        src = np.asarray(prediction.source_ranks, dtype=np.int64)
        if src.size and (int(src.min()) < 0 or int(src.max()) >= int(self.world_size)):
            raise ValueError("source rank outside world")
        out = np.zeros((int(self.world_size), int(self.world_size)), dtype=np.float64)
        mass = np.asarray(prediction.expected_assignment_mass, dtype=np.float64)
        for token in range(mass.shape[0]):
            source = int(src[token])
            for expert in np.flatnonzero(mass[token] > 0):
                out[source, int(mapping[expert])] += float(mass[token, expert])
        return out

    def remote_hint(self, prediction: FateExpertPrediction, *, confidence: float) -> TrafficHint:
        full = self.full_assignment_matrix(prediction)
        # Balanced row rounding preserves each source rank's integer top-k
        # assignment mass before local expert copies are removed.
        rounded_full = np.zeros_like(full, dtype=np.int32)
        for source in range(full.shape[0]):
            target = int(round(float(full[source].sum())))
            row = np.maximum(full[source], 0.0)
            if target > 0 and float(row.sum()) > 0:
                scaled = row * (float(target) / float(row.sum()))
                base = np.floor(scaled).astype(np.int32)
                remainder = target - int(base.sum())
                if remainder > 0:
                    order = np.argsort(-(scaled - base), kind="stable")
                    base[order[:remainder]] += 1
                rounded_full[source] = base
        rounded = rounded_full.copy(); np.fill_diagonal(rounded, 0)
        remote = full.copy(); np.fill_diagonal(remote, 0.0)
        return TrafficHint(
            predictor_id=prediction.predictor_id,
            target_dispatch_rows=rounded,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            hint_kind="expert_route",
            oracle=False,
            matrix_kind="remote_rows",
            metadata={
                "full_assignment_total": float(full.sum()),
                "remote_expected_total": float(remote.sum()),
                "top_k": int(prediction.top_k),
                "conversion": "expected_topk_mass_then_remove_local",
            },
        )


def perfect_routes_to_traffic(
    source_ranks: np.ndarray,
    selected_experts: np.ndarray,
    expert_to_rank: np.ndarray,
    world_size: int,
) -> np.ndarray:
    """Exact route→traffic conversion used to test adapter parity."""
    world = int(world_size)
    if world <= 0:
        raise ValueError("world_size must be positive")
    src = np.asarray(source_ranks, dtype=np.int64).reshape(-1)
    experts = np.asarray(selected_experts, dtype=np.int64)
    if experts.ndim == 1:
        experts = experts[:, None]
    if experts.ndim != 2 or experts.shape[0] != src.shape[0]:
        raise ValueError("token count mismatch")
    mapping = np.asarray(expert_to_rank, dtype=np.int64).reshape(-1)
    if mapping.size == 0:
        raise ValueError("expert_to_rank must be non-empty")
    if src.size and (int(src.min()) < 0 or int(src.max()) >= world):
        raise ValueError("source rank outside world")
    if mapping.size and (int(mapping.min()) < 0 or int(mapping.max()) >= world):
        raise ValueError("expert_to_rank mapping outside world")
    if experts.size and (int(experts.min()) < 0 or int(experts.max()) >= mapping.size):
        raise ValueError("selected expert outside expert_to_rank mapping")
    full = np.zeros((world, world), dtype=np.int32)
    for token in range(experts.shape[0]):
        for k in range(experts.shape[1]):
            full[int(src[token]), int(mapping[int(experts[token, k])])] += 1
    remote = full.copy(); np.fill_diagonal(remote, 0)
    return remote



def calibrate_fate_confidence(prediction: FateExpertPrediction, true_selected_experts: np.ndarray) -> dict:
    """Calibrate a FATE expert prediction against true next-layer top-k routes."""
    prediction.validate()
    truth = np.asarray(true_selected_experts, dtype=np.int64)
    if truth.ndim == 1:
        truth = truth[:, None]
    if truth.ndim != 2 or truth.shape[0] != prediction.expert_scores.shape[0]:
        raise ValueError("token count mismatch")
    if truth.shape[1] <= 0:
        raise ValueError("truth must contain at least one selected expert")
    if truth.size and (int(truth.min()) < 0 or int(truth.max()) >= prediction.expert_scores.shape[1]):
        raise ValueError("true selected expert outside prediction expert range")
    recalls = []
    for token in range(truth.shape[0]):
        predicted = set(np.flatnonzero(prediction.prefetch_mask[token]).tolist())
        actual = set(int(x) for x in truth[token])
        recalls.append(len(predicted & actual) / max(len(actual), 1))
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    exact_topk = np.argsort(-prediction.expert_scores, axis=1, kind="stable")[:, :truth.shape[1]]
    topk_overlap = float(np.mean([len(set(exact_topk[i]) & set(truth[i])) / truth.shape[1] for i in range(truth.shape[0])]))
    confidence = float(np.clip(0.5 * mean_recall + 0.5 * topk_overlap, 0.0, 1.0))
    return {"prefetch_recall": mean_recall, "topk_overlap": topk_overlap, "confidence": confidence}


@dataclass(frozen=True)
class FateObservation:
    """Runtime observation needed by the faithful FATE path."""

    current_gate_input: np.ndarray
    source_ranks: np.ndarray
    next_layer_gate: NextLayerGate | Callable[[np.ndarray], np.ndarray]
    expert_to_rank: np.ndarray
    world_size: int
    top_k: int
    gate_output_domain: str = "logits"
    layer_id: int | None = None
    request_id: str = "runtime"

    def validate(self) -> None:
        hidden = np.asarray(self.current_gate_input)
        source = np.asarray(self.source_ranks)
        mapping = np.asarray(self.expert_to_rank)
        world = int(self.world_size)
        if world <= 0:
            raise ValueError("world_size must be positive")
        if hidden.ndim != 2 or source.ndim != 1 or hidden.shape[0] != source.shape[0]:
            raise ValueError("FATE observation token dimensions do not match")
        if not np.issubdtype(hidden.dtype, np.number) or not np.isfinite(hidden).all():
            raise ValueError("current_gate_input must contain finite numeric values")
        if mapping.ndim != 1 or mapping.size == 0:
            raise ValueError("expert_to_rank must be a non-empty vector")
        for values, name in ((source, "source_ranks"), (mapping, "expert_to_rank")):
            if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain finite numeric ranks")
            if not np.allclose(values, np.rint(values), atol=0.0, rtol=0.0):
                raise ValueError(f"{name} must contain integral ranks")
        if source.size and (source.min() < 0 or source.max() >= world):
            raise ValueError("source rank outside world")
        if mapping.size and (mapping.min() < 0 or mapping.max() >= world):
            raise ValueError("expert placement outside world")
        if int(self.top_k) <= 0 or int(self.top_k) > int(mapping.size):
            raise ValueError("top_k must be within the next-layer expert count")
        if self.gate_output_domain not in {"logits", "probabilities", "nonnegative_scores"}:
            raise ValueError("unsupported gate_output_domain")
        if not str(self.request_id):
            raise ValueError("request_id must be non-empty")


@dataclass(frozen=True)
class FatePredictionBundle:
    expert_prediction: FateExpertPrediction
    traffic_hint: TrafficHint


@dataclass(frozen=True)
class FatePredictorService:
    """Production predictor: faithful FATE expert prediction followed by traffic aggregation."""

    predictor: FateCrossLayerGatePredictor = FateCrossLayerGatePredictor()
    default_confidence: float = 0.75

    def predict(self, observation: FateObservation, *, confidence: float | None = None) -> FatePredictionBundle:
        observation.validate()
        expert_prediction = self.predictor.predict_experts(
            observation.current_gate_input, observation.next_layer_gate, observation.source_ranks,
            top_k=observation.top_k, gate_output_domain=observation.gate_output_domain,
        )
        adapter = FateTrafficAdapter(observation.expert_to_rank, observation.world_size)
        base = adapter.remote_hint(
            expert_prediction, confidence=self.default_confidence if confidence is None else confidence
        )
        hint = TrafficHint(
            predictor_id=base.predictor_id,
            target_dispatch_rows=base.matrix(),
            confidence=base.confidence,
            hint_kind=base.hint_kind,
            oracle=False,
            matrix_kind=base.matrix_kind,
            metadata={
                **dict(base.metadata),
                "faithful_fate": True,
                "request_id": observation.request_id,
                "layer_id": observation.layer_id,
                "input": "Gate_in_i evaluated by Gate_{i+1}",
                "prefetch_percentile": self.predictor.percentile,
                "min_candidates": self.predictor.min_candidates,
            },
        )
        return FatePredictionBundle(expert_prediction=expert_prediction, traffic_hint=hint)

    def predict_from_gate_output(
        self, observation: FateObservation, gate_output: np.ndarray, *, confidence: float | None = None
    ) -> FatePredictionBundle:
        observation.validate()
        expert_prediction = self.predictor.predict_from_gate_output(
            gate_output, observation.source_ranks, top_k=observation.top_k,
            gate_output_domain=observation.gate_output_domain,
        )
        adapter = FateTrafficAdapter(observation.expert_to_rank, observation.world_size)
        base = adapter.remote_hint(
            expert_prediction, confidence=self.default_confidence if confidence is None else confidence
        )
        hint = TrafficHint(
            predictor_id=base.predictor_id, target_dispatch_rows=base.matrix(), confidence=base.confidence,
            hint_kind=base.hint_kind, oracle=False, matrix_kind=base.matrix_kind,
            metadata={
                **dict(base.metadata), "faithful_fate": True, "request_id": observation.request_id,
                "layer_id": observation.layer_id, "input": "Gate_in_i evaluated by Gate_{i+1}",
                "prefetch_percentile": self.predictor.percentile, "min_candidates": self.predictor.min_candidates,
            },
        )
        return FatePredictionBundle(expert_prediction=expert_prediction, traffic_hint=hint)

    def predict_hint(self, observation: FateObservation, *, confidence: float | None = None) -> TrafficHint:
        return self.predict(observation, confidence=confidence).traffic_hint


def fate_capture_schema() -> dict:
    """Machine-readable faithful FATE capture and replay contract."""
    return {
        "schema_version": "routersense.fate.capture.v2",
        "identity": {
            "canonical_instance_id": "model_slug:sample_id:layer_id:vepN",
            "duplicate_policy": "reject",
            "terminal_layer_policy": "explicit_zero_hint",
        },
        "per_token_required": [
            "request_id", "sample_id", "split", "layer_id", "token_position",
            "source_rank", "current_gate_input_or_replayable_reference",
        ],
        "per_layer_required": [
            "model_slug", "next_layer_gate_weight_or_callable_id", "expert_to_rank",
            "top_k", "gate_output_domain", "gate_callable_parity",
        ],
        "prediction_artifact_required": [
            "predicted_expert_scores", "prefetch_mask", "expected_assignment_mass",
            "confidence", "true_next_selected_experts", "true_next_routing_weights",
        ],
        "timing_required": [
            "next_gate_eval_start_ns", "next_gate_eval_done_ns",
            "prediction_start_ns", "prediction_done_ns", "target_frontier_ns",
        ],
        "corpus_required": [
            "utf8_text", "unique_sample_id", "development_or_validation_split",
            "normalized_corpus_digest", "unicode_integrity_passed",
        ],
        "calibration": {
            "fit_split": "development",
            "validation_mutation": "forbidden",
            "outputs": ["selected_percentile", "selected_confidence", "candidate_metrics"],
        },
    }

def fate_capture_readiness(record_keys: set[str]) -> dict[str, bool]:
    has_gate_input = bool({"gate_input", "current_gate_input", "gate_input_digest"} & record_keys)
    has_next_gate = bool({"next_gate_logits", "next_gate_scores", "next_gate_callable_id"} & record_keys)
    has_source = bool({"source_rank", "token_source_rank", "owner_rank"} & record_keys)
    return {
        "has_current_gate_input": has_gate_input,
        "has_next_gate_output_or_callable": has_next_gate,
        "has_token_source_rank": has_source,
        "faithful_fate_replay_ready": bool(has_gate_input and has_next_gate and has_source),
    }


__all__ = [
    "FateCrossLayerGatePredictor", "FateExpertPrediction", "FateObservation",
    "FatePredictionBundle", "FatePredictorService", "FateTrafficAdapter", "calibrate_fate_confidence",
    "fate_capture_readiness", "fate_capture_schema", "perfect_routes_to_traffic",
]
