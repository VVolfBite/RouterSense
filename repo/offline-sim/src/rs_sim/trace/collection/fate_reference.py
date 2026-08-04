from __future__ import annotations

"""Offline FATE cross-layer gate reference implementation.

This reference is intentionally separate from the bounded online sampled path.
It evaluates the current layer gate input with the next layer gate output,
selects a per-token percentile candidate set, normalizes its expected mass to
the model top-k assignment count, and aggregates experts to destination ranks.
The output is rank routing rows suitable for a ``FATE_P2`` artifact.
"""

from dataclasses import dataclass
from typing import Any


class FateReferenceError(ValueError):
    pass


def _balanced_round_row(values: Any, target: int) -> tuple[int, ...]:
    import numpy as np

    row = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if int(target) < 0:
        raise FateReferenceError("target assignment mass must be non-negative")
    if int(target) == 0:
        return tuple(0 for _ in range(int(row.size)))
    total = float(row.sum())
    if total <= 0.0:
        raise FateReferenceError("cannot round a zero expected-mass row to positive target")
    scaled = row * (float(target) / total)
    base = np.floor(scaled).astype(np.int64)
    remainder = int(target) - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(scaled - base), kind="stable")
        base[order[:remainder]] += 1
    result = tuple(int(item) for item in base.tolist())
    if sum(result) != int(target):
        raise FateReferenceError("balanced FATE rounding failed to preserve assignment mass")
    return result


@dataclass(frozen=True, slots=True)
class FateReferenceResult:
    routing_rows: tuple[tuple[int, ...], ...]
    predictor_id: str
    percentile: float
    min_candidates: int
    top_k: int
    estimator_kind: str
    evidence: tuple[tuple[str, Any], ...]


def predict_fate_routing_rows(
    gate_output: Any,
    source_ranks: Any,
    *,
    expert_to_rank: Any,
    world_size: int,
    top_k: int,
    percentile: float = 75.0,
    min_candidates: int | None = None,
    gate_output_domain: str = "logits",
) -> FateReferenceResult:
    """Return integer rank routing rows using the formal FATE reference rule."""

    import numpy as np

    raw = np.asarray(gate_output, dtype=np.float64)
    src = np.asarray(source_ranks, dtype=np.int64).reshape(-1)
    mapping = np.asarray(expert_to_rank, dtype=np.int64).reshape(-1)
    world = int(world_size)
    k = int(top_k)
    pct = float(percentile)
    if raw.ndim != 2 or raw.shape[0] != src.shape[0]:
        raise FateReferenceError("gate_output and source_ranks token counts must match")
    if raw.shape[1] != mapping.size or mapping.size == 0:
        raise FateReferenceError("expert_to_rank must match gate expert dimension")
    if world <= 0 or mapping.min(initial=0) < 0 or mapping.max(initial=0) >= world:
        raise FateReferenceError("expert_to_rank mapping outside world")
    if src.size and (int(src.min()) < 0 or int(src.max()) >= world):
        raise FateReferenceError("source rank outside world")
    if not np.isfinite(raw).all():
        raise FateReferenceError("gate_output must be finite")
    if not 0.0 <= pct <= 100.0:
        raise FateReferenceError("percentile must be within [0,100]")
    if k <= 0 or k > raw.shape[1]:
        raise FateReferenceError("top_k outside expert dimension")
    minimum = max(k, int(min_candidates or k))
    if minimum > raw.shape[1]:
        raise FateReferenceError("min_candidates exceeds expert dimension")

    domain = str(gate_output_domain).lower()
    if domain == "logits":
        shifted = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        scores = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    elif domain in {"probabilities", "nonnegative_scores"}:
        if (raw < 0).any():
            raise FateReferenceError("non-logit gate output must be non-negative")
        totals = raw.sum(axis=1, keepdims=True)
        scores = np.empty_like(raw)
        nonzero = totals[:, 0] > 0
        scores[nonzero] = raw[nonzero] / totals[nonzero]
        scores[~nonzero] = 1.0 / float(raw.shape[1])
    else:
        raise FateReferenceError(f"unsupported gate_output_domain {gate_output_domain!r}")

    threshold = np.percentile(scores, pct, axis=1, keepdims=True)
    mask = scores >= threshold
    for token in np.flatnonzero(mask.sum(axis=1) < minimum):
        indices = np.argsort(-scores[int(token)], kind="stable")[:minimum]
        mask[int(token), indices] = True
    selected = np.where(mask, scores, 0.0)
    expected = selected / np.maximum(selected.sum(axis=1, keepdims=True), 1e-12) * float(k)

    rank_mass = np.zeros((world, world), dtype=np.float64)
    source_token_counts = np.zeros(world, dtype=np.int64)
    for token in range(expected.shape[0]):
        source = int(src[token])
        source_token_counts[source] += 1
        for expert in np.flatnonzero(expected[token] > 0):
            rank_mass[source, int(mapping[int(expert)])] += float(expected[token, expert])
    rows = tuple(
        _balanced_round_row(rank_mass[source], int(source_token_counts[source]) * k)
        for source in range(world)
    )
    evidence = {
        "candidate_count_mean": float(mask.sum(axis=1).mean()) if mask.size else 0.0,
        "expected_assignment_total": float(expected.sum()),
        "routing_assignment_total": int(sum(sum(row) for row in rows)),
        "gate_output_domain": domain,
    }
    return FateReferenceResult(
        routing_rows=rows,
        predictor_id="fate_cross_layer_gate_percentile",
        percentile=pct,
        min_candidates=minimum,
        top_k=k,
        estimator_kind="EXPECTED_TOPK_MASS_PERCENTILE_TO_RANK_ROWS",
        evidence=tuple(sorted(evidence.items())),
    )


__all__ = [
    "FateReferenceError",
    "FateReferenceResult",
    "predict_fate_routing_rows",
]
