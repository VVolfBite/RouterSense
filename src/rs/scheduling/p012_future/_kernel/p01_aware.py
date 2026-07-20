from __future__ import annotations

import numpy as np

from .event_core import plan_event


def p01_release_vector(
    p0: np.ndarray, p1: np.ndarray, *, weights, edge_slope=None, edge_intercept=None,
    expert_compute_delay=0.0, wave_launch_b=0.0, max_waves=10000,
) -> tuple[np.ndarray, float]:
    zero = np.zeros_like(np.asarray(p0, dtype=np.int32))
    result = plan_event(
        [p0, p1, zero], hint=zero, scope="joint", full_truth_geometry=False,
        weights=weights, edge_slope=edge_slope, edge_intercept=edge_intercept,
        expert_compute_delay=expert_compute_delay, wave_launch_b=wave_launch_b, max_waves=max_waves,
    )
    release = np.asarray(result[11], dtype=np.float64)
    release = np.where(release < 0, float(result[0]), release)
    return release, max(float(result[0]), 1e-12)


def p01_aware_hint(
    p0: np.ndarray,
    p1: np.ndarray,
    forecast: np.ndarray,
    *,
    weights,
    tie_band: float = 0.12,
    release_weight: float = 0.15,
    edge_slope=None,
    edge_intercept=None,
    expert_compute_delay=0.0,
    wave_launch_b=0.0,
    max_waves=10000,
    p01_release2: np.ndarray | None = None,
    p01_horizon: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Use P01 release time only when predicted rank pressures are close.

    Large predicted P2 differences remain dominant. Within the same relative
    pressure band, a rank that P01 releases later receives a larger multiplier.
    """
    hint = np.asarray(forecast, dtype=np.float64).copy()
    np.fill_diagonal(hint, 0.0)
    incoming = hint.sum(axis=0)
    max_in = max(float(incoming.max(initial=0.0)), 1.0)
    pressure = hint.sum(axis=1) + 0.2 * (hint * (incoming[None, :] / max_in)).sum(axis=1)
    max_pressure = max(float(pressure.max(initial=0.0)), 1.0)
    band = max(float(tie_band), 1e-6)
    groups = np.floor((pressure / max_pressure) / band + 1e-12).astype(np.int32)
    if p01_release2 is None or p01_horizon is None:
        release, horizon = p01_release_vector(
            p0, p1, weights=weights, edge_slope=edge_slope, edge_intercept=edge_intercept,
            expert_compute_delay=expert_compute_delay, wave_launch_b=wave_launch_b, max_waves=max_waves,
        )
    else:
        release = np.asarray(p01_release2, dtype=np.float64).copy()
        horizon = max(float(p01_horizon), 1e-12)
        release = np.where(release < 0, horizon, release)
    multipliers = np.ones(len(pressure), dtype=np.float64)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            continue
        normalized = release[indices] / horizon
        centered = normalized - float(normalized.mean())
        denom = max(float(np.max(np.abs(centered))), 1e-12)
        multipliers[indices] = 1.0 + float(release_weight) * centered / denom
    adjusted = hint * multipliers[:, None]
    # Preserve predicted total mass; only near-tie row pressure moves.
    if float(adjusted.sum()) > 0:
        adjusted *= float(hint.sum()) / float(adjusted.sum())
    np.fill_diagonal(adjusted, 0.0)
    return adjusted, {
        "p01_release2": release.tolist(),
        "predicted_rank_pressure": pressure.tolist(),
        "tie_groups": groups.tolist(),
        "rank_multipliers": multipliers.tolist(),
        "tie_band": float(tie_band),
        "release_weight": float(release_weight),
    }


def release_delay(candidate, baseline) -> tuple[float, float]:
    c = np.asarray(candidate[11], dtype=np.float64)
    b = np.asarray(baseline[11], dtype=np.float64)
    c = np.where(c < 0, float(candidate[0]), c)
    b = np.where(b < 0, float(baseline[0]), b)
    positive = np.maximum(c - b, 0.0)
    horizon = max(float(baseline[0]), 1e-12)
    return float(positive.max(initial=0.0) / horizon), float(positive.mean() / horizon)


__all__ = ["p01_aware_hint", "p01_release_vector", "release_delay"]
