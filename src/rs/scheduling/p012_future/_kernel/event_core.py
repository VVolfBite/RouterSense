"""Public facade for the compiled event planning engine.

The heavy planning and binding kernels live in stage-focused modules; this
facade preserves the stable import surface used by planners and tests.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .contracts import _matrix
from .event_binding import _bind_future_template, _bind_prepared_order, _bind_template
from .event_planning import _event_plan, _event_plan_p0123

def _arrays(mats: Iterable[np.ndarray], hint: np.ndarray | None, slope: np.ndarray | None, intercept: np.ndarray | None):
    raw = tuple(mats)
    if len(raw) != 3:
        raise ValueError("exactly three P0/P1/P2 matrices are required")
    first = _matrix(raw[0], name="P0", zero_diagonal=True)
    n = first.shape[0]
    rows = (
        first,
        _matrix(raw[1], name="P1", world_size=n, zero_diagonal=True),
        _matrix(raw[2], name="P2", world_size=n, zero_diagonal=True),
    )
    actual = np.stack(rows)
    h = np.zeros((n, n), dtype=np.float64) if hint is None else np.asarray(hint, dtype=np.float64)
    s = np.ones((n, n), dtype=np.float64) if slope is None else np.asarray(slope, dtype=np.float64)
    b = np.zeros((n, n), dtype=np.float64) if intercept is None else np.asarray(intercept, dtype=np.float64)
    for name, value in (("hint", h), ("edge_slope", s), ("edge_intercept", b)):
        if value.shape != (n, n):
            raise ValueError(f"{name} shape must match traffic world size")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")
    if (h < 0).any():
        raise ValueError("hint must contain non-negative rows")
    if (s <= 0).any() or (b < 0).any():
        raise ValueError("edge costs require positive slopes and non-negative intercepts")
    return actual, h, s, b

def plan_event(
    mats: Iterable[np.ndarray], *, hint: np.ndarray | None = None, scope: str = "joint",
    full_truth_geometry: bool = False, weights=(0.25, 0.75, 4.0, 0.5),
    edge_slope: np.ndarray | None = None, edge_intercept: np.ndarray | None = None,
    expert_compute_delay: float = 0.0, wave_launch_b: float = 0.0, max_waves: int = 10000,
):
    actual, h, s, b = _arrays(mats, hint, edge_slope, edge_intercept)
    return _event_plan(actual, h, s, b, scope == "local", bool(full_truth_geometry), *map(float, weights),
                       float(expert_compute_delay), float(wave_launch_b), int(max_waves))

def plan_event_p0123(
    mats: Iterable[np.ndarray], *, hint: np.ndarray, p3_hint: np.ndarray | None = None,
    scope: str = "joint", full_truth_geometry: bool = False, p3_weight: float = 1.0,
    weights=(0.25, 0.75, 4.0, 0.5),
    edge_slope: np.ndarray | None = None, edge_intercept: np.ndarray | None = None,
    expert_compute_delay: float = 0.0, wave_launch_b: float = 0.0, max_waves: int = 10000,
):
    """Plan the executable P012 proxy while using P3 as advisory lookahead.

    P3 never appears in the returned phase arrays.  The function therefore
    preserves the existing P012 bind/replay/execution contract.
    """
    actual, h, s, b = _arrays(mats, hint, edge_slope, edge_intercept)
    p3 = np.ascontiguousarray(h.T if p3_hint is None else np.asarray(p3_hint, dtype=np.float64))
    if p3.shape != h.shape:
        raise ValueError("p3_hint shape must match P2 hint")
    np.fill_diagonal(p3, 0.0)
    return _event_plan_p0123(
        actual, h, p3, s, b, scope == "local", bool(full_truth_geometry),
        float(p3_weight), *map(float, weights), float(expert_compute_delay),
        float(wave_launch_b), int(max_waves),
    )

def bind_prepared_order(
    actual_mats: Iterable[np.ndarray], template, *,
    edge_slope: np.ndarray | None = None, edge_intercept: np.ndarray | None = None,
    expert_compute_delay: float = 0.0, wave_launch_b: float = 0.0, max_waves: int = 10000,
):
    """Consume a prepared P012 order with no online scoring or matching solver."""
    actual, _, slope, intercept = _arrays(actual_mats, None, edge_slope, edge_intercept)
    if not isinstance(template, tuple) or len(template) < 5:
        raise ValueError("prepared template must be a planner result tuple")
    t_phase = np.asarray(template[2], dtype=np.int8)
    t_dst = np.asarray(template[3], dtype=np.int16)
    t_size = np.asarray(template[4], dtype=np.int32)
    if t_phase.ndim != 2 or t_dst.shape != t_phase.shape or t_size.shape != t_phase.shape:
        raise ValueError("prepared template phase/dst/size shapes must match")
    if t_phase.shape[1] != actual.shape[1]:
        raise ValueError("prepared template world size mismatch")
    return _bind_prepared_order(
        actual, t_phase, t_dst, t_size, slope, intercept,
        float(expert_compute_delay), float(wave_launch_b), int(max_waves),
    )

def bind_future_template(
    actual_mats: Iterable[np.ndarray], template, *,
    edge_slope: np.ndarray | None = None, edge_intercept: np.ndarray | None = None,
    expert_compute_delay: float = 0.0, wave_launch_b: float = 0.0, max_waves: int = 10000,
    weights=(0.25, 0.75, 4.0, 0.5), use_template_slots: bool = True,
):
    """Legacy analysis binder retained for reproducibility.

    This helper permits truth differences and performs residual repair.  The
    production :class:`FutureP012Planner` does not call it; it uses
    :func:`bind_prepared_order` plus a one-shot fallback instead.
    """
    actual, _, slope, intercept = _arrays(actual_mats, None, edge_slope, edge_intercept)
    return _bind_future_template(
        actual, np.asarray(template[2], dtype=np.int8), np.asarray(template[3], dtype=np.int16),
        np.asarray(template[4], dtype=np.int32), slope, intercept,
        float(expert_compute_delay), float(wave_launch_b), int(max_waves),
        *map(float, weights), bool(use_template_slots),
    )

def bind_template(
    actual_mats: Iterable[np.ndarray], hint: np.ndarray, template,
    *, edge_slope: np.ndarray | None = None, edge_intercept: np.ndarray | None = None,
    expert_compute_delay: float = 0.0, wave_launch_b: float = 0.0, max_waves: int = 10000,
):
    actual, h, s, b = _arrays(actual_mats, hint, edge_slope, edge_intercept)
    if not isinstance(template, tuple) or len(template) < 8:
        raise ValueError("template must be a planner result tuple")
    phase = np.asarray(template[2], dtype=np.int8)
    dst = np.asarray(template[3], dtype=np.int16)
    size = np.asarray(template[4], dtype=np.int32)
    duration = np.asarray(template[6], dtype=np.float64)
    starts = np.asarray(template[7], dtype=np.float64)
    if phase.ndim != 2 or dst.shape != phase.shape or size.shape != phase.shape:
        raise ValueError("template phase/dst/size shapes must match")
    if phase.shape[1] != actual.shape[1] or duration.shape != (phase.shape[0],) or starts.shape != (phase.shape[0],):
        raise ValueError("template world size or wave-vector shape mismatch")
    return _bind_template(actual, h, phase, dst, size, duration, starts, s, b,
                          float(expert_compute_delay), float(wave_launch_b), int(max_waves))

def forecast_plan(
    actual_mats: Iterable[np.ndarray], forecast: np.ndarray, *, weights=(0.25, 0.75, 4.0, 0.5),
    edge_slope: np.ndarray | None = None, edge_intercept: np.ndarray | None = None,
    expert_compute_delay: float = 0.0, wave_launch_b: float = 0.0, max_waves: int = 10000,
):
    actual = [np.asarray(x, dtype=np.int32) for x in actual_mats]
    proxy = [actual[0], actual[1], np.asarray(forecast, dtype=np.int32)]
    template = plan_event(proxy, hint=forecast, scope="joint", full_truth_geometry=True, weights=weights,
                          edge_slope=edge_slope, edge_intercept=edge_intercept,
                          expert_compute_delay=expert_compute_delay, wave_launch_b=wave_launch_b, max_waves=max_waves)
    bound = bind_template(actual, forecast, template, edge_slope=edge_slope, edge_intercept=edge_intercept,
                          expert_compute_delay=expert_compute_delay, wave_launch_b=wave_launch_b, max_waves=max_waves)
    return template, bound

def rank_hint_plan(
    actual_mats: Iterable[np.ndarray], forecast: np.ndarray, *, weights=(0.25, 0.75, 4.0, 0.5),
    edge_slope: np.ndarray | None = None, edge_intercept: np.ndarray | None = None,
    expert_compute_delay: float = 0.0, wave_launch_b: float = 0.0, max_waves: int = 10000,
):
    actual = [np.asarray(x, dtype=np.int32) for x in actual_mats]
    zero = np.zeros_like(actual[0])
    template = plan_event([actual[0], actual[1], zero], hint=forecast, scope="joint", full_truth_geometry=False,
                          weights=weights, edge_slope=edge_slope, edge_intercept=edge_intercept,
                          expert_compute_delay=expert_compute_delay, wave_launch_b=wave_launch_b, max_waves=max_waves)
    bound = bind_template(actual, forecast, template, edge_slope=edge_slope, edge_intercept=edge_intercept,
                          expert_compute_delay=expert_compute_delay, wave_launch_b=wave_launch_b, max_waves=max_waves)
    return template, bound

def local_plan(mats: Iterable[np.ndarray], *, weights=(0.25, 0.75, 4.0, 0.5),
               edge_slope=None, edge_intercept=None, expert_compute_delay=0.0, wave_launch_b=0.0, max_waves=10000):
    matrices = [np.asarray(x, dtype=np.int32) for x in mats]
    n = matrices[0].shape[0]
    zero = np.zeros((n, n), dtype=np.int32)
    parts = []
    total = 0.0
    waves = 0
    valid = 1
    for phase, matrix in enumerate(matrices):
        r = plan_event([matrix, zero, zero], scope="local", weights=weights,
                       edge_slope=edge_slope, edge_intercept=edge_intercept,
                       expert_compute_delay=0.0, wave_launch_b=wave_launch_b, max_waves=max_waves)
        total += float(r[0]); waves += int(r[1]); valid *= int(r[-1]); parts.append(r)
        if phase == 0:
            total += float(expert_compute_delay)
    return total, waves, valid, parts

def warmup() -> None:
    matrix = np.array([[0, 3, 1], [2, 0, 1], [1, 2, 0]], dtype=np.int32)
    mats = [matrix, matrix.T.copy(), matrix.copy()]
    plan_event(mats, hint=matrix, full_truth_geometry=True)
    forecast_plan(mats, matrix)

__all__ = [
    "bind_prepared_order", "bind_template", "forecast_plan",
    "local_plan", "plan_event", "plan_event_p0123", "rank_hint_plan", "warmup",
]
