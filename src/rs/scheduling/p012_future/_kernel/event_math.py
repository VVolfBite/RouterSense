"""Compiled matching and release primitives for the event engine."""

from __future__ import annotations

import math
import numpy as np
from numba import njit

@njit(cache=True)
def _softmax(values: np.ndarray, tau: float = 0.25) -> np.ndarray:
    n = values.shape[0]
    out = np.zeros(n, np.float64)
    scale = 1.0
    for i in range(n):
        if values[i] > scale:
            scale = values[i]
    maxlog = -1e300
    for i in range(n):
        z = values[i] / scale / tau
        if z > maxlog:
            maxlog = z
    total = 0.0
    for i in range(n):
        z = values[i] / scale / tau - maxlog
        if z < -60:
            z = -60
        elif z > 60:
            z = 60
        out[i] = math.exp(z)
        total += out[i]
    if total <= 0:
        total = 1.0
    for i in range(n):
        out[i] /= total
    return out

@njit(cache=True)
def _hungarian_max(weights: np.ndarray) -> np.ndarray:
    n = weights.shape[0]
    u = np.zeros(n + 1, np.float64)
    v = np.zeros(n + 1, np.float64)
    p = np.zeros(n + 1, np.int64)
    way = np.zeros(n + 1, np.int64)
    inf = 1e300
    for i in range(1, n + 1):
        p[0] = i
        minv = np.full(n + 1, inf, np.float64)
        used = np.zeros(n + 1, np.uint8)
        j0 = 0
        while True:
            used[j0] = 1
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j] == 0:
                    cur = -weights[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta - 1e-15 or (abs(minv[j] - delta) <= 1e-15 and (j1 == 0 or j < j1)):
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j] != 0:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.full(n, -1, np.int64)
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment

@njit(cache=True)
def _sum_all(a: np.ndarray) -> int:
    total = 0
    for p in range(a.shape[0]):
        for s in range(a.shape[1]):
            for d in range(a.shape[2]):
                total += a[p, s, d]
    return total

@njit(cache=True)
def _refresh_release(now: float, release: np.ndarray, release_at: np.ndarray) -> None:
    for p in range(1, 3):
        for r in range(release.shape[1]):
            if release[p, r] == 0 and release_at[p, r] < 1e290 and now + 1e-12 >= release_at[p, r]:
                release[p, r] = 1

@njit(cache=True)
def _next_release(now: float, release: np.ndarray, release_at: np.ndarray) -> float:
    nxt = 1e300
    for p in range(1, 3):
        for r in range(release.shape[1]):
            t = release_at[p, r]
            if release[p, r] == 0 and t > now + 1e-12 and t < nxt:
                nxt = t
    return nxt

@njit(cache=True)
def _wave_duration(amount: int, assignment: np.ndarray, best: np.ndarray, weights: np.ndarray,
                   slope: np.ndarray, intercept: np.ndarray, launch_b: float) -> float:
    duration = 0.0
    for s in range(assignment.shape[0]):
        d = assignment[s]
        if d >= 0 and best[s, d] >= 0 and weights[s, d] > 1e-12:
            t = intercept[s, d] + slope[s, d] * float(amount)
            if t > duration:
                duration = t
    return launch_b + duration

__all__ = ["_softmax", "_hungarian_max", "_sum_all", "_refresh_release", "_next_release", "_wave_duration"]
