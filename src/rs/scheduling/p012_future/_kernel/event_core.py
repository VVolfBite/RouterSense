from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from numba import njit

from .contracts import _matrix


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


@njit(cache=True)
def _event_plan(
    mats: np.ndarray,
    hint: np.ndarray,
    edge_slope: np.ndarray,
    edge_intercept: np.ndarray,
    scope_local: bool,
    full_truth: bool,
    wres: float,
    wcp: float,
    wunlock: float,
    wend: float,
    expert_compute_delay: float,
    wave_launch_b: float,
    max_waves: int,
):
    res = mats.copy()
    n = res.shape[1]
    maxw = max_waves
    out_phase = np.full((maxw, n), -1, np.int8)
    out_dst = np.full((maxw, n), -1, np.int16)
    out_size = np.zeros((maxw, n), np.int32)
    quantum = np.zeros(maxw, np.int32)
    duration = np.zeros(maxw, np.float64)
    starts = np.zeros(maxw, np.float64)
    ends = np.zeros(maxw, np.float64)

    release = np.zeros((3, n), np.uint8)
    release[0, :] = 1
    release_at = np.full((3, n), 1e300, np.float64)
    release_at[0, :] = 0.0
    rel1 = np.full(n, -1.0, np.float64)
    rel2 = np.full(n, -1.0, np.float64)
    for r in range(n):
        in0 = 0
        in1 = 0
        for s in range(n):
            in0 += res[0, s, r]
            in1 += res[1, s, r]
        if in0 == 0:
            release_at[1, r] = expert_compute_delay
            rel1[r] = expert_compute_delay
            if in1 == 0:
                release_at[2, r] = expert_compute_delay
                rel2[r] = expert_compute_delay
    now = 0.0
    _refresh_release(now, release, release_at)
    wave = 0
    active_phase = 0
    phase_done = np.zeros(3, np.float64)

    while _sum_all(res) > 0 and wave < maxw:
        _refresh_release(now, release, release_at)
        if scope_local:
            while active_phase < 3:
                z = 0
                for s in range(n):
                    for d in range(n):
                        z += res[active_phase, s, d]
                if z > 0:
                    break
                active_phase += 1

        geom = np.zeros((3, n, n), np.float64)
        outgoing = np.zeros((3, n), np.float64)
        incoming = np.zeros((3, n), np.float64)
        maxres = 1.0
        for p in range(2):
            for s in range(n):
                for d in range(n):
                    if res[p, s, d] > 0:
                        geom[p, s, d] = edge_intercept[s, d] + edge_slope[s, d] * res[p, s, d]
        for s in range(n):
            for d in range(n):
                raw = res[2, s, d] if (full_truth or release[2, s]) else hint[s, d]
                if raw > 0:
                    geom[2, s, d] = edge_intercept[s, d] + edge_slope[s, d] * raw
        for p in range(3):
            for s in range(n):
                for d in range(n):
                    z = geom[p, s, d]
                    outgoing[p, s] += z
                    incoming[p, d] += z
                    if z > maxres:
                        maxres = z

        p2tail = np.zeros(n, np.float64)
        maxin = 1.0
        for d in range(n):
            if incoming[2, d] > maxin:
                maxin = incoming[2, d]
        for r in range(n):
            exposure = 0.0
            for d in range(n):
                exposure += geom[2, r, d] * (incoming[2, d] / maxin)
            p2tail[r] = outgoing[2, r] + 0.2 * exposure
        p1tail = np.zeros(n, np.float64)
        for r in range(n):
            mx = 0.0
            for d in range(n):
                if geom[1, r, d] > 0 and p2tail[d] > mx:
                    mx = p2tail[d]
            p1tail[r] = outgoing[1, r] + 0.5 * mx

        path0 = np.zeros(n, np.float64)
        path1 = np.zeros(n, np.float64)
        eta = np.zeros(n, np.float64)
        for s in range(n):
            eta[s] = 0.0 if release[1, s] else incoming[0, s]
        for d in range(n):
            mx = 0.0
            for s in range(n):
                if geom[1, s, d] > 0 and eta[s] > mx:
                    mx = eta[s]
            path0[d] = incoming[0, d] + p1tail[d]
            path1[d] = max(incoming[1, d], mx) + p2tail[d]
        bp0 = _softmax(path0)
        bp1 = _softmax(path1)

        send = np.zeros(n, np.float64)
        recv = np.zeros(n, np.float64)
        for p in range(3):
            for s in range(n):
                for d in range(n):
                    send[s] += geom[p, s, d]
                    recv[d] += geom[p, s, d]
        sp = _softmax(send)
        rp = _softmax(recv)
        maxt = 1.0
        for r in range(n):
            if p1tail[r] > maxt:
                maxt = p1tail[r]
            if p2tail[r] > maxt:
                maxt = p2tail[r]

        weights = np.zeros((n, n), np.float64)
        best = np.full((n, n), -1, np.int8)
        for p in range(3):
            if scope_local and p != active_phase:
                continue
            for s in range(n):
                if release[p, s] == 0:
                    continue
                for d in range(n):
                    rem = res[p, s, d]
                    if rem <= 0:
                        continue
                    cost_residual = edge_intercept[s, d] + edge_slope[s, d] * rem
                    residual = cost_residual / maxres
                    endpoint = sp[s] + rp[d]
                    barrier = 0.0
                    unlock = 0.0
                    if p == 0:
                        inrem = max(incoming[0, d], 1.0)
                        barrier = bp0[d]
                        unlock = (cost_residual / inrem) * p1tail[d] / maxt
                    elif p == 1:
                        inrem = max(incoming[1, d], 1.0)
                        barrier = bp1[d]
                        unlock = (cost_residual / inrem) * p2tail[d] / maxt
                    score = wres * residual + wcp * barrier + wunlock * unlock + wend * endpoint
                    score += 1e-9 * (3 - p) + 1e-12 * (n - s) + 1e-15 * (n - d)
                    if score > weights[s, d]:
                        weights[s, d] = score
                        best[s, d] = p

        assignment = _hungarian_max(weights)
        amount = 2147483647
        selected = 0
        for s in range(n):
            d = assignment[s]
            if d >= 0 and best[s, d] >= 0 and weights[s, d] > 1e-12:
                p = best[s, d]
                r = res[p, s, d]
                if r < amount:
                    amount = r
                selected += 1
        if selected == 0:
            nxt = _next_release(now, release, release_at)
            if nxt >= 1e290:
                break
            now = nxt
            continue

        wave_time = _wave_duration(amount, assignment, best, weights, edge_slope, edge_intercept, wave_launch_b)
        start = now
        end = now + wave_time
        for s in range(n):
            d = assignment[s]
            if d >= 0 and best[s, d] >= 0 and weights[s, d] > 1e-12:
                p = best[s, d]
                res[p, s, d] -= amount
                out_phase[wave, s] = p
                out_dst[wave, s] = d
                out_size[wave, s] = amount
                if end > phase_done[p]:
                    phase_done[p] = end
        for r in range(n):
            if release_at[1, r] >= 1e290:
                z = 0
                for s in range(n):
                    z += res[0, s, r]
                if z == 0:
                    release_at[1, r] = end + expert_compute_delay
                    rel1[r] = release_at[1, r]
            if release_at[2, r] >= 1e290:
                z = 0
                for s in range(n):
                    z += res[1, s, r]
                if z == 0 and release_at[1, r] < 1e290:
                    release_at[2, r] = max(end, release_at[1, r])
                    rel2[r] = release_at[2, r]
        quantum[wave] = amount
        duration[wave] = wave_time
        starts[wave] = start
        ends[wave] = end
        now = end
        wave += 1

    valid = 1 if _sum_all(res) == 0 else 0
    return (
        now, wave, out_phase[:wave], out_dst[:wave], out_size[:wave], quantum[:wave],
        duration[:wave], starts[:wave], ends[:wave], phase_done, rel1, rel2, valid,
    )


@njit(cache=True)
def _event_plan_p0123(
    mats: np.ndarray,
    hint: np.ndarray,
    p3_hint: np.ndarray,
    edge_slope: np.ndarray,
    edge_intercept: np.ndarray,
    scope_local: bool,
    full_truth: bool,
    p3_weight: float,
    wres: float,
    wcp: float,
    wunlock: float,
    wend: float,
    expert_compute_delay: float,
    wave_launch_b: float,
    max_waves: int,
):
    res = mats.copy()
    n = res.shape[1]
    maxw = max_waves
    out_phase = np.full((maxw, n), -1, np.int8)
    out_dst = np.full((maxw, n), -1, np.int16)
    out_size = np.zeros((maxw, n), np.int32)
    quantum = np.zeros(maxw, np.int32)
    duration = np.zeros(maxw, np.float64)
    starts = np.zeros(maxw, np.float64)
    ends = np.zeros(maxw, np.float64)

    release = np.zeros((3, n), np.uint8)
    release[0, :] = 1
    release_at = np.full((3, n), 1e300, np.float64)
    release_at[0, :] = 0.0
    rel1 = np.full(n, -1.0, np.float64)
    rel2 = np.full(n, -1.0, np.float64)
    for r in range(n):
        in0 = 0
        in1 = 0
        for s in range(n):
            in0 += res[0, s, r]
            in1 += res[1, s, r]
        if in0 == 0:
            release_at[1, r] = expert_compute_delay
            rel1[r] = expert_compute_delay
            if in1 == 0:
                release_at[2, r] = expert_compute_delay
                rel2[r] = expert_compute_delay
    now = 0.0
    _refresh_release(now, release, release_at)
    wave = 0
    active_phase = 0
    phase_done = np.zeros(3, np.float64)

    while _sum_all(res) > 0 and wave < maxw:
        _refresh_release(now, release, release_at)
        if scope_local:
            while active_phase < 3:
                z = 0
                for s in range(n):
                    for d in range(n):
                        z += res[active_phase, s, d]
                if z > 0:
                    break
                active_phase += 1

        geom = np.zeros((3, n, n), np.float64)
        outgoing = np.zeros((3, n), np.float64)
        incoming = np.zeros((3, n), np.float64)
        maxres = 1.0
        for p in range(2):
            for s in range(n):
                for d in range(n):
                    if res[p, s, d] > 0:
                        geom[p, s, d] = edge_intercept[s, d] + edge_slope[s, d] * res[p, s, d]
        for s in range(n):
            for d in range(n):
                raw = res[2, s, d] if (full_truth or release[2, s]) else hint[s, d]
                if raw > 0:
                    geom[2, s, d] = edge_intercept[s, d] + edge_slope[s, d] * raw
        for p in range(3):
            for s in range(n):
                for d in range(n):
                    z = geom[p, s, d]
                    outgoing[p, s] += z
                    incoming[p, d] += z
                    if z > maxres:
                        maxres = z

        # P3 is advisory-only: it never enters residual traffic or the emitted
        # executable template.  It only extends the downstream critical-tail
        # geometry used to rank P0/P1/P2 edges.  For standard MoE semantics
        # p3_hint is derived as p2_hint.T, but directed link costs are applied
        # after transposition so asymmetric topologies remain visible.
        p3_geom = np.zeros((n, n), np.float64)
        p3_outgoing = np.zeros(n, np.float64)
        p3_incoming = np.zeros(n, np.float64)
        p3_maxin = 1.0
        for s in range(n):
            for d in range(n):
                raw3 = p3_hint[s, d]
                if raw3 > 0:
                    z3 = edge_intercept[s, d] + edge_slope[s, d] * raw3
                    p3_geom[s, d] = z3
                    p3_outgoing[s] += z3
                    p3_incoming[d] += z3
        for d in range(n):
            if p3_incoming[d] > p3_maxin:
                p3_maxin = p3_incoming[d]
        p3tail = np.zeros(n, np.float64)
        for r in range(n):
            exposure3 = 0.0
            for d in range(n):
                exposure3 += p3_geom[r, d] * (p3_incoming[d] / p3_maxin)
            p3tail[r] = p3_weight * (p3_outgoing[r] + 0.2 * exposure3)

        p2tail = np.zeros(n, np.float64)
        maxin = 1.0
        for d in range(n):
            if incoming[2, d] > maxin:
                maxin = incoming[2, d]
        for r in range(n):
            exposure = 0.0
            downstream = 0.0
            for d in range(n):
                exposure += geom[2, r, d] * (incoming[2, d] / maxin)
                if geom[2, r, d] > 0 and p3tail[d] > downstream:
                    downstream = p3tail[d]
            p2tail[r] = outgoing[2, r] + 0.2 * exposure + 0.5 * downstream
        p1tail = np.zeros(n, np.float64)
        for r in range(n):
            mx = 0.0
            for d in range(n):
                if geom[1, r, d] > 0 and p2tail[d] > mx:
                    mx = p2tail[d]
            p1tail[r] = outgoing[1, r] + 0.5 * mx

        path0 = np.zeros(n, np.float64)
        path1 = np.zeros(n, np.float64)
        eta = np.zeros(n, np.float64)
        for s in range(n):
            eta[s] = 0.0 if release[1, s] else incoming[0, s]
        for d in range(n):
            mx = 0.0
            for s in range(n):
                if geom[1, s, d] > 0 and eta[s] > mx:
                    mx = eta[s]
            path0[d] = incoming[0, d] + p1tail[d]
            path1[d] = max(incoming[1, d], mx) + p2tail[d]
        bp0 = _softmax(path0)
        bp1 = _softmax(path1)

        send = np.zeros(n, np.float64)
        recv = np.zeros(n, np.float64)
        for p in range(3):
            for s in range(n):
                for d in range(n):
                    send[s] += geom[p, s, d]
                    recv[d] += geom[p, s, d]
        for s in range(n):
            for d in range(n):
                send[s] += p3_weight * p3_geom[s, d]
                recv[d] += p3_weight * p3_geom[s, d]
        sp = _softmax(send)
        rp = _softmax(recv)
        maxt = 1.0
        for r in range(n):
            if p1tail[r] > maxt:
                maxt = p1tail[r]
            if p2tail[r] > maxt:
                maxt = p2tail[r]

        weights = np.zeros((n, n), np.float64)
        best = np.full((n, n), -1, np.int8)
        for p in range(3):
            if scope_local and p != active_phase:
                continue
            for s in range(n):
                if release[p, s] == 0:
                    continue
                for d in range(n):
                    rem = res[p, s, d]
                    if rem <= 0:
                        continue
                    cost_residual = edge_intercept[s, d] + edge_slope[s, d] * rem
                    residual = cost_residual / maxres
                    endpoint = sp[s] + rp[d]
                    barrier = 0.0
                    unlock = 0.0
                    if p == 0:
                        inrem = max(incoming[0, d], 1.0)
                        barrier = bp0[d]
                        unlock = (cost_residual / inrem) * p1tail[d] / maxt
                    elif p == 1:
                        inrem = max(incoming[1, d], 1.0)
                        barrier = bp1[d]
                        unlock = (cost_residual / inrem) * p2tail[d] / maxt
                    score = wres * residual + wcp * barrier + wunlock * unlock + wend * endpoint
                    score += 1e-9 * (3 - p) + 1e-12 * (n - s) + 1e-15 * (n - d)
                    if score > weights[s, d]:
                        weights[s, d] = score
                        best[s, d] = p

        assignment = _hungarian_max(weights)
        amount = 2147483647
        selected = 0
        for s in range(n):
            d = assignment[s]
            if d >= 0 and best[s, d] >= 0 and weights[s, d] > 1e-12:
                p = best[s, d]
                r = res[p, s, d]
                if r < amount:
                    amount = r
                selected += 1
        if selected == 0:
            nxt = _next_release(now, release, release_at)
            if nxt >= 1e290:
                break
            now = nxt
            continue

        wave_time = _wave_duration(amount, assignment, best, weights, edge_slope, edge_intercept, wave_launch_b)
        start = now
        end = now + wave_time
        for s in range(n):
            d = assignment[s]
            if d >= 0 and best[s, d] >= 0 and weights[s, d] > 1e-12:
                p = best[s, d]
                res[p, s, d] -= amount
                out_phase[wave, s] = p
                out_dst[wave, s] = d
                out_size[wave, s] = amount
                if end > phase_done[p]:
                    phase_done[p] = end
        for r in range(n):
            if release_at[1, r] >= 1e290:
                z = 0
                for s in range(n):
                    z += res[0, s, r]
                if z == 0:
                    release_at[1, r] = end + expert_compute_delay
                    rel1[r] = release_at[1, r]
            if release_at[2, r] >= 1e290:
                z = 0
                for s in range(n):
                    z += res[1, s, r]
                if z == 0 and release_at[1, r] < 1e290:
                    release_at[2, r] = max(end, release_at[1, r])
                    rel2[r] = release_at[2, r]
        quantum[wave] = amount
        duration[wave] = wave_time
        starts[wave] = start
        ends[wave] = end
        now = end
        wave += 1

    valid = 1 if _sum_all(res) == 0 else 0
    return (
        now, wave, out_phase[:wave], out_dst[:wave], out_size[:wave], quantum[:wave],
        duration[:wave], starts[:wave], ends[:wave], phase_done, rel1, rel2, valid,
    )



@njit(cache=True)
def _forecast_priority(hint: np.ndarray, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    n = hint.shape[0]
    outgoing = np.zeros(n, np.float64)
    incoming = np.zeros(n, np.float64)
    scale = 1.0
    weighted = np.zeros((n, n), np.float64)
    for s in range(n):
        for d in range(n):
            if hint[s, d] > 0:
                weighted[s, d] = intercept[s, d] + slope[s, d] * hint[s, d]
            outgoing[s] += weighted[s, d]
            incoming[d] += weighted[s, d]
    for r in range(n):
        if outgoing[r] > scale:
            scale = outgoing[r]
        if incoming[r] > scale:
            scale = incoming[r]
    priority = np.zeros((n, n), np.float64)
    for s in range(n):
        for d in range(n):
            if s != d:
                priority[s, d] = (weighted[s, d] + 0.25 * outgoing[s] + 0.25 * incoming[d]) / scale
    return priority


@njit(cache=True)
def _bind_template(
    actual: np.ndarray,
    hint: np.ndarray,
    t_phase: np.ndarray,
    t_dst: np.ndarray,
    t_size: np.ndarray,
    t_duration: np.ndarray,
    t_starts: np.ndarray,
    edge_slope: np.ndarray,
    edge_intercept: np.ndarray,
    expert_compute_delay: float,
    wave_launch_b: float,
    max_waves: int,
):
    res = actual.copy()
    n = res.shape[1]
    maxw = max_waves
    out_phase = np.full((maxw, n), -1, np.int8)
    out_dst = np.full((maxw, n), -1, np.int16)
    out_size = np.zeros((maxw, n), np.int32)
    quantum = np.zeros(maxw, np.int32)
    duration = np.zeros(maxw, np.float64)
    starts = np.zeros(maxw, np.float64)
    ends = np.zeros(maxw, np.float64)

    release = np.zeros((3, n), np.uint8)
    release[0, :] = 1
    release_at = np.full((3, n), 1e300, np.float64)
    release_at[0, :] = 0.0
    rel1 = np.full(n, -1.0, np.float64)
    rel2 = np.full(n, -1.0, np.float64)
    for r in range(n):
        in0 = 0
        in1 = 0
        for s in range(n):
            in0 += res[0, s, r]
            in1 += res[1, s, r]
        if in0 == 0:
            release_at[1, r] = expert_compute_delay
            rel1[r] = expert_compute_delay
            if in1 == 0:
                release_at[2, r] = expert_compute_delay
                rel2[r] = expert_compute_delay
    priority = _forecast_priority(hint, edge_slope, edge_intercept)
    now = 0.0
    _refresh_release(now, release, release_at)
    wave = 0
    phase_done = np.zeros(3, np.float64)

    for tw in range(t_phase.shape[0]):
        if wave >= maxw:
            break
        if _sum_all(res) == 0:
            break
        # Preserve explicit idle/compute gaps from the forecast template.
        if now < t_starts[tw]:
            now = t_starts[tw]
        _refresh_release(now, release, release_at)
        cap = 0
        for s in range(n):
            if t_size[tw, s] > cap:
                cap = t_size[tw, s]
        if cap <= 0:
            continue
        used_d = np.zeros(n, np.uint8)
        has = np.zeros(n, np.uint8)
        served = np.zeros(n, np.int32)
        pp = np.full(n, -1, np.int8)
        dd = np.full(n, -1, np.int16)
        for s in range(n):
            p = t_phase[tw, s]
            d = t_dst[tw, s]
            if p < 0 or d < 0 or release[p, s] == 0 or used_d[d] != 0:
                continue
            available = res[p, s, d]
            if available <= 0:
                continue
            amount = available
            if amount > cap:
                amount = cap
            if t_size[tw, s] > 0 and amount > t_size[tw, s]:
                amount = t_size[tw, s]
            if amount > 0:
                has[s] = 1
                served[s] = amount
                pp[s] = p
                dd[s] = d
                used_d[d] = 1

        planned_transfer_budget = t_duration[tw] - wave_launch_b
        if planned_transfer_budget < 0.0:
            planned_transfer_budget = 0.0
        weights = np.zeros((n, n), np.float64)
        for s in range(n):
            if has[s] != 0 or release[2, s] == 0:
                continue
            for d in range(n):
                if used_d[d] == 0 and res[2, s, d] > 0:
                    weighted_res = edge_intercept[s, d] + edge_slope[s, d] * res[2, s, d]
                    weights[s, d] = priority[s, d] + 1e-6 * weighted_res + 1e-12 * (n - s) + 1e-15 * (n - d)
        assignment = _hungarian_max(weights)
        for s in range(n):
            if has[s] != 0:
                continue
            d = assignment[s]
            if d >= 0 and used_d[d] == 0 and weights[s, d] > 1e-12 and res[2, s, d] > 0:
                max_fit = 0
                if edge_slope[s, d] > 0.0 and planned_transfer_budget + 1e-12 >= edge_intercept[s, d]:
                    max_fit = int(math.floor((planned_transfer_budget - edge_intercept[s, d]) / edge_slope[s, d] + 1e-12))
                amount = res[2, s, d]
                if amount > max_fit:
                    amount = max_fit
                if amount > 0:
                    has[s] = 1
                    served[s] = amount
                    pp[s] = 2
                    dd[s] = d
                    used_d[d] = 1

        selected = 0
        for s in range(n):
            if has[s] != 0:
                selected += 1
        if selected == 0:
            nxt = _next_release(now, release, release_at)
            if nxt < 1e290:
                now = nxt
                _refresh_release(now, release, release_at)
            continue

        wave_time = wave_launch_b
        for s in range(n):
            if has[s] != 0:
                d = dd[s]
                t = edge_intercept[s, d] + edge_slope[s, d] * served[s]
                if wave_launch_b + t > wave_time:
                    wave_time = wave_launch_b + t
        start = now
        end = now + wave_time
        for s in range(n):
            if has[s] != 0:
                p = pp[s]
                d = dd[s]
                amount = served[s]
                res[p, s, d] -= amount
                out_phase[wave, s] = p
                out_dst[wave, s] = d
                out_size[wave, s] = amount
                if end > phase_done[p]:
                    phase_done[p] = end
        for r in range(n):
            if release_at[1, r] >= 1e290:
                z = 0
                for s in range(n):
                    z += res[0, s, r]
                if z == 0:
                    release_at[1, r] = end + expert_compute_delay
                    rel1[r] = release_at[1, r]
            if release_at[2, r] >= 1e290:
                z = 0
                for s in range(n):
                    z += res[1, s, r]
                if z == 0 and release_at[1, r] < 1e290:
                    release_at[2, r] = max(end, release_at[1, r])
                    rel2[r] = release_at[2, r]
        actual_quantum = 0
        for s in range(n):
            if served[s] > actual_quantum:
                actual_quantum = served[s]
        quantum[wave] = actual_quantum
        duration[wave] = wave_time
        starts[wave] = start
        ends[wave] = end
        now = end
        wave += 1

    known_left = 0
    for p in range(2):
        for s in range(n):
            for d in range(n):
                known_left += res[p, s, d]

    while wave < maxw:
        _refresh_release(now, release, release_at)
        left = 0
        for s in range(n):
            for d in range(n):
                left += res[2, s, d]
        if left == 0:
            break
        weights = np.zeros((n, n), np.float64)
        maxcost = 1.0
        send = np.zeros(n, np.float64)
        recv = np.zeros(n, np.float64)
        for s in range(n):
            for d in range(n):
                if res[2, s, d] > 0:
                    c = edge_intercept[s, d] + edge_slope[s, d] * res[2, s, d]
                    send[s] += c
                    recv[d] += c
                    if c > maxcost:
                        maxcost = c
        sp = _softmax(send, 0.2)
        rp = _softmax(recv, 0.2)
        for s in range(n):
            if release[2, s] == 0:
                continue
            for d in range(n):
                if res[2, s, d] > 0:
                    c = edge_intercept[s, d] + edge_slope[s, d] * res[2, s, d]
                    weights[s, d] = c / maxcost + 0.5 * (sp[s] + rp[d]) + 1e-12 * (n - s) + 1e-15 * (n - d)
        assignment = _hungarian_max(weights)
        amount = 2147483647
        selected = 0
        best = np.full((n, n), -1, np.int8)
        for s in range(n):
            d = assignment[s]
            if d >= 0 and weights[s, d] > 1e-12 and res[2, s, d] > 0:
                best[s, d] = 2
                if res[2, s, d] < amount:
                    amount = res[2, s, d]
                selected += 1
        if selected == 0:
            nxt = _next_release(now, release, release_at)
            if nxt >= 1e290:
                break
            now = nxt
            continue
        wave_time = _wave_duration(amount, assignment, best, weights, edge_slope, edge_intercept, wave_launch_b)
        start = now
        end = now + wave_time
        for s in range(n):
            d = assignment[s]
            if d >= 0 and weights[s, d] > 1e-12 and res[2, s, d] > 0:
                res[2, s, d] -= amount
                out_phase[wave, s] = 2
                out_dst[wave, s] = d
                out_size[wave, s] = amount
        phase_done[2] = end
        quantum[wave] = amount
        duration[wave] = wave_time
        starts[wave] = start
        ends[wave] = end
        now = end
        wave += 1

    valid = 1 if known_left == 0 and _sum_all(res) == 0 else 0
    return (
        now, wave, out_phase[:wave], out_dst[:wave], out_size[:wave], quantum[:wave],
        duration[:wave], starts[:wave], ends[:wave], phase_done, rel1, rel2, valid,
    )


@njit(cache=True)
def _future_template_priority(
    t_phase: np.ndarray,
    t_dst: np.ndarray,
    t_size: np.ndarray,
    n: int,
) -> np.ndarray:
    """Compile an immutable future template into a dense edge-priority tensor."""
    waves = t_phase.shape[0]
    priority = np.zeros((3, n, n), np.float64)
    max_size = 1.0
    for w in range(waves):
        for source in range(n):
            amount = t_size[w, source]
            if amount > max_size:
                max_size = float(amount)
    denom = float(max(waves, 1))
    for w in range(waves):
        order_score = float(waves - w) / denom
        for source in range(n):
            phase = t_phase[w, source]
            destination = t_dst[w, source]
            amount = t_size[w, source]
            if 0 <= phase < 3 and 0 <= destination < n and amount > 0:
                score = order_score + 0.05 * float(amount) / max_size
                if score > priority[phase, source, destination]:
                    priority[phase, source, destination] = score
    return priority


@njit(cache=True)
def _future_repair_weights(
    res: np.ndarray,
    release: np.ndarray,
    edge_slope: np.ndarray,
    edge_intercept: np.ndarray,
    static_priority: np.ndarray,
    wres: float,
    wcp: float,
    wunlock: float,
    wend: float,
):
    """Template-guided residual matching for Future-P012 target binding."""
    n = res.shape[1]
    geom = np.zeros((3, n, n), np.float64)
    outgoing = np.zeros((3, n), np.float64)
    incoming = np.zeros((3, n), np.float64)
    maxres = 1.0
    for p in range(3):
        for source in range(n):
            for destination in range(n):
                rows = res[p, source, destination]
                if rows > 0:
                    cost = edge_intercept[source, destination] + edge_slope[source, destination] * rows
                    geom[p, source, destination] = cost
                    outgoing[p, source] += cost
                    incoming[p, destination] += cost
                    if cost > maxres:
                        maxres = cost

    p2tail = np.zeros(n, np.float64)
    maxin = 1.0
    for destination in range(n):
        if incoming[2, destination] > maxin:
            maxin = incoming[2, destination]
    for rank in range(n):
        exposure = 0.0
        for destination in range(n):
            exposure += geom[2, rank, destination] * (incoming[2, destination] / maxin)
        p2tail[rank] = outgoing[2, rank] + 0.2 * exposure
    p1tail = np.zeros(n, np.float64)
    for rank in range(n):
        mx = 0.0
        for destination in range(n):
            if geom[1, rank, destination] > 0 and p2tail[destination] > mx:
                mx = p2tail[destination]
        p1tail[rank] = outgoing[1, rank] + 0.5 * mx

    path0 = np.zeros(n, np.float64)
    path1 = np.zeros(n, np.float64)
    eta = np.zeros(n, np.float64)
    for source in range(n):
        eta[source] = 0.0 if release[1, source] else incoming[0, source]
    for destination in range(n):
        mx = 0.0
        for source in range(n):
            if geom[1, source, destination] > 0 and eta[source] > mx:
                mx = eta[source]
        path0[destination] = incoming[0, destination] + p1tail[destination]
        path1[destination] = max(incoming[1, destination], mx) + p2tail[destination]
    bp0 = _softmax(path0)
    bp1 = _softmax(path1)

    send = np.zeros(n, np.float64)
    recv = np.zeros(n, np.float64)
    for p in range(3):
        for source in range(n):
            for destination in range(n):
                send[source] += geom[p, source, destination]
                recv[destination] += geom[p, source, destination]
    sp = _softmax(send)
    rp = _softmax(recv)
    maxt = 1.0
    for rank in range(n):
        if p1tail[rank] > maxt:
            maxt = p1tail[rank]
        if p2tail[rank] > maxt:
            maxt = p2tail[rank]

    weights = np.zeros((n, n), np.float64)
    best = np.full((n, n), -1, np.int8)
    for p in range(3):
        for source in range(n):
            if release[p, source] == 0:
                continue
            for destination in range(n):
                remaining = res[p, source, destination]
                if remaining <= 0:
                    continue
                cost = edge_intercept[source, destination] + edge_slope[source, destination] * remaining
                residual = cost / maxres
                endpoint = sp[source] + rp[destination]
                barrier = 0.0
                unlock = 0.0
                if p == 0:
                    denom = max(incoming[0, destination], 1.0)
                    barrier = bp0[destination]
                    unlock = (cost / denom) * p1tail[destination] / maxt
                elif p == 1:
                    denom = max(incoming[1, destination], 1.0)
                    barrier = bp1[destination]
                    unlock = (cost / denom) * p2tail[destination] / maxt
                score = (
                    wres * residual + wcp * barrier + wunlock * unlock + wend * endpoint
                    + 1.25 * static_priority[p, source, destination]
                )
                score += 1e-9 * (3 - p) + 1e-12 * (n - source) + 1e-15 * (n - destination)
                if score > weights[source, destination]:
                    weights[source, destination] = score
                    best[source, destination] = p
    return weights, best


@njit(cache=True)
def _bind_future_template(
    actual: np.ndarray,
    t_phase: np.ndarray,
    t_dst: np.ndarray,
    t_size: np.ndarray,
    edge_slope: np.ndarray,
    edge_intercept: np.ndarray,
    expert_compute_delay: float,
    wave_launch_b: float,
    max_waves: int,
    wres: float,
    wcp: float,
    wunlock: float,
    wend: float,
    use_template_slots: bool,
):
    """Fast target-side bind for a previous-layer Future-P012 template.

    The immutable template supplies edge order.  Target truth only scales legal
    slots and triggers one residual suffix repair; it never invokes the frozen
    on-demand P012 planner.
    """
    res = actual.copy()
    n = res.shape[1]
    maxw = max_waves
    out_phase = np.full((maxw, n), -1, np.int8)
    out_dst = np.full((maxw, n), -1, np.int16)
    out_size = np.zeros((maxw, n), np.int32)
    quantum = np.zeros(maxw, np.int32)
    duration = np.zeros(maxw, np.float64)
    starts = np.zeros(maxw, np.float64)
    ends = np.zeros(maxw, np.float64)

    release = np.zeros((3, n), np.uint8)
    release[0, :] = 1
    release_at = np.full((3, n), 1e300, np.float64)
    release_at[0, :] = 0.0
    rel1 = np.full(n, -1.0, np.float64)
    rel2 = np.full(n, -1.0, np.float64)
    for rank in range(n):
        in0 = 0
        in1 = 0
        for source in range(n):
            in0 += res[0, source, rank]
            in1 += res[1, source, rank]
        if in0 == 0:
            release_at[1, rank] = expert_compute_delay
            rel1[rank] = expert_compute_delay
            if in1 == 0:
                release_at[2, rank] = expert_compute_delay
                rel2[rank] = expert_compute_delay

    template_total = np.zeros((3, n, n), np.int64)
    template_remaining = np.zeros((3, n, n), np.int64)
    template_last = np.full((3, n, n), -1, np.int32)
    for tw in range(t_phase.shape[0]):
        for source in range(n):
            phase = t_phase[tw, source]
            destination = t_dst[tw, source]
            amount = t_size[tw, source]
            if 0 <= phase < 3 and 0 <= destination < n and amount > 0:
                template_total[phase, source, destination] += amount
                template_remaining[phase, source, destination] += amount
                template_last[phase, source, destination] = tw
    static_priority = _future_template_priority(t_phase, t_dst, t_size, n)

    now = 0.0
    _refresh_release(now, release, release_at)
    wave = 0
    phase_done = np.zeros(3, np.float64)
    template_rows = 0
    template_waves = 0
    projected_edges = 0

    template_loop_count = t_phase.shape[0] if use_template_slots else 0
    for tw in range(template_loop_count):
        if wave >= maxw or _sum_all(res) == 0:
            break
        _refresh_release(now, release, release_at)
        used_d = np.zeros(n, np.uint8)
        has = np.zeros(n, np.uint8)
        served = np.zeros(n, np.int32)
        pp = np.full(n, -1, np.int8)
        dd = np.full(n, -1, np.int16)

        for source in range(n):
            phase = t_phase[tw, source]
            destination = t_dst[tw, source]
            slot = t_size[tw, source]
            if not (0 <= phase < 3 and 0 <= destination < n and slot > 0):
                continue
            if release[phase, source] == 0 or used_d[destination] != 0:
                continue
            available = res[phase, source, destination]
            remaining_template = template_remaining[phase, source, destination]
            if available <= 0 or remaining_template <= 0:
                continue
            if tw == template_last[phase, source, destination]:
                amount = available
            else:
                amount = int(round(float(available) * float(slot) / float(remaining_template)))
                if amount < 1:
                    amount = 1
                if amount > available:
                    amount = available
            template_remaining[phase, source, destination] = max(remaining_template - slot, 0)
            has[source] = 1
            served[source] = amount
            pp[source] = phase
            dd[source] = destination
            used_d[destination] = 1
            template_rows += amount
            projected_edges += 1

        # Fill free endpoints according to precomputed priorities.  This remains
        # a bind operation: no fresh forecast geometry or candidate plan is built.
        weights, best = _future_repair_weights(
            res, release, edge_slope, edge_intercept, static_priority,
            wres, wcp, wunlock, wend,
        )
        for source in range(n):
            if has[source] != 0:
                for destination in range(n):
                    weights[source, destination] = 0.0
            else:
                for destination in range(n):
                    if used_d[destination] != 0:
                        weights[source, destination] = 0.0
        assignment = _hungarian_max(weights)
        cap = 0
        for source in range(n):
            if t_size[tw, source] > cap:
                cap = t_size[tw, source]
        for source in range(n):
            if has[source] != 0:
                continue
            destination = assignment[source]
            if destination < 0 or used_d[destination] != 0 or weights[source, destination] <= 1e-12:
                continue
            phase = best[source, destination]
            if phase < 0:
                continue
            amount = res[phase, source, destination]
            if cap > 0 and amount > cap:
                amount = cap
            if amount <= 0:
                continue
            has[source] = 1
            served[source] = amount
            pp[source] = phase
            dd[source] = destination
            used_d[destination] = 1

        selected = 0
        for source in range(n):
            if has[source] != 0:
                selected += 1
        if selected == 0:
            nxt = _next_release(now, release, release_at)
            if nxt < 1e290:
                now = nxt
                _refresh_release(now, release, release_at)
            continue

        wave_time = wave_launch_b
        actual_quantum = 0
        for source in range(n):
            if has[source] != 0:
                destination = dd[source]
                t = edge_intercept[source, destination] + edge_slope[source, destination] * served[source]
                if wave_launch_b + t > wave_time:
                    wave_time = wave_launch_b + t
                if served[source] > actual_quantum:
                    actual_quantum = served[source]
        start = now
        end = now + wave_time
        for source in range(n):
            if has[source] != 0:
                phase = pp[source]
                destination = dd[source]
                amount = served[source]
                res[phase, source, destination] -= amount
                out_phase[wave, source] = phase
                out_dst[wave, source] = destination
                out_size[wave, source] = amount
                if end > phase_done[phase]:
                    phase_done[phase] = end
        for rank in range(n):
            if release_at[1, rank] >= 1e290:
                incoming = 0
                for source in range(n):
                    incoming += res[0, source, rank]
                if incoming == 0:
                    release_at[1, rank] = end + expert_compute_delay
                    rel1[rank] = release_at[1, rank]
            if release_at[2, rank] >= 1e290:
                incoming = 0
                for source in range(n):
                    incoming += res[1, source, rank]
                if incoming == 0 and release_at[1, rank] < 1e290:
                    release_at[2, rank] = max(end, release_at[1, rank])
                    rel2[rank] = release_at[2, rank]
        quantum[wave] = actual_quantum
        duration[wave] = wave_time
        starts[wave] = start
        ends[wave] = end
        now = end
        wave += 1
        template_waves += 1

    repair_rows = _sum_all(res)
    repair_waves = 0
    while _sum_all(res) > 0 and wave < maxw:
        _refresh_release(now, release, release_at)
        weights, best = _future_repair_weights(
            res, release, edge_slope, edge_intercept, static_priority,
            wres, wcp, wunlock, wend,
        )
        assignment = _hungarian_max(weights)
        amount = 2147483647
        selected = 0
        for source in range(n):
            destination = assignment[source]
            if destination >= 0 and best[source, destination] >= 0 and weights[source, destination] > 1e-12:
                phase = best[source, destination]
                remaining = res[phase, source, destination]
                if remaining < amount:
                    amount = remaining
                selected += 1
        if selected == 0:
            nxt = _next_release(now, release, release_at)
            if nxt >= 1e290:
                break
            now = nxt
            continue
        wave_time = _wave_duration(amount, assignment, best, weights, edge_slope, edge_intercept, wave_launch_b)
        start = now
        end = now + wave_time
        for source in range(n):
            destination = assignment[source]
            if destination >= 0 and best[source, destination] >= 0 and weights[source, destination] > 1e-12:
                phase = best[source, destination]
                res[phase, source, destination] -= amount
                out_phase[wave, source] = phase
                out_dst[wave, source] = destination
                out_size[wave, source] = amount
                if end > phase_done[phase]:
                    phase_done[phase] = end
        for rank in range(n):
            if release_at[1, rank] >= 1e290:
                incoming = 0
                for source in range(n):
                    incoming += res[0, source, rank]
                if incoming == 0:
                    release_at[1, rank] = end + expert_compute_delay
                    rel1[rank] = release_at[1, rank]
            if release_at[2, rank] >= 1e290:
                incoming = 0
                for source in range(n):
                    incoming += res[1, source, rank]
                if incoming == 0 and release_at[1, rank] < 1e290:
                    release_at[2, rank] = max(end, release_at[1, rank])
                    rel2[rank] = release_at[2, rank]
        quantum[wave] = amount
        duration[wave] = wave_time
        starts[wave] = start
        ends[wave] = end
        now = end
        wave += 1
        repair_waves += 1

    valid = 1 if _sum_all(res) == 0 else 0
    support_total = 0
    support_hit = 0
    for phase in range(3):
        for source in range(n):
            for destination in range(n):
                if actual[phase, source, destination] > 0:
                    support_total += 1
                    if template_total[phase, source, destination] > 0:
                        support_hit += 1
    support = float(support_hit) / float(max(support_total, 1))
    return (
        now, wave, out_phase[:wave], out_dst[:wave], out_size[:wave], quantum[:wave],
        duration[:wave], starts[:wave], ends[:wave], phase_done, rel1, rel2, valid,
        template_rows, template_waves, projected_edges, repair_rows, repair_waves, support,
    )



@njit(cache=True)
def _bind_prepared_order(
    actual: np.ndarray,
    t_phase: np.ndarray,
    t_dst: np.ndarray,
    t_size: np.ndarray,
    edge_slope: np.ndarray,
    edge_intercept: np.ndarray,
    expert_compute_delay: float,
    wave_launch_b: float,
    max_waves: int,
):
    """Bind truth to an ahead-of-time matching skeleton without online search.

    The prepared template is consumed as a stable sequence of conflict-free
    slots.  Missing or delayed edges are not repaired with scoring/Hungarian;
    they are completed by a deterministic round-robin topology tail.  The
    target-side complexity is therefore O(template_slots + phases * N^2).
    """
    res = actual.copy()
    n = res.shape[1]
    # Every topology-tail wave removes at least one remaining edge.
    # This cap avoids allocating max_waves*N arrays (often 10k*N) on every bind.
    safe_bound = t_phase.shape[0] + 3 * n * max(n - 1, 1) + 8
    maxw = min(max_waves, safe_bound)
    out_phase = np.full((maxw, n), -1, np.int8)
    out_dst = np.full((maxw, n), -1, np.int16)
    out_size = np.zeros((maxw, n), np.int32)
    quantum = np.zeros(maxw, np.int32)
    duration = np.zeros(maxw, np.float64)
    starts = np.zeros(maxw, np.float64)
    ends = np.zeros(maxw, np.float64)

    release = np.zeros((3, n), np.uint8)
    release[0, :] = 1
    release_at = np.full((3, n), 1e300, np.float64)
    release_at[0, :] = 0.0
    rel1 = np.full(n, -1.0, np.float64)
    rel2 = np.full(n, -1.0, np.float64)
    for rank in range(n):
        in0 = 0
        in1 = 0
        for source in range(n):
            in0 += res[0, source, rank]
            in1 += res[1, source, rank]
        if in0 == 0:
            release_at[1, rank] = expert_compute_delay
            rel1[rank] = expert_compute_delay
            if in1 == 0:
                release_at[2, rank] = expert_compute_delay
                rel2[rank] = expert_compute_delay

    template_total = np.zeros((3, n, n), np.int64)
    template_remaining = np.zeros((3, n, n), np.int64)
    template_last = np.full((3, n, n), -1, np.int32)
    for tw in range(t_phase.shape[0]):
        for source in range(n):
            phase = t_phase[tw, source]
            destination = t_dst[tw, source]
            amount = t_size[tw, source]
            if 0 <= phase < 3 and 0 <= destination < n and amount > 0:
                template_total[phase, source, destination] += amount
                template_remaining[phase, source, destination] += amount
                template_last[phase, source, destination] = tw

    now = 0.0
    _refresh_release(now, release, release_at)
    wave = 0
    phase_done = np.zeros(3, np.float64)
    template_rows = 0
    template_waves = 0
    projected_edges = 0

    # Stable-filter the precomputed matching skeleton.  There is deliberately
    # no online candidate scoring and no matching solver in this loop.
    for tw in range(t_phase.shape[0]):
        if wave >= maxw or _sum_all(res) == 0:
            break
        _refresh_release(now, release, release_at)
        used_d = np.zeros(n, np.uint8)
        has = np.zeros(n, np.uint8)
        served = np.zeros(n, np.int32)
        pp = np.full(n, -1, np.int8)
        dd = np.full(n, -1, np.int16)
        for source in range(n):
            phase = t_phase[tw, source]
            destination = t_dst[tw, source]
            slot = t_size[tw, source]
            if not (0 <= phase < 3 and 0 <= destination < n and slot > 0):
                continue
            if release[phase, source] == 0 or used_d[destination] != 0:
                continue
            available = res[phase, source, destination]
            remaining_template = template_remaining[phase, source, destination]
            if available <= 0 or remaining_template <= 0:
                continue
            if tw == template_last[phase, source, destination]:
                amount = available
            else:
                amount = int(round(float(available) * float(slot) / float(remaining_template)))
                if amount < 1:
                    amount = 1
                if amount > available:
                    amount = available
            template_remaining[phase, source, destination] = max(remaining_template - slot, 0)
            has[source] = 1
            served[source] = amount
            pp[source] = phase
            dd[source] = destination
            used_d[destination] = 1
            template_rows += amount
            projected_edges += 1

        selected = 0
        for source in range(n):
            if has[source] != 0:
                selected += 1
        if selected == 0:
            continue

        wave_time = wave_launch_b
        actual_quantum = 0
        for source in range(n):
            if has[source] != 0:
                destination = dd[source]
                transfer = edge_intercept[source, destination] + edge_slope[source, destination] * served[source]
                if wave_launch_b + transfer > wave_time:
                    wave_time = wave_launch_b + transfer
                if served[source] > actual_quantum:
                    actual_quantum = served[source]
        start = now
        end = now + wave_time
        for source in range(n):
            if has[source] != 0:
                phase = pp[source]
                destination = dd[source]
                amount = served[source]
                res[phase, source, destination] -= amount
                out_phase[wave, source] = phase
                out_dst[wave, source] = destination
                out_size[wave, source] = amount
                if end > phase_done[phase]:
                    phase_done[phase] = end
        for rank in range(n):
            if release_at[1, rank] >= 1e290:
                incoming = 0
                for source in range(n):
                    incoming += res[0, source, rank]
                if incoming == 0:
                    release_at[1, rank] = end + expert_compute_delay
                    rel1[rank] = release_at[1, rank]
            if release_at[2, rank] >= 1e290:
                incoming = 0
                for source in range(n):
                    incoming += res[1, source, rank]
                if incoming == 0 and release_at[1, rank] < 1e290:
                    release_at[2, rank] = max(end, release_at[1, rank])
                    rel2[rank] = release_at[2, rank]
        quantum[wave] = actual_quantum
        duration[wave] = wave_time
        starts[wave] = start
        ends[wave] = end
        now = end
        wave += 1
        template_waves += 1

    topology_tail_rows = _sum_all(res)
    topology_tail_waves = 0

    # Complete uncovered truth with a solver-free edge-colouring tail.  For a
    # square rank graph, each cyclic shift is already a valid matching.
    for phase in range(3):
        while wave < maxw:
            phase_left = 0
            for source in range(n):
                for destination in range(n):
                    phase_left += res[phase, source, destination]
            if phase_left == 0:
                break
            _refresh_release(now, release, release_at)
            made_progress = 0
            for shift in range(1, n):
                if wave >= maxw:
                    break
                has = np.zeros(n, np.uint8)
                served = np.zeros(n, np.int32)
                dd = np.full(n, -1, np.int16)
                selected = 0
                common_amount = 2147483647
                for source in range(n):
                    if release[phase, source] == 0:
                        continue
                    destination = (source + shift) % n
                    amount = res[phase, source, destination]
                    if amount <= 0:
                        continue
                    has[source] = 1
                    dd[source] = destination
                    if amount < common_amount:
                        common_amount = amount
                    selected += 1
                if selected == 0:
                    continue
                for source in range(n):
                    if has[source] != 0:
                        served[source] = common_amount
                wave_time = wave_launch_b
                actual_quantum = 0
                for source in range(n):
                    if has[source] != 0:
                        destination = dd[source]
                        transfer = edge_intercept[source, destination] + edge_slope[source, destination] * served[source]
                        if wave_launch_b + transfer > wave_time:
                            wave_time = wave_launch_b + transfer
                        if served[source] > actual_quantum:
                            actual_quantum = served[source]
                start = now
                end = now + wave_time
                for source in range(n):
                    if has[source] != 0:
                        destination = dd[source]
                        amount = served[source]
                        res[phase, source, destination] -= amount
                        out_phase[wave, source] = phase
                        out_dst[wave, source] = destination
                        out_size[wave, source] = amount
                        if end > phase_done[phase]:
                            phase_done[phase] = end
                for rank in range(n):
                    if release_at[1, rank] >= 1e290:
                        incoming = 0
                        for source in range(n):
                            incoming += res[0, source, rank]
                        if incoming == 0:
                            release_at[1, rank] = end + expert_compute_delay
                            rel1[rank] = release_at[1, rank]
                    if release_at[2, rank] >= 1e290:
                        incoming = 0
                        for source in range(n):
                            incoming += res[1, source, rank]
                        if incoming == 0 and release_at[1, rank] < 1e290:
                            release_at[2, rank] = max(end, release_at[1, rank])
                            rel2[rank] = release_at[2, rank]
                quantum[wave] = actual_quantum
                duration[wave] = wave_time
                starts[wave] = start
                ends[wave] = end
                now = end
                wave += 1
                topology_tail_waves += 1
                made_progress = 1
                _refresh_release(now, release, release_at)
            if made_progress == 0:
                nxt = _next_release(now, release, release_at)
                if nxt >= 1e290:
                    break
                now = nxt
                _refresh_release(now, release, release_at)

    valid = 1 if _sum_all(res) == 0 else 0
    support_total = 0
    support_hit = 0
    for phase in range(3):
        for source in range(n):
            for destination in range(n):
                if actual[phase, source, destination] > 0:
                    support_total += 1
                    if template_total[phase, source, destination] > 0:
                        support_hit += 1
    support = float(support_hit) / float(max(support_total, 1))
    return (
        now, wave, out_phase[:wave], out_dst[:wave], out_size[:wave], quantum[:wave],
        duration[:wave], starts[:wave], ends[:wave], phase_done, rel1, rel2, valid,
        template_rows, template_waves, projected_edges, topology_tail_rows,
        topology_tail_waves, support,
    )


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
