"""Compiled P012/P0123 event-driven planning loops."""

from __future__ import annotations

import numpy as np
from numba import njit

from .event_math import _hungarian_max, _next_release, _refresh_release, _softmax, _sum_all, _wave_duration

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

__all__ = ["_event_plan", "_event_plan_p0123"]
