#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from routesense.scheduler import (
    EXECUTION_WINDOW_MODE,
    RUNTIME_LOOKAHEAD_MODE,
    fast_schedule_barrier_aware_birkhoff,
    fast_schedule_birkhoff,
    fast_schedule_ibbr,
    fast_schedule_lagrangian,
    fast_schedule_u_barrier_criticality_global_matching,
    fast_schedule_u_barrier_price_adaptive_matching,
    fast_schedule_u_gated_greedy_maximal,
    fast_schedule_u_gated_maxweight_matching,
    pairwise_oracle,
)


def _synthetic_case(name: str) -> tuple[list[list[int]], list[list[int]], list[list[int]], int]:
    if name == "unlock_hotspot":
        dispatch = [
            [0, 8, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 9, 0],
        ]
        combine = [list(row) for row in zip(*dispatch)]
        next_dispatch = [
            [0, 0, 0, 0],
            [0, 0, 20, 0],
            [0, 0, 0, 1],
            [8, 0, 0, 0],
        ]
        return dispatch, combine, next_dispatch, 4
    if name == "cross_phase_shift":
        dispatch = [
            [0, 12, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 8, 0, 0, 0, 0],
            [0, 0, 0, 0, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0],
            [4, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 7, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 3, 0],
        ]
        combine = [list(row) for row in zip(*dispatch)]
        next_dispatch = [
            [0, 0, 0, 0, 10, 0, 0, 0],
            [0, 0, 0, 0, 0, 9, 0, 0],
            [0, 0, 0, 0, 0, 0, 9, 0],
            [0, 0, 0, 0, 0, 0, 0, 8],
            [8, 0, 0, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 0, 0, 0, 0],
            [0, 0, 7, 0, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0],
        ]
        return dispatch, combine, next_dispatch, 8
    raise ValueError(f"unknown case {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiphase global-matching study on a synthetic workload.")
    parser.add_argument("--case", choices=["unlock_hotspot", "cross_phase_shift"], default="unlock_hotspot")
    parser.add_argument("--mode", choices=[EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE], default=EXECUTION_WINDOW_MODE)
    parser.add_argument("--prediction-confidence", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    dispatch, combine, next_dispatch, num_gpus = _synthetic_case(args.case)
    baselines = {
        "B_birkhoff": fast_schedule_birkhoff,
        "B_barrier_aware_birkhoff": fast_schedule_barrier_aware_birkhoff,
        "O_lagrangian": fast_schedule_lagrangian,
        "O_ibbr": fast_schedule_ibbr,
    }
    candidates = {
        "O_gated_greedy_maximal": fast_schedule_u_gated_greedy_maximal,
        "O_gated_maxweight_matching": fast_schedule_u_gated_maxweight_matching,
        "O_barrier_criticality_global_matching": fast_schedule_u_barrier_criticality_global_matching,
        "O_barrier_price_adaptive_matching": fast_schedule_u_barrier_price_adaptive_matching,
    }

    results: dict[str, dict[str, object]] = {}
    for name, fn in {**baselines, **candidates}.items():
        payload = fn(
            dispatch,
            combine,
            next_dispatch,
            num_gpus,
            mode=args.mode,
            prediction_confidence=args.prediction_confidence,
        ) if name.startswith("O_gated") or "barrier_criticality" in name or "barrier_price" in name else fn(dispatch, combine, next_dispatch, num_gpus)
        results[name] = {
            "makespan": payload["makespan"],
            "solve_time_ms": payload["solve_time_ms"],
            "wave_count": payload.get("wave_count"),
            "audit_valid": payload.get("audit", {}).get("valid"),
            "mode": payload.get("mode", EXECUTION_WINDOW_MODE),
        }

    oracle = pairwise_oracle(dispatch, combine, next_dispatch, num_gpus)
    summary = {
        "case": args.case,
        "mode": args.mode,
        "prediction_confidence": args.prediction_confidence,
        "semantic_note": (
            "Existing D/oracle schedulers are atomic per-(phase,src,dst) chunks; "
            "new U schedulers are fluid wave schedulers over residual flows. "
            "Cross-family makespan is therefore diagnostic, not an apples-to-apples upper-bound comparison."
        ),
        "results": results,
        "oracle": {
            "makespan": oracle.get("makespan"),
            "solver_status": oracle.get("solver_status"),
        },
    }
    text = json.dumps(summary, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
