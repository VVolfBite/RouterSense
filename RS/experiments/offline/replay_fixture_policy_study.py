#!/usr/bin/env python3
"""Run offline scheduling policies on replay-derived fixture files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.offline.policy_study import (
    build_replay_problem,
    expected_replay_flows,
    run_replay_policy_study,
)

# Compatibility exports while tests and legacy runners migrate to src/rs.
_build_problem = build_replay_problem
_expected_flows = expected_replay_flows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--policies", required=True, help="Comma-separated policy list")
    parser.add_argument("--mode", choices=("runtime_lookahead", "execution_window"), default="runtime_lookahead")
    parser.add_argument("--p2-source", choices=("zero_hint", "copy_current_dispatch", "perfect_trace", "actual_trace"), default="copy_current_dispatch")
    parser.add_argument("--expert-compute-delay", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fixture_path = Path(args.fixture)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    policies = [item.strip() for item in str(args.policies).split(",") if item.strip()]
    payload = run_replay_policy_study(
        fixture=fixture,
        policy_names=policies,
        mode=str(args.mode),
        p2_source=str(args.p2_source),
        expert_compute_delay=float(args.expert_compute_delay),
    )
    payload["fixture"] = str(fixture_path)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
