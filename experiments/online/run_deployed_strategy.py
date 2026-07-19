#!/usr/bin/env python3
"""Run one public comparison strategy under an already-started torchrun job.

This entrypoint is intentionally rank-local.  It converts one strategy from a
canonical comparison config into the existing single-run contract, then calls
the formal native or policy-correctness runner in the same process.  It avoids
nesting a second local ``torchrun`` inside a multi-node deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.online import collect_native_ep_trace, run_policy_correctness
from rs.experiments_support.deployed_strategy import prepare_deployed_strategy



def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-config", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="deployment")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = prepare_deployed_strategy(
        comparison_config=Path(args.comparison_config).resolve(),
        strategy_name=str(args.strategy),
        output_dir=Path(args.output_dir).resolve(),
        model_path=args.model_path,
    )
    payload["dry_run"] = bool(args.dry_run)
    payload["run_id"] = str(args.run_id)
    if int(payload["rank"]) == 0:
        print(json.dumps(payload, indent=2), flush=True)
    if args.dry_run:
        return 0
    runner_args = [
        "--config",
        str(payload["generated_config"]),
        "--run-id",
        str(args.run_id),
        "--output-dir",
        str(payload["strategy_output_dir"]),
    ]
    if payload["run_kind"] == "online_observe":
        return int(collect_native_ep_trace.main(runner_args))
    return int(run_policy_correctness.main(runner_args))


if __name__ == "__main__":
    raise SystemExit(main())
