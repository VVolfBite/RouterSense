#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.runner import SUPPORTED_MODELS, RunnerConfig, run_single_node_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the POC2 single-node 4-rank scheduler prototype.")
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default="olmoe")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "poc2_single_node_eval"))
    parser.add_argument("--prompts-path", type=str, default=None)
    parser.add_argument("--layer", type=str, default="auto")
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--microbatch-count", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock-routing", action="store_true")
    parser.add_argument(
        "--service-backend",
        choices=["mock-sleep", "synthetic-matmul", "synthetic-token-linear"],
        default="synthetic-matmul",
    )
    parser.add_argument("--service-scale", type=int, default=64)
    parser.add_argument("--per-token-compute-us", type=int, default=600)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fifo", "state-only", "dependency-only", "full"],
        choices=["fifo", "state-only", "dependency-only", "full"],
    )
    args = parser.parse_args(argv)

    config = RunnerConfig(
        model_key=args.model,
        model_path=args.model_path,
        output_dir=args.output_dir,
        prompts_path=args.prompts_path,
        layer=args.layer,
        microbatch_size=args.microbatch_size,
        microbatch_count=args.microbatch_count,
        max_length=args.max_length,
        precision=args.precision,
        world_size=args.world_size,
        top_k=args.top_k,
        seed=args.seed,
        mock_routing=bool(args.mock_routing),
        service_backend=args.service_backend,
        service_scale=args.service_scale,
        per_token_compute_us=args.per_token_compute_us,
        strategies=list(args.strategies),
    )
    run_single_node_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
