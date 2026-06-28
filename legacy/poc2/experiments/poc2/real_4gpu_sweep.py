#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.eval import build_sweep_aggregate, export_artifact_bundle, save_goal
from routesense_poc2.runner import RunnerConfig, run_single_node_experiment


MODEL_PATHS = {
    "olmoe": "/root/autodl-tmp/models/OLMoE-1B-7B-0924",
    "qwen_moe": "/root/autodl-tmp/models/Qwen1.5-MoE-A2.7B",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the POC2 real 4-GPU sweep and aggregate first conclusions.")
    parser.add_argument("--output-root", type=str, default=str(ROOT / "outputs"))
    parser.add_argument("--artifact-dir", type=str, default=str(ROOT / "artifacts" / "poc2_latest"))
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["synthetic-matmul", "synthetic-token-linear"],
        choices=["synthetic-matmul", "synthetic-token-linear", "transfer-aware"],
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    artifact_dir = Path(args.artifact_dir)
    aggregate_dir = output_root / "poc2_sweep_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    save_goal(
        aggregate_dir / "goal.md",
        "# POC2 Sweep Aggregate\n\n"
        "This sweep evaluates real routing + synthetic service under the single-node 4-GPU prototype.\n"
        "It covers two models and two synthetic service backends, while each run evaluates four schedulers.\n",
    )

    combos = [(model_key, backend) for model_key in ("olmoe", "qwen_moe") for backend in args.backends]
    run_summaries: list[dict[str, object]] = []
    for model_key, service_backend in combos:
        suffix = "olmoe" if model_key == "olmoe" else "qwen"
        backend_suffix = service_backend.replace("synthetic-", "").replace("-", "_")
        output_dir = output_root / f"poc2_single_node_real_{suffix}_{backend_suffix}"
        config = RunnerConfig(
            model_key=model_key,
            model_path=MODEL_PATHS[model_key],
            output_dir=str(output_dir),
            microbatch_size=args.microbatch_size,
            microbatch_count=args.microbatch_count,
            max_length=args.max_length,
            precision=args.precision,
            world_size=args.world_size,
            seed=args.seed,
            service_backend=service_backend,
        )
        payload = run_single_node_experiment(config)
        run_summaries.append(
            {
                "model_key": model_key,
                "service_backend": service_backend,
                "routing_mode": payload["routing_mode"],
                "service_mode": payload["service_mode"],
                "service_is_synthetic": payload["service_is_synthetic"],
                "backend_realism_level": payload["backend_realism_level"],
                "output_dir": str(output_dir),
                "summary": payload["summary"],
            }
        )

    aggregate = build_sweep_aggregate(run_summaries)
    (aggregate_dir / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    export_artifact_bundle(
        artifact_dir,
        summary=aggregate,
        config={
            "models": ["olmoe", "qwen_moe"],
            "backends": list(args.backends),
            "schedulers": ["fifo", "state-only", "dependency-only", "full"],
            "microbatch_size": args.microbatch_size,
            "microbatch_count": args.microbatch_count,
            "max_length": args.max_length,
            "precision": args.precision,
            "world_size": args.world_size,
            "seed": args.seed,
            "layer": "auto",
        },
        environment=_collect_environment(args.world_size),
        goal=(
            "# POC2 Latest Artifacts\n\n"
            "This bundle captures the latest single-node 4-GPU sweep under real routing and synthetic or transfer-aware service backends.\n"
            "Use the aggregate summary as an early scheduler signal, not as a final runtime latency conclusion.\n"
            "The current prototype is still below a full EP runtime, but the transfer-aware backend is one step closer to real execution.\n"
        ),
    )
    print(aggregate_dir / "summary.json")
    return 0


def _collect_environment(world_size: int) -> dict[str, object]:
    import socket

    import torch  # type: ignore

    gpu_count = torch.cuda.device_count()
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "visible_gpu_count": gpu_count,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(min(gpu_count, world_size))],
        "hostname": socket.gethostname(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
