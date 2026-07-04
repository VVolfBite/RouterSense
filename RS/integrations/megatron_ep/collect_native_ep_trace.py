#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.megatron_ep.routersense.trace_writer import write_json, write_jsonl
from integrations.megatron_ep.verify_env import main as verify_env_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--dispatcher", type=str, default="alltoall")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status = verify_env_main(["--model", args.model])
    if status != 0:
        write_json(
            output_dir / "summary.json",
            {
                "pipeline": "host_runtime_native_ep",
                "host_runtime": "megatron_core",
                "status": "blocked_environment",
                "reason": "verify_env_failed",
                "future_hint_mode": "none",
                "facade_mode": "not_started",
            },
        )
        write_jsonl(output_dir / "trace.jsonl", [])
        return status
    raise RuntimeError("Megatron native EP trace collection is not implemented until dependencies are installed.")


if __name__ == "__main__":
    raise SystemExit(main())
