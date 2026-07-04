#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.megatron_ep.routersense.contracts import NativeEPSummary
from integrations.megatron_ep.routersense.trace_writer import write_json
from integrations.megatron_ep.verify_env import main as verify_env_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--dispatcher", type=str, default="alltoall")
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status = verify_env_main(["--model", args.model])
    if status != 0:
        payload = NativeEPSummary(
            ep_size=int(args.ep_size),
            dispatcher=str(args.dispatcher),
            status="blocked_environment",
            reason="verify_env_failed",
            details={
                "model": args.model,
                "precision": args.precision,
                "prompt_file": args.prompt_file,
                "note": "Megatron Core / Bridge native smoke not executed because environment is blocked.",
            },
        ).to_dict()
        write_json(output_dir / "summary.json", payload)
        return status
    raise RuntimeError("Megatron native EP smoke execution is not implemented until dependencies are installed.")


if __name__ == "__main__":
    raise SystemExit(main())
