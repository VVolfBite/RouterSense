from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from .experiment import _run_one


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated RouterSense experiment worker")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    args.status.parent.mkdir(parents=True, exist_ok=True)
    try:
        row = _run_one(
            fixture_path=Path(spec["fixture_path"]),
            fixture_index=int(spec["fixture_index"]),
            treatment=dict(spec["treatment"]),
            treatment_index=int(spec["treatment_index"]),
            repeat_index=int(spec["repeat_index"]),
            warmup=bool(spec["warmup"]),
            config=dict(spec["config"]),
            run_dir=Path(spec["run_dir"]),
            dispose_runtime=False,
            trusted_fixture=(
                None if spec.get("trusted_fixture") is None
                else dict(spec["trusted_fixture"])
            ),
        )
        payload = {
            "schema_version": "RS_SIM_EXPERIMENT_WORKER_STATUS",
            "status": "PASS",
            "result_path": row["result_path"],
        }
        args.status.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except BaseException as exc:
        payload = {
            "schema_version": "RS_SIM_EXPERIMENT_WORKER_STATUS",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        args.status.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2


if __name__ == "__main__":
    # A few simulator dependencies may leave non-daemon helper threads alive
    # after all authoritative artifacts have been flushed.  This worker is a
    # deliberately isolated one-run process, so terminate the process boundary
    # explicitly instead of allowing a completed configuration to stall the
    # entire experiment matrix.
    code = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(int(code))
