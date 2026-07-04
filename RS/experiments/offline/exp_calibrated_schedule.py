#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.offline.calibration import assert_online_native_ep_observation
from rs.offline.reporting import build_calibrated_counterfactual_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase-1 calibrated offline schedule entrypoint placeholder."
    )
    parser.add_argument("--trace-metadata", type=str, required=True)
    parser.add_argument("--scheduler", type=str, default="phase_local")
    parser.add_argument("--output-dir", type=str, default="artifacts/offline/calibrated_schedule")
    args = parser.parse_args(argv)

    metadata = json.loads(Path(args.trace_metadata).read_text(encoding="utf-8"))
    assert_online_native_ep_observation(metadata)
    run_id = f"offline-calibrated-{uuid.uuid4().hex[:12]}"
    payload = build_calibrated_counterfactual_result(
        run_id=run_id,
        future_information_mode=metadata.get("future_information_mode", "none"),
        extra={
            "implemented": False,
            "scheduler": args.scheduler,
            "trace_metadata": args.trace_metadata,
            "implemented_reason": "calibrated offline simulator is not implemented yet",
        },
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
