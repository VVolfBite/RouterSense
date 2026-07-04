#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.offline.calibration import assert_online_native_ep_observation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate that calibrated offline analysis inputs come from online native EP observation.")
    parser.add_argument("--trace-metadata", type=str, required=True)
    args = parser.parse_args(argv)

    metadata_path = Path(args.trace_metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert_online_native_ep_observation(metadata)
    print(json.dumps({"status": "accepted", "trace_metadata": str(metadata_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
