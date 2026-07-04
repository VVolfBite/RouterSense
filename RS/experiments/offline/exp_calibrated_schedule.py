#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.offline.calibration import assert_online_native_ep_observation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase-1 calibrated offline schedule entrypoint placeholder."
    )
    parser.add_argument("--trace-metadata", type=str, required=True)
    parser.add_argument("--scheduler", type=str, default="phase_local")
    args = parser.parse_args(argv)
    import json
    from pathlib import Path

    metadata = json.loads(Path(args.trace_metadata).read_text(encoding="utf-8"))
    assert_online_native_ep_observation(metadata)
    raise NotImplementedError(
        "calibrated offline schedule simulation is not implemented yet; "
        "Phase 1 only enforces input provenance and boundary semantics"
    )


if __name__ == "__main__":
    raise SystemExit(main())
