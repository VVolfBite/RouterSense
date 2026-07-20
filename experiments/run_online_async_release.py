#!/usr/bin/env python3
"""Public online async-release entrypoint."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.formal_config_loader import load_formal_config
from rs.experiments_support.official_output import (
    initialize_official_output,
    update_official_status,
    write_official_configs,
)
from rs.experiments_support.online_evaluation_runner import run_online_evaluation
from rs.runtime.guards.artifact import write_failure_artifact
from rs.runtime.guards.errors import RouterSenseInvariantError

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()
def main() -> None:
    args = _parse_args()
    layout = None
    try:
        config_path = Path(args.config)
        resolved = load_formal_config(
            config_path=config_path,
            expected_runtime_line="async_release",
            official_entrypoint="experiments/run_online_async_release.py",
        )
        canonical_payload = resolved.normalized_config
        output_dir = (ROOT / str(args.output_dir)).resolve() if args.output_dir else (ROOT / "outputs/online/async_release" / canonical_payload["run"]["name"]).resolve()
        layout = initialize_official_output(
            repo_root=ROOT,
            output_dir=output_dir,
            run_type="online_async_release",
            official_entrypoint="experiments/run_online_async_release.py",
            config_snapshot=canonical_payload,
        )
        write_official_configs(
            layout,
            normalized_config=canonical_payload,
            consumed_config=canonical_payload,
        )
        rc = run_online_evaluation(
            config_path=config_path,
            output_dir=output_dir,
            dry_run=bool(args.dry_run),
        )
        update_official_status(
            layout,
            status="completed" if int(rc or 0) == 0 else "failed",
            extra={"completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        )
        raise SystemExit(int(rc or 0))
    except RouterSenseInvariantError as exc:
        if layout is not None:
            write_failure_artifact(layout.failures_dir / "startup_invariant_failure.json", error=exc)
            update_official_status(layout, status="failed", extra={"failure_codes": [exc.failure.error_code]})
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
