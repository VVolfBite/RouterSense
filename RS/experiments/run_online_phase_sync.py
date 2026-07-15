#!/usr/bin/env python3
"""Public online phase-sync entrypoint."""

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
from rs.experiments_support.strategy_comparison_runner import dump_yaml, run_strategy_comparison
from rs.experiments.output_schema import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    initialize_run_artifacts,
    update_status,
    write_resolved_configs,
)
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
            expected_runtime_line="phase_sync",
            official_entrypoint="experiments/run_online_phase_sync.py",
        )
        canonical_payload = resolved.normalized_config
        output_dir = (ROOT / str(args.output_dir)).resolve() if args.output_dir else (ROOT / "outputs/online/phase_sync" / canonical_payload["run"]["name"]).resolve()
        layout = initialize_run_artifacts(
            repo_root=ROOT,
            output_dir=output_dir,
            run_type="online_phase_sync",
            official_entrypoint="experiments/run_online_phase_sync.py",
            config_snapshot=canonical_payload,
        )
        payload = dict(resolved.legacy_bridge_config or {})
        payload.setdefault("runtime", {})
        payload["runtime"]["line"] = "phase_sync"
        payload["_normalized_public_bridge"] = True
        write_resolved_configs(
            layout,
            normalized_config=canonical_payload,
            consumed_config=canonical_payload,
            legacy_bridge_config=payload,
        )
        tmp_config = output_dir / "normalized_phase_sync_config.yaml"
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        dump_yaml(tmp_config, payload)
        rc = run_strategy_comparison(
            config_path=tmp_config,
            output_dir=output_dir,
            dry_run=bool(args.dry_run),
        )
        update_status(
            layout,
            status=RUN_STATUS_COMPLETED if int(rc or 0) == 0 else RUN_STATUS_FAILED,
            extra={"completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        )
        raise SystemExit(int(rc or 0))
    except RouterSenseInvariantError as exc:
        if layout is not None:
            write_failure_artifact(layout.failures_dir / "startup_invariant_failure.json", error=exc)
            update_status(layout, status=RUN_STATUS_FAILED, extra={"failure_codes": [exc.failure.error_code]})
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
