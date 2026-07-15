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

import yaml

from rs.core.config_normalization import canonical_online_comparison_payload, legacy_online_comparison_payload, normalize_run_config
from rs.experiments.output_schema import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    initialize_run_artifacts,
    update_status,
    validate_official_entrypoint_config,
    write_resolved_configs,
)
from rs.runtime.guards.artifact import write_failure_artifact
from rs.runtime.guards.errors import RouterSenseInvariantError

from experiments.online.run_strategy_comparison import main as run_strategy_comparison_main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {path}")
    return payload


def main() -> None:
    args = _parse_args()
    layout = None
    try:
        config_path = Path(args.config)
        normalized = normalize_run_config(_load_yaml(config_path), source_path=config_path)
        canonical_payload = canonical_online_comparison_payload(normalized)
        validate_official_entrypoint_config(
            config_snapshot=canonical_payload,
            expected_runtime_line="phase_sync",
            official_entrypoint="experiments/run_online_phase_sync.py",
        )
        output_dir = (ROOT / str(args.output_dir)).resolve() if args.output_dir else (ROOT / "outputs/online/phase_sync" / canonical_payload["run"]["name"]).resolve()
        layout = initialize_run_artifacts(
            repo_root=ROOT,
            output_dir=output_dir,
            run_type="online_phase_sync",
            official_entrypoint="experiments/run_online_phase_sync.py",
            config_snapshot=canonical_payload,
        )
        payload = legacy_online_comparison_payload(normalized)
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
        tmp_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        argv = ["--config", str(tmp_config), "--output-dir", str(output_dir)]
        if args.dry_run:
            argv.append("--dry-run")
        rc = run_strategy_comparison_main(argv)
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
