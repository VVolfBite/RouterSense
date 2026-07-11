#!/usr/bin/env python3
"""Public online async-release entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

import yaml

from rs.core.config_normalization import canonical_online_comparison_payload, legacy_online_comparison_payload, normalize_run_config

from experiments.online.run_strategy_comparison import main as run_strategy_comparison_main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {path}")
    return payload


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    normalized = normalize_run_config(_load_yaml(config_path))
    payload = legacy_online_comparison_payload(normalized)
    payload.setdefault("runtime", {})
    payload["runtime"]["line"] = "async_release"
    payload["_normalized_public_bridge"] = True
    tmp_config = Path(args.output_dir) / "normalized_async_release_config.yaml"
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (Path(args.output_dir) / "canonical_config_snapshot.yaml").write_text(
        yaml.safe_dump(canonical_online_comparison_payload(normalized), sort_keys=False),
        encoding="utf-8",
    )
    argv = ["--config", str(tmp_config), "--output-dir", str(args.output_dir)]
    if args.dry_run:
        argv.append("--dry-run")
    run_strategy_comparison_main(argv)


if __name__ == "__main__":
    main()
