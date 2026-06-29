from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .schemas import RunConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "poc1.yaml"


def load_config(path: str | None = None) -> RunConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        return RunConfig()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return RunConfig.from_dict(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RouteSense PoC1 CLI")
    parser.add_argument("command", choices=[
        "doctor",
        "prepare-data",
        "inspect-model",
        "collect-trace",
        "ablate",
        "analyze",
        "calibrate",
        "run-all",
    ])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace | None = None) -> RunConfig:
    config = load_config(getattr(args, "config", None) if args is not None else None)
    if args is None:
        return finalize_config(config)
    if args.run_id is not None:
        config.run_id = args.run_id
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    config.resume = bool(getattr(args, "resume", False))
    config.verbose = bool(getattr(args, "verbose", False))
    return finalize_config(config)


def finalize_config(config: RunConfig) -> RunConfig:
    if not config.run_id:
        config.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
