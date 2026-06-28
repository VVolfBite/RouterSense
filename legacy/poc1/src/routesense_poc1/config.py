from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from .schemas import RunConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "poc1.yaml"


def load_config(path: str | None = None) -> RunConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        config = RunConfig()
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        config = RunConfig(**raw)
    env_output_dir = os.environ.get("ROUTESENSE_OUTPUT_DIR")
    if env_output_dir:
        config.output_dir = env_output_dir
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RouteSense PoC1 CLI")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace | None = None) -> RunConfig:
    config = load_config(getattr(args, "config", None))
    if args is None:
        return config
    if getattr(args, "text", None) is not None:
        config.text = args.text
    if getattr(args, "layer", None) is not None:
        config.layer = args.layer
    if getattr(args, "rank", None) is not None:
        config.rank = args.rank
    if getattr(args, "output_dir", None) is not None:
        config.output_dir = args.output_dir
    elif os.environ.get("ROUTESENSE_OUTPUT_DIR"):
        config.output_dir = os.environ.get("ROUTESENSE_OUTPUT_DIR", config.output_dir)
    if getattr(args, "model_id", None) is not None:
        config.model_id = args.model_id
    config.mock = bool(getattr(args, "mock", False))
    return config


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
