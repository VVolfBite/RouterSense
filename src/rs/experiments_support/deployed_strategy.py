"""Prepare one canonical comparison strategy for rank-local deployment.

This module contains only reusable configuration materialization logic.  The
actual executable remains under :mod:`experiments.online`, which may import
Megatron/runner entrypoints.  Keeping preparation here preserves the formal
``src``/``experiments`` dependency boundary and makes deployment contracts
unit-testable without importing executable experiment modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rs.experiments_support.strategy_comparison_runner import (
    dump_yaml,
    load_yaml,
    model_config,
    normalize_strategy_entry,
    single_strategy_config,
    strategy_run_kind,
    topology_config,
    uses_public_runtime_surface,
    validate_public_runtime_surface,
)


def select_strategy(comparison: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one normalized strategy entry from a comparison config."""

    strategies = [
        normalize_strategy_entry(item)
        for item in list(comparison.get("strategies", []) or [])
    ]
    for strategy in strategies:
        if str(strategy.get("name", "")) == name:
            return strategy
    available = ", ".join(str(item.get("name", "")) for item in strategies)
    raise ValueError(f"unknown strategy {name!r}; available: {available}")


def prepare_deployed_strategy(
    *,
    comparison_config: Path,
    strategy_name: str,
    output_dir: Path,
    model_path: str | None,
) -> dict[str, object]:
    """Materialize the single-run files consumed by a rank-local launcher.

    The returned paths are absolute or rooted in ``output_dir`` and are safe
    for every rank to compute deterministically.  Rank 0 is responsible for
    emitting the human-readable dry-run payload; all ranks consume the same
    generated strategy contract when launched under ``torchrun``.
    """

    comparison_config = comparison_config.resolve()
    output_dir = output_dir.resolve()
    comparison = load_yaml(comparison_config)
    if uses_public_runtime_surface(comparison):
        validate_public_runtime_surface(comparison)
    strategy = select_strategy(comparison, strategy_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_root = output_dir / "deployment_generated"
    generated_model = Path(
        model_config(dict(comparison.get("model", {}) or {}), generated_root)
    )
    generated_topology = Path(
        topology_config(dict(comparison.get("topology", {}) or {}), generated_root)
    )
    if model_path:
        import yaml

        model_payload = yaml.safe_load(generated_model.read_text(encoding="utf-8")) or {}
        model_payload["local_path"] = str(model_path)
        dump_yaml(generated_model, model_payload)

    generated_config = single_strategy_config(
        comparison=comparison,
        strategy=strategy,
        repetition=0,
        output_dir=output_dir,
        model_config_path=str(generated_model),
        topology_config_path=str(generated_topology),
    )
    run_kind = strategy_run_kind(comparison=comparison, strategy=strategy)
    strategy_dir = output_dir / "per_strategy" / strategy_name
    return {
        "comparison_config": str(comparison_config),
        "strategy": strategy_name,
        "run_kind": run_kind,
        "generated_config": str(generated_config),
        "generated_model_config": str(generated_model),
        "generated_topology_config": str(generated_topology),
        "strategy_output_dir": str(strategy_dir),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }
