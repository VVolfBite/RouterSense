from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rs.core.config_normalization import (
    CanonicalRunConfig,
    canonical_offline_replay_payload,
    canonical_online_comparison_payload,
    normalize_run_config,
)
from rs.experiments.output_schema import validate_official_entrypoint_config


@dataclass(frozen=True)
class ResolvedFormalConfig:
    source_path: Path
    normalized: CanonicalRunConfig
    normalized_config: dict[str, Any]
    consumed_config: dict[str, Any]
    official_entrypoint: str
    expected_runtime_line: str | None
    invariant_mode: str


def load_formal_config(
    *,
    config_path: str | Path,
    expected_runtime_line: str | None,
    official_entrypoint: str,
) -> ResolvedFormalConfig:
    resolved_path = Path(config_path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {resolved_path}")
    normalized = normalize_run_config(payload, source_path=resolved_path)
    if expected_runtime_line == "offline_replay":
        normalized_config = canonical_offline_replay_payload(normalized)
    else:
        normalized_config = canonical_online_comparison_payload(normalized)
    invariant_mode = validate_official_entrypoint_config(
        config_snapshot=normalized_config,
        expected_runtime_line=expected_runtime_line,
        official_entrypoint=official_entrypoint,
    )
    return ResolvedFormalConfig(
        source_path=resolved_path,
        normalized=normalized,
        normalized_config=normalized_config,
        consumed_config=normalized_config,
        official_entrypoint=str(official_entrypoint),
        expected_runtime_line=expected_runtime_line,
        invariant_mode=str(invariant_mode),
    )


__all__ = ["ResolvedFormalConfig", "load_formal_config"]
