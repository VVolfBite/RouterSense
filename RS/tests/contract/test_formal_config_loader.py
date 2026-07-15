from __future__ import annotations

from pathlib import Path

import pytest

from rs.core.formal_config_loader import load_formal_config


ROOT = Path(__file__).resolve().parents[2]


def test_load_formal_offline_config_returns_canonical_and_legacy_payloads() -> None:
    resolved = load_formal_config(
        config_path=ROOT / "configs/official/offline_replay.yaml",
        expected_runtime_line="offline_replay",
        official_entrypoint="experiments/run_offline_replay.py",
    )
    assert resolved.normalized_config["schema_version"] == 1
    assert resolved.consumed_config == resolved.normalized_config
    assert resolved.legacy_bridge_config is not None
    assert "fixture_dir" in resolved.legacy_bridge_config
    assert resolved.invariant_mode


def test_load_formal_online_config_returns_canonical_and_legacy_payloads() -> None:
    resolved = load_formal_config(
        config_path=ROOT / "configs/official/online_phase_sync.yaml",
        expected_runtime_line="phase_sync",
        official_entrypoint="experiments/run_online_phase_sync.py",
    )
    assert resolved.normalized_config["schema_version"] == 1
    assert resolved.consumed_config == resolved.normalized_config
    assert resolved.legacy_bridge_config is not None
    assert "execution" in resolved.legacy_bridge_config
    assert resolved.normalized_config["runtime"]["line"] == "phase_sync"


def test_load_formal_config_rejects_wrong_runtime_line() -> None:
    with pytest.raises(Exception):
        load_formal_config(
            config_path=ROOT / "configs/official/online_phase_sync.yaml",
            expected_runtime_line="async_release",
            official_entrypoint="experiments/run_online_async_release.py",
        )
