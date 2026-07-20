from __future__ import annotations

from pathlib import Path

from rs.core.experiment_config import load_run_config
from rs.core.formal_config_loader import load_formal_config
from rs.experiments_support.gpu_runner_common import load_official_config


ROOT = Path(__file__).resolve().parents[2]


def test_official_formal_entrypoint_configs_load() -> None:
    for relative_path, expected_runtime_line, entrypoint in (
        ("configs/official/offline_replay.yaml", "offline_replay", "experiments/run_offline_replay.py"),
        ("configs/official/online_phase_sync.yaml", "phase_sync", "experiments/run_online_phase_sync.py"),
        ("configs/official/online_async_release.yaml", "async_release", "experiments/run_online_async_release.py"),
    ):
        resolved = load_formal_config(
            config_path=ROOT / relative_path,
            expected_runtime_line=expected_runtime_line,
            official_entrypoint=entrypoint,
        )
        assert resolved.normalized_config["schema_version"] == 1
        assert resolved.consumed_config == resolved.normalized_config


def test_official_gpu_validation_configs_load() -> None:
    for relative_path in (
        "configs/official/gpu_c2_correctness.yaml",
        "configs/official/gpu_a2_performance.yaml",
        "configs/official/gpu_first_bringup.yaml",
        "configs/official/gpu_hotpath_iteration.yaml",
        "configs/official/gpu_runtime_attribution.yaml",
        "configs/official/gpu_runtime_diag.yaml",
        "configs/official/gpu_runtime_timeline.yaml",
        "configs/official/gpu_shadow_retire_check.yaml",
    ):
        config = load_official_config(ROOT / relative_path)
        assert int(config["schema_version"]) == 1
        assert str(config["run"]["kind"]) == "gpu_validation"
        assert isinstance(config.get("evaluation", {}), dict)
