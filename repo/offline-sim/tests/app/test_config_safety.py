from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rs_sim.app.config_io import ConfigError, load_config
from rs_sim.app.experiment import normalize_experiment_config
from rs_sim.runtime import (
    load_runtime_profile_bundle_json,
    write_runtime_profile_bundle_json,
)


ROOT = Path(__file__).resolve().parents[2]


def _formal_config():
    return load_config(ROOT / "configs" / "experiment" / "formal" / "ep8_fate_comparison.yaml")


def test_unknown_simulation_field_fails_closed() -> None:
    config = copy.deepcopy(_formal_config())
    config["simulation"]["stagging"] = "0.25X"
    with pytest.raises(ConfigError, match="stagging"):
        normalize_experiment_config(config)


def test_string_boolean_fails_closed() -> None:
    config = copy.deepcopy(_formal_config())
    config["simulation"]["p0_p1_compute_end_barrier"] = "false"
    with pytest.raises(ConfigError, match="boolean"):
        normalize_experiment_config(config)


def test_runtime_profile_round_trip_and_digest_gate(tmp_path: Path) -> None:
    config = normalize_experiment_config(_formal_config())
    bundle = load_runtime_profile_bundle_json(config["simulation"]["runtime_profile"])
    target = tmp_path / "profile.json"
    write_runtime_profile_bundle_json(target, bundle)
    assert load_runtime_profile_bundle_json(target) == bundle
    text = target.read_text(encoding="utf-8").replace(bundle.profile_digest, "0" * 64)
    target.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_runtime_profile_bundle_json(target)


def test_omitted_execution_semantics_use_paper_defaults() -> None:
    config = copy.deepcopy(_formal_config())
    config["simulation"].pop("release_mode", None)
    config["simulation"].pop("p0_p1_compute_end_barrier", None)
    normalized = normalize_experiment_config(config)
    assert normalized["simulation"]["release_mode"] == "RANK_LOCAL"
    assert normalized["simulation"]["p0_p1_compute_end_barrier"] is True
    assert normalized["simulation"]["max_task_bytes"] == 262144
    assert normalized["simulation"]["alignment_bytes"] == 256


def test_paper_claim_rejects_rank_local_local_without_release_only_role(tmp_path: Path) -> None:
    from rs_sim.app.experiment import run_experiment

    config = copy.deepcopy(_formal_config())
    config["comparison"]["claim_mode"] = "PAPER"
    local = next(
        item for item in config["experiments"]["treatments"]
        if item["name"] == "FIFO-Local"
    )
    local["release_mode"] = "RANK_LOCAL"
    config["output"]["directory"] = str(tmp_path / "out")
    with pytest.raises(ConfigError, match="Local treatments require release_mode=PHASE_BARRIER"):
        run_experiment(config)
