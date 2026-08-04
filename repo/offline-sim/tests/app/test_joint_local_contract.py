from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rs_sim.app.config_io import ConfigError, load_config
from rs_sim.app.experiment import (
    RunProcessError,
    _expand_treatments,
    _validate_fate_joint_identity,
    _validate_fate_joint_runtime_contract,
    normalize_experiment_config,
)


def test_default_matrix_uses_local_baselines_and_rscf_joint() -> None:
    root = Path(__file__).resolve().parents[2]
    normalized = normalize_experiment_config(load_config(root / "configs" / "experiment" / "formal" / "ep8_fate_comparison.yaml"))
    treatments = _expand_treatments(
        normalized["experiments"],
        default_release_mode=normalized["simulation"]["release_mode"],
    )
    by_name = {item["name"]: item for item in treatments}

    assert set(by_name) == {
        "FIFO-Local",
        "Greedy-Local",
        "BvN-Local",
        "RSCF-Local",
        "RSCF-Joint-FATE",
        "RSCF-Joint-Perfect",
        "Oracle-Local",
        "Oracle-Joint",
    }
    assert by_name["FIFO-Local"]["scope"] == "PHASE_LOCAL"
    assert by_name["Greedy-Local"]["scope"] == "PHASE_LOCAL"
    assert by_name["BvN-Local"]["scope"] == "PHASE_LOCAL"
    assert by_name["RSCF-Local"]["scope"] == "PHASE_LOCAL"
    assert by_name["RSCF-Local"]["release_mode"] == "PHASE_BARRIER"
    assert by_name["RSCF-Joint-FATE"]["scope"] == "WINDOW_JOINT"
    assert by_name["RSCF-Joint-FATE"]["release_mode"] == "RANK_LOCAL"


def test_external_matched_joint_requires_explicit_diagnostic_opt_in() -> None:
    row = {
        "treatments": [
            {
                "name": "Birkhoff-Joint",
                "algorithm": "joint(global_(birkhoff()))",
                "information": "fate",
                "overlap": "overlap",
            }
        ]
    }
    with pytest.raises(ConfigError, match="External baselines must remain Local"):
        _expand_treatments(row, default_release_mode="RANK_LOCAL")

    row["treatments"][0]["allow_matched_joint_diagnostic"] = True
    treatments = _expand_treatments(row, default_release_mode="RANK_LOCAL")
    assert treatments[0]["algorithm"] == "joint(global_(birkhoff()))"
    assert treatments[0]["scope"] == "WINDOW_JOINT"
    assert treatments[0]["allow_matched_joint_diagnostic"] is True
    assert treatments[0]["experiment_role"] == "EXPLICIT_DIAGNOSTIC_ABLATION"


def test_legacy_split_algorithm_fields_are_rejected() -> None:
    with pytest.raises(ConfigError, match="one algorithm expression"):
        _expand_treatments({
            "treatments": [{
                "name": "legacy",
                "core": "rscf",
                "scope": "joint",
                "planning": "global",
                "information": "fate",
            }]
        }, default_release_mode="RANK_LOCAL")


def _window(**overrides):
    values = {
        "anchor_layer_id": 0,
        "information_mode": "FATE_P2",
        "predicted_p2_slot_count": 4,
        "bound_exact_p2_task_count": 4,
        "prediction_generated": True,
        "prediction_nonempty": True,
        "prediction_validated": True,
        "prediction_consumed": True,
        "prediction_fallback": False,
        "prediction_fallback_reason": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fate_joint_contract_fails_closed_on_empty_template() -> None:
    treatment = {
        "name": "RSCF-Joint",
        "scope": "WINDOW_JOINT",
        "information": "FATE_P2",
    }
    with pytest.raises(RunProcessError, match="empty P2 template"):
        _validate_fate_joint_runtime_contract(
            treatment,
            (_window(predicted_p2_slot_count=0, prediction_nonempty=False),),
        )


def test_fate_joint_contract_accepts_consumed_nonempty_prediction() -> None:
    treatment = {
        "name": "RSCF-Joint",
        "scope": "WINDOW_JOINT",
        "information": "FATE_P2",
    }
    _validate_fate_joint_runtime_contract(treatment, (_window(),))


def test_fate_joint_identity_requires_same_digest() -> None:
    base = {
        "fixture_truth_digest": "fixture",
        "repeat_index": 0,
        "treatment": {
            "name": "RSCF-Joint",
            "scope": "WINDOW_JOINT",
            "information": "FATE_P2",
        },
        "per_window_metrics": [
            {"anchor_layer_id": 0, "prediction_digest": "same"}
        ],
    }
    matched = {
        **base,
        "treatment": {
            "name": "FIFO-Joint-Diagnostic",
            "scope": "WINDOW_JOINT",
            "information": "FATE_P2",
        },
    }
    _validate_fate_joint_identity((base, matched))

    mismatched = {
        **matched,
        "per_window_metrics": [
            {"anchor_layer_id": 0, "prediction_digest": "different"}
        ],
    }
    with pytest.raises(RunProcessError, match="different P2 advisory digests"):
        _validate_fate_joint_identity((base, mismatched))
