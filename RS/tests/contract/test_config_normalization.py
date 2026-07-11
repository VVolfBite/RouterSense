from __future__ import annotations

from rs.core.config_normalization import (
    canonical_offline_replay_payload,
    canonical_online_comparison_payload,
    legacy_offline_replay_payload,
    legacy_online_comparison_payload,
    normalize_run_config,
)


def _without_migrations(payload: dict) -> dict:
    cloned = dict(payload)
    cloned["applied_migrations"] = []
    return cloned


def test_offline_replay_v0_and_v1_normalize_equivalently() -> None:
    v0 = {
        "fixture_dir": "tests/fixtures/offline_replay_smoke",
        "max_windows": 2,
        "bucket_rows": [512, 1024],
        "policies": ["birkhoff_phase_local", "greedy_ready_set"],
        "hints": ["zero_hint", "copy_current_dispatch"],
        "scheduling_mode": "execution_window",
        "expert_compute_delay": 0.0,
        "output_dir": "outputs/offline/offline_replay_smoke",
    }
    normalized_v0 = normalize_run_config(v0)
    v1 = canonical_offline_replay_payload(normalized_v0)
    normalized_v1 = normalize_run_config(v1)
    assert _without_migrations(normalized_v0.to_dict()) == _without_migrations(normalized_v1.to_dict())
    assert legacy_offline_replay_payload(normalized_v1)["bucket_rows"] == [512, 1024]


def test_online_comparison_v0_and_v1_normalize_equivalently() -> None:
    v0 = {
        "model": {"path": "/tmp/model"},
        "topology": {"ep_size": 2, "launcher": {"kind": "torchrun"}},
        "runtime": {"line": "phase_sync", "output_mode": "paper", "precision": "fp16", "dispatcher": "alltoall"},
        "workload": {"prompts": "configs/workload/smoke_prompts.json"},
        "strategies": [{"name": "disabled"}, {"name": "birkhoff_phase_local_async_p2p"}],
        "execution": {
            "repetitions": 1,
            "warmup": 0,
            "bucket_rows": 1024,
            "p0_weight": 1.0,
            "p1_reservation_weight": 1.0,
            "p2_hint_weight": 1.0,
            "schedule_layer_selector": "all",
            "schedule_phase_selector": "both",
        },
        "comparison": {"baseline_strategy": "disabled"},
    }
    normalized_v0 = normalize_run_config(v0)
    v1 = canonical_online_comparison_payload(normalized_v0)
    normalized_v1 = normalize_run_config(v1)
    assert _without_migrations(normalized_v0.to_dict()) == _without_migrations(normalized_v1.to_dict())
    assert legacy_online_comparison_payload(normalized_v1)["execution"]["bucket_rows"] == 1024


def test_unknown_or_conflicting_values_raise() -> None:
    bad = {
        "schema_version": 1,
        "run": {"kind": "offline_replay"},
        "model": {},
        "topology": {},
        "workload": {},
        "runtime": {},
        "traffic": {},
        "policy": {},
        "prediction": {},
        "evaluation": {},
        "replay": {},
        "oracle": {},
        "regime_analysis": {},
        "fixture_dir": "legacy/conflict",
    }
    try:
        normalize_run_config(bad)
    except ValueError as exc:
        assert "legacy fields" in str(exc) or "conflict" in str(exc) or "mapping" in str(exc)
    else:
        raise AssertionError("expected ValueError for mixed v1/legacy config")
