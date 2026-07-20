from __future__ import annotations

from rs.core.config_normalization import (
    canonical_offline_replay_payload,
    canonical_online_comparison_payload,
    legacy_offline_replay_payload,
    legacy_online_comparison_payload,
    normalize_run_config,
)
from pathlib import Path


def _without_migrations(payload: dict) -> dict:
    cloned = dict(payload)
    cloned["applied_migrations"] = []
    return cloned


def test_offline_replay_v0_and_v1_normalize_equivalently() -> None:
    v0 = {
        "fixture_dir": "tests/fixtures/offline_replay_smoke",
        "max_windows": 2,
        "bucket_rows": [512, 1024],
        "policies": ["birkhoff_bucket_phase_local", "greedy_bucket"],
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


def test_normalize_rejects_string_schema_version() -> None:
    try:
        normalize_run_config({"schema_version": "1"})
    except ValueError as exc:
        assert "schema_version must be an integer" in str(exc)
    else:
        raise AssertionError("expected strict schema_version rejection")


def test_normalize_rejects_bool_bucket_rows() -> None:
    bad = {
        "fixture_dir": "tests/fixtures/offline_replay_smoke",
        "bucket_rows": [True],
    }
    try:
        normalize_run_config(bad)
    except ValueError as exc:
        assert "bucket_rows[0] must be an integer" in str(exc)
    else:
        raise AssertionError("expected strict bucket_rows rejection")


def test_v1_component_references_are_resolved_recursively(tmp_path: Path) -> None:
    topology_component = tmp_path / "topology.yaml"
    topology_component.write_text(
        "\n".join(
            (
                "launcher:",
                "  kind: torchrun",
                "  nnodes: 1",
                "  nproc_per_node: 4",
                "ep_size: 4",
            )
        ),
        encoding="utf-8",
    )
    model_component = tmp_path / "model.yaml"
    model_component.write_text(
        "\n".join(
            (
                "model_id: fixture_model",
                "local_path: /models/fixture",
                "precision: fp16",
            )
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "official.yaml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "run:",
                "  kind: online_strategy_comparison",
                "  name: test",
                "model:",
                f"  path: {model_component.name}",
                "topology:",
                f"  component: {topology_component.name}",
                "workload: {}",
                "runtime:",
                "  line: phase_sync",
                "traffic:",
                "  matrix_unit: rows",
                "  bucket_rows: 1024",
                "policy: {}",
                "prediction: {}",
                "evaluation: {}",
                "replay: {}",
                "oracle: {}",
                "regime_analysis: {}",
            )
        ),
        encoding="utf-8",
    )
    normalized = normalize_run_config(
        {
            "schema_version": 1,
            "run": {"kind": "online_strategy_comparison", "name": "test"},
            "model": {"path": model_component.name},
            "topology": {"component": topology_component.name},
            "workload": {},
            "runtime": {"line": "phase_sync"},
            "traffic": {"matrix_unit": "rows", "bucket_rows": 1024},
            "policy": {},
            "prediction": {},
            "evaluation": {},
            "replay": {},
            "oracle": {},
            "regime_analysis": {},
        },
        source_path=config_path,
    )
    assert normalized.model["model_id"] == "fixture_model"
    assert normalized.model["local_path"] == "/models/fixture"
    assert normalized.topology["ep_size"] == 4
    assert normalized.topology["launcher"]["nproc_per_node"] == 4


def test_model_component_can_recurse_through_local_path_yaml(tmp_path: Path) -> None:
    concrete_model = tmp_path / "concrete-model.yaml"
    concrete_model.write_text(
        "\n".join(
            (
                "model_id: nested_fixture_model",
                "local_path: /models/nested-fixture",
                "trust_remote_code: false",
            )
        ),
        encoding="utf-8",
    )
    model_component = tmp_path / "model-component.yaml"
    model_component.write_text(
        "\n".join(
            (
                "model_id: model_component",
                f"local_path: {concrete_model.name}",
                "trust_remote_code: false",
            )
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "official.yaml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "run: {kind: online_strategy_comparison, name: nested}",
                f"model: {{path: {model_component.name}}}",
                "topology: {}",
                "workload: {}",
                "runtime: {line: phase_sync}",
                "traffic: {matrix_unit: rows, bucket_rows: 1024}",
                "policy: {}",
                "prediction: {}",
                "evaluation: {}",
                "replay: {}",
                "oracle: {}",
                "regime_analysis: {}",
            )
        ),
        encoding="utf-8",
    )
    normalized = normalize_run_config(yaml_like(config_path), source_path=config_path)
    assert normalized.model["model_id"] == "nested_fixture_model"
    assert normalized.model["local_path"] == "/models/nested-fixture"


def test_component_reference_cycle_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("component: b.yaml\n", encoding="utf-8")
    b.write_text("component: a.yaml\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "run: {kind: online_strategy_comparison, name: cyc}",
                "model: {path: a.yaml}",
                "topology: {}",
                "workload: {}",
                "runtime: {line: phase_sync}",
                "traffic: {matrix_unit: rows, bucket_rows: 1024}",
                "policy: {}",
                "prediction: {}",
                "evaluation: {}",
                "replay: {}",
                "oracle: {}",
                "regime_analysis: {}",
            )
        ),
        encoding="utf-8",
    )
    try:
        normalize_run_config(yaml_like(config_path), source_path=config_path)
    except ValueError as exc:
        assert "cyclic config component reference" in str(exc)
    else:
        raise AssertionError("expected cyclic component reference failure")


def yaml_like(path: Path) -> dict:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
