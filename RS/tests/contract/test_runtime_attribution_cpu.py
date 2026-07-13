from __future__ import annotations

from pathlib import Path

import yaml

from experiments.distributed._gpu_runner_common import build_policy_correctness_config
from rs.core.experiment_config import load_run_config
from rs.runtime.online.megatron_ep.observation.attribution import (
    ForwardCostTree,
    PhaseCostTree,
    SelectedLayerCostTree,
    aggregate_sync_callsite_cost,
    attribution_schema,
    legacy_outside_measured_hooks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phase_cost_tree_accepts_non_overlapping_components() -> None:
    row = PhaseCostTree(
        strategy="routersense_u_core_zero_raw_async",
        rank=3,
        forward_epoch=2,
        layer_id="1",
        phase="P0_dispatch",
        hook_total_us=10_000.0,
        components={
            "context_us": 100.0,
            "observation_us": 200.0,
            "local_matrix_us": 300.0,
            "global_matrix_us": 400.0,
            "prediction_us": 0.0,
            "plan_input_us": 500.0,
            "scheduler_solve_us": 1_000.0,
            "plan_postprocess_us": 700.0,
            "plan_store_us": 100.0,
            "agreement_us": 200.0,
            "materialization_us": 800.0,
            "preflight_us": 200.0,
            "pack_us": 300.0,
            "executor_wall_us": 1_000.0,
            "unpack_us": 300.0,
            "state_update_us": 200.0,
            "summary_us": 100.0,
        },
    ).to_dict()
    assert row["phase_tree_valid"] is True
    assert row["unattributed_us"] == 3_600.0
    assert row["explained_ratio"] == 0.64


def test_phase_cost_tree_rejects_negative_unattributed_beyond_tolerance() -> None:
    row = PhaseCostTree(
        strategy="x",
        rank=0,
        forward_epoch=0,
        layer_id="0",
        phase="P1_return",
        hook_total_us=1_000.0,
        components={"context_us": 900.0, "executor_wall_us": 300.0},
    ).to_dict()
    assert row["phase_tree_valid"] is False
    assert "exceeds" in row["validation_failure"]


def test_selected_layer_tree_uses_real_expert_boundaries() -> None:
    row = SelectedLayerCostTree(
        strategy="native",
        rank=1,
        forward_epoch=1,
        layer_id="0",
        selected_layer_total_us=25_000.0,
        p0_total_routerSense_us=0.0,
        p0_to_expert_us=2_000.0,
        expert_module_wall_us=18_000.0,
        expert_to_p1_us=1_000.0,
        p1_total_routerSense_us=0.0,
    ).to_dict()
    assert row["selected_layer_tree_valid"] is True
    assert row["selected_layer_unattributed_us"] == 4_000.0


def test_forward_tree_validates_top_level_partition() -> None:
    row = ForwardCostTree(
        strategy="routersense_b_core_independent_async",
        rank=0,
        forward_epoch=0,
        full_forward_wall_us=220_000.0,
        outside_selected_layers_us=130_000.0,
        selected_layer_0_total_us=35_000.0,
        inter_selected_layer_gap_us=20_000.0,
        selected_layer_1_total_us=35_000.0,
    ).to_dict()
    assert row["forward_tree_valid"] is True
    assert row["forward_unattributed_us"] == 0.0


def test_forward_tree_rejects_mismatched_partition() -> None:
    row = ForwardCostTree(
        strategy="x",
        rank=0,
        forward_epoch=0,
        full_forward_wall_us=100_000.0,
        outside_selected_layers_us=50_000.0,
        selected_layer_0_total_us=30_000.0,
        inter_selected_layer_gap_us=30_000.0,
        selected_layer_1_total_us=30_000.0,
    ).to_dict()
    assert row["forward_tree_valid"] is False


def test_legacy_expert_compute_formula_is_deprecated() -> None:
    row = legacy_outside_measured_hooks(
        full_forward_us=220_000.0,
        dispatch_hook_us=50_000.0,
        combine_hook_us=40_000.0,
    )
    assert row["outside_measured_hooks_us"] == 130_000.0
    assert row["measurement_status"] == "derived_legacy"
    assert row["deprecated"] is True


def test_sync_callsite_aggregation_preserves_measurement_induced_flag() -> None:
    rows = aggregate_sync_callsite_cost(
        [
            {"callsite_id": "WAIT_P2P_BATCH", "wall_us": 10.0, "execution_required": True},
            {"callsite_id": "WAIT_P2P_BATCH", "wall_us": 5.0, "execution_required": True},
            {"callsite_id": "SYNC_DEBUG", "wall_us": 7.0, "measurement_induced": True},
        ]
    )
    by_id = {row["callsite_id"]: row for row in rows}
    assert by_id["WAIT_P2P_BATCH"]["call_count"] == 2
    assert by_id["WAIT_P2P_BATCH"]["wall_us"] == 15.0
    assert by_id["SYNC_DEBUG"]["measurement_induced"] is True


def test_attribution_schema_marks_bad_legacy_fields() -> None:
    schema = attribution_schema()
    assert schema["profile"] == "attribution_light"
    assert schema["legacy_fields"]["outside_measured_hooks_us"]["deprecated"] is True
    assert schema["legacy_fields"]["selected_window_span_us"]["deprecated_as_communication"] is True


def test_gpu_runtime_attribution_config_is_fixed_small_matrix() -> None:
    payload = yaml.safe_load((REPO_ROOT / "configs/official/gpu_runtime_attribution.yaml").read_text())
    assert payload["world_size"] == 4
    assert payload["profile"] == "attribution_light"
    assert payload["preflight_mode"] == "compact"
    assert payload["evaluation"]["warmup"] == 1
    assert payload["evaluation"]["repeats"] == 2
    assert payload["workload"]["tokenization"]["expected_batch_rows"] == 8
    assert payload["workload"]["tokenization"]["expected_seq_len"] == 16
    assert payload["strategies"] == [
        "native",
        "routersense_b_core_independent_async",
        "routersense_u_core_zero_raw_async",
    ]
    assert payload["diagnostics"]["profiler"] is False
    assert payload["diagnostics"]["raw_task_jsonl"] is False
    assert payload["diagnostics"]["tensor_dump"] is False


def test_gpu_runtime_attribution_child_config_loads_profile(tmp_path: Path) -> None:
    base = yaml.safe_load((REPO_ROOT / "configs/official/gpu_runtime_attribution.yaml").read_text())
    base["model"] = {
        "model_id": "test-model",
        "local_path": "/tmp/test-model",
        "trust_remote_code": False,
    }
    child = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="routersense_u_core_zero_raw_async",
        run_name="attribution_child",
        output_root=tmp_path / "out",
        profile="attribution_light",
        selected_layers="0,1",
        save_logits=False,
        preflight_mode="compact",
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(yaml.safe_dump(child, sort_keys=False))
    config = load_run_config(config_path=child_path)
    assert config.observation.profile == "attribution_light"
    assert config.execution.preflight_mode == "compact"
    assert config.execution.schedule.selected_layer_ids == ("0", "1")
