from __future__ import annotations

import json
from pathlib import Path

from experiments.distributed._gpu_runner_common import build_policy_correctness_config, load_official_config
from experiments.distributed.run_gpu_a2_strategy_compare import _build_strategy_result, _metric_series, aggregate_hotpath_rank_counts
from rs.core.layer_ids import stable_layer_ids
from rs.core.layer_selection import resolve_layer_selector


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_official_gpu_c2_matrix_includes_four_joint_async_candidates() -> None:
    payload = load_official_config(REPO_ROOT / "configs/official/gpu_c2_correctness.yaml")
    assert payload["candidate_strategies"] == [
        "birkhoff_phase_local_async_p2p",
        "routersense_b_core_independent_async",
        "routersense_u_core_zero_raw_async",
        "routersense_u_core_predicted_raw_async",
        "routersense_u_core_predicted_safe_async",
    ]


def test_a2_metric_series_uses_repeat_records_without_transport_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "repeat_records": [
            {
                "forward_epoch": 0,
                "warmup": False,
                "global_max_forward_us": 100.0,
                "dispatch_transport_us": 10.0,
                "return_transport_us": 12.0,
                "batch_submit_us": 3.0,
                "wait_us": 4.0,
                "prediction_us": 5.0,
                "raw_u_build_us": 6.0,
                "paired_b_build_us": 7.0,
                "host_projection_us": 8.0,
                "safe_selection_us": 9.0,
                "plan_agreement_us": 1.0,
                "local_materialization_us": 2.0,
                "preflight_us": 3.0,
            }
        ]
    }
    metrics = _metric_series(summary, run_dir)
    assert metrics["total_forward_us"] == [100.0]
    assert metrics["dispatch_transport_us"] == [10.0]
    assert metrics["return_transport_us"] == [12.0]
    assert metrics["p2p_enqueue_us"] == [3.0]
    assert metrics["p2p_wait_us"] == [4.0]


def test_synthetic_hotpath_aggregate_uses_all_ranks() -> None:
    ranks = [
        {
            "rank": rank,
            "selected_p0_hook_count": 4,
            "selected_p1_hook_count": 4,
            "prediction_source_p0_hook_count": 0,
            "none_heavy_hook_count": 0,
            "real_p0_execution_count": 4,
            "real_p1_execution_count": 4,
            "shadow_dispatch_execution_count": 0,
            "shadow_combine_execution_count": 0,
            "observation_finalize_dispatch_count": 4,
            "observation_finalize_combine_count": 4,
            "shadow_policy_agreement_count": 0,
            "shadow_plan_build_count": 0,
            "shadow_control_collective_count": 0,
            "raw_u_build_count": 4,
            "paired_b_build_count": 0,
            "predict_count": 0,
        }
        for rank in range(4)
    ]
    aggregate = aggregate_hotpath_rank_counts(
        ranks,
        expected_world_size=4,
        warmup_iters=1,
        measure_iters=1,
        selected_layer_ids=["0", "1"],
        prediction_source_layer_ids=[],
        strategy="routersense_u_core_zero_raw_async",
    )
    assert aggregate["selected_p0_hook_count_per_rank"] == [4, 4, 4, 4]
    assert aggregate["selected_p0_hook_count_all_rank_sum"] == 16
    assert aggregate["selected_p1_hook_count_per_rank"] == [4, 4, 4, 4]
    assert aggregate["selected_p1_hook_count_all_rank_sum"] == 16
    assert aggregate["expected_selected_p0_hook_count_per_rank"] == 4
    assert aggregate["selected_p0_hook_count_exact"] is True
    assert aggregate["expected_prediction_source_p0_hook_count_per_rank"] == 0
    assert aggregate["prediction_source_p0_hook_count_all_rank_exact"] is True
    assert aggregate["none_heavy_hook_count_all_rank_sum"] == 0
    assert aggregate["raw_u_build_count_all_rank_sum"] == 16
    assert aggregate["expected_raw_u_build_upper_bound_all_rank"] == 16
    assert aggregate["raw_u_build_count_all_rank_valid"] is True
    assert aggregate["raw_u_build_count_by_layer_per_rank_valid"] is True
    assert aggregate["rank_count_observed"] == 4
    assert aggregate["hotpath_eligible"] is True


def test_synthetic_hotpath_exact_count_mismatch_is_ineligible() -> None:
    ranks = [
        {
            "rank": rank,
            "selected_p0_hook_count": 3 if rank == 1 else 4,
            "selected_p1_hook_count": 4,
            "prediction_source_p0_hook_count": 0,
            "none_heavy_hook_count": 0,
            "real_p0_execution_count": 4,
            "real_p1_execution_count": 4,
            "shadow_dispatch_execution_count": 0,
            "shadow_combine_execution_count": 0,
            "observation_finalize_dispatch_count": 4,
            "observation_finalize_combine_count": 4,
            "shadow_policy_agreement_count": 0,
            "shadow_plan_build_count": 0,
            "shadow_control_collective_count": 0,
            "raw_u_build_count": 4,
            "raw_u_build_count_by_layer_per_rank": {"0": 2, "1": 2},
            "paired_b_build_count": 0,
            "predict_count": 0,
        }
        for rank in range(4)
    ]
    aggregate = aggregate_hotpath_rank_counts(
        ranks,
        expected_world_size=4,
        warmup_iters=1,
        measure_iters=1,
        selected_layer_ids=["0", "1"],
        prediction_source_layer_ids=[],
        strategy="routersense_u_core_zero_raw_async",
    )
    assert aggregate["hotpath_eligible"] is False
    assert any("selected_p0_hook_count:rank=1:expected=4:actual=3" in item for item in aggregate["eligibility_reasons"])


def test_synthetic_hotpath_raw_u_layer_upper_bound_is_ineligible() -> None:
    ranks = [
        {
            "rank": rank,
            "selected_p0_hook_count": 4,
            "selected_p1_hook_count": 4,
            "prediction_source_p0_hook_count": 0,
            "none_heavy_hook_count": 0,
            "real_p0_execution_count": 4,
            "real_p1_execution_count": 4,
            "shadow_dispatch_execution_count": 0,
            "shadow_combine_execution_count": 0,
            "observation_finalize_dispatch_count": 4,
            "observation_finalize_combine_count": 4,
            "shadow_policy_agreement_count": 0,
            "shadow_plan_build_count": 0,
            "shadow_control_collective_count": 0,
            "raw_u_build_count": 5 if rank == 0 else 4,
            "raw_u_build_count_by_layer_per_rank": {"0": 3, "1": 2} if rank == 0 else {"0": 2, "1": 2},
            "paired_b_build_count": 0,
            "predict_count": 0,
        }
        for rank in range(4)
    ]
    aggregate = aggregate_hotpath_rank_counts(
        ranks,
        expected_world_size=4,
        warmup_iters=1,
        measure_iters=1,
        selected_layer_ids=["0", "1"],
        prediction_source_layer_ids=[],
        strategy="routersense_u_core_zero_raw_async",
    )
    assert aggregate["hotpath_eligible"] is False
    assert aggregate["raw_u_build_count_per_rank_valid"] is False
    assert aggregate["raw_u_build_count_by_layer_per_rank_valid"] is False
    assert any("raw_u_build_count_by_layer:rank=0:layer=0:upper=2:actual=3" in item for item in aggregate["eligibility_reasons"])


def test_synthetic_hotpath_aggregate_rejects_missing_duplicate_and_none_heavy() -> None:
    missing = aggregate_hotpath_rank_counts([{"rank": 0, "selected_p0_hook_count": 4}], expected_world_size=4)
    assert missing["hotpath_eligible"] is False
    assert any("rank_count_mismatch" in item for item in missing["eligibility_reasons"])
    duplicate = aggregate_hotpath_rank_counts(
        [
            {"rank": 0, "selected_p0_hook_count": 4, "none_heavy_hook_count": 0},
            {"rank": 0, "selected_p0_hook_count": 4, "none_heavy_hook_count": 0},
        ],
        expected_world_size=2,
    )
    assert duplicate["hotpath_eligible"] is False
    assert any("duplicate_rank" in item for item in duplicate["eligibility_reasons"])
    none_heavy = aggregate_hotpath_rank_counts(
        [
            {
                "rank": rank,
                "selected_p0_hook_count": 4,
                "selected_p1_hook_count": 4,
                "prediction_source_p0_hook_count": 0,
                "none_heavy_hook_count": 1 if rank == 2 else 0,
                "real_p0_execution_count": 4,
                "real_p1_execution_count": 4,
                "shadow_dispatch_execution_count": 0,
                "shadow_combine_execution_count": 0,
                "observation_finalize_dispatch_count": 4,
                "observation_finalize_combine_count": 4,
                "shadow_policy_agreement_count": 0,
                "shadow_plan_build_count": 0,
                "shadow_control_collective_count": 0,
                "raw_u_build_count": 4,
                "paired_b_build_count": 0,
                "predict_count": 0,
            }
            for rank in range(4)
        ],
        expected_world_size=4,
    )
    assert none_heavy["hotpath_eligible"] is False
    assert "none_heavy_hook_count_positive" in none_heavy["eligibility_reasons"]


def test_stable_layer_id_sorting_numeric_and_mixed_names() -> None:
    assert stable_layer_ids({"10", "2", "1"}) == ["1", "2", "10"]
    assert stable_layer_ids({"layer10", "layer2", "alpha"}) == ["alpha", "layer2", "layer10"]


def test_raw_and_safe_child_configs_diverge_in_safe_projection_mode() -> None:
    base = {
        "model": {"model_id": "fixture/model", "local_path": "/tmp/model", "trust_remote_code": False},
        "topology": {"world_size": 4, "ep_size": 4},
        "runtime": {"line": "async_release", "invariant_mode": "evaluation_strict", "precision": "bf16", "dispatcher": "alltoall"},
        "traffic": {"bucket_mode": "dynamic_current", "bucket_rows": 0},
        "policy": {"options": {"p0_weight": 1.0, "p1_reservation_weight": 1.0, "p2_hint_weight": 1.0, "residual_weight": 0.75, "barrier_weight": 1.75, "age_weight": 0.15, "prediction_weight": 0.35}},
        "workload": {"prompts": "configs/workload/smoke_prompts.json"},
        "execution": {"schedule_phase_selector": "both"},
        "evaluation": {"selected_layer_ids": [0, 1]},
    }
    raw_cfg = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="routersense_u_core_predicted_raw_async",
        run_name="raw",
        output_root=Path("/tmp/raw"),
        profile="execution",
        selected_layers="selected",
        save_logits=False,
    )
    safe_cfg = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="routersense_u_core_predicted_safe_async",
        run_name="safe",
        output_root=Path("/tmp/safe"),
        profile="execution",
        selected_layers="selected",
        save_logits=False,
    )
    assert raw_cfg["execution"]["safe_projection_mode"] == "disabled"
    assert safe_cfg["execution"]["safe_projection_mode"] == "host_select"
    assert json.dumps(raw_cfg, sort_keys=True) != json.dumps(safe_cfg, sort_keys=True)


def test_selected_layer_selector_resolves_to_explicit_ids_in_child_config() -> None:
    base = {
        "model": {"model_id": "fixture/model", "local_path": "/tmp/model", "trust_remote_code": False},
        "topology": {"world_size": 4, "ep_size": 4},
        "runtime": {"line": "async_release", "invariant_mode": "evaluation_strict", "precision": "bf16", "dispatcher": "alltoall"},
        "traffic": {"bucket_mode": "dynamic_current", "bucket_rows": 0},
        "policy": {"options": {}},
        "workload": {"prompts": "configs/workload/smoke_prompts.json"},
        "execution": {"schedule_phase_selector": "both"},
        "evaluation": {"selected_layer_ids": [0, 1]},
    }
    cfg = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="routersense_u_core_predicted_raw_async",
        run_name="selected",
        output_root=Path("/tmp/selected"),
        profile="execution",
        selected_layers="selected",
        save_logits=False,
    )
    assert cfg["requested_layer_selector"] == "selected"
    assert cfg["resolved_layer_selector"] == "explicit"
    assert cfg["resolved_layer_ids"] == ["0", "1"]
    assert cfg["execution"]["schedule"]["selected_layer_ids"] == ["0", "1"]


def test_selected_layer_selector_requires_ids_in_strict_mode() -> None:
    try:
        resolve_layer_selector("selected", invariant_mode="evaluation_strict")
    except ValueError as exc:
        assert "selected_layer_ids" in str(exc)
    else:
        raise AssertionError("expected strict selected resolver to fail without selected_layer_ids")


def test_official_gpu_first_bringup_config_uses_selected_layers_and_core_strategies() -> None:
    payload = load_official_config(REPO_ROOT / "configs/official/gpu_first_bringup.yaml")
    assert payload["selected_layers"] == "0,1"
    assert payload["evaluation"]["selected_layer_ids"] == [0, 1]
    assert payload["workload"]["prompts"] == "configs/components/workloads/bringup_2_short_prompts.json"
    assert payload["strategies"] == [
        "native",
        "birkhoff_phase_local_sync",
        "birkhoff_phase_local_async_p2p",
        "routersense_b_core_independent_async",
        "routersense_u_core_zero_raw_async",
        "routersense_u_core_predicted_raw_async",
        "routersense_u_core_predicted_safe_async",
    ]


def test_gpu_hotpath_iteration_config_uses_compact_preflight_and_three_strategies() -> None:
    payload = load_official_config(REPO_ROOT / "configs/official/gpu_hotpath_iteration.yaml")
    assert payload["preflight_mode"] == "compact"
    assert payload["workload"]["prompts"] == "configs/components/workloads/comparison_8x16_prompts.json"
    assert payload["workload"]["tokenization"] == {
        "padding": "max_length",
        "truncation": True,
        "max_length": 16,
        "expected_prompt_count": 8,
        "expected_batch_rows": 8,
        "expected_seq_len": 16,
    }
    assert payload["strategies"] == [
        "native",
        "routersense_b_core_independent_async",
        "routersense_u_core_zero_raw_async",
    ]


def test_child_config_carries_requested_and_effective_preflight_mode() -> None:
    base = {
        "model": {"model_id": "fixture/model", "local_path": "/tmp/model", "trust_remote_code": False},
        "topology": {"world_size": 4, "ep_size": 4},
        "runtime": {"line": "async_release", "invariant_mode": "evaluation_strict", "precision": "bf16", "dispatcher": "alltoall"},
        "traffic": {"bucket_mode": "dynamic_current", "bucket_rows": 0},
        "policy": {"options": {}},
        "workload": {"prompts": "configs/workload/smoke_prompts.json"},
        "execution": {"schedule_phase_selector": "both"},
        "evaluation": {"selected_layer_ids": [0, 1]},
    }
    cfg = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="routersense_b_core_independent_async",
        run_name="preflight",
        output_root=Path("/tmp/preflight"),
        profile="perf",
        selected_layers="selected",
        save_logits=False,
        preflight_mode="compact",
    )
    assert cfg["execution"]["preflight_mode"] == "compact"
    assert cfg["requested_preflight_mode"] == "compact"
    assert cfg["effective_preflight_mode"] == "compact"


def test_child_config_carries_workload_tokenization_contract() -> None:
    cfg = build_policy_correctness_config(
        base_comparison={
            "model": {"model_id": "fixture/model", "local_path": "/tmp/model", "trust_remote_code": False},
            "topology": {"world_size": 4, "ep_size": 4},
            "runtime": {"line": "async_release", "invariant_mode": "evaluation_strict", "precision": "bf16", "dispatcher": "alltoall"},
            "traffic": {"bucket_mode": "dynamic_current", "bucket_rows": 0},
            "policy": {"options": {}},
            "workload": {
                "prompts": "configs/components/workloads/comparison_8x16_prompts.json",
                "tokenization": {
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": 16,
                    "expected_prompt_count": 8,
                    "expected_batch_rows": 8,
                    "expected_seq_len": 16,
                },
            },
            "execution": {"schedule_phase_selector": "both"},
            "evaluation": {"selected_layer_ids": [0, 1]},
        },
        strategy_name="routersense_b_core_independent_async",
        run_name="tokenization",
        output_root=Path("/tmp/tokenization"),
        profile="perf",
        selected_layers="selected",
        save_logits=False,
        preflight_mode="compact",
    )
    assert cfg["workload"]["tokenization"]["padding"] == "max_length"
    assert cfg["workload"]["tokenization"]["max_length"] == 16
    assert cfg["workload"]["tokenization"]["expected_batch_rows"] == 8


def test_child_config_rejects_invalid_preflight_mode() -> None:
    try:
        build_policy_correctness_config(
            base_comparison={
                "model": {"model_id": "fixture/model", "local_path": "/tmp/model", "trust_remote_code": False},
                "topology": {"world_size": 4, "ep_size": 4},
                "runtime": {"line": "async_release", "invariant_mode": "evaluation_strict"},
                "traffic": {"bucket_mode": "dynamic_current", "bucket_rows": 0},
                "policy": {"options": {}},
                "workload": {"prompts": "configs/workload/smoke_prompts.json"},
                "execution": {"schedule_phase_selector": "both"},
                "evaluation": {"selected_layer_ids": [0, 1]},
            },
            strategy_name="routersense_b_core_independent_async",
            run_name="bad",
            output_root=Path("/tmp/bad"),
            profile="perf",
            selected_layers="selected",
            save_logits=False,
            preflight_mode="invalid",
        )
    except ValueError as exc:
        assert "preflight_mode" in str(exc)
    else:
        raise AssertionError("expected invalid preflight mode to fail")
