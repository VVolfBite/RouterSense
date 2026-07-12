from __future__ import annotations

import json
from pathlib import Path

from experiments.distributed._gpu_runner_common import build_policy_correctness_config, load_official_config
from experiments.distributed.run_gpu_a2_strategy_compare import _metric_series


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_official_gpu_c2_matrix_includes_four_joint_async_candidates() -> None:
    payload = load_official_config(REPO_ROOT / "configs/official/gpu_c2_correctness.yaml")
    assert payload["candidate_strategies"] == [
        "birkhoff_phase_local_async_p2p",
        "routersense_joint_zero_raw_async",
        "routersense_joint_predicted_raw_async",
        "routersense_joint_zero_safe_async",
        "routersense_joint_predicted_safe_async",
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


def test_raw_and_safe_child_configs_diverge_in_safe_projection_mode() -> None:
    base = {
        "model": {"model_id": "fixture/model", "local_path": "/tmp/model", "trust_remote_code": False},
        "topology": {"world_size": 4, "ep_size": 4},
        "runtime": {"line": "async_release", "invariant_mode": "evaluation_strict", "precision": "bf16", "dispatcher": "alltoall"},
        "traffic": {"bucket_mode": "dynamic_current", "bucket_rows": 0},
        "policy": {"options": {"p0_weight": 1.0, "p1_reservation_weight": 1.0, "p2_hint_weight": 1.0, "residual_weight": 0.75, "barrier_weight": 1.75, "age_weight": 0.15, "prediction_weight": 0.35}},
        "workload": {"prompts": "configs/workload/smoke_prompts.json"},
        "execution": {"schedule_phase_selector": "both"},
    }
    raw_cfg = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="routersense_joint_predicted_raw_async",
        run_name="raw",
        output_root=Path("/tmp/raw"),
        profile="execution",
        selected_layers="selected",
        save_logits=False,
    )
    safe_cfg = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="routersense_joint_predicted_safe_async",
        run_name="safe",
        output_root=Path("/tmp/safe"),
        profile="execution",
        selected_layers="selected",
        save_logits=False,
    )
    assert raw_cfg["execution"]["safe_projection_mode"] == "disabled"
    assert safe_cfg["execution"]["safe_projection_mode"] == "host_select"
    assert json.dumps(raw_cfg, sort_keys=True) != json.dumps(safe_cfg, sort_keys=True)
