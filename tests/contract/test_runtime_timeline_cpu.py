from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rs.experiments_support.gpu_runner_common import build_policy_correctness_config
from rs.core.experiment_config import load_run_config
from rs.runtime.online.megatron_ep.observation.runtime_timeline import (
    MEASUREMENT_STATUSES,
    RuntimePhaseTimeline,
    interval_us,
    phase_label,
    summarize_rank_imbalance,
    summarize_task_granularity,
)
from rs.runtime.online.megatron_ep.observation.tokenization import compute_token_count_contract


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_token_count_contract_8x16_uses_attention_mask_sum() -> None:
    contract = compute_token_count_contract(
        actual_batch_rows=8,
        actual_seq_len=16,
        attention_mask_sum=72,
    )
    payload = contract.to_dict()
    assert payload["total_token_slots"] == 128
    assert payload["valid_token_count"] == 72
    assert payload["padding_token_count"] == 56
    assert payload["valid_token_count"] + payload["padding_token_count"] == payload["total_token_slots"]
    assert payload["token_count_status"] == "measured"
    assert payload["padded_token_count_deprecated"] is True
    assert payload["padded_token_count_unit"] == "padding_token_count"


def test_token_count_contract_missing_attention_mask_is_unavailable() -> None:
    payload = compute_token_count_contract(
        actual_batch_rows=8,
        actual_seq_len=16,
        attention_mask_sum=None,
    ).to_dict()
    assert payload["total_token_slots"] == 128
    assert payload["valid_token_count"] is None
    assert payload["padding_token_count"] is None
    assert payload["token_count_status"] == "unavailable"


def test_runtime_phase_timeline_derives_intervals() -> None:
    timeline = RuntimePhaseTimeline(
        rank=0,
        forward_epoch=3,
        layer_id="1",
        phase="P0_dispatch",
        strategy="routersense_current_p012_joint_event_rscf_async",
        plan_origin="current_window",
        timestamps={
            "hook_enter_ns": 0,
            "observation_begin_ns": 10,
            "observation_end_ns": 30,
            "plan_begin_ns": 30,
            "plan_end_ns": 80,
            "materialize_begin_ns": 80,
            "materialize_end_ns": 100,
            "pack_begin_ns": 100,
            "pack_end_ns": 140,
            "submit_begin_ns": 150,
            "first_request_submitted_ns": 170,
            "last_request_submitted_ns": 210,
            "first_request_completed_ns": 260,
            "all_requests_completed_ns": 310,
            "unpack_begin_ns": 310,
            "unpack_end_ns": 340,
            "hook_exit_ns": 400,
        },
    )
    assert timeline.validate() == ()
    derived = timeline.derived_intervals()
    assert derived["observation_us"] == {"value_us": 0.02, "measurement_status": "derived"}
    assert derived["submit_queue_us"] == {"value_us": 0.02, "measurement_status": "derived"}
    assert derived["submit_span_us"] == {"value_us": 0.04, "measurement_status": "derived"}
    assert derived["request_wait_us"] == {"value_us": 0.1, "measurement_status": "derived"}
    assert derived["post_transport_us"] == {"value_us": 0.09, "measurement_status": "derived"}
    assert derived["first_submit_to_all_complete_us"] == {"value_us": 0.14, "measurement_status": "derived"}


def test_runtime_phase_timeline_rejects_reverse_timestamp_order() -> None:
    timeline = RuntimePhaseTimeline(
        rank=0,
        forward_epoch=0,
        layer_id="0",
        phase="P1_return",
        strategy="s",
        timestamps={"hook_enter_ns": 10, "observation_begin_ns": 9},
    )
    assert timeline.validate()
    assert "precedes" in timeline.validate()[0]


def test_missing_request_completion_is_unavailable_not_zero() -> None:
    result = interval_us(
        {"last_request_submitted_ns": 100, "all_requests_completed_ns": None},
        "last_request_submitted_ns",
        "all_requests_completed_ns",
    )
    assert result["value_us"] is None
    assert result["measurement_status"] == "unavailable"


def test_phase_interval_not_applicable_status() -> None:
    result = interval_us({}, "pack_begin_ns", "pack_end_ns", not_applicable=True)
    assert result == {"value_us": None, "measurement_status": "not_applicable"}
    assert phase_label("P0") == "P0_dispatch"
    assert phase_label("P1") == "P1_return"


def test_task_granularity_summary_uses_compact_statistics() -> None:
    summary = summarize_task_granularity(
        [
            {"row_count": 1, "byte_count": 128, "wave_id": 0, "op_kind": "send"},
            {"row_count": 8, "byte_count": 4096, "wave_id": 0, "op_kind": "recv"},
            {"row_count": 16, "byte_count": 131072, "wave_id": 1, "op_kind": "send"},
        ],
        small_task_bytes_threshold=64 * 1024,
    )
    assert summary["task_count"] == 3
    assert summary["wave_count"] == 2
    assert summary["send_task_count"] == 2
    assert summary["recv_task_count"] == 1
    assert summary["total_rows"] == 25
    assert summary["total_wire_bytes"] == 135296
    assert summary["min_task_rows"] == 1
    assert summary["median_task_rows"] == 8.0
    assert summary["max_task_bytes"] == 131072
    assert summary["single_row_task_count"] == 1
    assert summary["small_task_count"] == 2


def test_rank_imbalance_summary_handles_zero_min_ratio() -> None:
    summary = summarize_rank_imbalance(
        [
            {"rank": 0, "layer_id": "0", "phase": "P0_dispatch", "rows": 10, "wire_bytes": 0, "task_count": 0, "submit_span_us": 2, "request_wait_us": 5, "hook_total_us": 10},
            {"rank": 1, "layer_id": "0", "phase": "P0_dispatch", "rows": 20, "wire_bytes": 100, "task_count": 4, "submit_span_us": 4, "request_wait_us": 10, "hook_total_us": 20},
            {"rank": 2, "layer_id": "0", "phase": "P0_dispatch", "rows": 30, "wire_bytes": 200, "task_count": 8, "submit_span_us": 8, "request_wait_us": 15, "hook_total_us": 30},
            {"rank": 3, "layer_id": "0", "phase": "P0_dispatch", "rows": 40, "wire_bytes": 300, "task_count": 12, "submit_span_us": 16, "request_wait_us": 20, "hook_total_us": 40},
        ]
    )
    assert summary["rank_count"] == 4
    assert summary["max_to_min_bytes_ratio"]["ratio_status"] == "undefined_due_to_zero_min"
    assert summary["max_to_min_task_count_ratio"]["ratio_status"] == "undefined_due_to_zero_min"
    assert summary["max_to_min_wait_ratio"]["value"] == 4.0
    assert summary["critical_rank"] == 3


def test_measurement_status_vocabulary_is_fixed() -> None:
    assert MEASUREMENT_STATUSES == {"measured", "derived", "not_applicable", "unavailable"}
    timeline = RuntimePhaseTimeline(
        rank=0,
        forward_epoch=0,
        layer_id="0",
        phase="P0_dispatch",
        strategy="s",
        statuses={"host_pack_us": "guessed"},
    )
    assert "invalid measurement_status" in timeline.validate()[0]


def test_gpu_runtime_timeline_config_is_timeline_light_and_perf_safe() -> None:
    payload = yaml.safe_load((REPO_ROOT / "configs/official/gpu_runtime_timeline.yaml").read_text())
    assert payload["world_size"] == 4
    assert payload["selected_layers"] == "0,1"
    assert payload["profile"] == "timeline_light"
    assert payload["preflight_mode"] == "compact"
    assert payload["evaluation"]["warmup"] == 1
    assert payload["evaluation"]["repeats"] == 1
    assert payload["workload"]["tokenization"]["expected_batch_rows"] == 8
    assert payload["workload"]["tokenization"]["expected_seq_len"] == 16
    assert payload["timeline"]["raw_task_jsonl"] is False
    assert payload["timeline"]["profiler"] is False
    assert payload["timeline"]["correctness_replay"] is False
    assert payload["strategies"] == [
        "routersense_current_p012_local_event_rscf_async",
        "routersense_current_p012_joint_event_rscf_async",
    ]


def test_gpu_runtime_timeline_child_config_loads_timeline_light_profile(tmp_path: Path) -> None:
    base = yaml.safe_load((REPO_ROOT / "configs/official/gpu_runtime_timeline.yaml").read_text())
    base["model"] = {
        "model_id": "test-model",
        "local_path": "/tmp/test-model",
        "trust_remote_code": False,
    }
    child = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="routersense_current_p012_local_event_rscf_async",
        run_name="timeline_child",
        output_root=tmp_path / "out",
        profile="timeline_light",
        selected_layers="0,1",
        save_logits=False,
        preflight_mode="compact",
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(yaml.safe_dump(child, sort_keys=False))
    config = load_run_config(config_path=child_path)
    assert config.observation.profile == "timeline_light"
    assert config.execution.preflight_mode == "compact"
    assert config.execution.schedule.selected_layer_ids == ("0", "1")
