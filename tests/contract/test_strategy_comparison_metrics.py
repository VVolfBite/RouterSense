from __future__ import annotations

import json
from dataclasses import replace

from rs.core.contracts.result import RunIdentity
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle
from rs.reporting.comparison_metrics import (
    add_baseline_deltas,
    build_comparison_report,
    communication_collective_time_from_timeline,
    communication_phase_window_from_timeline,
    communication_makespan_from_timeline,
    metrics_from_rank_dir,
    native_communication_makespan_from_observer,
)
from rs.reporting.shadow_plan_analysis import build_shadow_plan_alignment
from rs.runtime.online.megatron_ep.control import plan_agreement as plan_agreement_mod
from rs.scheduling.phase_execution import FutureDemandHint
from rs.scheduling.registry import resolve_phase_policy
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def _write_result_bundle(run_dir) -> None:
    bundle = build_result_bundle(
        ResultBundleDraft(
            run_identity=RunIdentity(
                run_id="run",
                pipeline="online",
                claim_scope="formal",
                trace_origin="runtime",
                future_information_mode="predicted",
            ),
            status="success",
            correctness_status="valid",
            performance_status="ineligible",
            commit_sha="abc123",
            git_clean=True,
            instrumentation_mode="contract",
            audit_evidence_level="summary_only",
            measurement_complete=True,
            summary={
                "all_work_completed": True,
                "fallback_count": 0,
                "timeout_count": 0,
                "check_failure_count": 0,
                "execution_outcome_count": 1,
            },
            details={},
            extensions={},
        )
    )
    (run_dir / "result_bundle.json").write_text(json.dumps(bundle.to_dict()), encoding="utf-8")


def test_communication_makespan_from_timeline() -> None:
    timeline = [
        {"event": "before_wave", "ts_us": 100},
        {"event": "after_wave", "ts_us": 180},
        {"event": "before_wave", "ts_us": 210},
        {"event": "after_wave", "ts_us": 260},
    ]
    assert communication_makespan_from_timeline(timeline) == 160.0


def test_communication_phase_window_from_timeline() -> None:
    timeline = [
        {"event": "before_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 100},
        {"event": "after_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 140},
        {"event": "before_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 1, "tensor_role": "hidden_states", "ts_us": 170},
        {"event": "after_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 1, "tensor_role": "hidden_states", "ts_us": 210},
        {"event": "before_payload_collective", "layer": "layer0", "phase": "P1", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 300},
        {"event": "after_payload_collective", "layer": "layer0", "phase": "P1", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 360},
    ]
    assert communication_phase_window_from_timeline(timeline) == 170.0


def test_communication_collective_time_from_timeline() -> None:
    timeline = [
        {"event": "before_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 100},
        {"event": "after_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 140},
        {"event": "before_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 1, "tensor_role": "hidden_states", "ts_us": 170},
        {"event": "after_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 1, "tensor_role": "hidden_states", "ts_us": 210},
        {"event": "before_payload_collective", "layer": "layer0", "phase": "P1", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 300},
        {"event": "after_payload_collective", "layer": "layer0", "phase": "P1", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 360},
    ]
    assert communication_collective_time_from_timeline(timeline) == 140.0


def test_net_benefit_formula() -> None:
    strategy = {
        "communication_makespan_us": {"mean": 800.0},
        "scheduling_overhead_us": {"mean": 50.0},
        "total_forward_us": {"mean": 1200.0},
    }
    baseline = {
        "communication_makespan_us": {"mean": 1000.0},
        "total_forward_us": {"mean": 1300.0},
    }
    out = add_baseline_deltas(strategy, baseline)
    assert out["net_comm_savings_us"]["mean"] == 200.0
    assert out["net_benefit_us"]["mean"] == 150.0
    assert out["benefit_ratio"]["mean"] == 4.0


def test_native_communication_makespan_from_observer() -> None:
    observer_rows = [
        {"layer": "layer0", "phase": "token_dispatch_enter", "ts_us": 100},
        {"layer": "layer0", "phase": "P0_comm", "ts_us": 180},
        {"layer": "layer0", "phase": "token_combine_enter", "ts_us": 220},
        {"layer": "layer0", "phase": "P1_comm", "ts_us": 300},
        {"layer": "layer1", "phase": "token_dispatch_enter", "ts_us": 350},
        {"layer": "layer1", "phase": "P0_comm", "ts_us": 410},
        {"layer": "layer1", "phase": "token_combine_enter", "ts_us": 470},
        {"layer": "layer1", "phase": "P1_comm", "ts_us": 540},
    ]
    assert native_communication_makespan_from_observer(observer_rows) == 290.0


def test_comparison_report_structure() -> None:
    report = build_comparison_report(
        run_id="run",
        baseline="disabled",
        strategies=[
            {
                "name": "disabled",
                "description": "baseline",
                "repetitions": 1,
                "metrics": {"communication_makespan_us": {"mean": 100.0}, "total_wave_count": {"mean": 10.0}, "total_forward_us": {"mean": 200.0}},
            },
            {
                "name": "prepared_priority",
                "description": "candidate",
                "repetitions": 1,
                "metrics": {"communication_makespan_us": {"mean": 80.0}, "total_wave_count": {"mean": 8.0}, "scheduling_overhead_us": {"mean": 5.0}, "total_forward_us": {"mean": 180.0}},
            },
        ],
    )
    assert report["run_id"] == "run"
    assert report["baseline"] == "disabled"
    assert report["strategies"][1]["metrics"]["net_benefit_us"]["mean"] == 15.0
    assert report["pairwise_vs_baseline"]["prepared_priority"]["comm_makespan_delta_pct"] == -20.0
    assert "prepared_priority_vs_birkhoff_phase_local" not in report["pairwise_head_to_head"]


def test_plan_agreement_timing_in_metrics(monkeypatch) -> None:
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 2), (3, 0)))
    local_context = contexts[0]
    policy = resolve_phase_policy(policy_name="fifo_bucket", bucket_rows=0)

    monkeypatch.setattr(plan_agreement_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(plan_agreement_mod.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(plan_agreement_mod.dist, "get_process_group_ranks", lambda group=None: [0, 1])
    monkeypatch.setattr(plan_agreement_mod.dist, "get_backend", lambda group=None: "gloo")

    class _Group:
        WORLD = object()

    monkeypatch.setattr(plan_agreement_mod.dist, "group", _Group)

    all_gather_calls: list[object] = []

    encoded = [
        plan_agreement_mod._encode_planning_summary_tensor(
            ctx.to_planning_summary(),
            world_size=2,
            device=plan_agreement_mod.torch.device("cpu"),
        )
        for ctx in contexts
    ]

    def all_gather(output, value, group=None):
        all_gather_calls.append(value.clone())
        output[0].copy_(encoded[0])
        output[1].copy_(encoded[1])

    broadcast_state: dict[str, object] = {}

    def broadcast(tensor, src=0, group=None):
        if tensor.numel() == 1:
            broadcast_state["length"] = int(tensor.item())
            return None
        if "payload" not in broadcast_state:
            root_plan = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
            payload = plan_agreement_mod._encode_abstract_plan_tensor(
                root_plan.to_abstract_plan(),
                device=plan_agreement_mod.torch.device("cpu"),
            )
            broadcast_state["payload"] = payload
            tensor.copy_(payload)
            return None
            tensor.copy_(broadcast_state["payload"])
            return None

    def broadcast_object_list(payload, src=0, group=None):
        payload[0] = {"success": True, "root_global_rank": 0}
        return None

    monkeypatch.setattr(plan_agreement_mod.dist, "all_gather", all_gather)
    monkeypatch.setattr(plan_agreement_mod.dist, "broadcast", broadcast)
    monkeypatch.setattr(plan_agreement_mod.dist, "broadcast_object_list", broadcast_object_list)

    plan = plan_agreement_mod.run_phase_plan_agreement(local_context=local_context, policy=policy, group=None)
    for key in (
        "all_gather_time_us",
        "all_gather_submit_time_us",
        "all_gather_sync_time_us",
        "build_plan_time_us",
        "broadcast_time_us",
        "broadcast_length_submit_time_us",
        "broadcast_length_sync_time_us",
        "broadcast_payload_submit_time_us",
        "broadcast_payload_sync_time_us",
        "verify_time_us",
        "total_agreement_time_us",
        "summary_stack_time_us",
        "summary_tensor_to_cpu_time_us",
        "summary_object_decode_time_us",
        "abstract_tensor_to_cpu_time_us",
        "abstract_object_decode_time_us",
        "planning_summary_tensor_len",
        "planning_summary_total_elements",
        "abstract_plan_tensor_len",
        "abstract_plan_total_elements",
        "abstract_plan_task_ref_count",
        "broadcast_payload_elements",
    ):
        assert key in plan.metrics
        assert float(plan.metrics[key]) >= 0.0
    assert len(all_gather_calls) == 1


def test_planning_summary_tensor_length_is_not_quadratic() -> None:
    assert plan_agreement_mod._summary_tensor_length(2) == 3 + 2 + (2 * 4)
    assert plan_agreement_mod._summary_tensor_length(4) == 3 + 4 + (4 * 4)
    assert plan_agreement_mod._summary_tensor_length(8) == 3 + 8 + (8 * 4)


def test_planning_summary_encoding_ignores_prepared_hint_metadata() -> None:
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 2), (3, 0)), p2_hint_mode="deterministic_stub")
    base = contexts[0]
    hinted = replace(
        base,
        p2_hint=FutureDemandHint(
            hint_mode="calibrated_artifact",
            hint_digest="digest",
            hint_source="prepared",
            metadata={
                "preferred_edges": [
                    {"phase": "P0", "src_rank": 0, "dst_rank": 1, "priority": 0},
                    {"phase": "P0", "src_rank": 1, "dst_rank": 0, "priority": 1},
                ],
                "preferred_waves": [{"wave_id": 0, "edges": []}],
            },
        ),
    )
    encoded_base = plan_agreement_mod._encode_planning_summary_tensor(
        base.to_planning_summary(),
        world_size=2,
        device=plan_agreement_mod.torch.device("cpu"),
    )
    encoded_hinted = plan_agreement_mod._encode_planning_summary_tensor(
        hinted.to_planning_summary(),
        world_size=2,
        device=plan_agreement_mod.torch.device("cpu"),
    )
    assert encoded_base.tolist() == encoded_hinted.tolist()


def test_shadow_plan_alignment_exact_match() -> None:
    rows = build_shadow_plan_alignment(
        prepared_phase_plan_shadow=[
            {
                "ts_us": 10,
                "layer_name": "model.layers.1.mlp",
                "phase": "P0",
                "prepared_window_key": "window-1",
                "compiled_plan_hash": "plan-a",
                "compiled_bucket_order": ["P0:1->0:0:0", "P0:0->1:0:0"],
                "prepared_plan_order_preserved": True,
                "hint_edges_consumed": 2,
                "hint_match_rate": 1.0,
            }
        ],
        scheduled_phase_plans=[
            {
                "ts_us": 11,
                "layer_name": "model.layers.1.mlp",
                "phase": "P0",
                "plan_hash": "plan-a",
                "metrics": {
                    "bucket_order": ["P0:1->0:0:0", "P0:0->1:0:0"],
                    "ordered_by_prepared_plan": True,
                    "hint_edges_consumed": 2,
                    "hint_match_rate": 1.0,
                },
            }
        ],
        transport_execution=[
            {"layer_name": "model.layers.1.mlp", "phase": "P0", "task_id": "P0:1->0:0:0"},
            {"layer_name": "model.layers.1.mlp", "phase": "P0", "task_id": "P0:0->1:0:0"},
        ],
    )
    assert len(rows) == 1
    assert rows[0]["plan_hash_match"] is True
    assert rows[0]["prepared_to_actual_exact_match"] is True
    assert rows[0]["actual_plan_to_execution_exact_match"] is True


def test_metrics_from_rank_dir_includes_shadow_alignment_summary(tmp_path) -> None:
    run_dir = tmp_path
    (run_dir / "rank0_control_timeline.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_transport_bundles.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_plan_arrival_records.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_planning_timing.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_phase_contexts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_observer.jsonl").write_text("", encoding="utf-8")
    _write_result_bundle(run_dir)
    (run_dir / "rank0_prepared_phase_plan_shadow.jsonl").write_text(
        json.dumps(
            {
                "ts_us": 10,
                "layer_name": "model.layers.1.mlp",
                "phase": "P0",
                "prepared_window_key": "window-1",
                "compiled_plan_hash": "plan-a",
                "compiled_bucket_order": ["P0:1->0:0:0"],
                "prepared_plan_order_preserved": True,
                "hint_edges_consumed": 1,
                "hint_match_rate": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "rank0_scheduled_phase_plans.jsonl").write_text(
        json.dumps(
            {
                "ts_us": 11,
                "layer_name": "model.layers.1.mlp",
                "phase": "P0",
                "plan_hash": "plan-a",
                "waves": [],
                "metrics": {
                    "bucket_order": ["P0:1->0:0:0"],
                    "ordered_by_prepared_plan": True,
                    "hint_edges_consumed": 1,
                    "hint_match_rate": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "rank0_transport_execution.jsonl").write_text(
        json.dumps({"layer_name": "model.layers.1.mlp", "phase": "P0", "task_id": "P0:1->0:0:0"}) + "\n",
        encoding="utf-8",
    )

    metrics = metrics_from_rank_dir(run_dir, rank=0)
    assert metrics["prepared_shadow_phase_count"] == 1.0
    assert metrics["prepared_shadow_plan_hash_match_count"] == 1.0
    assert metrics["prepared_shadow_exact_order_match_count"] == 1.0
    assert metrics["prepared_shadow_execution_exact_match_count"] == 1.0


def test_metrics_from_rank_dir_uses_native_observer_fallback(tmp_path) -> None:
    run_dir = tmp_path
    (run_dir / "rank0_control_timeline.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_scheduled_phase_plans.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_transport_execution.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_transport_bundles.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_plan_arrival_records.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_planning_timing.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_phase_contexts.jsonl").write_text("", encoding="utf-8")
    _write_result_bundle(run_dir)
    (run_dir / "rank0_observer.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"layer": "layer0", "phase": "token_dispatch_enter", "ts_us": 100}),
                json.dumps({"layer": "layer0", "phase": "P0_comm", "ts_us": 180}),
                json.dumps({"layer": "layer0", "phase": "token_combine_enter", "ts_us": 220}),
                json.dumps({"layer": "layer0", "phase": "P1_comm", "ts_us": 300}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = metrics_from_rank_dir(run_dir, rank=0)
    assert metrics["communication_makespan_us"] == 160.0
    assert metrics["communication_phase_window_us"] == 160.0
    assert metrics["communication_collective_active_us"] == 160.0


def test_metrics_from_rank_dir_prefers_timeline_phase_window(tmp_path) -> None:
    run_dir = tmp_path
    (run_dir / "rank0_control_timeline.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "before_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 100}),
                json.dumps({"event": "after_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 140}),
                json.dumps({"event": "before_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 1, "tensor_role": "hidden_states", "ts_us": 170}),
                json.dumps({"event": "after_payload_collective", "layer": "layer0", "phase": "P0", "wave_id": 1, "tensor_role": "hidden_states", "ts_us": 210}),
                json.dumps({"event": "before_payload_collective", "layer": "layer0", "phase": "P1", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 300}),
                json.dumps({"event": "after_payload_collective", "layer": "layer0", "phase": "P1", "wave_id": 0, "tensor_role": "hidden_states", "ts_us": 360}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "rank0_scheduled_phase_plans.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_transport_execution.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_transport_bundles.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_plan_arrival_records.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_planning_timing.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_phase_contexts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_observer.jsonl").write_text("", encoding="utf-8")
    _write_result_bundle(run_dir)
    metrics = metrics_from_rank_dir(run_dir, rank=0)
    assert metrics["communication_makespan_us"] == 170.0
    assert metrics["communication_phase_window_us"] == 170.0
    assert metrics["communication_collective_active_us"] == 140.0


def test_metrics_from_rank_dir_includes_planning_stage_metrics(tmp_path) -> None:
    run_dir = tmp_path
    (run_dir / "rank0_control_timeline.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_scheduled_phase_plans.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_transport_execution.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_transport_bundles.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_plan_arrival_records.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_phase_contexts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_observer.jsonl").write_text("", encoding="utf-8")
    _write_result_bundle(run_dir)
    (run_dir / "rank0_planning_timing.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"stage": "build_p2_hint", "duration_us": 10.5}),
                json.dumps({"stage": "build_p2_hint", "duration_us": 9.5}),
                json.dumps({"stage": "run_phase_plan_agreement", "duration_us": 100.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = metrics_from_rank_dir(run_dir, rank=0)
    assert metrics["build_p2_hint_time_us"] == 20.0
    assert metrics["avg_build_p2_hint_time_us"] == 10.0
    assert metrics["build_p2_hint_count"] == 2.0
    assert metrics["run_phase_plan_agreement_time_us"] == 100.0


def test_metrics_from_rank_dir_reads_prepared_plan_summary(tmp_path) -> None:
    run_dir = tmp_path
    (run_dir / "rank0_control_timeline.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_transport_execution.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_transport_bundles.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_plan_arrival_records.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_planning_timing.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_phase_contexts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rank0_observer.jsonl").write_text("", encoding="utf-8")
    _write_result_bundle(run_dir)
    (run_dir / "rank0_scheduled_phase_plans.jsonl").write_text(
        json.dumps(
            {
                "waves": [{"wave_id": 0, "task_count": 2}],
                "metrics": {
                    "bucket_count": 4,
                    "nonzero_edge_count": 2,
                    "total_row_count": 128,
                    "total_byte_count": 4096,
                    "avg_buckets_per_edge": 2.0,
                    "max_buckets_per_edge": 3,
                    "expected_collective_count": 8,
                    "max_wave_task_count": 2,
                    "hint_edges_available": 3,
                    "hint_edges_matched": 2,
                    "hint_match_rate": 2.0 / 3.0,
                    "preferred_wave_count": 1,
                    "preferred_edge_count": 2,
                    "all_gather_submit_time_us": 10.0,
                    "all_gather_sync_time_us": 20.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "rank0_prepared_plan_summary.json").write_text(
        json.dumps(
            {
                "p2_matrix_source": "replicated_local_row",
                "p2_matrix_total_bytes": 777,
                "p2_matrix_is_replicated_local_row": True,
                "p2_matrix_row_sums": [1, 2],
                "p2_matrix_col_sums": [3, 4],
            }
        ),
        encoding="utf-8",
    )
    metrics = metrics_from_rank_dir(run_dir, rank=0)
    assert metrics["p2_matrix_source"] == "replicated_local_row"
    assert metrics["p2_matrix_total_bytes"] == 777.0
    assert metrics["p2_matrix_is_replicated_local_row"] is True
    assert metrics["p2_matrix_row_sums"] == [1, 2]
    assert metrics["p2_matrix_col_sums"] == [3, 4]
    assert metrics["bucket_count"] == 4.0
    assert metrics["nonzero_edge_count"] == 2.0
    assert metrics["avg_buckets_per_edge"] == 2.0
    assert metrics["collective_count"] == 8.0
    assert metrics["hint_edges_available"] == 3.0
    assert metrics["hint_edges_matched"] == 2.0


def test_shadow_plan_alignment_uses_transport_plan_hash_to_recover_layer_name() -> None:
    rows = build_shadow_plan_alignment(
        prepared_phase_plan_shadow=[
            {
                "ts_us": 10,
                "layer_name": "model.layers.1.mlp",
                "phase": "P0",
                "compiled_plan_hash": "plan-a",
                "compiled_bucket_order": ["P0:1->0:0:0"],
            }
        ],
        scheduled_phase_plans=[
            {
                "ts_us": 11,
                "plan_key": {"layer_id": "1", "phase": "P0"},
                "phase": "P0",
                "plan_hash": "plan-a",
                "metrics": {"bucket_order": ["P0:1->0:0:0"]},
            }
        ],
        transport_execution=[
            {
                "layer_name": "model.layers.1.mlp",
                "phase": "P0",
                "plan_hash": "plan-a",
                "task_id": "P0:1->0:0:0",
            }
        ],
    )
    assert len(rows) == 1
    assert rows[0]["layer_name"] == "model.layers.1.mlp"
    assert rows[0]["has_actual_scheduled_plan"] is True
    assert rows[0]["plan_hash_match"] is True


def test_shadow_plan_alignment_summary_excludes_compile_failures() -> None:
    rows = build_shadow_plan_alignment(
        prepared_phase_plan_shadow=[
            {
                "ts_us": 10,
                "layer_name": "model.layers.1.mlp",
                "phase": "P0",
                "compile_status": "failed",
                "exception": "ValueError: missing incoming slot",
            },
            {
                "ts_us": 12,
                "layer_name": "model.layers.2.mlp",
                "phase": "P0",
                "compile_status": "ok",
                "compiled_plan_hash": "plan-b",
                "compiled_bucket_order": ["P0:1->0:0:0"],
            },
        ],
        scheduled_phase_plans=[
            {
                "ts_us": 13,
                "layer_name": "model.layers.2.mlp",
                "phase": "P0",
                "plan_hash": "plan-b",
                "metrics": {"bucket_order": ["P0:1->0:0:0"]},
            }
        ],
        transport_execution=[
            {"layer_name": "model.layers.2.mlp", "phase": "P0", "plan_hash": "plan-b", "task_id": "P0:1->0:0:0"}
        ],
    )
    considered = [row for row in rows if row["layer_name"] == "model.layers.2.mlp"]
    assert len(considered) == 1
    assert considered[0]["prepared_compile_status"] == "ok"
    from rs.reporting.shadow_plan_analysis import summarize_shadow_plan_alignment

    summary = summarize_shadow_plan_alignment(rows)
    assert summary["prepared_shadow_phase_count"] == 1.0
    assert summary["prepared_shadow_compile_failed_count"] == 1.0


def test_shadow_plan_alignment_deduplicates_transport_task_ids() -> None:
    rows = build_shadow_plan_alignment(
        prepared_phase_plan_shadow=[],
        scheduled_phase_plans=[
            {
                "ts_us": 11,
                "layer_name": "model.layers.1.mlp",
                "phase": "P0",
                "plan_hash": "plan-a",
                "metrics": {"bucket_order": ["P0:1->0:0:0", "P0:0->1:0:0"]},
            }
        ],
        transport_execution=[
            {"layer_name": "model.layers.1.mlp", "phase": "P0", "plan_hash": "plan-a", "task_id": "P0:1->0:0:0", "tensor_role": "hidden_states"},
            {"layer_name": "model.layers.1.mlp", "phase": "P0", "plan_hash": "plan-a", "task_id": "P0:1->0:0:0", "tensor_role": "routing_probs"},
            {"layer_name": "model.layers.1.mlp", "phase": "P0", "plan_hash": "plan-a", "task_id": "P0:0->1:0:0", "tensor_role": "hidden_states"},
        ],
    )
    assert rows[0]["actual_execution_order"] == ["P0:1->0:0:0", "P0:0->1:0:0"]
