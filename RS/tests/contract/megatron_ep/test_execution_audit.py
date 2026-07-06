from __future__ import annotations

from rs.runtime.online.megatron_ep.execution.audit import ExecutionAuditInput, build_execution_audit


def _plan() -> dict[str, object]:
    return {
        "policy_name": "trivial_reverse_bucket",
        "plan_hash": "plan-1",
        "transport_mutation": True,
        "waves": [
            {
                "wave_id": 0,
                "bucket_tasks": [
                    {"task_id": "a", "row_count": 4, "byte_count": 16},
                    {"task_id": "b", "row_count": 4, "byte_count": 16},
                ],
            }
        ],
    }


def test_execution_audit_detects_missing_task() -> None:
    audit = build_execution_audit(
        ExecutionAuditInput(
            execution_plan=_plan(),
            transport_events=(
                {"wave_id": 0, "task_id": "a", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
            ),
            phase_contract={"phase": "P1", "layer_id": "0", "policy_enabled": True},
        )
    )
    assert audit.status == "failed"
    assert audit.missing_tasks == ("b",)


def test_execution_audit_detects_order_mismatch() -> None:
    audit = build_execution_audit(
        ExecutionAuditInput(
            execution_plan=_plan(),
            transport_events=(
                {"wave_id": 0, "task_id": "b", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
                {"wave_id": 0, "task_id": "a", "src_rank": 1, "dst_rank": 0, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
            ),
            phase_contract={"phase": "P1", "layer_id": "0", "policy_enabled": True},
        )
    )
    assert audit.status == "failed"
    assert audit.order_mismatches


def test_execution_audit_accepts_valid_reverse_plan() -> None:
    plan = {
        "policy_name": "trivial_reverse_bucket",
        "plan_hash": "plan-1",
        "transport_mutation": True,
        "waves": [
            {"wave_id": 0, "bucket_tasks": [{"task_id": "b", "row_count": 4, "byte_count": 16}]},
            {"wave_id": 1, "bucket_tasks": [{"task_id": "a", "row_count": 4, "byte_count": 16}]},
        ],
    }
    audit = build_execution_audit(
        ExecutionAuditInput(
            execution_plan=plan,
            transport_events=(
                {"wave_id": 0, "task_id": "b", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
                {"wave_id": 1, "task_id": "a", "src_rank": 1, "dst_rank": 0, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
            ),
            phase_contract={"phase": "P1", "layer_id": "0", "policy_enabled": True},
        )
    )
    assert audit.status == "passed"


def test_execution_audit_preserves_p0_bundle_atomicity() -> None:
    plan = {
        "policy_name": "bucketed_fifo",
        "plan_hash": "plan-1",
        "transport_mutation": True,
        "waves": [{"wave_id": 0, "bucket_tasks": [{"task_id": "a", "row_count": 4, "byte_count": 32}]}],
    }
    audit = build_execution_audit(
        ExecutionAuditInput(
            execution_plan=plan,
            transport_events=(
                {"wave_id": 0, "task_id": "a", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
                {"wave_id": 0, "task_id": "a", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16, "tensor_role": "routing_probs"},
            ),
            phase_contract={"phase": "P0", "layer_id": "0", "policy_enabled": True},
        )
    )
    assert audit.status == "passed"
    assert audit.p0_bundle_atomicity_preserved is True


def test_execution_audit_detects_wave_mismatch() -> None:
    plan = {
        "policy_name": "bucketed_fifo",
        "plan_hash": "plan-1",
        "transport_mutation": True,
        "waves": [
            {"wave_id": 0, "bucket_tasks": [{"task_id": "a", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16}]},
            {"wave_id": 1, "bucket_tasks": [{"task_id": "b", "src_rank": 1, "dst_rank": 0, "row_count": 4, "byte_count": 16}]},
        ],
    }
    audit = build_execution_audit(
        ExecutionAuditInput(
            execution_plan=plan,
            transport_events=(
                {"wave_id": 0, "task_id": "a", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
                {"wave_id": 0, "task_id": "b", "src_rank": 1, "dst_rank": 0, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
            ),
            phase_contract={"phase": "P1", "layer_id": "0", "policy_enabled": True},
        )
    )
    assert audit.status == "failed"
    assert audit.executed_wave_count != audit.planned_wave_count


def test_execution_audit_records_prepared_plan_lineage() -> None:
    plan = {
        "policy_name": "routersense_p0p1p2_hint",
        "plan_hash": "plan-1",
        "transport_mutation": True,
        "metrics": {
            "compiled_from_prepared_plan": True,
            "prepared_plan_order_preserved": True,
            "prepared_window_key": "window-1",
            "source_logical_plan_hash": "logical-1",
            "hint_edges_consumed": 1,
            "hint_match_rate": 1.0,
        },
        "waves": [{"wave_id": 0, "bucket_tasks": [{"task_id": "a", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16}]}],
    }
    audit = build_execution_audit(
        ExecutionAuditInput(
            execution_plan=plan,
            transport_events=(
                {"wave_id": 0, "task_id": "a", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
            ),
            phase_contract={"phase": "P1", "layer_id": "0", "policy_enabled": True},
        )
    )
    assert audit.status == "passed"
    assert audit.details["compiled_from_prepared_plan"] is True
    assert audit.details["prepared_window_key"] == "window-1"
    assert audit.details["source_logical_plan_hash"] == "logical-1"


def test_execution_audit_fails_when_prepared_plan_order_not_preserved() -> None:
    plan = {
        "policy_name": "routersense_p0p1p2_hint",
        "plan_hash": "plan-1",
        "transport_mutation": True,
        "metrics": {
            "compiled_from_prepared_plan": True,
            "prepared_plan_order_preserved": False,
        },
        "waves": [{"wave_id": 0, "bucket_tasks": [{"task_id": "a", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16}]}],
    }
    audit = build_execution_audit(
        ExecutionAuditInput(
            execution_plan=plan,
            transport_events=(
                {"wave_id": 0, "task_id": "a", "src_rank": 0, "dst_rank": 1, "row_count": 4, "byte_count": 16, "tensor_role": "hidden_states"},
            ),
            phase_contract={"phase": "P1", "layer_id": "0", "policy_enabled": True},
        )
    )
    assert audit.status == "failed"
    assert audit.details["prepared_plan_order_preserved"] is False
