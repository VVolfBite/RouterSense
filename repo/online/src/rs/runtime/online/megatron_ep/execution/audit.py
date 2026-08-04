"""在线 phase 执行后的审计逻辑。

主要入口：
- build_execution_audit()
用于比较“计划里要做什么”和“实际执行了什么”。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from rs.runtime.online.megatron_ep.observation import ExecutionAudit
from rs.scheduling.phase_execution import PhaseExecutionPlan


@dataclass(frozen=True)
class ExecutionAuditInput:
    execution_plan: PhaseExecutionPlan | dict[str, Any] | None
    transport_events: tuple[dict[str, Any], ...]
    phase_contract: dict[str, Any]


def build_execution_audit(audit_input: ExecutionAuditInput) -> ExecutionAudit:
    plan = audit_input.execution_plan
    plan_dict = plan.to_dict() if isinstance(plan, PhaseExecutionPlan) else plan
    phase = str(audit_input.phase_contract.get("phase", "unknown"))
    layer_id = str(audit_input.phase_contract.get("layer_id", "unknown"))
    policy_enabled = bool(audit_input.phase_contract.get("policy_enabled", False))
    if not policy_enabled or plan_dict is None or not bool(plan_dict.get("transport_mutation", False)):
        return ExecutionAudit(
            status="not_applicable",
            policy_name=(plan_dict or {}).get("policy_name", "disabled"),
            plan_hash=(plan_dict or {}).get("plan_hash", ""),
            phase=phase,
            layer_id=layer_id,
            planned_wave_count=len((plan_dict or {}).get("waves", [])),
            executed_wave_count=0,
        )

    planned_waves = tuple(plan_dict.get("waves", []) or [])
    planned_tasks = [task for wave in planned_waves for task in wave.get("bucket_tasks", []) or []]
    if planned_tasks:
        planned_task_ids = tuple(str(task.get("task_id", "")) for task in planned_tasks if str(task.get("task_id", "")))
    else:
        planned_task_ids = tuple(
            str(task_id)
            for wave in planned_waves
            for task_id in (wave.get("task_ids", []) or [])
            if str(task_id)
        )
    planned_rows = sum(int(task.get("row_count", 0)) for task in planned_tasks)
    planned_bytes = 0
    planned_bytes_source = "task_payloads"
    for task in planned_tasks:
        payload_slices = task.get("payload_slices", []) or []
        if payload_slices:
            planned_bytes += sum(int(payload.get("payload_byte_count", 0)) for payload in payload_slices)
        else:
            planned_bytes += int(task.get("byte_count", 0))
    if not planned_tasks:
        planned_bytes_source = "not_recorded_in_perf"
    planned_wave_count_total = len(planned_waves)
    planned_wave_count = sum(1 for wave in planned_waves if (wave.get("bucket_tasks") or wave.get("task_ids")))
    planned_local_rows = sum(int(task.get("row_count", 0)) for task in planned_tasks if int(task.get("src_rank", -1)) == int(task.get("dst_rank", -2)))
    planned_remote_rows = planned_rows - planned_local_rows

    transport_events = tuple(audit_input.transport_events)
    native_fallback_events = sum(1 for row in transport_events if row.get("event") == "native_fallback")
    contract_violation_events = sum(1 for row in transport_events if row.get("event") == "contract_violation")
    result_summaries = [row for row in transport_events if row.get("record_type") == "result_summary"]
    execution_entries = [row for row in transport_events if row.get("record_type") != "result_summary"]
    task_entries = [row for row in execution_entries if row.get("record_type") in {None, "task"}]
    logical_task_entries = [row for row in task_entries if "op_kind" not in row]
    audited_task_entries = logical_task_entries if logical_task_entries else task_entries
    task_id_none_event_count = 0
    raw_executed_task_ids: list[str] = []
    payload_roles_by_task: dict[str, set[str]] = defaultdict(set)
    duplicate_payload_keys: list[tuple[str, str]] = []
    seen_payload_keys: set[tuple[str, str]] = set()
    for row in audited_task_entries:
        task_id = str(row.get("task_id") or row.get("bucket_id") or "")
        if not task_id:
            task_id_none_event_count += 1
            continue
        raw_executed_task_ids.append(task_id)
        tensor_role = str(row.get("tensor_role") or "__none__")
        payload_roles_by_task[task_id].add(tensor_role)
        payload_key = (task_id, tensor_role)
        if payload_key in seen_payload_keys:
            duplicate_payload_keys.append(payload_key)
        else:
            seen_payload_keys.add(payload_key)
    logical_executed: list[str] = []
    for task_id in raw_executed_task_ids:
        if task_id not in logical_executed:
            logical_executed.append(task_id)
    executed_task_ids = tuple(logical_executed)
    execution_summary_used = False
    if not executed_task_ids and planned_task_ids and result_summaries:
        summary_complete = all(
            bool(row.get("all_work_completed", row.get("result", {}).get("all_work_completed", True)))
            and not bool(row.get("fallback_used", False))
            and not bool(row.get("timeout", False))
            for row in result_summaries
        )
        if summary_complete:
            executed_task_ids = planned_task_ids
            execution_summary_used = True

    unique_task_rows: dict[str, int] = {}
    unique_task_bytes: dict[str, int] = {}
    for row in audited_task_entries:
        task_id = str(row.get("task_id") or row.get("bucket_id") or "")
        if not task_id:
            continue
        unique_task_rows[task_id] = max(unique_task_rows.get(task_id, 0), int(row.get("row_count", 0)))
        unique_task_bytes[task_id] = unique_task_bytes.get(task_id, 0) + int(row.get("byte_count", 0))
    executed_rows = sum(unique_task_rows.values())
    executed_bytes = sum(unique_task_bytes.values())
    executed_wave_count = len(
        {
            int(row.get("wave_id", -1))
            for row in audited_task_entries
            if "wave_id" in row and (row.get("task_id") or row.get("bucket_id"))
        }
    )
    if execution_summary_used:
        executed_wave_count = max(
            int((row.get("result") or {}).get("active_wave_count", row.get("active_wave_count", 0)) or 0)
            for row in result_summaries
        )
        executed_rows = planned_rows
        executed_bytes = planned_bytes
    actual_local_rows = 0
    actual_remote_rows = 0
    classified_tasks: set[str] = set()
    for row in audited_task_entries:
        task_id = str(row.get("task_id") or row.get("bucket_id") or "")
        if not task_id or task_id in classified_tasks:
            continue
        classified_tasks.add(task_id)
        if int(row.get("src_rank", -1)) == int(row.get("dst_rank", -2)):
            actual_local_rows += int(row.get("row_count", 0))
        else:
            actual_remote_rows += int(row.get("row_count", 0))

    planned_set = set(planned_task_ids)
    executed_set = set(executed_task_ids)
    missing = tuple(sorted(planned_set - executed_set))
    unexpected = tuple(sorted(executed_set - planned_set))
    duplicate = tuple(sorted({task_id for task_id, _ in duplicate_payload_keys}))

    order_mismatches: list[str] = []
    for index, task_id in enumerate(executed_task_ids):
        if index >= len(planned_task_ids):
            break
        if task_id != planned_task_ids[index]:
            order_mismatches.append(f"{index}:{planned_task_ids[index]}!={task_id}")

    p0_bundle_atomicity_preserved = True
    if phase == "P0":
        grouped: dict[tuple[int, str], list[str]] = {}
        for row in audited_task_entries:
            key = (int(row.get("wave_id", -1)), str(row.get("task_id") or row.get("bucket_id") or ""))
            grouped.setdefault(key, []).append(str(row.get("tensor_role", "")))
        p0_bundle_atomicity_preserved = all(tuple(roles) == ("hidden_states", "routing_probs") for roles in grouped.values())

    local_copy_coverage_passed = planned_local_rows == actual_local_rows if planned_tasks else True
    remote_flow_coverage_passed = not missing and not unexpected
    metrics = plan_dict.get("metrics", {}) or {}
    compiled_from_prepared_plan = bool(metrics.get("compiled_from_prepared_plan", False))
    prepared_plan_order_preserved_metric = bool(metrics.get("prepared_plan_order_preserved", False)) if compiled_from_prepared_plan else True
    prepared_plan_order_preserved = bool(prepared_plan_order_preserved_metric and not order_mismatches)
    p2_matrix_source = str(metrics.get("pending_window_p2_matrix_source", metrics.get("p2_matrix_source", "")))
    p2_matrix_is_replicated_local_row = bool(metrics.get("p2_matrix_is_replicated_local_row", False))
    multi_payload_task_count = sum(1 for roles in payload_roles_by_task.values() if len(roles) > 1)
    payload_roles_by_first_5_task_ids = {
        task_id: sorted(payload_roles_by_task.get(task_id, set()))
        for task_id in list(executed_task_ids)[:5]
    }

    status = "passed"
    if (
        planned_wave_count != executed_wave_count
        or missing
        or unexpected
        or duplicate
        or order_mismatches
        or (planned_tasks and planned_rows != executed_rows)
        or (planned_tasks and planned_bytes != executed_bytes)
        or not local_copy_coverage_passed
        or not remote_flow_coverage_passed
        or not p0_bundle_atomicity_preserved
        or not prepared_plan_order_preserved
        or native_fallback_events
        or contract_violation_events
    ):
        status = "failed"

    return ExecutionAudit(
        status=status,
        policy_name=str(plan_dict.get("policy_name", "")),
        plan_hash=str(plan_dict.get("plan_hash", "")),
        phase=phase,
        layer_id=layer_id,
        planned_wave_count=planned_wave_count,
        executed_wave_count=executed_wave_count,
        planned_task_ids=planned_task_ids,
        executed_task_ids=executed_task_ids,
        missing_tasks=missing,
        unexpected_tasks=unexpected,
        duplicate_tasks=duplicate,
        order_mismatches=tuple(order_mismatches),
        planned_rows=planned_rows,
        actual_rows=executed_rows,
        planned_bytes=planned_bytes,
        actual_bytes=executed_bytes,
        native_fallback_events=native_fallback_events,
        contract_violation_events=contract_violation_events,
        p0_bundle_atomicity_preserved=p0_bundle_atomicity_preserved,
        local_copy_coverage_passed=local_copy_coverage_passed,
        remote_flow_coverage_passed=remote_flow_coverage_passed,
        details={
            "planned_execution_order": planned_task_ids,
            "actual_execution_order": executed_task_ids,
            "planned_task_count": len(planned_task_ids),
            "executed_task_count": len(executed_task_ids),
            "planned_wave_count_total": planned_wave_count_total,
            "planned_local_rows": planned_local_rows,
            "actual_local_rows": actual_local_rows,
            "planned_remote_rows": planned_remote_rows,
            "actual_remote_rows": actual_remote_rows,
            "planned_bytes_source": planned_bytes_source,
            "first_10_planned_task_ids": list(planned_task_ids[:10]),
            "first_10_executed_task_ids": list(executed_task_ids[:10]),
            "task_id_none_event_count": task_id_none_event_count,
            "multi_payload_task_count": multi_payload_task_count,
            "payload_roles_by_first_5_task_ids": payload_roles_by_first_5_task_ids,
            "audited_task_entry_source": "logical_task_entries" if logical_task_entries else "task_entries",
            "execution_summary_used": execution_summary_used,
            "compiled_from_prepared_plan": compiled_from_prepared_plan,
            "prepared_plan_order_preserved": prepared_plan_order_preserved,
            "prepared_plan_order_preserved_metric": prepared_plan_order_preserved_metric,
            "prepared_window_key": str(metrics.get("prepared_window_key", "")),
            "source_logical_plan_hash": str(metrics.get("source_logical_plan_hash", "")),
            "hint_edges_consumed": int(metrics.get("hint_edges_consumed", 0) or 0),
            "hint_match_rate": float(metrics.get("hint_match_rate", 0.0) or 0.0),
            "p2_matrix_source": p2_matrix_source,
            "p2_matrix_is_replicated_local_row": p2_matrix_is_replicated_local_row,
        },
    )


__all__ = ["ExecutionAuditInput", "build_execution_audit"]
