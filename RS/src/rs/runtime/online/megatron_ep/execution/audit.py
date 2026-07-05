"""Execution audit for scheduled online phase execution."""

from __future__ import annotations

from collections import Counter
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

    planned_tasks = [
        task
        for wave in plan_dict.get("waves", [])
        for task in wave.get("bucket_tasks", []) or []
    ]
    planned_task_ids = tuple(str(task.get("task_id", "")) for task in planned_tasks)
    planned_rows = sum(int(task.get("row_count", 0)) for task in planned_tasks)
    planned_bytes = 0
    for task in planned_tasks:
        payload_slices = task.get("payload_slices", []) or []
        if payload_slices:
            planned_bytes += sum(int(payload.get("payload_byte_count", 0)) for payload in payload_slices)
        else:
            planned_bytes += int(task.get("byte_count", 0))
    planned_wave_count = len(plan_dict.get("waves", []))
    planned_local_rows = sum(int(task.get("row_count", 0)) for task in planned_tasks if int(task.get("src_rank", -1)) == int(task.get("dst_rank", -2)))
    planned_remote_rows = planned_rows - planned_local_rows

    transport_events = tuple(audit_input.transport_events)
    native_fallback_events = sum(1 for row in transport_events if row.get("event") == "native_fallback")
    contract_violation_events = sum(1 for row in transport_events if row.get("event") == "contract_violation")
    execution_entries = [row for row in transport_events if row.get("record_type") != "result_summary"]
    raw_executed_task_ids = tuple(
        str(row.get("task_id") or row.get("bucket_id") or "")
        for row in execution_entries
        if row.get("task_id") or row.get("bucket_id")
    )
    if phase == "P0":
        logical_executed: list[str] = []
        for task_id in raw_executed_task_ids:
            if task_id and task_id not in logical_executed:
                logical_executed.append(task_id)
        executed_task_ids = tuple(logical_executed)
    else:
        executed_task_ids = raw_executed_task_ids

    unique_task_rows: dict[str, int] = {}
    unique_task_bytes: dict[str, int] = {}
    for row in execution_entries:
        task_id = str(row.get("task_id") or row.get("bucket_id") or "")
        if not task_id:
            continue
        unique_task_rows[task_id] = max(unique_task_rows.get(task_id, 0), int(row.get("row_count", 0)))
        unique_task_bytes[task_id] = unique_task_bytes.get(task_id, 0) + int(row.get("byte_count", 0))
    executed_rows = sum(unique_task_rows.values())
    executed_bytes = sum(unique_task_bytes.values())
    executed_wave_count = len({int(row.get("wave_id", -1)) for row in execution_entries if "wave_id" in row})
    actual_local_rows = 0
    actual_remote_rows = 0
    classified_tasks: set[str] = set()
    for row in execution_entries:
        task_id = str(row.get("task_id") or row.get("bucket_id") or "")
        if not task_id or task_id in classified_tasks:
            continue
        classified_tasks.add(task_id)
        if int(row.get("src_rank", -1)) == int(row.get("dst_rank", -2)):
            actual_local_rows += int(row.get("row_count", 0))
        else:
            actual_remote_rows += int(row.get("row_count", 0))

    planned_counter = Counter(planned_task_ids)
    executed_counter = Counter(executed_task_ids)
    missing = tuple(sorted(task_id for task_id, count in (planned_counter - executed_counter).items() for _ in range(count)))
    unexpected = tuple(sorted(task_id for task_id, count in (executed_counter - planned_counter).items() for _ in range(count)))
    duplicate = tuple(sorted(task_id for task_id, count in executed_counter.items() if count > planned_counter.get(task_id, 0)))

    order_mismatches: list[str] = []
    for index, task_id in enumerate(executed_task_ids):
        if index >= len(planned_task_ids):
            break
        if task_id != planned_task_ids[index]:
            order_mismatches.append(f"{index}:{planned_task_ids[index]}!={task_id}")

    p0_bundle_atomicity_preserved = True
    if phase == "P0":
        grouped: dict[tuple[int, str], list[str]] = {}
        for row in execution_entries:
            key = (int(row.get("wave_id", -1)), str(row.get("task_id") or row.get("bucket_id") or ""))
            grouped.setdefault(key, []).append(str(row.get("tensor_role", "")))
        p0_bundle_atomicity_preserved = all(tuple(roles) == ("hidden_states", "routing_probs") for roles in grouped.values())

    local_copy_coverage_passed = planned_local_rows == actual_local_rows
    remote_flow_coverage_passed = planned_remote_rows == actual_remote_rows

    status = "passed"
    if (
        planned_wave_count != executed_wave_count
        or missing
        or unexpected
        or duplicate
        or order_mismatches
        or planned_rows != executed_rows
        or planned_bytes != executed_bytes
        or not local_copy_coverage_passed
        or not remote_flow_coverage_passed
        or not p0_bundle_atomicity_preserved
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
            "planned_local_rows": planned_local_rows,
            "actual_local_rows": actual_local_rows,
            "planned_remote_rows": planned_remote_rows,
            "actual_remote_rows": actual_remote_rows,
        },
    )


__all__ = ["ExecutionAuditInput", "build_execution_audit"]
