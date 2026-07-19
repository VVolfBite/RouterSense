from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from rs.scheduling.contracts import LogicalSchedulePlan, MultiPhaseSchedulingProblem
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext
from rs.scheduling.validation import stable_hash

from ..capabilities import PolicyCapabilities
from .common import (
    build_phase_serial_release_aware_plan,
    build_logical_plan_from_order,
    build_transfer_layouts_and_tasks,
    finalize_execution_plan,
    flows_from_matrix,
    include_real_p2_phase,
    pack_phase_tasks,
)


def _two_choice_hash(*, step: int, item_id: str) -> str:
    return stable_hash({"step": int(step), "item_id": str(item_id)})


def _pending_endpoint_pressure(items: list[Any]) -> tuple[dict[int, int], dict[int, int]]:
    outgoing = defaultdict(int)
    incoming = defaultdict(int)
    for item in items:
        outgoing[int(item.src_rank)] += int(item.byte_count)
        incoming[int(item.dst_rank)] += int(item.byte_count)
    return dict(outgoing), dict(incoming)


def _least_loaded_choice_key(item: Any, *, outgoing: dict[int, int], incoming: dict[int, int]) -> tuple[int, int, int, int, str]:
    src_rank = int(item.src_rank)
    dst_rank = int(item.dst_rank)
    src_pressure = int(outgoing.get(src_rank, 0))
    dst_pressure = int(incoming.get(dst_rank, 0))
    return (
        min(src_pressure, dst_pressure),
        src_pressure + dst_pressure,
        src_rank,
        dst_rank,
        str(getattr(item, "flow_id", getattr(item, "task_id", ""))),
    )


def _power_of_two_order(items: list[Any]) -> list[Any]:
    pending = list(items)
    ordered: list[Any] = []
    step = 0
    while pending:
        outgoing, incoming = _pending_endpoint_pressure(pending)
        sampled = sorted(
            pending,
            key=lambda item: _two_choice_hash(
                step=step,
                item_id=str(getattr(item, "flow_id", getattr(item, "task_id", ""))),
            ),
        )[:2]
        if not sampled:
            raise ValueError("power_of_two_choices encountered empty candidate set")
        chosen = min(
            sampled,
            key=lambda item: _least_loaded_choice_key(item, outgoing=outgoing, incoming=incoming),
        )
        ordered.append(chosen)
        pending.remove(chosen)
        step += 1
    return ordered


class PowerOfTwoChoicesPolicy:
    policy_name = "power_of_two_choices"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=True,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=False,
        uses_p2_forecast=False,
        requires_fixed_placement=False,
        evaluation_eligible=True,
    )

    def __init__(self, *, bucket_rows: int) -> None:
        self.bucket_rows = int(bucket_rows)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        ordered_p0 = _power_of_two_order(
            list(
                flows_from_matrix(
                    problem.p0_dispatch_matrix,
                    phase="p0_dispatch",
                    release_state="ready",
                    executable=True,
                )
            )
        )
        ordered_p1 = _power_of_two_order(
            list(
                flows_from_matrix(
                    problem.p1_return_matrix,
                    phase="p1_return",
                    release_state="ready",
                    executable=True,
                )
            )
        )
        ordered_p2 = []
        if include_real_p2_phase(problem):
            ordered_p2 = _power_of_two_order(
                list(
                    flows_from_matrix(
                        problem.p2_next_dispatch_forecast_matrix,
                        phase="p2_next_dispatch",
                        release_state="ready",
                        executable=True,
                    )
                )
            )
        base_plan = build_logical_plan_from_order(
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            ordered_p0=ordered_p0,
            ordered_p1=ordered_p1,
            ordered_p2=ordered_p2,
            information_mode="load_balance_power_of_two",
            p2_source=problem.forecast.source if problem.forecast is not None else "none",
            evaluation_eligible=True,
            priority_components={"selection_rule": "deterministic power-of-two least-loaded endpoint choice"},
            tie_break_rule="min(endpoint_pressure), sum(endpoint_pressure), src_rank, dst_rank, flow_id",
        )
        p0_waves = tuple(wave for wave in base_plan.waves if all(flow.phase == "p0_dispatch" for flow in wave.flows))
        p1_waves = tuple(wave for wave in base_plan.waves if all(flow.phase == "p1_return" for flow in wave.flows))
        p2_waves = tuple(wave for wave in base_plan.waves if all(flow.phase == "p2_next_dispatch" for flow in wave.flows))
        return build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="load_balance_power_of_two",
            tie_break_rule="min(endpoint_pressure), sum(endpoint_pressure), src_rank, dst_rank, flow_id",
            priority_components={"selection_rule": "deterministic power-of-two least-loaded endpoint choice"},
            p0_waves=p0_waves,
            p1_waves=p1_waves,
            p2_waves=p2_waves,
            service_model="phase_serial_atomic",
        )

    def build_plan(
        self,
        *,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> PhaseExecutionPlan:
        transfer_layouts, all_tasks, build_stats = build_transfer_layouts_and_tasks(
            local_context=local_context,
            global_contexts=global_contexts,
            bucket_rows=self.bucket_rows,
            return_stats=True,
        )
        sort_started_ns = time.perf_counter_ns()
        ordered_tasks = _power_of_two_order(all_tasks)
        sort_time_us = (time.perf_counter_ns() - sort_started_ns) / 1000.0
        waves, pack_stats = pack_phase_tasks(ordered_tasks, phase=local_context.phase, return_stats=True)
        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "information_mode": "load_balance_power_of_two",
            "bucket_order": [task.task_id for task in ordered_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": [float(sum(int(task.byte_count) for task in wave.bucket_tasks)) for wave in waves],
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "p1_dependency_used": False,
            "p2_forecast_used": False,
            "p2_source": local_context.p2_hint.hint_source,
            "evaluation_eligible": True,
            "priority_components": {"selection_rule": "deterministic power-of-two least-loaded endpoint choice"},
            "tie_break_rule": "min(endpoint_pressure), sum(endpoint_pressure), src_rank, dst_rank, task_id",
            "fallback_reason": "",
        }
        return finalize_execution_plan(
            local_context=local_context,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            transfer_layouts=transfer_layouts,
            all_tasks=ordered_tasks,
            waves=waves,
            diagnostics=diagnostics,
            timing_metrics={
                **build_stats,
                **pack_stats,
                "sort_tasks_time_us": sort_time_us,
            },
        )
