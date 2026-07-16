"""Certified exact reference for tiny discrete bucket scheduling instances."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rs.core.contracts import PlanWave, PlannedFlow, WindowPlan
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem


MAX_RANK_COUNT = 4
MAX_BUCKET_TASK_COUNT = 12


@dataclass(frozen=True)
class UnsupportedExactSolve(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return f"exact small-instance solve unsupported: {self.reason}"


def solve_exact_small_instance(
    *,
    flows: tuple[FlowDemand, ...],
    rank_count: int,
    time_limit_ms: int = 5000,
) -> dict[str, Any]:
    if int(rank_count) > MAX_RANK_COUNT or len(flows) > MAX_BUCKET_TASK_COUNT:
        return {
            "reference_model": "discrete_bucket_phase_sync_wave",
            "supported": False,
            "solver_status": "unsupported_scale",
            "certified_optimal": False,
            "objective_logical_makespan": None,
            "wave_count": None,
            "best_bound": None,
            "optimality_gap": None,
            "search_nodes": 0,
            "time_limit_ms": int(time_limit_ms),
            "schedule": [],
        }
    canonical_flows = tuple(sorted(flows, key=lambda flow: (flow.phase, flow.src_rank, flow.dst_rank, flow.byte_count, flow.flow_id)))
    start_ns = time.monotonic_ns()
    deadline_ns = start_ns + int(time_limit_ms) * 1_000_000
    all_mask = (1 << len(canonical_flows)) - 1
    compatible_masks = _compatible_wave_masks(canonical_flows)
    search_nodes = 0

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, tuple[int, ...]] | None:
        nonlocal search_nodes
        search_nodes += 1
        if time.monotonic_ns() > deadline_ns:
            return None
        if mask == 0:
            return (0, ())
        best: tuple[int, tuple[int, ...]] | None = None
        sub = mask
        while sub:
            if sub in compatible_masks:
                duration = _mask_duration(canonical_flows, sub)
                remaining = solve(mask ^ sub)
                if remaining is None:
                    return None
                candidate = (duration + remaining[0], (sub, *remaining[1]))
                if best is None or _candidate_key(canonical_flows, candidate) < _candidate_key(canonical_flows, best):
                    best = candidate
            sub = (sub - 1) & mask
        return best

    result = solve(all_mask)
    if result is None:
        return {
            "reference_model": "discrete_bucket_phase_sync_wave",
            "supported": True,
            "solver_status": "time_limit",
            "certified_optimal": False,
            "objective_logical_makespan": None,
            "wave_count": None,
            "best_bound": None,
            "optimality_gap": None,
            "search_nodes": search_nodes,
            "time_limit_ms": int(time_limit_ms),
            "schedule": [],
        }
    objective, masks = result
    schedule = [_mask_to_wave_record(canonical_flows, wave_id, mask) for wave_id, mask in enumerate(masks)]
    return {
        "reference_model": "discrete_bucket_phase_sync_wave",
        "supported": True,
        "solver_status": "optimal",
        "certified_optimal": True,
        "objective_logical_makespan": int(objective),
        "wave_count": len(schedule),
        "best_bound": int(objective),
        "optimality_gap": 0,
        "search_nodes": int(search_nodes),
        "time_limit_ms": int(time_limit_ms),
        "schedule": schedule,
    }


def solve_problem_exact(problem: MultiPhaseSchedulingProblem, *, time_limit_ms: int = 5000) -> dict[str, Any]:
    flows = tuple(problem.flow_window.ready_flows + problem.flow_window.blocked_flows)
    return solve_exact_small_instance(flows=flows, rank_count=problem.topology.num_gpus, time_limit_ms=time_limit_ms)


def exact_result_to_logical_plan(result: dict[str, Any], *, policy_name: str = "exact_small_instance_reference") -> LogicalSchedulePlan:
    waves = []
    for row in result.get("schedule", []):
        flows = tuple(
            FlowDemand(
                flow_id=str(item["flow_id"]),
                phase=str(item["phase"]),
                src_rank=int(item["src_rank"]),
                dst_rank=int(item["dst_rank"]),
                byte_count=int(item["byte_count"]),
                release_state="ready",
                is_executable=True,
            )
            for item in row.get("flows", [])
        )
        waves.append(LogicalWave(wave_id=int(row["wave_id"]), flows=flows, duration=float(row["duration"])))
    return LogicalSchedulePlan(
        policy_name=policy_name,
        waves=tuple(waves),
        diagnostics={
            "policy_name": policy_name,
            "policy_version": "v1",
            "logical_model": "discrete_bucket_phase_sync_wave",
            "reference_result": result,
            "evaluation_eligible": True,
            "certified_optimal": bool(result.get("certified_optimal", False)),
        },
    )


def exact_result_to_window_plan(
    result: dict[str, Any],
    *,
    planner_id: str,
    planner_family: str,
    request_digest: str,
) -> WindowPlan:
    waves = []
    for row in result.get("schedule", []):
        flows = tuple(
            PlannedFlow(
                flow_id=str(item["flow_id"]),
                phase=str(item["phase"]),
                src_rank=int(item["src_rank"]),
                dst_rank=int(item["dst_rank"]),
                row_count=int(item["byte_count"]),
                release_state="ready",
                executable=True,
            )
            for item in row.get("flows", [])
        )
        waves.append(
            PlanWave(
                wave_id=int(row["wave_id"]),
                flows=flows,
                estimated_duration=float(row.get("duration", 0.0)),
            )
        )
    metadata = {
        "policy_name": str(planner_id),
        "policy_version": "v1",
        "logical_model": "discrete_bucket_phase_sync_wave",
        "reference_result": dict(result),
        "evaluation_eligible": True,
        "certified_optimal": bool(result.get("certified_optimal", False)),
    }
    return WindowPlan(
        planner_id=str(planner_id),
        planner_family=str(planner_family),
        request_digest=str(request_digest),
        waves=tuple(waves),
        metadata=metadata,
    )


def _compatible_wave_masks(flows: tuple[FlowDemand, ...]) -> set[int]:
    masks: set[int] = set()
    for mask in range(1, 1 << len(flows)):
        used_src: set[int] = set()
        used_dst: set[int] = set()
        ok = True
        for index, flow in enumerate(flows):
            if not (mask & (1 << index)):
                continue
            if flow.src_rank in used_src or flow.dst_rank in used_dst:
                ok = False
                break
            used_src.add(flow.src_rank)
            used_dst.add(flow.dst_rank)
        if ok:
            masks.add(mask)
    return masks


def _mask_duration(flows: tuple[FlowDemand, ...], mask: int) -> int:
    return max(int(flow.byte_count) for index, flow in enumerate(flows) if mask & (1 << index))


def _candidate_key(flows: tuple[FlowDemand, ...], candidate: tuple[int, tuple[int, ...]]) -> tuple[Any, ...]:
    objective, masks = candidate
    return (
        int(objective),
        len(masks),
        tuple(tuple(flow.flow_id for index, flow in enumerate(flows) if mask & (1 << index)) for mask in masks),
    )


def _mask_to_wave_record(flows: tuple[FlowDemand, ...], wave_id: int, mask: int) -> dict[str, Any]:
    selected = [flow for index, flow in enumerate(flows) if mask & (1 << index)]
    return {
        "wave_id": int(wave_id),
        "duration": _mask_duration(flows, mask),
        "flows": [
            {
                "flow_id": flow.flow_id,
                "phase": flow.phase,
                "src_rank": flow.src_rank,
                "dst_rank": flow.dst_rank,
                "byte_count": flow.byte_count,
            }
            for flow in selected
        ],
    }


__all__ = [
    "MAX_BUCKET_TASK_COUNT",
    "MAX_RANK_COUNT",
    "UnsupportedExactSolve",
    "exact_result_to_logical_plan",
    "exact_result_to_window_plan",
    "solve_exact_small_instance",
    "solve_problem_exact",
]
