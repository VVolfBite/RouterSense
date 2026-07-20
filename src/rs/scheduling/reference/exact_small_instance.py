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

EXACT_REFERENCE_MODEL_ID = "routersense_exact_bucket_wave_release_v2"
EXACT_SINGLE_PHASE_MODEL_ID = "routersense_exact_bucket_wave_single_phase_v2"
EXACT_TASK_MODEL_ID = "canonical_remote_edge_bucket_v1"
EXACT_COST_MODEL_ID = "full_duplex_matching_wave_max_rows_v1"
EXACT_RELEASE_MODEL_ID = "rank_local_p0_to_p1_to_p2_v1"


def exact_model_contract(*, scope: str) -> dict[str, Any]:
    normalized_scope = str(scope)
    if normalized_scope not in {"local", "joint", "single_phase"}:
        raise ValueError(f"unsupported exact model scope {scope!r}")
    return {
        "reference_model": EXACT_SINGLE_PHASE_MODEL_ID if normalized_scope == "single_phase" else EXACT_REFERENCE_MODEL_ID,
        "task_model_id": EXACT_TASK_MODEL_ID,
        "cost_model_id": EXACT_COST_MODEL_ID,
        "release_model_id": EXACT_RELEASE_MODEL_ID,
        "scope": normalized_scope,
        "scope_only_difference": normalized_scope in {"local", "joint"},
    }


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
            **exact_model_contract(scope="single_phase"),
            "supported": False,
            "solver_status": "unsupported_scale",
            "certified_optimal": False,
            "objective_logical_makespan": None,
            "wave_count": None,
            "best_bound": None,
            "optimality_gap": None,
            "search_nodes": 0,
            "task_count": int(len(flows)),
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
            **exact_model_contract(scope="single_phase"),
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
        **exact_model_contract(scope="single_phase"),
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
    return solve_problem_exact_with_scope(problem, time_limit_ms=time_limit_ms, scope="joint")


def solve_problem_exact_with_scope(
    problem: MultiPhaseSchedulingProblem,
    *,
    time_limit_ms: int = 5000,
    scope: str = "joint",
) -> dict[str, Any]:
    normalized_scope = str(scope)
    if normalized_scope not in {"joint", "local"}:
        raise ValueError(f"unsupported exact scope {scope!r}")
    started_ns = time.monotonic_ns()
    rank_count = int(problem.topology.num_gpus)
    phase_flows = _phase_grouped_flows(problem)
    total_flow_count = sum(len(items) for items in phase_flows.values())
    if rank_count > MAX_RANK_COUNT or total_flow_count > MAX_BUCKET_TASK_COUNT:
        return {
            **exact_model_contract(scope=normalized_scope),
            "bucket_rows": int(problem.options.bucket_rows),
            "supported": False,
            "solver_status": "unsupported_scale",
            "certified_optimal": False,
            "objective_logical_makespan": None,
            "wave_count": None,
            "best_bound": None,
            "optimality_gap": None,
            "search_nodes": 0,
            "time_limit_ms": int(time_limit_ms),
            "solver_runtime_ms_wall": (time.monotonic_ns() - started_ns) / 1_000_000.0,
            "schedule": [],
            "scope": normalized_scope,
        }
    if normalized_scope == "local":
        result = _solve_problem_exact_local(
            problem,
            phase_flows=phase_flows,
            time_limit_ms=time_limit_ms,
            started_ns=started_ns,
        )
    else:
        result = _solve_problem_exact_joint(
            problem,
            phase_flows=phase_flows,
            time_limit_ms=time_limit_ms,
            started_ns=started_ns,
        )
    return {
        **result,
        "task_count": int(total_flow_count),
        "bucket_rows": int(problem.options.bucket_rows),
    }


def _bucketize_flows(flows: tuple[FlowDemand, ...], *, bucket_rows: int) -> tuple[FlowDemand, ...]:
    normalized_bucket_rows = int(bucket_rows)
    bucketed: list[FlowDemand] = []
    for flow in flows:
        total = int(flow.byte_count)
        if total <= 0:
            continue
        step = total if normalized_bucket_rows <= 0 else normalized_bucket_rows
        offset = 0
        ordinal = 0
        while offset < total:
            current = min(step, total - offset)
            bucketed.append(
                FlowDemand(
                    flow_id=f"{flow.flow_id}:bucket:{ordinal}",
                    phase=str(flow.phase),
                    src_rank=int(flow.src_rank),
                    dst_rank=int(flow.dst_rank),
                    byte_count=int(current),
                    release_state=str(flow.release_state),
                    is_executable=bool(flow.is_executable),
                    dependency_metadata={
                        **dict(flow.dependency_metadata),
                        "origin_flow_id": str(flow.flow_id),
                        "row_offset": int(offset),
                        "bucket_ordinal": int(ordinal),
                        "bucket_rows": int(normalized_bucket_rows),
                    },
                )
            )
            offset += current
            ordinal += 1
    return tuple(bucketed)


def _phase_grouped_flows(problem: MultiPhaseSchedulingProblem) -> dict[str, tuple[FlowDemand, ...]]:
    bucket_rows = int(problem.options.bucket_rows)
    p0 = _bucketize_flows(tuple(problem.flow_window.ready_flows), bucket_rows=bucket_rows)
    p1 = _bucketize_flows(tuple(problem.flow_window.blocked_flows), bucket_rows=bucket_rows)
    p2 = _bucketize_flows(
        tuple(
            flow
            for flow in problem.flow_window.forecast_pressure
            if bool(flow.is_executable) and str(flow.release_state) != "advisory_only"
        ),
        bucket_rows=bucket_rows,
    )
    return {"p0": p0, "p1": p1, "p2": p2}


def _solve_problem_exact_local(
    problem: MultiPhaseSchedulingProblem,
    *,
    phase_flows: dict[str, tuple[FlowDemand, ...]],
    time_limit_ms: int,
    started_ns: int,
) -> dict[str, Any]:
    rank_count = int(problem.topology.num_gpus)
    phase_results: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    wave_offset = 0
    search_nodes = 0
    elapsed_wall_ms = 0.0
    for phase_name in ("p0", "p1", "p2"):
        flows = phase_flows[phase_name]
        if not flows:
            continue
        elapsed_total_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
        remaining_time_ms = max(0, int(float(time_limit_ms) - elapsed_total_ms))
        if remaining_time_ms <= 0:
            return {
                **exact_model_contract(scope="local"),
                "bucket_rows": int(problem.options.bucket_rows),
                "supported": True,
                "solver_status": "time_limit",
                "certified_optimal": False,
                "objective_logical_makespan": None,
                "wave_count": None,
                "best_bound": None,
                "optimality_gap": None,
                "search_nodes": int(search_nodes),
                "time_limit_ms": int(time_limit_ms),
                "solver_runtime_ms_wall": (time.monotonic_ns() - started_ns) / 1_000_000.0,
                "schedule": [],
                "scope": "local",
                "phase_solver_results": phase_results,
                "combined_validation": {"valid": False, "reason": "shared_time_budget_exhausted"},
            }
        phase_started_ns = time.monotonic_ns()
        phase_result = solve_exact_small_instance(
            flows=flows,
            rank_count=rank_count,
            time_limit_ms=remaining_time_ms,
        )
        phase_wall_ms = (time.monotonic_ns() - phase_started_ns) / 1_000_000.0
        phase_result = {**phase_result, "solver_runtime_ms_wall": phase_wall_ms, "phase_name": phase_name}
        phase_results.append(phase_result)
        elapsed_wall_ms += phase_wall_ms
        search_nodes += int(phase_result.get("search_nodes", 0) or 0)
        if (
            not bool(phase_result.get("supported", False))
            or str(phase_result.get("solver_status")) != "optimal"
            or not bool(phase_result.get("certified_optimal", False))
            or phase_result.get("objective_logical_makespan") is None
        ):
            return {
                **exact_model_contract(scope="local"),
                "bucket_rows": int(problem.options.bucket_rows),
                "supported": bool(phase_result.get("supported", False)),
                "solver_status": str(phase_result.get("solver_status", "unknown")),
                "certified_optimal": False,
                "objective_logical_makespan": None,
                "wave_count": None,
                "best_bound": None,
                "optimality_gap": None,
                "search_nodes": int(search_nodes),
                "time_limit_ms": int(time_limit_ms),
                "solver_runtime_ms_wall": (time.monotonic_ns() - started_ns) / 1_000_000.0,
                "schedule": [],
                "scope": "local",
                "phase_solver_results": phase_results,
                "combined_validation": {"valid": False, "reason": "phase_exact_not_optimal"},
            }
        for wave in phase_result.get("schedule", []):
            schedule.append({**dict(wave), "wave_id": int(wave_offset + int(wave["wave_id"]))})
        wave_offset += len(tuple(phase_result.get("schedule", ())))
    objective = float(sum(float(wave.get("duration", 0.0) or 0.0) for wave in schedule))
    return {
        **exact_model_contract(scope="local"),
        "bucket_rows": int(problem.options.bucket_rows),
        "supported": True,
        "solver_status": "optimal",
        "certified_optimal": True,
        "objective_logical_makespan": objective,
        "wave_count": len(schedule),
        "best_bound": objective,
        "optimality_gap": 0.0,
        "search_nodes": int(search_nodes),
        "time_limit_ms": int(time_limit_ms),
        "solver_runtime_ms_wall": (time.monotonic_ns() - started_ns) / 1_000_000.0,
        "schedule": schedule,
        "scope": "local",
        "phase_solver_results": phase_results,
        "combined_validation": {"valid": True, "objective": objective},
    }


def _solve_problem_exact_joint(
    problem: MultiPhaseSchedulingProblem,
    *,
    phase_flows: dict[str, tuple[FlowDemand, ...]],
    time_limit_ms: int,
    started_ns: int,
) -> dict[str, Any]:
    phase_order = {"p0_dispatch": 0, "p1_return": 1, "p2_next_dispatch": 2}
    flows = tuple(
        sorted(
            phase_flows["p0"] + phase_flows["p1"] + phase_flows["p2"],
            key=lambda flow: (
                phase_order.get(str(flow.phase), 99),
                int(flow.src_rank),
                int(flow.dst_rank),
                int(flow.byte_count),
                str(flow.flow_id),
            ),
        )
    )
    if not flows:
        return {
            **exact_model_contract(scope="joint"),
            "bucket_rows": int(problem.options.bucket_rows),
            "supported": True,
            "solver_status": "optimal",
            "certified_optimal": True,
            "objective_logical_makespan": 0.0,
            "wave_count": 0,
            "best_bound": 0.0,
            "optimality_gap": 0.0,
            "search_nodes": 1,
            "time_limit_ms": int(time_limit_ms),
            "solver_runtime_ms_wall": (time.monotonic_ns() - started_ns) / 1_000_000.0,
            "schedule": [],
            "scope": "joint",
        }
    deadline_ns = started_ns + int(time_limit_ms) * 1_000_000
    all_mask = (1 << len(flows)) - 1
    search_nodes = 0

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, tuple[int, ...]] | None:
        nonlocal search_nodes
        search_nodes += 1
        if time.monotonic_ns() > deadline_ns:
            return None
        if mask == 0:
            return (0, ())
        available_indices = _available_flow_indices(flows, mask)
        if not available_indices:
            return None
        compatible_masks = _compatible_wave_masks_subset(flows, mask, available_indices)
        best: tuple[int, tuple[int, ...]] | None = None
        for sub in compatible_masks:
            duration = _mask_duration(flows, sub)
            remaining = solve(mask ^ sub)
            if remaining is None:
                return None
            candidate = (duration + remaining[0], (sub, *remaining[1]))
            if best is None or _candidate_key(flows, candidate) < _candidate_key(flows, best):
                best = candidate
        return best

    result = solve(all_mask)
    if result is None:
        return {
            **exact_model_contract(scope="joint"),
            "bucket_rows": int(problem.options.bucket_rows),
            "supported": True,
            "solver_status": "time_limit",
            "certified_optimal": False,
            "objective_logical_makespan": None,
            "wave_count": None,
            "best_bound": None,
            "optimality_gap": None,
            "search_nodes": int(search_nodes),
            "time_limit_ms": int(time_limit_ms),
            "solver_runtime_ms_wall": (time.monotonic_ns() - started_ns) / 1_000_000.0,
            "schedule": [],
            "scope": "joint",
        }
    objective, masks = result
    schedule = [_mask_to_wave_record(flows, wave_id, mask) for wave_id, mask in enumerate(masks)]
    return {
        **exact_model_contract(scope="joint"),
        "bucket_rows": int(problem.options.bucket_rows),
        "supported": True,
        "solver_status": "optimal",
        "certified_optimal": True,
        "objective_logical_makespan": float(objective),
        "wave_count": len(schedule),
        "best_bound": float(objective),
        "optimality_gap": 0.0,
        "search_nodes": int(search_nodes),
        "time_limit_ms": int(time_limit_ms),
        "solver_runtime_ms_wall": (time.monotonic_ns() - started_ns) / 1_000_000.0,
        "schedule": schedule,
        "scope": "joint",
    }


def _available_flow_indices(flows: tuple[FlowDemand, ...], mask: int) -> tuple[int, ...]:
    remaining = [flows[index] for index in range(len(flows)) if mask & (1 << index)]
    remaining_p0_by_dst = {int(flow.dst_rank) for flow in remaining if str(flow.phase) == "p0_dispatch"}
    remaining_p1_by_dst = {int(flow.dst_rank) for flow in remaining if str(flow.phase) == "p1_return"}
    available: list[int] = []
    for index, flow in enumerate(flows):
        if not (mask & (1 << index)):
            continue
        phase = str(flow.phase)
        if phase == "p0_dispatch":
            available.append(index)
            continue
        if phase == "p1_return" and int(flow.src_rank) not in remaining_p0_by_dst:
            available.append(index)
            continue
        if phase == "p2_next_dispatch" and int(flow.src_rank) not in remaining_p1_by_dst:
            available.append(index)
            continue
    return tuple(available)


def _compatible_wave_masks_subset(flows: tuple[FlowDemand, ...], mask: int, available_indices: tuple[int, ...]) -> tuple[int, ...]:
    compatible: list[int] = []
    candidates = [1 << index for index in available_indices]
    limit = 1 << len(candidates)
    for sub_index in range(1, limit):
        submask = 0
        for bit_index, bit_mask in enumerate(candidates):
            if sub_index & (1 << bit_index):
                submask |= bit_mask
        if submask & ~mask:
            continue
        used_src: set[int] = set()
        used_dst: set[int] = set()
        ok = True
        for index, flow in enumerate(flows):
            if not (submask & (1 << index)):
                continue
            if int(flow.src_rank) in used_src or int(flow.dst_rank) in used_dst:
                ok = False
                break
            used_src.add(int(flow.src_rank))
            used_dst.add(int(flow.dst_rank))
        if ok:
            compatible.append(submask)
    return tuple(sorted(compatible, key=lambda item: (_mask_duration(flows, item), item)))


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
            "logical_model": str(result.get("reference_model", EXACT_REFERENCE_MODEL_ID)),
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
        "logical_model": str(result.get("reference_model", EXACT_REFERENCE_MODEL_ID)),
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
    "EXACT_COST_MODEL_ID",
    "EXACT_REFERENCE_MODEL_ID",
    "EXACT_RELEASE_MODEL_ID",
    "EXACT_SINGLE_PHASE_MODEL_ID",
    "EXACT_TASK_MODEL_ID",
    "MAX_BUCKET_TASK_COUNT",
    "MAX_RANK_COUNT",
    "UnsupportedExactSolve",
    "exact_model_contract",
    "exact_result_to_logical_plan",
    "exact_result_to_window_plan",
    "solve_exact_small_instance",
    "solve_problem_exact",
    "solve_problem_exact_with_scope",
]
