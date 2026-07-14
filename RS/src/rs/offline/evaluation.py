from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rs.core.contracts import EvaluationSpec, EvaluationTaskSet, ExecutionTruth, PlanEvaluation, WindowPlan


def _normalize_phase(name: str) -> str:
    if str(name) in {"p2_next_dispatch_forecast", "p2_next_dispatch"}:
        return "p2_next_dispatch"
    return str(name)


@dataclass(frozen=True)
class OfflineEvaluator:
    def evaluate(
        self,
        plan: WindowPlan,
        truth: ExecutionTruth,
        spec: EvaluationSpec,
    ) -> PlanEvaluation:
        plan.validate()
        truth.validate()
        spec.validate()
        phase_filter = {"p0_dispatch", "p1_return"} if str(spec.track) == "runtime_lookahead" else {"p0_dispatch", "p1_return", "p2_next_dispatch"}
        return evaluate_window_plan_against_task_set(
            plan=plan,
            task_set=truth.task_set,
            phase_filter=phase_filter,
            launch_cost=float(spec.launch_cost),
            bytes_per_row=int(spec.bytes_per_row),
            bandwidth=float(spec.bandwidth),
        )


def evaluate_window_plan_against_task_set(
    *,
    plan: WindowPlan,
    task_set: EvaluationTaskSet,
    phase_filter: set[str],
    launch_cost: float,
    bytes_per_row: int,
    bandwidth: float,
) -> PlanEvaluation:
    plan.validate()
    task_set.validate()
    expected_by_edge: dict[tuple[str, int, int], int] = {}
    dependencies_by_edge: dict[tuple[str, int, int], tuple[str, ...]] = {}
    for task in task_set.tasks:
        phase = _normalize_phase(task.phase)
        if phase not in phase_filter:
            continue
        edge_key = (phase, int(task.src_rank), int(task.dst_rank))
        expected_by_edge[edge_key] = int(task.row_count)
        dependencies_by_edge[edge_key] = tuple(task.release_dependencies)
    seen_by_edge = {key: 0 for key in expected_by_edge}
    completed_edges: dict[tuple[str, int, int], float] = {}
    completed_ids: list[str] = []
    dependency_violations: list[str] = []
    current_time = 0.0
    for wave in plan.waves:
        used_src: set[int] = set()
        used_dst: set[int] = set()
        wave_tasks = []
        for flow in wave.flows:
            phase = _normalize_phase(flow.phase)
            if phase not in phase_filter:
                continue
            edge_key = (phase, int(flow.src_rank), int(flow.dst_rank))
            expected_rows = expected_by_edge.get(edge_key)
            if expected_rows is None:
                return PlanEvaluation(
                    valid=False,
                    reason=f"unexpected_task:{phase}:{int(flow.src_rank)}->{int(flow.dst_rank)}",
                    realized_makespan=None,
                    completed_tasks=tuple(completed_ids),
                    unresolved_tasks=tuple(sorted(_edge_name(item) for item, value in seen_by_edge.items() if value != expected_by_edge[item])),
                    dependency_violations=tuple(dependency_violations),
                    coverage_valid=False,
                    port_valid=False,
                )
            if int(flow.src_rank) in used_src or int(flow.dst_rank) in used_dst:
                return PlanEvaluation(
                    valid=False,
                    reason=f"port_conflict:wave_{wave.wave_id}",
                    realized_makespan=None,
                    completed_tasks=tuple(completed_ids),
                    unresolved_tasks=tuple(sorted(_edge_name(item) for item, value in seen_by_edge.items() if value != expected_by_edge[item])),
                    dependency_violations=tuple(dependency_violations),
                    coverage_valid=False,
                    port_valid=False,
                )
            new_rows = int(seen_by_edge[edge_key]) + int(flow.row_count)
            if int(flow.row_count) < 0 or new_rows > int(expected_rows):
                return PlanEvaluation(
                    valid=False,
                    reason=f"row_count_mismatch:{phase}:{int(flow.src_rank)}->{int(flow.dst_rank)}",
                    realized_makespan=None,
                    completed_tasks=tuple(completed_ids),
                    unresolved_tasks=tuple(sorted(_edge_name(item) for item, value in seen_by_edge.items() if value != expected_by_edge[item])),
                    dependency_violations=tuple(dependency_violations),
                    coverage_valid=False,
                    port_valid=False,
                )
            used_src.add(int(flow.src_rank))
            used_dst.add(int(flow.dst_rank))
            missing_dependencies = []
            for dependency in dependencies_by_edge.get(edge_key, ()):
                dependency_edge = _edge_from_task_id(dependency)
                if dependency_edge is not None and seen_by_edge.get(dependency_edge, 0) < expected_by_edge.get(dependency_edge, 0):
                    missing_dependencies.append(dependency)
            if missing_dependencies:
                dependency_violations.append(f"{_edge_name(edge_key)}:{','.join(sorted(missing_dependencies))}")
            seen_by_edge[edge_key] = new_rows
            wave_tasks.append((edge_key, int(flow.row_count)))
        if dependency_violations:
            return PlanEvaluation(
                valid=False,
                reason="dependency_violation",
                realized_makespan=None,
                completed_tasks=tuple(completed_ids),
                unresolved_tasks=tuple(sorted(_edge_name(item) for item, value in seen_by_edge.items() if value != expected_by_edge[item])),
                dependency_violations=tuple(dependency_violations),
                coverage_valid=False,
                port_valid=True,
            )
        if not wave_tasks:
            continue
        wave_duration = float(launch_cost) + max(float(row_count * int(bytes_per_row)) / float(bandwidth) for _edge, row_count in wave_tasks)
        end_time = current_time + wave_duration
        for edge_key, _row_count in wave_tasks:
            completed_edges[edge_key] = end_time
            completed_ids.append(_edge_name(edge_key))
        current_time = end_time
    unresolved = tuple(sorted(_edge_name(item) for item, value in seen_by_edge.items() if value != expected_by_edge[item]))
    return PlanEvaluation(
        valid=len(unresolved) == 0 and len(dependency_violations) == 0,
        reason=None if len(unresolved) == 0 and len(dependency_violations) == 0 else "incomplete_coverage",
        realized_makespan=current_time if len(unresolved) == 0 and len(dependency_violations) == 0 else None,
        completed_tasks=tuple(completed_ids),
        unresolved_tasks=unresolved,
        dependency_violations=tuple(dependency_violations),
        coverage_valid=len(unresolved) == 0,
        port_valid=True,
        metrics={
            "completed_task_count": len(completed_ids),
            "expected_task_count": len(expected_by_edge),
        },
    )


def _edge_from_task_id(task_id: str) -> tuple[str, int, int] | None:
    try:
        phase, ranks = str(task_id).split(":", 1)
        src_rank, dst_rank = ranks.split("->", 1)
        return (_normalize_phase(phase), int(src_rank), int(dst_rank))
    except Exception:
        return None


def _edge_name(edge: tuple[str, int, int]) -> str:
    return f"{edge[0]}:{edge[1]}->{edge[2]}"

__all__ = ["OfflineEvaluator", "evaluate_window_plan_against_task_set"]
