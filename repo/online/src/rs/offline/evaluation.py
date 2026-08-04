from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import EvaluationSpec, EvaluationTask, EvaluationTaskSet, ExecutionTruth, PlanEvaluation, WindowPlan
from rs.offline.cost_model import EvaluationCostModel


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
        return evaluate_window_plan_against_task_set(plan=plan, task_set=truth.task_set, phase_filter=phase_filter, spec=spec)


def evaluate_window_plan_against_task_set(
    *,
    plan: WindowPlan,
    task_set: EvaluationTaskSet,
    phase_filter: set[str],
    spec: EvaluationSpec,
) -> PlanEvaluation:
    plan.validate()
    task_set.validate()
    spec.validate()
    tasks_by_edge: dict[tuple[str, int, int], EvaluationTask] = {}
    expected_by_edge: dict[tuple[str, int, int], int] = {}
    dependencies_by_edge: dict[tuple[str, int, int], tuple[str, ...]] = {}
    release_by_edge: dict[tuple[str, int, int], float] = {}
    for task in task_set.tasks:
        phase = _normalize_phase(task.phase)
        if phase not in phase_filter:
            continue
        edge_key = (phase, int(task.src_rank), int(task.dst_rank))
        tasks_by_edge[edge_key] = task
        expected_by_edge[edge_key] = int(task.row_count)
        dependencies_by_edge[edge_key] = tuple(task.release_dependencies)
        release_by_edge[edge_key] = float(task.release_time)
    served_by_edge = {key: 0 for key in expected_by_edge}
    completed_edges: dict[tuple[str, int, int], float] = {}
    completed_tasks: list[str] = []
    dependency_violations: list[str] = []
    wave_metrics: list[dict[str, object]] = []
    current_time = 0.0

    for wave in plan.waves:
        wave_flows = [flow for flow in wave.flows if _normalize_phase(flow.phase) in phase_filter]
        if not wave_flows:
            continue
        used_src: set[int] = set()
        used_dst: set[int] = set()
        completed_snapshot = dict(completed_edges)
        served_snapshot = dict(served_by_edge)
        validated_tasks: list[tuple[tuple[str, int, int], EvaluationTask, int]] = []
        ready_times: list[float] = []
        wave_served_delta: dict[tuple[str, int, int], int] = {}
        for flow in wave_flows:
            phase = _normalize_phase(flow.phase)
            edge_key = (phase, int(flow.src_rank), int(flow.dst_rank))
            task = tasks_by_edge.get(edge_key)
            if task is None:
                return _invalid_evaluation(
                    reason=f"unexpected_task:{phase}:{int(flow.src_rank)}->{int(flow.dst_rank)}",
                    completed_tasks=completed_tasks,
                    served_by_edge=served_by_edge,
                    expected_by_edge=expected_by_edge,
                    dependency_violations=dependency_violations,
                    port_valid=False,
                )
            if int(flow.src_rank) in used_src or int(flow.dst_rank) in used_dst:
                return _invalid_evaluation(
                    reason=f"port_conflict:wave_{wave.wave_id}",
                    completed_tasks=completed_tasks,
                    served_by_edge=served_by_edge,
                    expected_by_edge=expected_by_edge,
                    dependency_violations=dependency_violations,
                    port_valid=False,
                )
            used_src.add(int(flow.src_rank))
            used_dst.add(int(flow.dst_rank))
            if int(flow.row_count) <= 0:
                return _invalid_evaluation(
                    reason=f"non_positive_row_count:{phase}:{int(flow.src_rank)}->{int(flow.dst_rank)}",
                    completed_tasks=completed_tasks,
                    served_by_edge=served_by_edge,
                    expected_by_edge=expected_by_edge,
                    dependency_violations=dependency_violations,
                    port_valid=True,
                )
            already_served = int(served_snapshot.get(edge_key, 0))
            if edge_key in completed_snapshot:
                return _invalid_evaluation(
                    reason=f"duplicate_task:{phase}:{int(flow.src_rank)}->{int(flow.dst_rank)}",
                    completed_tasks=completed_tasks,
                    served_by_edge=served_by_edge,
                    expected_by_edge=expected_by_edge,
                    dependency_violations=dependency_violations,
                    port_valid=True,
                )
            wave_delta = int(wave_served_delta.get(edge_key, 0))
            if already_served + wave_delta + int(flow.row_count) > int(expected_by_edge[edge_key]):
                return _invalid_evaluation(
                    reason=f"row_count_mismatch:{phase}:{int(flow.src_rank)}->{int(flow.dst_rank)}",
                    completed_tasks=completed_tasks,
                    served_by_edge=served_by_edge,
                    expected_by_edge=expected_by_edge,
                    dependency_violations=dependency_violations,
                    port_valid=True,
                )
            wave_served_delta[edge_key] = wave_delta + int(flow.row_count)
            dep_completion_times: list[float] = []
            for dependency in dependencies_by_edge.get(edge_key, ()):
                dependency_edge = _edge_from_task_id(dependency)
                if dependency_edge is None:
                    dependency_violations.append(f"{_edge_name(edge_key)}:unknown_dependency:{dependency}")
                    continue
                if dependency_edge not in completed_snapshot:
                    dependency_violations.append(f"{_edge_name(edge_key)}:same_wave_or_missing:{dependency}")
                    continue
                dep_completion_times.append(float(completed_snapshot[dependency_edge]))
            if dependency_violations:
                return _invalid_evaluation(
                    reason="dependency_violation",
                    completed_tasks=completed_tasks,
                    served_by_edge=served_by_edge,
                    expected_by_edge=expected_by_edge,
                    dependency_violations=dependency_violations,
                    port_valid=True,
                )
            ready_times.append(_task_ready_time(task=task, dependency_completion_times=tuple(dep_completion_times), spec=spec))
            validated_tasks.append((edge_key, task, int(flow.row_count)))
        wave_start = max([current_time, *ready_times])
        wave_duration = float(spec.launch_cost) + max(
            EvaluationCostModel.flow_duration(
                row_count=row_count,
                bytes_per_row=max(int(task.byte_count) // max(int(task.row_count), 1), int(spec.bytes_per_row)),
                spec=spec,
            )
            for _, task, row_count in validated_tasks
        )
        wave_end = wave_start + wave_duration
        for edge_key, task, row_count in validated_tasks:
            served_by_edge[edge_key] = int(served_by_edge.get(edge_key, 0)) + int(row_count)
            if int(served_by_edge[edge_key]) == int(expected_by_edge[edge_key]):
                completed_edges[edge_key] = wave_end
                completed_tasks.append(str(task.task_id))
        wave_metrics.append(
            {
                "wave_id": int(wave.wave_id),
                "start": float(wave_start),
                "end": float(wave_end),
                "idle_time": float(max(wave_start - current_time, 0.0)),
            }
        )
        current_time = wave_end

    unresolved = tuple(sorted(_edge_name(item) for item, value in served_by_edge.items() if value != expected_by_edge[item]))
    return PlanEvaluation(
        valid=len(unresolved) == 0 and len(dependency_violations) == 0,
        reason=None if len(unresolved) == 0 and len(dependency_violations) == 0 else "incomplete_coverage",
        realized_makespan=current_time if len(unresolved) == 0 and len(dependency_violations) == 0 else None,
        completed_tasks=tuple(completed_tasks),
        unresolved_tasks=unresolved,
        dependency_violations=tuple(dependency_violations),
        coverage_valid=len(unresolved) == 0,
        port_valid=True,
        metrics={
            "completed_task_count": len(completed_tasks),
            "expected_task_count": len(expected_by_edge),
            "served_row_count": int(sum(served_by_edge.values())),
            "expected_row_count": int(sum(expected_by_edge.values())),
            "wave_metrics": tuple(wave_metrics),
            "tail_completion": float(current_time),
        },
    )


def _task_ready_time(*, task: EvaluationTask, dependency_completion_times: tuple[float, ...], spec: EvaluationSpec) -> float:
    release_time = float(task.release_time)
    if str(task.phase) == "p0_dispatch":
        return release_time
    dep_ready = max(dependency_completion_times) if dependency_completion_times else 0.0
    if str(task.phase) == "p1_return":
        return max(release_time, dep_ready + float(spec.compute_delay))
    return max(release_time, dep_ready)


def _invalid_evaluation(
    *,
    reason: str,
    completed_tasks: list[str],
    served_by_edge: dict[tuple[str, int, int], int],
    expected_by_edge: dict[tuple[str, int, int], int],
    dependency_violations: list[str],
    port_valid: bool,
) -> PlanEvaluation:
    return PlanEvaluation(
        valid=False,
        reason=reason,
        realized_makespan=None,
        completed_tasks=tuple(completed_tasks),
        unresolved_tasks=tuple(sorted(_edge_name(item) for item, value in served_by_edge.items() if value != expected_by_edge[item])),
        dependency_violations=tuple(dependency_violations),
        coverage_valid=False,
        port_valid=bool(port_valid),
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
