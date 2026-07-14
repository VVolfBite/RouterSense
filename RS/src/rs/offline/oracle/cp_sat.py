from __future__ import annotations

from dataclasses import dataclass
import itertools
import time
from typing import Any

from rs.core.contracts import EvaluationSpec, EvaluationTask, EvaluationTaskSet
from rs.core.hashing import stable_hash_dict
from rs.offline.cost_model import EvaluationCostModel


@dataclass(frozen=True)
class OracleResult:
    solver_id: str
    solver_status: str
    certified_optimal: bool
    objective: float | None
    best_bound: float | None
    relative_gap: float | None
    solve_time: float
    task_count: int
    model_digest: str
    schedule: tuple[dict[str, Any], ...]


def solve_cp_sat(
    task_set: EvaluationTaskSet,
    *,
    spec: EvaluationSpec,
    mode: str,
    time_limit_s: float = 30.0,
) -> OracleResult:
    task_set.validate()
    spec.validate()
    model_digest = stable_hash_dict(
        {
            "oracle_cp_sat_v2": True,
            "mode": str(mode),
            "spec": spec.to_dict(),
            "task_set": task_set.to_dict(),
        }
    )
    if float(spec.launch_cost) != 0.0:
        return OracleResult(
            solver_id="cp_sat",
            solver_status="UNSUPPORTED_COST_MODEL",
            certified_optimal=False,
            objective=None,
            best_bound=None,
            relative_gap=None,
            solve_time=0.0,
            task_count=len(task_set.tasks),
            model_digest=model_digest,
            schedule=(),
        )
    if str(mode) not in {"local", "joint"}:
        raise ValueError(f"unsupported oracle mode {mode!r}")
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        objective, schedule = tiny_exact_solve(task_set=task_set, spec=spec, mode=str(mode))
        return OracleResult(
            solver_id="tiny_exact",
            solver_status="UNSUPPORTED",
            certified_optimal=objective is not None,
            objective=objective,
            best_bound=objective,
            relative_gap=0.0 if objective is not None else None,
            solve_time=0.0,
            task_count=len(task_set.tasks),
            model_digest=model_digest,
            schedule=schedule,
        )
    tasks = tuple(task_set.tasks)
    if not tasks:
        return OracleResult(
            solver_id="cp_sat",
            solver_status="OPTIMAL",
            certified_optimal=True,
            objective=0.0,
            best_bound=0.0,
            relative_gap=0.0,
            solve_time=0.0,
            task_count=0,
            model_digest=model_digest,
            schedule=(),
        )
    durations = [max(1, int(round(EvaluationCostModel.task_duration(task, spec)))) for task in tasks]
    horizon = max(1, int(sum(durations) + sum(int(task.release_time) for task in tasks) + max(int(spec.compute_delay), 0) * len(tasks) + 1))
    model = cp_model.CpModel()
    starts = []
    ends = []
    intervals = []
    for index, task in enumerate(tasks):
        start = model.NewIntVar(0, horizon, f"start_{index}")
        end = model.NewIntVar(0, horizon, f"end_{index}")
        interval = model.NewIntervalVar(start, durations[index], end, f"interval_{index}")
        starts.append(start)
        ends.append(end)
        intervals.append(interval)
        model.Add(start >= int(task.release_time))
    for rank in range(task_set.world_size):
        model.AddNoOverlap([intervals[idx] for idx, task in enumerate(tasks) if int(task.src_rank) == rank])
        model.AddNoOverlap([intervals[idx] for idx, task in enumerate(tasks) if int(task.dst_rank) == rank])
    end_by_id = {task.task_id: ends[idx] for idx, task in enumerate(tasks)}
    for idx, task in enumerate(tasks):
        if str(mode) == "local":
            if task.phase == "p1_return":
                for dep in [item for item in tasks if item.phase == "p0_dispatch"]:
                    model.Add(starts[idx] >= end_by_id[dep.task_id] + int(spec.compute_delay))
            elif task.phase == "p2_next_dispatch":
                for dep in [item for item in tasks if item.phase == "p1_return"]:
                    model.Add(starts[idx] >= end_by_id[dep.task_id])
        else:
            for dep_id in task.release_dependencies:
                delay = int(spec.compute_delay) if str(task.phase) == "p1_return" else 0
                model.Add(starts[idx] >= end_by_id[dep_id] + delay)
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, ends)
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 1
    started = time.perf_counter()
    status = solver.Solve(model)
    solve_time = time.perf_counter() - started
    status_name = str(solver.StatusName(status))
    certified_optimal = status == cp_model.OPTIMAL
    objective = float(solver.Value(makespan)) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    best_bound = float(solver.BestObjectiveBound()) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    relative_gap = 0.0 if certified_optimal and objective is not None else None
    schedule = tuple(
        {
            "task_id": task.task_id,
            "start": int(solver.Value(starts[idx])),
            "end": int(solver.Value(ends[idx])),
            "phase": str(task.phase),
        }
        for idx, task in enumerate(tasks)
    ) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else ()
    return OracleResult(
        solver_id="cp_sat",
        solver_status=status_name,
        certified_optimal=certified_optimal,
        objective=objective,
        best_bound=best_bound,
        relative_gap=relative_gap,
        solve_time=float(solve_time),
        task_count=len(tasks),
        model_digest=model_digest,
        schedule=schedule,
    )


def tiny_exact_solve(task_set: EvaluationTaskSet, *, spec: EvaluationSpec, mode: str) -> tuple[float | None, tuple[dict[str, Any], ...]]:
    tasks = tuple(task_set.tasks)
    if len(tasks) > 8:
        return None, ()
    best_objective: float | None = None
    best_schedule: tuple[dict[str, Any], ...] = ()
    for permutation in itertools.permutations(tasks):
        current_time = 0.0
        schedule: list[dict[str, Any]] = []
        completion_by_id: dict[str, float] = {}
        feasible = True
        for task in permutation:
            if str(mode) == "local":
                if str(task.phase) == "p1_return" and any(prev.phase == "p0_dispatch" and prev.task_id not in completion_by_id for prev in tasks):
                    feasible = False
                    break
                if str(task.phase) == "p2_next_dispatch" and any(prev.phase == "p1_return" and prev.task_id not in completion_by_id for prev in tasks):
                    feasible = False
                    break
            dep_times: list[float] = []
            for dep_id in task.release_dependencies:
                if dep_id not in completion_by_id:
                    feasible = False
                    break
                dep_times.append(completion_by_id[dep_id])
            if not feasible:
                break
            ready = float(task.release_time)
            if dep_times:
                ready = max(ready, max(dep_times) + (float(spec.compute_delay) if str(task.phase) == "p1_return" else 0.0))
            start = max(current_time, ready)
            end = start + EvaluationCostModel.task_duration(task, spec)
            schedule.append({"task_id": task.task_id, "start": start, "end": end, "phase": task.phase})
            completion_by_id[task.task_id] = end
            current_time = end
        if feasible and (best_objective is None or current_time < best_objective):
            best_objective = current_time
            best_schedule = tuple(schedule)
    return best_objective, best_schedule


__all__ = ["OracleResult", "solve_cp_sat", "tiny_exact_solve"]
