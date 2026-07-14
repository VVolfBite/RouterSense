from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from rs.core.contracts import EvaluationTaskSet
from rs.core.hashing import stable_hash_dict


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


def solve_cp_sat(task_set: EvaluationTaskSet, *, mode: str, time_limit_s: float = 30.0) -> OracleResult:
    task_set.validate()
    model_digest = stable_hash_dict(
        {
            "oracle_cp_sat_v1": True,
            "mode": str(mode),
            "task_set": task_set.to_dict(),
        }
    )
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        return OracleResult(
            solver_id="cp_sat",
            solver_status="UNSUPPORTED",
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
    horizon = max(1, len(tasks) * 4)
    model = cp_model.CpModel()
    starts = []
    ends = []
    intervals = []
    for index, task in enumerate(tasks):
        start = model.NewIntVar(0, horizon, f"start_{index}")
        end = model.NewIntVar(0, horizon, f"end_{index}")
        interval = model.NewIntervalVar(start, max(int(task.row_count), 1), end, f"interval_{index}")
        starts.append(start)
        ends.append(end)
        intervals.append(interval)
    for rank in range(task_set.world_size):
        model.AddNoOverlap([intervals[idx] for idx, task in enumerate(tasks) if int(task.src_rank) == rank])
        model.AddNoOverlap([intervals[idx] for idx, task in enumerate(tasks) if int(task.dst_rank) == rank])
    end_by_id = {task.task_id: ends[idx] for idx, task in enumerate(tasks)}
    for idx, task in enumerate(tasks):
        dependencies = tuple(task.release_dependencies)
        if str(mode) == "local":
            if task.phase == "p1_return":
                for dep in [item for item in tasks if item.phase == "p0_dispatch"]:
                    model.Add(starts[idx] >= end_by_id[dep.task_id])
            elif task.phase == "p2_next_dispatch":
                for dep in [item for item in tasks if item.phase in {"p0_dispatch", "p1_return"}]:
                    model.Add(starts[idx] >= end_by_id[dep.task_id])
        else:
            for dep_id in dependencies:
                model.Add(starts[idx] >= end_by_id[dep_id])
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
    if objective is None or best_bound is None or objective == 0.0:
        relative_gap = 0.0 if certified_optimal else None
    else:
        relative_gap = abs(float(objective) - float(best_bound)) / abs(float(objective))
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


__all__ = ["OracleResult", "solve_cp_sat"]
