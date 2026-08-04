from __future__ import annotations

from rs_sim.scheduler.core.oracle import SOLVER_BACKEND, solve_exact_wire
from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel
from rs_sim.scheduler.planning.planner import FairnessContract, SchedulingProblem, SchedulingTask
from rs_sim.scheduler.stable import stable_digest


def _problem(rank_count: int, tasks: tuple[SchedulingTask, ...]) -> SchedulingProblem:
    phase_tokens = tuple(dict.fromkeys(task.phase_token for task in tasks))
    return SchedulingProblem(
        rank_count=rank_count,
        tasks=tasks,
        phase_tokens=phase_tokens,
        fairness=FairnessContract(
            task_catalogue_digest=stable_digest(tuple(task.task_id for task in tasks)),
            task_boundary_digest=stable_digest(tuple(
                (
                    task.task_id,
                    task.phase_token,
                    task.src_rank,
                    task.dst_rank,
                    task.chunk_index,
                    task.byte_offset,
                    task.payload_bytes,
                )
                for task in tasks
            )),
            taskization_digest="taskization",
            receiver_contract_rule_digest="receiver",
            buffer_profile_digest="buffer",
            compiler_digest="compiler",
            transport_digest="transport",
            release_model_digest="release",
            information_digest="information",
            cost_model_digest="cost",
        ),
    )


def _task(
    task_id: str,
    phase: int,
    src: int,
    dst: int,
    duration: int,
    chunk: int = 0,
) -> SchedulingTask:
    return SchedulingTask(
        task_id=task_id,
        phase_token=f"phase-{phase}",
        phase_ordinal=phase,
        src_rank=src,
        dst_rank=dst,
        payload_bytes=duration,
        chunk_index=chunk,
        byte_offset=chunk * duration,
        ready_at_ns=0,
    )


def test_milp_preserves_full_duplex_matching() -> None:
    problem = _problem(2, (
        _task("a", 0, 0, 1, 10),
        _task("b", 0, 1, 0, 20),
    ))
    result = solve_exact_wire(
        problem,
        wire_cost_model=RSCFWireCostModel(default_slope=1.0),
        release_mode="PHASE_BARRIER",
        time_limit_ms=5_000,
    )
    assert result.solver_backend == SOLVER_BACKEND
    assert result.certified_optimal
    assert result.objective_units == 20
    assert len(result.waves) == 1
    assert set(result.waves[0].task_ids) == {"a", "b"}


def test_rank_local_release_can_overlap_unrelated_upstream_work() -> None:
    tasks = (
        _task("short-to-r1", 0, 0, 1, 1),
        _task("long-to-r0", 0, 2, 0, 100),
        _task("p2-from-r1", 1, 1, 2, 100),
    )
    problem = _problem(3, tasks)
    model = RSCFWireCostModel(default_slope=1.0)
    rank_local = solve_exact_wire(
        problem,
        wire_cost_model=model,
        release_mode="RANK_LOCAL",
        time_limit_ms=5_000,
    )
    barrier = solve_exact_wire(
        problem,
        wire_cost_model=model,
        release_mode="PHASE_BARRIER",
        time_limit_ms=5_000,
    )
    assert rank_local.certified_optimal and barrier.certified_optimal
    assert rank_local.objective_units == 101
    assert barrier.objective_units == 200


def test_formal_solver_has_no_sixteen_task_cutoff() -> None:
    tasks = tuple(
        _task(
            f"task-{round_id}-{src}",
            0,
            src,
            (src + 1) % 5,
            1,
            chunk=round_id,
        )
        for round_id in range(4)
        for src in range(5)
    )
    problem = _problem(5, tasks)
    result = solve_exact_wire(
        problem,
        wire_cost_model=RSCFWireCostModel(default_slope=1.0),
        release_mode="PHASE_BARRIER",
        time_limit_ms=10_000,
    )
    assert result.supported
    assert result.has_feasible_schedule
    assert result.certified_optimal
    assert result.objective_units == 4
    assert len(tuple(task for wave in result.waves for task in wave.task_ids)) == 20
    assert result.variable_count > 0
    assert result.constraint_count > 0


def test_large_phase_local_window_uses_fast_bounded_reference() -> None:
    tasks = tuple(
        _task(
            f"large-{round_id}-{src}",
            0,
            src,
            (src + (round_id % 7) + 1) % 8,
            1 + (round_id % 3),
            chunk=round_id,
        )
        for round_id in range(24)
        for src in range(8)
    )
    result = solve_exact_wire(
        _problem(8, tasks),
        wire_cost_model=RSCFWireCostModel(default_slope=1.0),
        release_mode="PHASE_BARRIER",
        time_limit_ms=250,
    )
    assert result.supported and result.has_feasible_schedule
    assert result.solver_status in {
        "optimal_by_matching_bound",
        "bounded_matching_reference",
    }
    assert result.variable_count == 0
    assert result.constraint_count == 0
    assert result.canonical_task_count == len(tasks)
    assert sorted(task_id for wave in result.waves for task_id in wave.task_ids) == sorted(
        task.task_id for task in tasks
    )


def test_joint_bounded_reference_preserves_all_tasks_and_release_scope() -> None:
    from rs_sim.scheduler.core.oracle import BOUNDED_SOLVER_BACKEND, solve_bounded_wire

    tasks = (
        _task("p1-r1", 0, 0, 1, 2),
        _task("p1-r0", 0, 2, 0, 7),
        _task("p2-r1", 1, 1, 2, 5),
        _task("p2-r0", 1, 0, 1, 3),
    )
    result = solve_bounded_wire(
        _problem(3, tasks),
        wire_cost_model=RSCFWireCostModel(default_slope=1.0),
        release_mode="RANK_LOCAL",
    )
    assert result.supported and result.has_feasible_schedule
    assert result.solver_backend == BOUNDED_SOLVER_BACKEND
    assert result.variable_count == 0 and result.constraint_count == 0
    assert sorted(task_id for wave in result.waves for task_id in wave.task_ids) == sorted(
        task.task_id for task in tasks
    )
    assert result.best_bound is not None
    assert result.objective_units is not None
    assert result.best_bound <= result.objective_units
