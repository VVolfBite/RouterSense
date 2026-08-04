from __future__ import annotations

from rs_sim.contracts.schema import AuthorityStamp
from rs_sim.scheduler.core.oracle import solve_exact_wire
from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel
from rs_sim.scheduler.execution.live import _merge_phase_local_rank_release_waves
from rs_sim.scheduler.execution.window_arbiter import (
    PhaseFrontier,
    ReleaseFrontierWindowArbiter,
    WindowArbitrationContext,
)
from rs_sim.scheduler.planning.planner import (
    FairnessContract,
    OrderOnlyPlanner,
    PlannerScope,
    SchedulingProblem,
    SchedulingTask,
    _critical_completion_objective_for_waves,
)
from rs_sim.scheduler.stable import stable_digest


def _problem(rank_count: int, tasks: tuple[SchedulingTask, ...]) -> SchedulingProblem:
    return SchedulingProblem(
        rank_count=rank_count,
        tasks=tasks,
        phase_tokens=tuple(dict.fromkeys(task.phase_token for task in tasks)),
        fairness=FairnessContract(
            task_catalogue_digest=stable_digest(tuple(task.task_id for task in tasks)),
            task_boundary_digest=stable_digest(
                tuple(
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
                )
            ),
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
    phase_token: str,
    phase_ordinal: int,
    src: int,
    dst: int,
    duration: int,
    *,
    ready_at: int = 0,
) -> SchedulingTask:
    return SchedulingTask(
        task_id=task_id,
        phase_token=phase_token,
        phase_ordinal=phase_ordinal,
        src_rank=src,
        dst_rank=dst,
        payload_bytes=duration,
        chunk_index=0,
        byte_offset=0,
        ready_at_ns=ready_at,
    )


def test_local_matching_algorithms_improve_a_hand_solved_phase() -> None:
    problem = _problem(
        3,
        (
            _task("A_long", "P1", 0, 0, 1, 10),
            _task("a_short", "P1", 0, 0, 2, 1),
            _task("B_long", "P1", 0, 1, 2, 10),
            _task("b_short", "P1", 0, 1, 0, 1),
        ),
    )
    model = RSCFWireCostModel(default_slope=1.0)
    planner = OrderOnlyPlanner()
    fifo = planner.plan(
        problem,
        algorithm_id="fifo",
        planner_scope=PlannerScope.PHASE_LOCAL,
        rscf_wire_cost_model=model,
        rscf_semantic_phase_ordinal=1,
    )
    assert _critical_completion_objective_for_waves(problem, fifo.waves, model) == 21
    for algorithm in ("greedy", "birkhoff", "residual_mwm", "rscf"):
        plan = planner.plan(
            problem,
            algorithm_id=algorithm,
            planner_scope=PlannerScope.PHASE_LOCAL,
            rscf_wire_cost_model=model,
            rscf_semantic_phase_ordinal=1,
        )
        assert _critical_completion_objective_for_waves(problem, plan.waves, model) == 11
    exact = solve_exact_wire(
        problem,
        wire_cost_model=model,
        release_mode="PHASE_BARRIER",
        semantic_phase_ordinal=1,
        time_limit_ms=5_000,
    )
    assert exact.certified_optimal and exact.objective_units == 11


def test_rscf_local_and_oracle_prioritize_p2_terminal_tail() -> None:
    problem = _problem(
        3,
        (
            _task("normal", "P2", 0, 0, 1, 10),
            _task("tail-critical", "P2", 0, 0, 2, 10),
        ),
    )
    model = RSCFWireCostModel(
        default_slope=1.0,
        p2_completion_tail_by_rank=((2, 100.0),),
    )
    plan = OrderOnlyPlanner().plan(
        problem,
        algorithm_id="rscf",
        planner_scope=PlannerScope.PHASE_LOCAL,
        rscf_wire_cost_model=model,
        rscf_semantic_phase_ordinal=2,
    )
    assert plan.ordered_task_ids == ("tail-critical", "normal")
    assert _critical_completion_objective_for_waves(
        problem,
        plan.waves,
        model,
        semantic_phase_ordinal=2,
    ) == 110
    exact = solve_exact_wire(
        problem,
        wire_cost_model=model,
        release_mode="PHASE_BARRIER",
        semantic_phase_ordinal=2,
        time_limit_ms=5_000,
    )
    assert exact.certified_optimal and exact.objective_units == 110
    assert exact.waves[0].task_ids == ("tail-critical",)


def test_wire_oracle_respects_canonical_task_ready_time() -> None:
    problem = _problem(
        3,
        (
            _task("future", "P2", 0, 0, 1, 10, ready_at=100),
            _task("ready", "P2", 0, 0, 2, 10, ready_at=0),
        ),
    )
    result = solve_exact_wire(
        problem,
        wire_cost_model=RSCFWireCostModel(default_slope=1.0),
        release_mode="PHASE_BARRIER",
        semantic_phase_ordinal=2,
        time_limit_ms=5_000,
    )
    assert result.certified_optimal
    assert result.objective_units == 110
    assert tuple(task for wave in result.waves for task in wave.task_ids) == (
        "ready",
        "future",
    )


def test_rank_local_local_merge_allows_released_p2_to_overtake_later_p1() -> None:
    merged = _merge_phase_local_rank_release_waves(
        (("p1-fast",), ("p1-slow",), ("p2-released",)),
        p1_task_ids=("p1-fast", "p1-slow"),
    )
    assert merged == (("p1-fast", "p2-released"), ("p1-slow",))
    preferred = tuple(task for wave in merged for task in wave)
    p1 = PhaseFrontier.build(
        phase_key="P1",
        authority_stamp=AuthorityStamp("P1", "p1-plan", 0, "p1-authority"),
        ready_task_ids=("p1-slow",),
    )
    p2 = PhaseFrontier.build(
        phase_key="P2",
        authority_stamp=AuthorityStamp("P2", "p2-plan", 0, "p2-authority"),
        ready_task_ids=("p2-released",),
    )
    context = WindowArbitrationContext.build(
        window_key="window",
        frontiers=(p1, p2),
        transport_snapshot_digest="snapshot",
        observed_at_ns=1,
    )
    decision = ReleaseFrontierWindowArbiter(preferred_task_ids=preferred).select(context)
    assert decision.selected_phase_token == "P2"
    assert decision.selected_task_ids == ("p2-released",)
