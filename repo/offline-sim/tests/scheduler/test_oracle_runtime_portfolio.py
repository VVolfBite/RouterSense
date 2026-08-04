from __future__ import annotations

from rs_sim.scheduler.core.oracle import OracleResult, OracleWave
from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel
from rs_sim.scheduler.planning.planner import (
    FairnessContract,
    OrderOnlyPlanner,
    PlannerScope,
    SchedulingProblem,
    SchedulingTask,
)
from rs_sim.scheduler.stable import stable_digest


def _problem(tasks: tuple[SchedulingTask, ...]) -> SchedulingProblem:
    return SchedulingProblem(
        rank_count=3,
        tasks=tasks,
        phase_tokens=tuple(dict.fromkeys(task.phase_token for task in tasks)),
        fairness=FairnessContract(
            task_catalogue_digest=stable_digest(tuple(task.task_id for task in tasks)),
            task_boundary_digest=stable_digest(tuple(
                (task.task_id, task.phase_token, task.src_rank, task.dst_rank,
                 task.chunk_index, task.byte_offset, task.payload_bytes)
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


def _task(task_id: str, *, dst: int, ordinal: int = 2) -> SchedulingTask:
    return SchedulingTask(
        task_id=task_id,
        phase_token="p2-only",
        phase_ordinal=ordinal,
        src_rank=0,
        dst_rank=dst,
        payload_bytes=10,
        chunk_index=0,
        byte_offset=0,
        ready_at_ns=0,
    )


def test_phase_local_groups_by_phase_token_not_dense_ordinal() -> None:
    problem = _problem((_task("only-p2", dst=1, ordinal=2),))
    plan = OrderOnlyPlanner().plan(
        problem,
        algorithm_id="fifo",
        planner_scope=PlannerScope.PHASE_LOCAL,
    )
    assert plan.ordered_task_ids == ("only-p2",)


def test_oracle_selects_better_ready_aware_candidate_without_relabeling_certification(monkeypatch) -> None:
    critical = _task("a-critical", dst=1)
    ordinary = _task("b-ordinary", dst=2)
    problem = _problem((critical, ordinary))

    # The mocked solver is exact only for its own serial-wave surrogate, but
    # deliberately puts the long-tail destination second.
    fake = OracleResult(
        supported=True,
        solver_status="optimal",
        certified_optimal=True,
        objective_units=20,
        best_bound=20,
        optimality_gap=0.0,
        search_nodes=1,
        waves=(
            OracleWave(0, ("b-ordinary",), 10, 0, 10),
            OracleWave(1, ("a-critical",), 10, 10, 20),
        ),
        model_id="fake",
        cost_model_id="fake",
        release_model_id="phase_barrier",
        result_digest="fake-result",
        solver_backend="FAKE",
        solve_time_ms=0.1,
        canonical_task_count=2,
        incumbent_source="FAKE_CERTIFIED",
    )
    monkeypatch.setattr("rs_sim.scheduler.core.oracle.solve_exact_wire", lambda *a, **k: fake)

    model = RSCFWireCostModel(
        default_slope=1.0,
        p2_completion_tail_by_rank=((1, 100.0),),
    )
    plan = OrderOnlyPlanner().plan(
        problem,
        algorithm_id="oracle",
        planner_scope=PlannerScope.PHASE_LOCAL,
        rscf_wire_cost_model=model,
        rscf_semantic_phase_ordinal=2,
        release_mode="PHASE_BARRIER",
        oracle_require_certified=True,
    )
    diagnostics = dict(plan.diagnostics)
    assert diagnostics["oracle_solver_model_certified"] == (True,)
    assert diagnostics["oracle_selected_candidate_source"] != ("solver",)
    assert diagnostics["oracle_solver_ready_aware_objective_ns"] == (120,)
    assert diagnostics["oracle_ready_aware_objective_ns"] == (110,)
    assert plan.ordered_task_ids[0] == "a-critical"
