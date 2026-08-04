from __future__ import annotations

from rs_sim.scheduler.core.rscf_core import (
    RSCFTask,
    RSCFWireCostModel,
    order_rscf,
    phase_barrier_release_dependency,
    rank_local_release_dependency,
)
from rs_sim.scheduler.planning.planner import (
    AlgorithmWave,
    FairnessContract,
    SchedulingProblem,
    SchedulingTask,
    _critical_completion_objective_for_waves,
    _estimated_p1_release_tail_for_waves,
)
from rs_sim.scheduler.stable import stable_digest


def test_phase_barrier_does_not_release_one_rank_early() -> None:
    tasks = (
        RSCFTask("p1-release", 1, 0, 1, 1),
        RSCFTask("p1-other", 1, 0, 2, 1),
        RSCFTask("p2-after-r1", 2, 1, 0, 1),
    )
    local = rank_local_release_dependency(
        upstream_phase=1,
        downstream_phase=2,
        rank_count=3,
        delay_provider=lambda _rank: 0,
    )
    barrier = phase_barrier_release_dependency(
        upstream_phase=1,
        downstream_phase=2,
        rank_count=3,
        delay_provider=lambda _rank: 0,
    )
    local_plan = order_rscf(tasks, rank_count=3, release_dependencies=(local,))
    barrier_plan = order_rscf(tasks, rank_count=3, release_dependencies=(barrier,))
    assert tuple(wave.task_ids for wave in local_plan.waves) == (
        ("p1-release",),
        ("p1-other", "p2-after-r1"),
    )
    assert tuple(wave.task_ids for wave in barrier_plan.waves) == (
        ("p1-release",),
        ("p1-other",),
        ("p2-after-r1",),
    )


def _problem() -> SchedulingProblem:
    tasks = (
        SchedulingTask("a", "p1", 0, 0, 2, 1, 0, 0, 0),
        SchedulingTask("b", "p1", 0, 1, 3, 1, 0, 0, 0),
    )
    catalogue = stable_digest(tuple(item.task_id for item in tasks))
    boundaries = stable_digest(
        tuple(
            (
                item.task_id,
                item.phase_token,
                item.src_rank,
                item.dst_rank,
                item.chunk_index,
                item.byte_offset,
                item.payload_bytes,
            )
            for item in tasks
        )
    )
    fairness = FairnessContract(
        task_catalogue_digest=catalogue,
        task_boundary_digest=boundaries,
        taskization_digest="taskization",
        receiver_contract_rule_digest="receiver",
        buffer_profile_digest="buffer",
        compiler_digest="compiler",
        transport_digest="transport",
        release_model_digest="release",
        information_digest="information",
        cost_model_digest="cost",
    )
    return SchedulingProblem(
        rank_count=4,
        tasks=tasks,
        phase_tokens=("p1",),
        fairness=fairness,
    )


def test_safe_objectives_use_max_of_per_task_launch_plus_duration() -> None:
    problem = _problem()
    wave = AlgorithmWave(
        wave_id=0,
        task_ids=("a", "b"),
        phase_tokens=("p1",),
    )
    model = RSCFWireCostModel(
        default_slope=0.0,
        edge_intercept=((0, 2, 1.0), (1, 3, 100.0)),
        edge_launch=((0, 2, 100.0), (1, 3, 1.0)),
    )
    assert _critical_completion_objective_for_waves(problem, (wave,), model) == 101
    tail, release_times = _estimated_p1_release_tail_for_waves(
        problem, (wave,), model
    )
    assert tail == 101
    assert release_times == ((2, 101), (3, 101))


def test_safe_objective_respects_phase_barrier_release() -> None:
    tasks = (
        SchedulingTask("p1-r1", "p1", 1, 0, 1, 1, 0, 0, 0),
        SchedulingTask("p1-r2", "p1", 1, 0, 2, 100, 0, 0, 0),
        SchedulingTask("p2-r1", "p2", 2, 1, 3, 1, 0, 0, 0),
    )
    fairness = FairnessContract(
        task_catalogue_digest=stable_digest(tuple(item.task_id for item in tasks)),
        task_boundary_digest=stable_digest(
            tuple(
                (
                    item.task_id,
                    item.phase_token,
                    item.src_rank,
                    item.dst_rank,
                    item.chunk_index,
                    item.byte_offset,
                    item.payload_bytes,
                )
                for item in tasks
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
    )
    problem = SchedulingProblem(
        rank_count=4, tasks=tasks, phase_tokens=("p1", "p2"), fairness=fairness
    )
    waves = (
        AlgorithmWave(0, ("p1-r1",), ("p1",)),
        AlgorithmWave(1, ("p2-r1",), ("p2",)),
        AlgorithmWave(2, ("p1-r2",), ("p1",)),
    )
    model = RSCFWireCostModel(default_intercept=0.0, default_slope=1.0)
    assert _critical_completion_objective_for_waves(
        problem, waves, model, release_mode="RANK_LOCAL"
    ) == 102
    assert _critical_completion_objective_for_waves(
        problem, waves, model, release_mode="PHASE_BARRIER"
    ) == 2**63 - 1
