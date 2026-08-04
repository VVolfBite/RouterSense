from __future__ import annotations

from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel
from rs_sim.scheduler.metrics.priority_replay import replay_ready_aware_priority
from rs_sim.scheduler.planning.planner import (
    AlgorithmWave,
    FairnessContract,
    SchedulingProblem,
    SchedulingTask,
)
from rs_sim.scheduler.stable import stable_digest


def _task(task_id: str, *, phase: int = 0, src: int, dst: int, size: int, ready: int = 0) -> SchedulingTask:
    return SchedulingTask(
        task_id=task_id,
        phase_token=f"phase-{phase}",
        phase_ordinal=phase,
        src_rank=src,
        dst_rank=dst,
        payload_bytes=size,
        chunk_index=0,
        byte_offset=0,
        ready_at_ns=ready,
    )


def _problem(tasks: tuple[SchedulingTask, ...], rank_count: int = 4) -> SchedulingProblem:
    return SchedulingProblem(
        rank_count=rank_count,
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


def _wave(*task_ids: str) -> tuple[AlgorithmWave, ...]:
    return (
        AlgorithmWave(
            wave_id=0,
            task_ids=tuple(task_ids),
            phase_tokens=(),
            logical_duration_units=0,
        ),
    )


def test_replay_skips_not_ready_head_and_uses_independent_endpoints() -> None:
    problem = _problem((
        _task("late", src=0, dst=1, size=1, ready=100),
        _task("ready", src=2, dst=3, size=10),
    ))
    result = replay_ready_aware_priority(
        problem,
        _wave("late", "ready"),
        RSCFWireCostModel(default_slope=1.0),
    )
    assert result.feasible
    assert result.launched_task_ids == ("ready", "late")
    assert result.completion_ns == 101


def test_replay_preserves_full_duplex_and_rank_local_release() -> None:
    opposite = _problem((
        _task("a", src=0, dst=1, size=100),
        _task("b", src=1, dst=0, size=100),
    ), rank_count=2)
    assert replay_ready_aware_priority(
        opposite, _wave("a", "b"), RSCFWireCostModel(default_slope=1.0)
    ).completion_ns == 100

    joint = _problem((
        _task("p1-fast", phase=0, src=2, dst=0, size=1),
        _task("p1-slow", phase=0, src=3, dst=1, size=100),
        _task("p2-r0", phase=1, src=0, dst=2, size=100),
    ))
    waves = _wave("p1-fast", "p1-slow", "p2-r0")
    model = RSCFWireCostModel(default_slope=1.0)
    assert replay_ready_aware_priority(
        joint, waves, model, release_mode="PHASE_BARRIER"
    ).completion_ns == 200
    assert replay_ready_aware_priority(
        joint, waves, model, release_mode="RANK_LOCAL"
    ).completion_ns == 101
