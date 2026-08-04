from __future__ import annotations

from rs_sim.scheduler.core.birkhoff_core import BirkhoffTask, order_birkhoff
from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel
from rs_sim.scheduler.metrics.communication_stall import communication_stall_for_waves
from rs_sim.scheduler.planning.planner import (
    AlgorithmWave,
    FairnessContract,
    SchedulingProblem,
    SchedulingTask,
)
from rs_sim.scheduler.stable import stable_digest


def _task(task_id: str, *, src: int, dst: int, size: int, ready: int = 0) -> SchedulingTask:
    return SchedulingTask(
        task_id=task_id,
        phase_token="phase-0",
        phase_ordinal=0,
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
        phase_tokens=("phase-0",),
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


def _waves(order: tuple[str, ...]) -> tuple[AlgorithmWave, ...]:
    return (
        AlgorithmWave(
            wave_id=0,
            task_ids=order,
            phase_tokens=("phase-0",),
            logical_duration_units=0,
        ),
    )


def test_zero_transport_preserves_canonical_ready_time() -> None:
    problem = _problem((_task("t", src=0, dst=1, size=10, ready=100),), rank_count=2)
    result = communication_stall_for_waves(
        problem,
        _waves(("t",)),
        RSCFWireCostModel(default_slope=1.0),
    )
    assert result.actual.task_completion_ns == (("t", 110),)
    assert result.zero_transport.task_completion_ns == (("t", 100),)
    assert result.stall_ns_by_rank == (0, 10)
    assert result.mean_stall_ns == 5
    assert result.phase_stall_ns == 10


def test_birkhoff_reduces_mean_stall_when_order_avoids_head_of_line_blocking() -> None:
    edges = (
        ("t0", 0, 1),
        ("t1", 0, 2),
        ("t2", 0, 3),
        ("t3", 2, 1),
    )
    tasks = tuple(_task(task_id, src=src, dst=dst, size=100) for task_id, src, dst in edges)
    problem = _problem(tasks)
    fifo_order = tuple(
        task.task_id
        for task in sorted(tasks, key=lambda task: (task.ready_at_ns, task.src_rank, task.dst_rank, task.task_id))
    )
    birkhoff_order, _waves_result, _certificate = order_birkhoff(
        (BirkhoffTask(task_id, src, dst, 1) for task_id, src, dst in edges),
        rank_count=4,
    )
    cost = RSCFWireCostModel(default_slope=1.0)
    fifo = communication_stall_for_waves(problem, _waves(fifo_order), cost)
    birkhoff = communication_stall_for_waves(problem, _waves(birkhoff_order), cost)

    assert fifo.actual.completion_ns == birkhoff.actual.completion_ns == 300
    assert fifo.mean_stall_ns == 175
    assert birkhoff.mean_stall_ns == 150
    assert (fifo.mean_stall_ns - birkhoff.mean_stall_ns) / fifo.mean_stall_ns > 0.10
