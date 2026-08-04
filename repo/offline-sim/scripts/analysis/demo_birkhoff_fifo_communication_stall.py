from __future__ import annotations

"""Hand-checkable FIFO/Birkhoff ordering example for communication stall."""

import json

from rs_sim.scheduler.core.birkhoff_core import BirkhoffTask, order_birkhoff
from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel
from rs_sim.scheduler.metrics.communication_stall import communication_stall_for_waves
from rs_sim.scheduler.planning.planner import AlgorithmWave, FairnessContract, SchedulingProblem, SchedulingTask
from rs_sim.scheduler.stable import stable_digest


def main() -> None:
    edges = (("t0", 0, 1), ("t1", 0, 2), ("t2", 0, 3), ("t3", 2, 1))
    tasks = tuple(
        SchedulingTask(task_id, "phase-0", 0, src, dst, 100, 0, 0, 0)
        for task_id, src, dst in edges
    )
    problem = SchedulingProblem(
        rank_count=4,
        tasks=tasks,
        phase_tokens=("phase-0",),
        fairness=FairnessContract(
            task_catalogue_digest=stable_digest(tuple(task.task_id for task in tasks)),
            task_boundary_digest=stable_digest(tuple((task.task_id, task.src_rank, task.dst_rank) for task in tasks)),
            taskization_digest="demo",
            receiver_contract_rule_digest="demo",
            buffer_profile_digest="demo",
            compiler_digest="demo",
            transport_digest="demo",
            release_model_digest="demo",
            information_digest="demo",
            cost_model_digest="demo",
        ),
    )
    fifo_order = tuple(task.task_id for task in sorted(tasks, key=lambda task: (task.src_rank, task.dst_rank, task.task_id)))
    birkhoff_order, _, _ = order_birkhoff(
        (BirkhoffTask(task_id, src, dst, 1) for task_id, src, dst in edges), rank_count=4
    )
    cost = RSCFWireCostModel(default_slope=1.0)
    rows = {}
    for name, order in (("fifo", fifo_order), ("birkhoff", birkhoff_order)):
        waves = (AlgorithmWave(0, tuple(order), ("phase-0",), 0),)
        result = communication_stall_for_waves(problem, waves, cost)
        rows[name] = {
            "order": order,
            "stall_ns_by_rank": result.stall_ns_by_rank,
            "mean_communication_stall_ns": result.mean_stall_ns,
            "p95_communication_stall_ns": result.p95_stall_ns,
            "communication_makespan_ns": result.actual.completion_ns,
        }
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
