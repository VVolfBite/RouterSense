from __future__ import annotations

from rs_sim.scheduler.planning.planner import (
    FairnessContract,
    OrderOnlyPlanner,
    PlannerScope,
    SchedulingProblem,
    SchedulingTask,
)
from rs_sim.scheduler.stable import stable_digest


def _problem(tasks: tuple[SchedulingTask, ...], *, rank_count: int = 4) -> SchedulingProblem:
    phase_tokens = tuple(dict.fromkeys(task.phase_token for task in tasks))
    fairness = FairnessContract(
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
    )
    return SchedulingProblem(
        rank_count=rank_count,
        tasks=tasks,
        phase_tokens=phase_tokens,
        fairness=fairness,
    )


def _task(
    task_id: str,
    *,
    phase: int,
    src: int,
    dst: int,
    payload: int,
    chunk: int = 0,
) -> SchedulingTask:
    return SchedulingTask(
        task_id=task_id,
        phase_token=f"P{phase}",
        phase_ordinal=phase - 1,
        src_rank=src,
        dst_rank=dst,
        payload_bytes=payload,
        chunk_index=chunk,
        byte_offset=chunk * payload,
        ready_at_ns=0,
    )


def _plan(
    tasks: tuple[SchedulingTask, ...],
    *,
    algorithm: str,
    scope: PlannerScope,
):
    return OrderOnlyPlanner().plan(
        _problem(tasks),
        algorithm_id=algorithm,
        planner_scope=scope,
        rank_to_node=(0, 0, 1, 1),
        oracle_require_certified=False,
    )


def test_greedy_joint_uses_predicted_p2_as_longest_work_hint() -> None:
    # P1 tasks have identical canonical payloads.  Canonical Greedy alone would
    # choose p1-a first.  The large predicted P2 outgoing from rank 3 is released
    # by completing P1 inbound to destination 3, so Joint Greedy must promote
    # p1-b without importing any RSCF score component.
    p1 = (
        _task("p1-a", phase=1, src=0, dst=1, payload=10),
        _task("p1-b", phase=1, src=2, dst=3, payload=10),
    )
    p2 = (
        _task("p2-heavy", phase=2, src=3, dst=0, payload=100),
    )

    local = _plan(p1, algorithm="greedy", scope=PlannerScope.PHASE_LOCAL)
    joint = _plan(p1 + p2, algorithm="greedy", scope=PlannerScope.WINDOW_JOINT)

    assert local.ordered_task_ids[:2] == ("p1-a", "p1-b")
    first_p1 = next(item for item in joint.ordered_task_ids if item.startswith("p1-"))
    assert first_p1 == "p1-b"
    diagnostics = dict(joint.diagnostics)
    assert diagnostics["joint_p2_adaptation"] == (
        "LONGEST_WORK_PLUS_PROPORTIONAL_DOWNSTREAM_P2_SHARE"
    )
    assert diagnostics["joint_p2_task_count"] == 1
    assert diagnostics["joint_p2_payload_units"] == 100


def test_zero_p2_joint_reduces_exactly_to_p1_only_core_order() -> None:
    p1 = (
        _task("a", phase=1, src=0, dst=2, payload=8),
        _task("b", phase=1, src=1, dst=3, payload=16),
        _task("c", phase=1, src=2, dst=1, payload=8),
        _task("d", phase=1, src=3, dst=0, payload=4),
    )

    for algorithm in (
        "greedy",
        "birkhoff",
        "islip",
        "residual_mwm",
        "fast",
        "aurora",
    ):
        local = _plan(p1, algorithm=algorithm, scope=PlannerScope.PHASE_LOCAL)
        zero_joint = _plan(p1, algorithm=algorithm, scope=PlannerScope.WINDOW_JOINT)
        assert zero_joint.ordered_task_ids == local.ordered_task_ids, algorithm
        assert tuple(wave.task_ids for wave in zero_joint.waves) == tuple(
            wave.task_ids for wave in local.waves
        ), algorithm
        diagnostics = dict(zero_joint.diagnostics)
        assert diagnostics["joint_p2_task_count"] == 0
        assert diagnostics["joint_p2_payload_units"] == 0
        assert diagnostics["zero_p2_reduction_contract"] == "EXACT_P1_ONLY_CORE_ORDER"


def test_residual_mwm_joint_naturally_uses_combined_p1_p2_residual() -> None:
    p1 = (
        _task("p1-a", phase=1, src=0, dst=2, payload=10),
        _task("p1-b", phase=1, src=1, dst=3, payload=10),
    )
    # Add enough P2 residual to edge 0->3 so the first maximum-residual matching
    # changes from the P1-only matching.  This is the core's own residual idea,
    # not a release-criticality score.
    p2 = (
        _task("p2-0", phase=2, src=0, dst=3, payload=40, chunk=0),
        _task("p2-1", phase=2, src=0, dst=3, payload=40, chunk=1),
    )

    local = _plan(p1, algorithm="residual_mwm", scope=PlannerScope.PHASE_LOCAL)
    joint = _plan(p1 + p2, algorithm="residual_mwm", scope=PlannerScope.WINDOW_JOINT)

    assert tuple(wave.task_ids for wave in joint.waves) != tuple(
        wave.task_ids for wave in local.waves
    )
    assert dict(joint.diagnostics)["joint_p2_adaptation"] == (
        "COMBINED_P1_P2_EDGE_RESIDUAL_MATCHING"
    )
