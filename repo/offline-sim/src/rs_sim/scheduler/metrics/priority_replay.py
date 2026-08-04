from __future__ import annotations

"""Ready-aware replay of a frozen ORDER_ONLY priority plan.

The live executor treats logical waves as priority evidence, not as rigid
barriers.  It skips tasks that are not source-ready, starts every compatible
per-rank TX/RX transfer, and revisits the same frozen priority after each
completion or release event.  This module provides a deterministic, lightweight
replay of that contract for diagnostics and hand-checkable validation.

It deliberately does not model Receiver posting/drain, ControlPlane delivery,
shared-lane contention, or planning/binding service lines.  Its result is a
transport-priority diagnostic, not a replacement for the full Runtime.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from rs_sim.contracts.paper_defaults import PAPER_RELEASE_MODE
from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel

if TYPE_CHECKING:  # pragma: no cover
    from rs_sim.scheduler.planning.planner import AlgorithmWave, SchedulingProblem, SchedulingTask


_INF_NS = 2**63 - 1


@dataclass(frozen=True, slots=True)
class PriorityReplayResult:
    completion_ns: int
    transfer_finish_ns: int
    deadlocked: bool
    launched_task_ids: tuple[str, ...]
    task_completion_ns: tuple[tuple[str, int], ...]
    p2_source_release_ns_by_rank: tuple[tuple[int, int], ...]
    p2_destination_finish_ns_by_rank: tuple[tuple[int, int], ...]

    @property
    def feasible(self) -> bool:
        return not self.deadlocked and self.completion_ns < _INF_NS


def _semantic_phase_map(
    problem: "SchedulingProblem",
    *,
    semantic_phase_ordinal: int | None,
) -> tuple[dict[str, int], int, int | None, tuple[int, ...]]:
    ordinals = sorted({int(task.phase_ordinal) for task in problem.tasks})
    if len(ordinals) == 1 and semantic_phase_ordinal is not None:
        semantic_by_id = {
            task.task_id: int(semantic_phase_ordinal) for task in problem.tasks
        }
    else:
        # SchedulingProblem phase ordinals are often dense 0/1 while the paper
        # names the two communication phases P1/P2.  Preserve either form.
        offset = 1 if ordinals and max(ordinals) <= 1 else 0
        semantic_by_id = {
            task.task_id: int(task.phase_ordinal) + offset for task in problem.tasks
        }
    semantic_values = tuple(sorted(set(semantic_by_id.values())))
    if semantic_values == (2,):
        return semantic_by_id, 1, 2, semantic_values
    if 1 in semantic_values and 2 in semantic_values:
        return semantic_by_id, 1, 2, semantic_values
    if len(semantic_values) >= 2:
        return semantic_by_id, semantic_values[0], semantic_values[1], semantic_values
    return semantic_by_id, semantic_values[0], None, semantic_values


def replay_ready_aware_priority(
    problem: "SchedulingProblem",
    waves: Iterable["AlgorithmWave"],
    wire_cost_model: RSCFWireCostModel | None = None,
    *,
    release_mode: str = PAPER_RELEASE_MODE,
    semantic_phase_ordinal: int | None = None,
) -> PriorityReplayResult:
    """Replay one complete frozen priority under per-rank full-duplex TX/RX.

    Every task must appear exactly once in ``waves``.  A task may start when its
    own ``ready_at_ns``, wire-model source-ready time, P1→P2 release dependency,
    source TX, and destination RX are all ready.  The first compatible tasks in
    the frozen priority are launched together.
    """

    cost = wire_cost_model or RSCFWireCostModel()
    by_id = {task.task_id: task for task in problem.tasks}
    if not by_id:
        return PriorityReplayResult(0, 0, False, (), (), (), ())

    priority = tuple(
        task_id
        for wave in waves
        for task_id in wave.task_ids
        if task_id in by_id
    )
    if len(priority) != len(by_id) or set(priority) != set(by_id):
        return PriorityReplayResult(_INF_NS, _INF_NS, True, (), (), (), ())

    semantic_by_id, p1_phase, p2_phase, semantic_values = _semantic_phase_map(
        problem,
        semantic_phase_ordinal=semantic_phase_ordinal,
    )
    normalized_release = str(release_mode).upper()
    if normalized_release not in {"PHASE_BARRIER", "RANK_LOCAL"}:
        raise ValueError("release_mode must be PHASE_BARRIER or RANK_LOCAL")

    rank_count = int(problem.rank_count)
    tx_free = [0.0] * rank_count
    rx_free = [0.0] * rank_count
    unscheduled = set(priority)
    running: list[tuple[float, str]] = []
    completion_by_id: dict[str, float] = {}
    launched: list[str] = []

    p1_pending_by_destination = {rank: set() for rank in range(rank_count)}
    p2_pending_by_destination = {rank: set() for rank in range(rank_count)}
    for task in problem.tasks:
        semantic = semantic_by_id[task.task_id]
        if p2_phase is not None and semantic == p1_phase:
            p1_pending_by_destination[task.dst_rank].add(task.task_id)
        elif p2_phase is not None and semantic == p2_phase:
            p2_pending_by_destination[task.dst_rank].add(task.task_id)

    p2_source_ready = {
        rank: (0.0 if not p1_pending_by_destination[rank] else float("inf"))
        for rank in range(rank_count)
    }
    p2_destination_finish: dict[int, float] = {}
    barrier_released = p2_phase is None or all(
        not pending for pending in p1_pending_by_destination.values()
    )
    now = 0.0

    def base_ready(task: "SchedulingTask") -> float:
        semantic = semantic_by_id[task.task_id]
        ready = max(
            float(task.ready_at_ns or 0),
            float(cost.source_ready(semantic, task.src_rank)),
        )
        if p2_phase is not None and semantic == p2_phase:
            ready = max(ready, p2_source_ready[task.src_rank])
        return ready

    while unscheduled or running:
        finished = sorted(
            (finish, task_id)
            for finish, task_id in running
            if finish <= now + 1.0e-9
        )
        if finished:
            finished_ids = {task_id for _finish, task_id in finished}
            running = [item for item in running if item[1] not in finished_ids]
            for finish, task_id in finished:
                completion_by_id[task_id] = float(finish)
                task = by_id[task_id]
                semantic = semantic_by_id[task_id]
                if p2_phase is not None and semantic == p1_phase:
                    pending = p1_pending_by_destination[task.dst_rank]
                    pending.discard(task_id)
                    if normalized_release == "RANK_LOCAL" and not pending:
                        p2_source_ready[task.dst_rank] = min(
                            p2_source_ready[task.dst_rank],
                            float(finish) + cost.p1_to_p2_delay(task.dst_rank),
                        )
                elif p2_phase is not None and semantic == p2_phase:
                    pending = p2_pending_by_destination[task.dst_rank]
                    pending.discard(task_id)
                    if not pending:
                        p2_destination_finish[task.dst_rank] = float(finish)

            if (
                normalized_release == "PHASE_BARRIER"
                and p2_phase is not None
                and not barrier_released
                and all(not pending for pending in p1_pending_by_destination.values())
            ):
                barrier_released = True
                p1_finishes = [
                    completion_by_id[task.task_id]
                    for task in problem.tasks
                    if semantic_by_id[task.task_id] == p1_phase
                ]
                barrier_time = max(p1_finishes, default=now)
                for rank in range(rank_count):
                    p2_source_ready[rank] = (
                        barrier_time + cost.p1_to_p2_delay(rank)
                    )

        if not unscheduled and not running:
            break

        selected: list["SchedulingTask"] = []
        selected_src: set[int] = set()
        selected_dst: set[int] = set()
        for task_id in priority:
            if task_id not in unscheduled:
                continue
            task = by_id[task_id]
            ready = base_ready(task)
            if ready == float("inf") or ready > now + 1.0e-9:
                continue
            if tx_free[task.src_rank] > now + 1.0e-9:
                continue
            if rx_free[task.dst_rank] > now + 1.0e-9:
                continue
            if task.src_rank in selected_src or task.dst_rank in selected_dst:
                continue
            selected.append(task)
            selected_src.add(task.src_rank)
            selected_dst.add(task.dst_rank)

        if selected:
            for task in selected:
                duration = max(
                    0.0,
                    cost.launch(task.src_rank, task.dst_rank)
                    + cost.duration(task.src_rank, task.dst_rank, task.payload_bytes),
                )
                finish = now + duration
                tx_free[task.src_rank] = finish
                rx_free[task.dst_rank] = finish
                unscheduled.remove(task.task_id)
                running.append((finish, task.task_id))
                launched.append(task.task_id)
            continue

        next_times: list[float] = [
            finish for finish, _task_id in running if finish > now + 1.0e-9
        ]
        for task_id in unscheduled:
            task = by_id[task_id]
            ready = base_ready(task)
            if ready != float("inf") and ready > now + 1.0e-9:
                next_times.append(ready)
            resource_ready = max(tx_free[task.src_rank], rx_free[task.dst_rank])
            if resource_ready > now + 1.0e-9:
                next_times.append(resource_ready)
        if not next_times:
            return PriorityReplayResult(
                _INF_NS,
                _INF_NS,
                True,
                tuple(launched),
                tuple(sorted((key, int(round(value))) for key, value in completion_by_id.items())),
                tuple(),
                tuple(),
            )
        now = min(next_times)

    if len(completion_by_id) != len(by_id):
        return PriorityReplayResult(_INF_NS, _INF_NS, True, tuple(launched), (), (), ())

    transfer_finish = max(completion_by_id.values(), default=0.0)
    completion = transfer_finish
    if p2_phase is not None:
        for rank, finish in p2_destination_finish.items():
            completion = max(completion, finish + cost.p2_completion_tail(rank))
    elif semantic_values == (2,):
        finish_by_destination: dict[int, float] = {}
        for task in problem.tasks:
            finish_by_destination[task.dst_rank] = max(
                finish_by_destination.get(task.dst_rank, 0.0),
                completion_by_id[task.task_id],
            )
        for rank, finish in finish_by_destination.items():
            completion = max(completion, finish + cost.p2_completion_tail(rank))

    finite_releases = tuple(
        (rank, int(round(value)))
        for rank, value in sorted(p2_source_ready.items())
        if value != float("inf")
    )
    return PriorityReplayResult(
        completion_ns=max(0, int(round(completion))),
        transfer_finish_ns=max(0, int(round(transfer_finish))),
        deadlocked=False,
        launched_task_ids=tuple(launched),
        task_completion_ns=tuple(
            sorted((task_id, int(round(value))) for task_id, value in completion_by_id.items())
        ),
        p2_source_release_ns_by_rank=finite_releases,
        p2_destination_finish_ns_by_rank=tuple(
            (rank, int(round(value))) for rank, value in sorted(p2_destination_finish.items())
        ),
    )


def ready_aware_completion_objective(
    problem: "SchedulingProblem",
    waves: Iterable["AlgorithmWave"],
    wire_cost_model: RSCFWireCostModel | None = None,
    *,
    release_mode: str = PAPER_RELEASE_MODE,
    semantic_phase_ordinal: int | None = None,
) -> int:
    """Return only the completion objective, or ``2**63-1`` if infeasible."""

    return replay_ready_aware_priority(
        problem,
        waves,
        wire_cost_model,
        release_mode=release_mode,
        semantic_phase_ordinal=semantic_phase_ordinal,
    ).completion_ns


__all__ = [
    "PriorityReplayResult",
    "ready_aware_completion_objective",
    "replay_ready_aware_priority",
]
