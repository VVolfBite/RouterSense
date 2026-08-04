from __future__ import annotations

"""Communication-stall diagnostics for a frozen scheduling priority.

This module reuses :mod:`priority_replay` rather than maintaining a second
scheduler model.  The actual and zero-transport runs use the same tasks,
ready times, release mode, priority order, and local timing metadata.  Only
wire launch and transfer service are set to zero in the counterfactual.

The result is a transport-priority diagnostic.  It is not the full Runtime
counterfactual because Receiver posting/drain, staging, ControlPlane, and
planning/binding lines are not simulated by the lightweight replay.
"""

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Iterable

from rs_sim.contracts.paper_defaults import PAPER_RELEASE_MODE
from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel
from rs_sim.scheduler.metrics.priority_replay import (
    PriorityReplayResult,
    _semantic_phase_map,
    replay_ready_aware_priority,
)

if TYPE_CHECKING:  # pragma: no cover
    from rs_sim.scheduler.planning.planner import AlgorithmWave, SchedulingProblem


def _nearest_rank_percentile_ns(values: tuple[int, ...], percentile: int) -> int:
    rows = sorted(int(value) for value in values)
    if not rows:
        return 0
    if percentile <= 0:
        return rows[0]
    index = min(len(rows) - 1, max(0, math.ceil(percentile * len(rows) / 100) - 1))
    return rows[index]


@dataclass(frozen=True, slots=True)
class CommunicationStallResult:
    actual: PriorityReplayResult
    zero_transport: PriorityReplayResult
    actual_rank_completion_ns: tuple[int, ...]
    zero_transport_rank_completion_ns: tuple[int, ...]
    stall_ns_by_rank: tuple[int, ...]
    mean_stall_ns: int
    p95_stall_ns: int
    max_stall_ns: int
    phase_stall_ns: int

    @property
    def feasible(self) -> bool:
        return self.actual.feasible and self.zero_transport.feasible


def zero_transport_cost_model(cost: RSCFWireCostModel) -> RSCFWireCostModel:
    """Return the same readiness/local timing model with zero wire service."""

    return RSCFWireCostModel(
        default_slope=0.0,
        default_intercept=0.0,
        wave_launch_cost=0.0,
        source_ready_by_phase_rank=tuple(cost.source_ready_by_phase_rank),
        p1_to_p2_delay_by_rank=tuple(cost.p1_to_p2_delay_by_rank),
        p2_completion_tail_by_rank=tuple(cost.p2_completion_tail_by_rank),
    )


def _rank_completion_ns(
    problem: "SchedulingProblem",
    replay: PriorityReplayResult,
    cost: RSCFWireCostModel,
    *,
    semantic_phase_ordinal: int | None,
) -> tuple[int, ...]:
    if not replay.feasible:
        return tuple(2**63 - 1 for _ in range(int(problem.rank_count)))

    completion_by_id = dict(replay.task_completion_ns)
    semantic_by_id, _p1_phase, p2_phase, semantic_values = _semantic_phase_map(
        problem,
        semantic_phase_ordinal=semantic_phase_ordinal,
    )
    rank_completion = [0 for _ in range(int(problem.rank_count))]
    rank_has_p2 = [False for _ in range(int(problem.rank_count))]
    for task in problem.tasks:
        finish = int(completion_by_id[task.task_id])
        rank_completion[task.dst_rank] = max(rank_completion[task.dst_rank], finish)
        semantic = semantic_by_id[task.task_id]
        if (p2_phase is not None and semantic == p2_phase) or semantic_values == (2,):
            rank_has_p2[task.dst_rank] = True

    for rank, has_p2 in enumerate(rank_has_p2):
        if has_p2:
            rank_completion[rank] += int(round(cost.p2_completion_tail(rank)))
    return tuple(rank_completion)


def communication_stall_for_waves(
    problem: "SchedulingProblem",
    waves: Iterable["AlgorithmWave"],
    wire_cost_model: RSCFWireCostModel | None = None,
    *,
    release_mode: str = PAPER_RELEASE_MODE,
    semantic_phase_ordinal: int | None = None,
) -> CommunicationStallResult:
    """Evaluate per-rank and phase communication stall for one frozen plan."""

    cost = wire_cost_model or RSCFWireCostModel()
    frozen_waves = tuple(waves)
    actual = replay_ready_aware_priority(
        problem,
        frozen_waves,
        cost,
        release_mode=release_mode,
        semantic_phase_ordinal=semantic_phase_ordinal,
    )
    zero_cost = zero_transport_cost_model(cost)
    zero = replay_ready_aware_priority(
        problem,
        frozen_waves,
        zero_cost,
        release_mode=release_mode,
        semantic_phase_ordinal=semantic_phase_ordinal,
    )
    actual_rank = _rank_completion_ns(
        problem,
        actual,
        cost,
        semantic_phase_ordinal=semantic_phase_ordinal,
    )
    zero_rank = _rank_completion_ns(
        problem,
        zero,
        zero_cost,
        semantic_phase_ordinal=semantic_phase_ordinal,
    )
    if not actual.feasible or not zero.feasible:
        stalls = tuple(2**63 - 1 for _ in range(int(problem.rank_count)))
        return CommunicationStallResult(
            actual=actual,
            zero_transport=zero,
            actual_rank_completion_ns=actual_rank,
            zero_transport_rank_completion_ns=zero_rank,
            stall_ns_by_rank=stalls,
            mean_stall_ns=2**63 - 1,
            p95_stall_ns=2**63 - 1,
            max_stall_ns=2**63 - 1,
            phase_stall_ns=2**63 - 1,
        )

    stalls = tuple(max(0, int(a) - int(z)) for a, z in zip(actual_rank, zero_rank))
    return CommunicationStallResult(
        actual=actual,
        zero_transport=zero,
        actual_rank_completion_ns=actual_rank,
        zero_transport_rank_completion_ns=zero_rank,
        stall_ns_by_rank=stalls,
        mean_stall_ns=int(round(sum(stalls) / len(stalls))) if stalls else 0,
        p95_stall_ns=_nearest_rank_percentile_ns(stalls, 95),
        max_stall_ns=max(stalls, default=0),
        phase_stall_ns=max(0, int(actual.completion_ns) - int(zero.completion_ns)),
    )


__all__ = [
    "CommunicationStallResult",
    "communication_stall_for_waves",
    "zero_transport_cost_model",
]
