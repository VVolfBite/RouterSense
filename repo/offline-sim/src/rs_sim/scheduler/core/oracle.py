from __future__ import annotations

"""Canonical-task Oracle backed by a symmetry-compressed SciPy/HiGHS MILP.

The solver preserves every canonical transfer task and its fixed wire cost.  It
only removes solver symmetry: tasks that are identical with respect to phase,
source, destination, service time, release time, and terminal tail share one
multiplicity row in the MILP.  A solved multiplicity is expanded back to the
original stable task IDs before returning the plan.

Each logical wave is a full-duplex bipartite matching.  A rank may transmit one
canonical task and receive one canonical task concurrently, while no rank may
transmit two or receive two tasks in the same wave.  Wave execution is
non-preemptive and sequential, matching the formal Oracle contract used by the
runtime priority executor.
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel
from rs_sim.scheduler.core.birkhoff_core import BirkhoffTask, decompose_integer_matrix, order_birkhoff
from rs_sim.scheduler.core.literature_cores import (
    LiteratureTask,
    order_aurora,
    order_islip,
    order_residual_mwm,
)
from rs_sim.scheduler.planning.planner import SchedulingProblem, SchedulingTask
from rs_sim.scheduler.stable import stable_digest

MILP_MODEL_ID = "canonical_multiplicity_matching_wave_highs_milp"
PAYLOAD_COST_MODEL_ID = "canonical_payload_wave_makespan"
WIRE_COST_MODEL_ID = "canonical_wire_release_aware_wave_makespan"
SOLVER_BACKEND = "SCIPY_HIGHS_MILP"
BOUNDED_REFERENCE_MODEL_ID = "canonical_release_aware_bounded_reference"
BOUNDED_SOLVER_BACKEND = "DETERMINISTIC_RELEASE_AWARE_BOUND"


@dataclass(frozen=True, slots=True)
class OracleWave:
    wave_id: int
    task_ids: tuple[str, ...]
    duration_units: int
    start_units: int = 0
    finish_units: int = 0


@dataclass(frozen=True, slots=True)
class OracleResult:
    supported: bool
    solver_status: str
    certified_optimal: bool
    objective_units: int | None
    best_bound: int | None
    optimality_gap: float | None
    search_nodes: int
    waves: tuple[OracleWave, ...]
    model_id: str
    cost_model_id: str
    release_model_id: str
    result_digest: str
    failure_reason: str | None = None
    solver_backend: str = SOLVER_BACKEND
    solve_time_ms: float | None = None
    variable_count: int = 0
    constraint_count: int = 0
    canonical_task_count: int = 0
    symmetry_group_count: int = 0
    candidate_slot_count: int = 0
    incumbent_source: str | None = None

    @property
    def has_feasible_schedule(self) -> bool:
        return self.objective_units is not None and (
            bool(self.waves) or self.objective_units == 0
        )


@dataclass(frozen=True, slots=True)
class _TaskGroup:
    group_id: int
    task_ids: tuple[str, ...]
    phase_token: str
    phase: int
    src_rank: int
    dst_rank: int
    duration: int
    source_ready: int
    terminal_tail: int

    @property
    def count(self) -> int:
        return len(self.task_ids)


@dataclass(frozen=True, slots=True)
class _Index:
    group_count: int
    slot_count: int
    release_count: int

    @property
    def x0(self) -> int:
        return 0

    @property
    def d0(self) -> int:
        return self.group_count * self.slot_count

    @property
    def s0(self) -> int:
        return self.d0 + self.slot_count

    @property
    def y0(self) -> int:
        return self.s0 + self.slot_count

    @property
    def r0(self) -> int:
        return self.y0 + self.slot_count

    @property
    def cmax(self) -> int:
        return self.r0 + self.release_count

    @property
    def variable_count(self) -> int:
        return self.cmax + 1

    def x(self, group: int, slot: int) -> int:
        return self.x0 + group * self.slot_count + slot

    def d(self, slot: int) -> int:
        return self.d0 + slot

    def s(self, slot: int) -> int:
        return self.s0 + slot

    def y(self, slot: int) -> int:
        return self.y0 + slot

    def release(self, release_index: int) -> int:
        return self.r0 + release_index


class _Rows:
    def __init__(self) -> None:
        self.row: list[int] = []
        self.col: list[int] = []
        self.data: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(
        self,
        coefficients: Iterable[tuple[int, float]],
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> None:
        row_id = len(self.lower)
        for column, value in coefficients:
            if value:
                self.row.append(row_id)
                self.col.append(int(column))
                self.data.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    @property
    def count(self) -> int:
        return len(self.lower)

    def constraint(self, variable_count: int) -> LinearConstraint:
        matrix = coo_array(
            (np.asarray(self.data, dtype=float), (self.row, self.col)),
            shape=(self.count, int(variable_count)),
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )


def _sorted_tasks(problem: SchedulingProblem) -> tuple[SchedulingTask, ...]:
    return tuple(
        sorted(
            problem.tasks,
            key=lambda task: (
                task.phase_ordinal,
                task.src_rank,
                task.dst_rank,
                task.chunk_index,
                task.byte_offset,
                task.payload_bytes,
                task.task_id,
            ),
        )
    )


def _semantic_phases(
    tasks: tuple[SchedulingTask, ...],
    semantic_phase_ordinal: int | None,
) -> tuple[int, ...]:
    ordinals = tuple(sorted({int(task.phase_ordinal) for task in tasks}))
    if len(ordinals) == 1 and semantic_phase_ordinal is not None:
        return tuple(int(semantic_phase_ordinal) for _ in tasks)
    offset = 1 if ordinals and max(ordinals) <= 1 else 0
    return tuple(int(task.phase_ordinal) + offset for task in tasks)


def _release_model_id(release_mode: str) -> str:
    normalized = str(release_mode).upper()
    if normalized == "PHASE_BARRIER":
        return "phase_barrier_release_with_rank_local_delay"
    if normalized == "RANK_LOCAL":
        return "rank_local_release_with_rank_local_delay"
    raise ValueError("release_mode must be PHASE_BARRIER or RANK_LOCAL")


def _build_groups(
    tasks: tuple[SchedulingTask, ...],
    phases: tuple[int, ...],
    durations: tuple[int, ...],
    source_ready: tuple[int, ...],
    terminal_tail: tuple[int, ...],
) -> tuple[_TaskGroup, ...]:
    buckets: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for task, phase, duration, ready, tail in zip(
        tasks, phases, durations, source_ready, terminal_tail
    ):
        key = (
            str(task.phase_token),
            int(phase),
            int(task.src_rank),
            int(task.dst_rank),
            int(duration),
            int(ready),
            int(tail),
        )
        buckets[key].append(str(task.task_id))
    groups: list[_TaskGroup] = []
    for group_id, key in enumerate(sorted(buckets)):
        phase_token, phase, src, dst, duration, ready, tail = key
        groups.append(
            _TaskGroup(
                group_id=group_id,
                task_ids=tuple(sorted(buckets[key])),
                phase_token=phase_token,
                phase=phase,
                src_rank=src,
                dst_rank=dst,
                duration=duration,
                source_ready=ready,
                terminal_tail=tail,
            )
        )
    return tuple(groups)


def _candidate_slot_count(groups: tuple[_TaskGroup, ...], rank_count: int) -> int:
    """Safe compact upper bound: sum of phase-local bipartite degrees.

    Scheduling every phase sequentially with a bipartite edge colouring is
    always feasible in this many waves.  Rank-local overlap can use fewer waves
    but never needs more to beat that phase-sequential feasible schedule.
    """

    total = 0
    for phase in sorted({group.phase for group in groups}):
        phase_groups = tuple(group for group in groups if group.phase == phase)
        source_degree = [0] * rank_count
        destination_degree = [0] * rank_count
        for group in phase_groups:
            source_degree[group.src_rank] += group.count
            destination_degree[group.dst_rank] += group.count
        total += max(max(source_degree, default=0), max(destination_degree, default=0))
    return max(1, int(total))


def _release_keys(
    groups: tuple[_TaskGroup, ...],
    rank_count: int,
    release_mode: str,
) -> tuple[tuple[int, int | None], ...]:
    phases = sorted({group.phase for group in groups})
    normalized = str(release_mode).upper()
    keys: list[tuple[int, int | None]] = []
    for boundary, _ in enumerate(zip(phases, phases[1:])):
        if normalized == "PHASE_BARRIER":
            keys.append((boundary, None))
        elif normalized == "RANK_LOCAL":
            keys.extend((boundary, rank) for rank in range(rank_count))
        else:
            raise ValueError("release_mode must be PHASE_BARRIER or RANK_LOCAL")
    return tuple(keys)



def _group_priority(group: _TaskGroup) -> tuple[int, int, int, int]:
    return (
        int(group.terminal_tail),
        int(group.duration),
        int(group.phase),
        -int(group.group_id),
    )


def _maximum_cardinality_group_matching(
    eligible: tuple[_TaskGroup, ...],
) -> tuple[_TaskGroup, ...]:
    """Deterministic maximum-cardinality matching over eligible rank edges."""

    best_by_edge: dict[tuple[int, int], _TaskGroup] = {}
    for group in eligible:
        edge = (group.src_rank, group.dst_rank)
        incumbent = best_by_edge.get(edge)
        if incumbent is None or _group_priority(group) > _group_priority(incumbent):
            best_by_edge[edge] = group
    adjacency: dict[int, list[tuple[int, _TaskGroup]]] = defaultdict(list)
    for (src, dst), group in best_by_edge.items():
        adjacency[src].append((dst, group))
    for src in adjacency:
        adjacency[src].sort(
            key=lambda item: (_group_priority(item[1]), -item[0]),
            reverse=True,
        )

    match_by_destination: dict[int, tuple[int, _TaskGroup]] = {}

    def augment(src: int, seen: set[int]) -> bool:
        for dst, group in adjacency.get(src, ()):
            if dst in seen:
                continue
            seen.add(dst)
            previous = match_by_destination.get(dst)
            if previous is None or augment(previous[0], seen):
                match_by_destination[dst] = (src, group)
                return True
        return False

    source_order = sorted(
        adjacency,
        key=lambda src: (len(adjacency[src]), src),
    )
    for src in source_order:
        augment(src, set())
    return tuple(
        sorted(
            (item[1] for item in match_by_destination.values()),
            key=lambda group: (group.src_rank, group.dst_rank, group.group_id),
        )
    )


def _perfect_matching_from_counts(
    counts: list[list[int]],
) -> tuple[tuple[int, int], ...]:
    """Return one deterministic perfect matching from a regular multigraph."""

    rank_count = len(counts)
    match_by_destination: dict[int, int] = {}

    def augment(src: int, seen: set[int]) -> bool:
        destinations = [
            dst for dst in range(rank_count) if counts[src][dst] > 0
        ]
        for dst in destinations:
            if dst in seen:
                continue
            seen.add(dst)
            previous = match_by_destination.get(dst)
            if previous is None or augment(previous, seen):
                match_by_destination[dst] = src
                return True
        return False

    for src in range(rank_count):
        if not augment(src, set()):
            raise RuntimeError("regular bipartite multigraph has no perfect matching")
    return tuple(sorted((src, dst) for dst, src in match_by_destination.items()))


def _phase_edge_coloring(
    phase_groups: tuple[_TaskGroup, ...],
    *,
    rank_count: int,
) -> tuple[tuple[_TaskGroup, ...], ...]:
    """Exact Delta-edge-colouring of one bipartite task multigraph.

    The old implementation recomputed a perfect matching once per logical
    wave.  Large canonical traces can contain thousands of equal chunks, so
    that path scaled with ``Delta * matching_cost``.  This implementation uses
    the coefficient-compressed Birkhoff certificate and expands coefficients
    only when producing the final canonical waves.  The schedule and task
    boundaries are unchanged.
    """

    count_matrix = [[0 for _ in range(rank_count)] for _ in range(rank_count)]
    groups_by_edge: dict[tuple[int, int], list[_TaskGroup]] = defaultdict(list)
    remaining_by_group = {group.group_id: group.count for group in phase_groups}
    for group in phase_groups:
        count_matrix[group.src_rank][group.dst_rank] += group.count
        groups_by_edge[(group.src_rank, group.dst_rank)].append(group)
    for edge in groups_by_edge:
        groups_by_edge[edge].sort(key=_group_priority, reverse=True)

    certificate = decompose_integer_matrix(count_matrix)
    edge_cursor = {edge: 0 for edge in groups_by_edge}
    matchings: list[tuple[_TaskGroup, ...]] = []
    for interval in certificate.intervals:
        for _ in range(int(interval.coefficient)):
            selected: list[_TaskGroup] = []
            for edge in interval.real_edges:
                candidates = groups_by_edge[edge]
                cursor = edge_cursor[edge]
                while cursor < len(candidates) and remaining_by_group[candidates[cursor].group_id] <= 0:
                    cursor += 1
                if cursor >= len(candidates):
                    raise RuntimeError("edge colouring over-served a canonical edge")
                group = candidates[cursor]
                remaining_by_group[group.group_id] -= 1
                edge_cursor[edge] = cursor
                selected.append(group)
            if selected:
                matchings.append(tuple(sorted(selected, key=lambda g: (g.src_rank, g.dst_rank, g.group_id))))
    if any(remaining_by_group.values()):
        raise RuntimeError("edge colouring omitted canonical task multiplicities")
    return tuple(matchings)


def _greedy_feasible_waves(
    groups: tuple[_TaskGroup, ...],
    *,
    rank_count: int,
    release_mode: str,
    wire_cost_model: RSCFWireCostModel | None,
) -> tuple[tuple[OracleWave, ...], int]:
    """Construct a phase-sequential Delta-wave feasible upper bound.

    This bound intentionally does not assume rank-local overlap.  It is always
    feasible for both release modes and uses exactly the sum of the phase-local
    bipartite maximum degrees, avoiding task-identity and empty-slot symmetry.
    """

    phases = sorted({group.phase for group in groups})
    by_phase: dict[int, tuple[_TaskGroup, ...]] = {
        phase: tuple(group for group in groups if group.phase == phase)
        for phase in phases
    }
    cursor = {group.group_id: 0 for group in groups}
    completion_by_group = {group.group_id: 0 for group in groups}
    current = 0
    objective = 0
    waves: list[OracleWave] = []
    normalized = str(release_mode).upper()

    for phase_index, phase in enumerate(phases):
        phase_groups = by_phase[phase]
        matchings = _phase_edge_coloring(
            phase_groups, rank_count=rank_count
        )
        previous = () if phase_index == 0 else by_phase[phases[phase_index - 1]]
        for selected in matchings:
            required_start = current
            for group in selected:
                required_start = max(required_start, group.source_ready)
                if phase_index == 0:
                    continue
                if normalized == "PHASE_BARRIER":
                    predecessors = previous
                elif normalized == "RANK_LOCAL":
                    predecessors = tuple(
                        item for item in previous if item.dst_rank == group.src_rank
                    )
                else:
                    raise ValueError(
                        "release_mode must be PHASE_BARRIER or RANK_LOCAL"
                    )
                release_base = max(
                    (completion_by_group[item.group_id] for item in predecessors),
                    default=0,
                )
                delay = (
                    max(
                        0,
                        int(round(wire_cost_model.p1_to_p2_delay(group.src_rank))),
                    )
                    if wire_cost_model is not None
                    else 0
                )
                required_start = max(required_start, release_base + delay)
            duration = max(group.duration for group in selected)
            finish = required_start + duration
            task_ids: list[str] = []
            for group in selected:
                position = cursor[group.group_id]
                task_id = group.task_ids[position]
                cursor[group.group_id] = position + 1
                completion_by_group[group.group_id] = finish
                objective = max(objective, finish + group.terminal_tail)
                task_ids.append(task_id)
            waves.append(
                OracleWave(
                    wave_id=len(waves),
                    task_ids=tuple(task_ids),
                    duration_units=duration,
                    start_units=required_start,
                    finish_units=finish,
                )
            )
            current = finish
    if any(cursor[group.group_id] != group.count for group in groups):
        raise RuntimeError("feasible upper bound omitted canonical tasks")
    return tuple(waves), int(objective)


def _release_aware_analytical_lower_bound(
    groups: tuple[_TaskGroup, ...],
    *,
    rank_count: int,
    release_mode: str,
    wire_cost_model: RSCFWireCostModel | None,
) -> int:
    """Resource and release-chain lower bound for the MILP objective."""

    normalized = str(release_mode).upper()
    phases = sorted({group.phase for group in groups})
    previous_global_completion = 0
    previous_destination_completion = [0] * rank_count
    objective_bound = 0

    for phase_index, phase in enumerate(phases):
        phase_groups = tuple(group for group in groups if group.phase == phase)
        earliest: dict[int, int] = {}
        for group in phase_groups:
            release_at = 0
            if phase_index > 0:
                delay = (
                    max(
                        0,
                        int(round(wire_cost_model.p1_to_p2_delay(group.src_rank))),
                    )
                    if wire_cost_model is not None
                    else 0
                )
                if normalized == "PHASE_BARRIER":
                    release_at = previous_global_completion + delay
                elif normalized == "RANK_LOCAL":
                    release_at = previous_destination_completion[group.src_rank] + delay
                else:
                    raise ValueError(
                        "release_mode must be PHASE_BARRIER or RANK_LOCAL"
                    )
            earliest[group.group_id] = max(group.source_ready, release_at)
            objective_bound = max(
                objective_bound,
                earliest[group.group_id] + group.duration + group.terminal_tail,
            )

        source_completion = [0] * rank_count
        destination_completion = [0] * rank_count
        for rank in range(rank_count):
            source_groups = tuple(
                group for group in phase_groups if group.src_rank == rank
            )
            if source_groups:
                source_completion[rank] = max(
                    min(earliest[group.group_id] for group in source_groups)
                    + sum(group.count * group.duration for group in source_groups),
                    max(
                        earliest[group.group_id] + group.duration
                        for group in source_groups
                    ),
                )
            destination_groups = tuple(
                group for group in phase_groups if group.dst_rank == rank
            )
            if destination_groups:
                destination_completion[rank] = max(
                    min(earliest[group.group_id] for group in destination_groups)
                    + sum(group.count * group.duration for group in destination_groups),
                    max(
                        earliest[group.group_id] + group.duration
                        for group in destination_groups
                    ),
                )
                # Every canonical task targeting one rank shares that rank's
                # terminal continuation tail.  Since the destination RX resource
                # serializes all of those tasks, the final inbound completion for
                # that rank must be followed by the same tail.  Binding the tail
                # to the destination workload gives a strict, substantially
                # tighter P2 lower bound than considering one task plus tail.
                objective_bound = max(
                    objective_bound,
                    destination_completion[rank]
                    + max(group.terminal_tail for group in destination_groups),
                )
        phase_completion = max(
            max(source_completion, default=0),
            max(destination_completion, default=0),
            max(
                (
                    earliest[group.group_id]
                    + group.duration
                    + group.terminal_tail
                    for group in phase_groups
                ),
                default=0,
            ),
        )
        objective_bound = max(objective_bound, phase_completion)
        previous_global_completion = phase_completion
        previous_destination_completion = destination_completion
    return int(objective_bound)


def _materialize_single_phase_order(
    matchings: tuple[tuple[_TaskGroup, ...], ...],
    order: tuple[int, ...],
) -> tuple[tuple[OracleWave, ...], int]:
    cursor: dict[int, int] = defaultdict(int)
    current = 0
    objective = 0
    waves: list[OracleWave] = []
    for matching_index in order:
        selected = matchings[matching_index]
        if not selected:
            continue
        start = max(
            current,
            max((group.source_ready for group in selected), default=current),
        )
        duration = max(group.duration for group in selected)
        finish = start + duration
        task_ids: list[str] = []
        for group in selected:
            position = cursor[group.group_id]
            if position >= group.count:
                raise RuntimeError("single-phase Oracle reused a canonical task group")
            task_ids.append(group.task_ids[position])
            cursor[group.group_id] = position + 1
            objective = max(objective, finish + group.terminal_tail)
        waves.append(
            OracleWave(
                wave_id=len(waves),
                task_ids=tuple(task_ids),
                duration_units=duration,
                start_units=start,
                finish_units=finish,
            )
        )
        current = finish
    return tuple(waves), int(objective)


def _single_phase_order_candidates(
    matchings: tuple[tuple[_TaskGroup, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    count = len(matchings)
    if count <= 1:
        return (tuple(range(count)),)
    features: dict[int, tuple[int, int, int]] = {}
    for index, selected in enumerate(matchings):
        features[index] = (
            max((group.source_ready for group in selected), default=0),
            max((group.terminal_tail for group in selected), default=0),
            max((group.duration for group in selected), default=0),
        )
    candidates: set[tuple[int, ...]] = {
        tuple(range(count)),
        tuple(reversed(range(count))),
        tuple(sorted(range(count), key=lambda i: (features[i][0], -features[i][1], -features[i][2], i))),
        tuple(sorted(range(count), key=lambda i: (-features[i][1], features[i][0], -features[i][2], i))),
        tuple(sorted(range(count), key=lambda i: (-features[i][2], -features[i][1], features[i][0], i))),
        tuple(sorted(range(count), key=lambda i: (features[i][2], -features[i][1], features[i][0], i))),
    }

    def greedy(kind: str) -> tuple[int, ...]:
        remaining = set(range(count))
        chosen: list[int] = []
        current = 0
        objective = 0
        while remaining:
            ranked: list[tuple[tuple[int, ...], int, int, int]] = []
            for index in remaining:
                ready, tail, duration = features[index]
                start = max(current, ready)
                finish = start + duration
                projected = max(objective, finish + tail)
                if kind == "completion":
                    key = (projected, start - current, -tail, -duration, index)
                elif kind == "tail":
                    key = (start - current, -tail, -duration, projected, index)
                else:
                    key = (start - current, -duration, -tail, projected, index)
                ranked.append((key, index, finish, projected))
            _, selected, finish, projected = min(ranked)
            remaining.remove(selected)
            chosen.append(selected)
            current = finish
            objective = projected
        return tuple(chosen)

    candidates.add(greedy("completion"))
    candidates.add(greedy("tail"))
    candidates.add(greedy("duration"))
    return tuple(sorted(candidates))



def _single_phase_membership_candidates(
    tasks: tuple[SchedulingTask, ...],
    groups: tuple[_TaskGroup, ...],
    *,
    rank_count: int,
) -> tuple[tuple[tuple[_TaskGroup, ...], ...], ...]:
    group_by_task = {
        task_id: group
        for group in groups
        for task_id in group.task_ids
    }
    duration_by_task = {
        task_id: group.duration
        for group in groups
        for task_id in group.task_ids
    }
    task_by_id = {task.task_id: task for task in tasks}
    raw: list[tuple[tuple[_TaskGroup, ...], ...]] = [
        _phase_edge_coloring(groups, rank_count=rank_count)
    ]

    def add_task_waves(waves: Iterable[Iterable[str]]) -> None:
        converted: list[tuple[_TaskGroup, ...]] = []
        seen: list[str] = []
        for wave in waves:
            ids = tuple(str(item) for item in wave)
            if not ids:
                continue
            seen.extend(ids)
            converted.append(tuple(group_by_task[item] for item in ids))
        if sorted(seen) == sorted(task_by_id):
            raw.append(tuple(converted))

    def packed(order: tuple[SchedulingTask, ...]) -> tuple[tuple[str, ...], ...]:
        waves: list[tuple[str, ...]] = []
        current: list[str] = []
        sources: set[int] = set()
        destinations: set[int] = set()
        for task in order:
            if current and (
                task.src_rank in sources or task.dst_rank in destinations
            ):
                waves.append(tuple(current))
                current = []
                sources = set()
                destinations = set()
            current.append(task.task_id)
            sources.add(task.src_rank)
            destinations.add(task.dst_rank)
        if current:
            waves.append(tuple(current))
        return tuple(waves)

    fifo_order = tuple(sorted(tasks, key=lambda task: (
        int(task.ready_at_ns or 0), task.phase_ordinal, task.src_rank,
        task.dst_rank, task.chunk_index, task.byte_offset, task.task_id,
    )))
    greedy_order = tuple(sorted(tasks, key=lambda task: (
        -duration_by_task[task.task_id], task.phase_ordinal, task.src_rank,
        task.dst_rank, task.chunk_index, task.byte_offset, task.task_id,
    )))
    add_task_waves(packed(fifo_order))
    add_task_waves(packed(greedy_order))

    literature = tuple(
        LiteratureTask(
            task_id=task.task_id,
            phase=task.phase_ordinal,
            src_rank=task.src_rank,
            dst_rank=task.dst_rank,
            payload_units=duration_by_task[task.task_id],
            chunk_index=task.chunk_index,
            byte_offset=task.byte_offset,
            ready_at=float(task.ready_at_ns or 0),
        )
        for task in tasks
    )
    # Large canonical traces need a bounded reference that scales with edge
    # multiplicity rather than re-running every literature baseline.  The
    # exact task catalogue and all task costs remain unchanged; only the set of
    # feasible incumbent generators is reduced.  Residual-MWM complements the
    # deterministic Delta edge-colouring, while FIFO/greedy packing above keep
    # two cheap order-sensitive alternatives.
    if len(tasks) > 512:
        plans = ()
    else:
        plans = (
            order_islip(literature, rank_count=rank_count),
            order_residual_mwm(literature, rank_count=rank_count),
            order_aurora(literature, rank_count=rank_count),
        )
    for plan in plans:
        add_task_waves(wave.task_ids for wave in plan.waves)

    if len(tasks) <= 512:
        _, birkhoff_waves, _ = order_birkhoff(
            (
                BirkhoffTask(
                    task_id=task.task_id,
                    src_rank=task.src_rank,
                    dst_rank=task.dst_rank,
                    payload_units=duration_by_task[task.task_id],
                    chunk_index=task.chunk_index,
                    byte_offset=task.byte_offset,
                )
                for task in tasks
            ),
            rank_count=rank_count,
        )
        add_task_waves(birkhoff_waves)

    unique: dict[tuple[tuple[int, ...], ...], tuple[tuple[_TaskGroup, ...], ...]] = {}
    for candidate in raw:
        key = tuple(
            tuple(sorted(group.group_id for group in wave))
            for wave in candidate
            if wave
        )
        unique.setdefault(key, candidate)
    return tuple(unique[key] for key in sorted(unique))


def _single_phase_bounded_reference(
    *,
    problem: SchedulingProblem,
    tasks: tuple[SchedulingTask, ...],
    groups: tuple[_TaskGroup, ...],
    model_id: str,
    cost_model_id: str,
    release_model_id: str,
    release_mode: str,
    wire_cost_model: RSCFWireCostModel | None,
    started_ns: int,
    extra_digest_payload: dict[str, Any] | None,
) -> OracleResult:
    candidates = _single_phase_membership_candidates(
        tasks,
        groups,
        rank_count=problem.rank_count,
    )
    best_waves: tuple[OracleWave, ...] = ()
    best_objective: int | None = None
    best_order: tuple[int, ...] = ()
    best_membership = 0
    for membership_index, matchings in enumerate(candidates):
        for order in _single_phase_order_candidates(matchings):
            waves, objective = _materialize_single_phase_order(matchings, order)
            tie = (membership_index, order)
            incumbent_tie = (best_membership, best_order)
            if best_objective is None or (objective, tie) < (best_objective, incumbent_tie):
                best_waves = waves
                best_objective = objective
                best_order = order
                best_membership = membership_index
    matchings = candidates[best_membership] if candidates else ()
    objective_units = int(best_objective or 0)
    source_work = [0] * problem.rank_count
    destination_work = [0] * problem.rank_count
    for group in groups:
        source_work[group.src_rank] += group.count * group.duration
        destination_work[group.dst_rank] += group.count * group.duration
    analytical_bound = max(
        max(source_work, default=0),
        max(destination_work, default=0),
        max(
            (
                group.source_ready + group.duration + group.terminal_tail
                for group in groups
            ),
            default=0,
        ),
        _release_aware_analytical_lower_bound(
            groups,
            rank_count=problem.rank_count,
            release_mode=release_mode,
            wire_cost_model=wire_cost_model,
        ),
    )
    best_bound = min(objective_units, int(analytical_bound))
    certified = objective_units == best_bound
    gap = 0.0 if certified else max(
        0.0,
        (objective_units - best_bound) / max(1, objective_units),
    )
    status = "optimal_by_matching_bound" if certified else "bounded_matching_reference"
    payload: dict[str, Any] = {
        "problem_digest": problem.problem_digest,
        "objective_units": objective_units,
        "best_bound": best_bound,
        "optimality_gap": repr(gap),
        "waves": best_waves,
        "matching_order": best_order,
        "membership_candidate_index": best_membership,
        "membership_candidate_count": len(candidates),
        "model_id": model_id,
        "cost_model_id": cost_model_id,
        "release_model_id": release_model_id,
        "solver_backend": "BIPARTITE_EDGE_COLORING_BOUND",
        "solver_status": status,
        "canonical_task_count": len(tasks),
        "symmetry_group_count": len(groups),
        "candidate_slot_count": len(matchings),
        "incumbent_source": "MULTISTART_LOCAL_PLAN_REFERENCE",
    }
    if extra_digest_payload:
        payload.update(extra_digest_payload)
    return OracleResult(
        supported=True,
        solver_status=status,
        certified_optimal=certified,
        objective_units=objective_units,
        best_bound=best_bound,
        optimality_gap=gap,
        search_nodes=0,
        waves=best_waves,
        model_id=model_id,
        cost_model_id=cost_model_id,
        release_model_id=release_model_id,
        result_digest=stable_digest(payload),
        solver_backend="BIPARTITE_EDGE_COLORING_BOUND",
        solve_time_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
        variable_count=0,
        constraint_count=0,
        canonical_task_count=len(tasks),
        symmetry_group_count=len(groups),
        candidate_slot_count=len(matchings),
        incumbent_source="MULTISTART_LOCAL_PLAN_REFERENCE",
    )

def _failure(
    *,
    status: str,
    reason: str,
    model_id: str,
    cost_model_id: str,
    release_model_id: str,
    supported: bool = True,
    search_nodes: int = 0,
    solve_time_ms: float | None = None,
    variable_count: int = 0,
    constraint_count: int = 0,
    canonical_task_count: int = 0,
    symmetry_group_count: int = 0,
    candidate_slot_count: int = 0,
) -> OracleResult:
    payload = {
        "supported": bool(supported),
        "solver_status": status,
        "reason": reason,
        "search_nodes": int(search_nodes),
        "model_id": model_id,
        "cost_model_id": cost_model_id,
        "release_model_id": release_model_id,
        "solver_backend": SOLVER_BACKEND,
        "variable_count": int(variable_count),
        "constraint_count": int(constraint_count),
        "canonical_task_count": int(canonical_task_count),
        "symmetry_group_count": int(symmetry_group_count),
        "candidate_slot_count": int(candidate_slot_count),
    }
    return OracleResult(
        supported=bool(supported),
        solver_status=str(status),
        certified_optimal=False,
        objective_units=None,
        best_bound=None,
        optimality_gap=None,
        search_nodes=int(search_nodes),
        waves=(),
        model_id=str(model_id),
        cost_model_id=str(cost_model_id),
        release_model_id=str(release_model_id),
        result_digest=stable_digest(payload),
        failure_reason=str(reason),
        solve_time_ms=solve_time_ms,
        variable_count=int(variable_count),
        constraint_count=int(constraint_count),
        canonical_task_count=int(canonical_task_count),
        symmetry_group_count=int(symmetry_group_count),
        candidate_slot_count=int(candidate_slot_count),
        incumbent_source=None,
    )


def _solve_milp(
    problem: SchedulingProblem,
    *,
    task_duration: Callable[[SchedulingTask], int],
    model_id: str,
    cost_model_id: str,
    release_mode: str,
    time_limit_ms: int,
    relative_gap: float,
    wire_cost_model: RSCFWireCostModel | None,
    semantic_phase_ordinal: int | None,
    extra_digest_payload: dict[str, Any] | None = None,
) -> OracleResult:
    started_ns = time.monotonic_ns()
    release_model_id = _release_model_id(release_mode)
    if int(time_limit_ms) <= 0:
        return _failure(
            status="invalid_configuration",
            reason="time_limit_ms must be positive",
            model_id=model_id,
            cost_model_id=cost_model_id,
            release_model_id=release_model_id,
        )
    if not math.isfinite(float(relative_gap)) or not 0.0 <= float(relative_gap) < 1.0:
        return _failure(
            status="invalid_configuration",
            reason="relative_gap must be finite and in [0, 1)",
            model_id=model_id,
            cost_model_id=cost_model_id,
            release_model_id=release_model_id,
        )

    tasks = _sorted_tasks(problem)
    if not tasks:
        payload = {
            "problem_digest": problem.problem_digest,
            "objective_units": 0,
            "waves": (),
            "model_id": model_id,
            "cost_model_id": cost_model_id,
            "release_model_id": release_model_id,
            "solver_backend": SOLVER_BACKEND,
        }
        if extra_digest_payload:
            payload.update(extra_digest_payload)
        return OracleResult(
            supported=True,
            solver_status="optimal",
            certified_optimal=True,
            objective_units=0,
            best_bound=0,
            optimality_gap=0.0,
            search_nodes=0,
            waves=(),
            model_id=model_id,
            cost_model_id=cost_model_id,
            release_model_id=release_model_id,
            result_digest=stable_digest(payload),
            solve_time_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
            variable_count=1,
            constraint_count=0,
        )

    cost = wire_cost_model or RSCFWireCostModel()
    durations = tuple(max(0, int(task_duration(task))) for task in tasks)
    phases = _semantic_phases(tasks, semantic_phase_ordinal)
    source_ready = tuple(
        max(
            0,
            int(tasks[i].ready_at_ns or 0),
            (
                int(round(cost.source_ready(phases[i], tasks[i].src_rank)))
                if wire_cost_model is not None
                else 0
            ),
        )
        for i in range(len(tasks))
    )
    final_phase = max(phases)
    terminal_tail = tuple(
        max(0, int(round(cost.p2_completion_tail(tasks[i].dst_rank))))
        if wire_cost_model is not None and final_phase == 2 and phases[i] == final_phase
        else 0
        for i in range(len(tasks))
    )
    groups = _build_groups(tasks, phases, durations, source_ready, terminal_tail)
    # Large phase-local windows are dominated by slot-assignment symmetry.  A
    # bounded bipartite edge-colouring reference is both much faster and more
    # useful for trace sweeps: it returns a strict feasible incumbent, a valid
    # analytical lower bound, and certifies optimality whenever they meet.
    # Joint/release-coupled problems continue to use the full MILP.
    if len(set(phases)) == 1 and (
        int(time_limit_ms) <= 1_000 or len(tasks) >= 512
    ):
        return _single_phase_bounded_reference(
            problem=problem,
            tasks=tasks,
            groups=groups,
            model_id=model_id,
            cost_model_id=cost_model_id,
            release_model_id=release_model_id,
            release_mode=release_mode,
            wire_cost_model=wire_cost_model,
            started_ns=started_ns,
            extra_digest_payload=extra_digest_payload,
        )
    heuristic_waves, heuristic_objective = _greedy_feasible_waves(
        groups,
        rank_count=problem.rank_count,
        release_mode=release_mode,
        wire_cost_model=wire_cost_model,
    )
    slot_count = max(
        _candidate_slot_count(groups, problem.rank_count),
        len(heuristic_waves),
    )
    release_keys = _release_keys(groups, problem.rank_count, release_mode)
    release_index = {key: index for index, key in enumerate(release_keys)}
    index = _Index(len(groups), slot_count, len(release_keys))

    max_delay = (
        max(
            (max(0, int(round(cost.p1_to_p2_delay(rank)))) for rank in range(problem.rank_count)),
            default=0,
        )
        if wire_cost_model is not None
        else 0
    )
    horizon = max(
        1,
        max(source_ready, default=0)
        + sum(durations)
        + max(0, len(set(phases)) - 1) * max_delay
        + max(terminal_tail, default=0)
        + 1,
    )
    big_m = float(horizon)
    max_duration = max(durations, default=0)

    objective = np.zeros(index.variable_count, dtype=float)
    objective[index.cmax] = 1.0
    # The total secondary weight stays below one time unit and therefore cannot
    # change the integral makespan optimum.  It only gives HiGHS a stable
    # preference for a compact prefix and deterministic group placement.
    wave_epsilon = 1.0 / (8.0 * (slot_count + 1))
    assignment_epsilon = 1.0 / (32.0 * (len(groups) + 1) * (slot_count + 1) ** 2)
    for slot in range(slot_count):
        objective[index.y(slot)] = wave_epsilon
    for group in range(len(groups)):
        for slot in range(slot_count):
            objective[index.x(group, slot)] = assignment_epsilon * (group + 1) * (slot + 1)

    integrality = np.zeros(index.variable_count, dtype=np.uint8)
    integrality[index.x0:index.d0] = 1
    integrality[index.y0:index.r0] = 1

    lower_bounds = np.zeros(index.variable_count, dtype=float)
    source_work = [0] * problem.rank_count
    destination_work = [0] * problem.rank_count
    for group in groups:
        source_work[group.src_rank] += group.count * group.duration
        destination_work[group.dst_rank] += group.count * group.duration
    critical_lower_bound = max(
        max(source_work, default=0),
        max(destination_work, default=0),
        max(
            (
                group.source_ready + group.duration + group.terminal_tail
                for group in groups
            ),
            default=0,
        ),
        _release_aware_analytical_lower_bound(
            groups,
            rank_count=problem.rank_count,
            release_mode=release_mode,
            wire_cost_model=wire_cost_model,
        ),
    )
    lower_bounds[index.cmax] = float(critical_lower_bound)
    upper_bounds = np.full(index.variable_count, float(horizon), dtype=float)
    upper_bounds[index.cmax] = float(heuristic_objective)
    upper_bounds[index.x0:index.d0] = 1.0
    upper_bounds[index.y0:index.r0] = 1.0
    for slot in range(slot_count):
        upper_bounds[index.d(slot)] = float(max_duration)

    rows = _Rows()

    # Every canonical task remains represented: a group with multiplicity m
    # must occupy exactly m distinct matching waves.
    for group in groups:
        rows.add(
            ((index.x(group.group_id, slot), 1.0) for slot in range(slot_count)),
            lower=float(group.count),
            upper=float(group.count),
        )

    # Per-rank independent TX and RX resources.  TX and RX are intentionally
    # separate, so a rank may send and receive in the same wave.
    for slot in range(slot_count):
        for rank in range(problem.rank_count):
            rows.add(
                (
                    (index.x(group.group_id, slot), 1.0)
                    for group in groups
                    if group.src_rank == rank
                ),
                upper=1.0,
            )
            rows.add(
                (
                    (index.x(group.group_id, slot), 1.0)
                    for group in groups
                    if group.dst_rank == rank
                ),
                upper=1.0,
            )

    # Used waves form a gap-free prefix.
    for slot in range(slot_count):
        occupancy = [(index.x(group.group_id, slot), 1.0) for group in groups]
        rows.add((*occupancy, (index.y(slot), -float(len(groups)))), upper=0.0)
        rows.add((*occupancy, (index.y(slot), -1.0)), lower=0.0)
        rows.add(
            ((index.d(slot), 1.0), (index.y(slot), -float(max_duration))),
            upper=0.0,
        )
    for slot in range(slot_count - 1):
        rows.add(
            ((index.y(slot), 1.0), (index.y(slot + 1), -1.0)),
            lower=0.0,
        )

    # Wave duration is the maximum exact canonical service time selected in it.
    for group in groups:
        for slot in range(slot_count):
            rows.add(
                (
                    (index.d(slot), 1.0),
                    (index.x(group.group_id, slot), -float(group.duration)),
                ),
                lower=0.0,
            )

    # Non-preemptive sequential logical waves; release constraints may add idle
    # time between waves.
    for slot in range(slot_count - 1):
        rows.add(
            (
                (index.s(slot + 1), 1.0),
                (index.s(slot), -1.0),
                (index.d(slot), -1.0),
            ),
            lower=0.0,
        )

    # Source readiness for every selected canonical task class.
    for group in groups:
        for slot in range(slot_count):
            rows.add(
                (
                    (index.s(slot), 1.0),
                    (index.x(group.group_id, slot), -big_m),
                ),
                lower=float(group.source_ready) - big_m,
            )

    phase_values = sorted({group.phase for group in groups})
    normalized_release = str(release_mode).upper()
    for boundary, (upstream_phase, downstream_phase) in enumerate(
        zip(phase_values, phase_values[1:])
    ):
        upstream = tuple(group for group in groups if group.phase == upstream_phase)
        downstream = tuple(group for group in groups if group.phase == downstream_phase)
        if normalized_release == "PHASE_BARRIER":
            release_var = index.release(release_index[(boundary, None)])
            for group in upstream:
                for slot in range(slot_count):
                    rows.add(
                        (
                            (release_var, 1.0),
                            (index.s(slot), -1.0),
                            (index.x(group.group_id, slot), -big_m),
                        ),
                        lower=float(group.duration) - big_m,
                    )
            for group in downstream:
                delay = (
                    max(0, int(round(cost.p1_to_p2_delay(group.src_rank))))
                    if wire_cost_model is not None
                    else 0
                )
                for slot in range(slot_count):
                    rows.add(
                        (
                            (index.s(slot), 1.0),
                            (release_var, -1.0),
                            (index.x(group.group_id, slot), -big_m),
                        ),
                        lower=float(delay) - big_m,
                    )
        else:
            for rank in range(problem.rank_count):
                release_var = index.release(release_index[(boundary, rank)])
                for group in upstream:
                    if group.dst_rank != rank:
                        continue
                    for slot in range(slot_count):
                        rows.add(
                            (
                                (release_var, 1.0),
                                (index.s(slot), -1.0),
                                (index.x(group.group_id, slot), -big_m),
                            ),
                            lower=float(group.duration) - big_m,
                        )
                delay = (
                    max(0, int(round(cost.p1_to_p2_delay(rank))))
                    if wire_cost_model is not None
                    else 0
                )
                for group in downstream:
                    if group.src_rank != rank:
                        continue
                    for slot in range(slot_count):
                        rows.add(
                            (
                                (index.s(slot), 1.0),
                                (release_var, -1.0),
                                (index.x(group.group_id, slot), -big_m),
                            ),
                            lower=float(delay) - big_m,
                        )

    # Terminal critical completion, including P2 destination-local tail.
    for group in groups:
        for slot in range(slot_count):
            rows.add(
                (
                    (index.cmax, 1.0),
                    (index.s(slot), -1.0),
                    (index.x(group.group_id, slot), -big_m),
                ),
                lower=float(group.duration + group.terminal_tail) - big_m,
            )

    constraint = rows.constraint(index.variable_count)
    options = {
        "time_limit": float(time_limit_ms) / 1000.0,
        "mip_rel_gap": float(relative_gap),
        "presolve": bool(index.variable_count <= 5_000),
        "disp": False,
    }
    try:
        raw = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=constraint,
            options=options,
        )
    except Exception as exc:
        return _failure(
            status="solver_error",
            reason=f"{type(exc).__name__}: {exc}",
            model_id=model_id,
            cost_model_id=cost_model_id,
            release_model_id=release_model_id,
            solve_time_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
            variable_count=index.variable_count,
            constraint_count=rows.count,
            canonical_task_count=len(tasks),
            symmetry_group_count=len(groups),
            candidate_slot_count=slot_count,
        )

    solve_time_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
    search_nodes = int(getattr(raw, "mip_node_count", 0) or 0)
    raw_gap = getattr(raw, "mip_gap", None)
    gap = None if raw_gap is None or not math.isfinite(float(raw_gap)) else float(raw_gap)
    raw_bound = getattr(raw, "mip_dual_bound", None)
    best_bound = (
        None
        if raw_bound is None or not math.isfinite(float(raw_bound))
        else max(0, int(math.floor(float(raw_bound) + 1.0e-7)))
    )

    analytical_bound = int(critical_lower_bound)
    if best_bound is None:
        best_bound = analytical_bound
    else:
        best_bound = max(best_bound, analytical_bound)

    if raw.x is None:
        if int(raw.status) in {2, 3, 4}:
            statuses = {2: "infeasible", 3: "unbounded", 4: "solver_error"}
            return _failure(
                status=statuses[int(raw.status)],
                reason=str(raw.message),
                model_id=model_id,
                cost_model_id=cost_model_id,
                release_model_id=release_model_id,
                search_nodes=search_nodes,
                solve_time_ms=solve_time_ms,
                variable_count=index.variable_count,
                constraint_count=rows.count,
                canonical_task_count=len(tasks),
                symmetry_group_count=len(groups),
                candidate_slot_count=slot_count,
            )
        objective_units = int(heuristic_objective)
        fallback_gap = max(0.0, (objective_units - best_bound) / max(1, objective_units))
        payload: dict[str, Any] = {
            "problem_digest": problem.problem_digest,
            "objective_units": objective_units,
            "best_bound": best_bound,
            "optimality_gap": repr(fallback_gap),
            "waves": heuristic_waves,
            "search_nodes": search_nodes,
            "model_id": model_id,
            "cost_model_id": cost_model_id,
            "release_model_id": release_model_id,
            "solver_backend": SOLVER_BACKEND,
            "solver_status": "time_limit_heuristic_feasible",
            "variable_count": index.variable_count,
            "constraint_count": rows.count,
            "canonical_task_count": len(tasks),
            "symmetry_group_count": len(groups),
            "candidate_slot_count": slot_count,
            "incumbent_source": "DETERMINISTIC_FEASIBLE_BOUND",
        }
        if extra_digest_payload:
            payload.update(extra_digest_payload)
        return OracleResult(
            supported=True,
            solver_status="time_limit_heuristic_feasible",
            certified_optimal=False,
            objective_units=objective_units,
            best_bound=best_bound,
            optimality_gap=fallback_gap,
            search_nodes=search_nodes,
            waves=heuristic_waves,
            model_id=model_id,
            cost_model_id=cost_model_id,
            release_model_id=release_model_id,
            result_digest=stable_digest(payload),
            failure_reason=str(raw.message),
            solve_time_ms=solve_time_ms,
            variable_count=index.variable_count,
            constraint_count=rows.count,
            canonical_task_count=len(tasks),
            symmetry_group_count=len(groups),
            candidate_slot_count=slot_count,
            incumbent_source="DETERMINISTIC_FEASIBLE_BOUND",
        )

    values = np.asarray(raw.x, dtype=float)
    selected_slots_by_group: dict[int, list[int]] = {}
    for group in groups:
        selected = [
            slot
            for slot in range(slot_count)
            if values[index.x(group.group_id, slot)] >= 0.5
        ]
        if len(selected) != group.count:
            return _failure(
                status="invalid_incumbent",
                reason=(
                    f"group {group.group_id} assigned {len(selected)} slots for "
                    f"{group.count} canonical tasks"
                ),
                model_id=model_id,
                cost_model_id=cost_model_id,
                release_model_id=release_model_id,
                search_nodes=search_nodes,
                solve_time_ms=solve_time_ms,
                variable_count=index.variable_count,
                constraint_count=rows.count,
                canonical_task_count=len(tasks),
                symmetry_group_count=len(groups),
                candidate_slot_count=slot_count,
            )
        selected_slots_by_group[group.group_id] = sorted(selected)

    task_ids_by_slot: dict[int, list[str]] = defaultdict(list)
    group_by_id = {group.group_id: group for group in groups}
    for group_id, selected_slots in selected_slots_by_group.items():
        group = group_by_id[group_id]
        for task_id, slot in zip(group.task_ids, selected_slots):
            task_ids_by_slot[slot].append(task_id)

    used_slots = sorted(task_ids_by_slot)
    waves: list[OracleWave] = []
    task_by_id = {task.task_id: task for task in tasks}
    duration_by_task = {task.task_id: duration for task, duration in zip(tasks, durations)}
    tail_by_task = {task.task_id: tail for task, tail in zip(tasks, terminal_tail)}
    for wave_id, slot in enumerate(used_slots):
        members = tuple(
            sorted(
                task_ids_by_slot[slot],
                key=lambda task_id: (
                    task_by_id[task_id].src_rank,
                    task_by_id[task_id].dst_rank,
                    task_by_id[task_id].phase_ordinal,
                    task_by_id[task_id].chunk_index,
                    task_by_id[task_id].byte_offset,
                    task_id,
                ),
            )
        )
        start_units = max(0, int(round(values[index.s(slot)])))
        duration_units = max((duration_by_task[task_id] for task_id in members), default=0)
        waves.append(
            OracleWave(
                wave_id=wave_id,
                task_ids=members,
                duration_units=int(duration_units),
                start_units=start_units,
                finish_units=start_units + int(duration_units),
            )
        )

    milp_objective = max(
        (
            wave.start_units
            + duration_by_task[task_id]
            + tail_by_task[task_id]
            for wave in waves
            for task_id in wave.task_ids
        ),
        default=0,
    )
    if int(heuristic_objective) < int(milp_objective):
        waves = list(heuristic_waves)
        objective_units = int(heuristic_objective)
        incumbent_source = "DETERMINISTIC_FEASIBLE_BOUND"
    else:
        objective_units = int(milp_objective)
        incumbent_source = "HIGHS_MILP_INCUMBENT"
    certified = int(raw.status) == 0
    if certified:
        status = "optimal"
        best_bound = objective_units
        gap = 0.0
        incumbent_source = "HIGHS_CERTIFIED_OPTIMUM"
    elif int(raw.status) == 1:
        status = "time_limit_feasible"
        gap = max(0.0, (objective_units - best_bound) / max(1, objective_units))
    else:
        status = "feasible_not_certified"
        gap = max(0.0, (objective_units - best_bound) / max(1, objective_units))

    payload: dict[str, Any] = {
        "problem_digest": problem.problem_digest,
        "objective_units": objective_units,
        "best_bound": best_bound,
        "optimality_gap": None if gap is None else repr(gap),
        "waves": tuple(waves),
        "search_nodes": search_nodes,
        "model_id": model_id,
        "cost_model_id": cost_model_id,
        "release_model_id": release_model_id,
        "solver_backend": SOLVER_BACKEND,
        "solver_status": status,
        "variable_count": index.variable_count,
        "constraint_count": rows.count,
        "canonical_task_count": len(tasks),
        "symmetry_group_count": len(groups),
        "candidate_slot_count": slot_count,
        "incumbent_source": incumbent_source,
    }
    if extra_digest_payload:
        payload.update(extra_digest_payload)
    return OracleResult(
        supported=True,
        solver_status=status,
        certified_optimal=certified,
        objective_units=int(objective_units),
        best_bound=best_bound,
        optimality_gap=gap,
        search_nodes=search_nodes,
        waves=tuple(waves),
        model_id=model_id,
        cost_model_id=cost_model_id,
        release_model_id=release_model_id,
        result_digest=stable_digest(payload),
        failure_reason=None if certified else str(raw.message),
        solve_time_ms=solve_time_ms,
        variable_count=index.variable_count,
        constraint_count=rows.count,
        canonical_task_count=len(tasks),
        symmetry_group_count=len(groups),
        candidate_slot_count=slot_count,
        incumbent_source=incumbent_source,
    )


def solve_exact(
    problem: SchedulingProblem,
    *,
    time_limit_ms: int = 30_000,
    relative_gap: float = 0.0,
    release_mode: str = "RANK_LOCAL",
    semantic_phase_ordinal: int | None = None,
) -> OracleResult:
    """Solve the exact canonical matching-wave problem in payload units."""

    return _solve_milp(
        problem,
        task_duration=lambda task: int(task.payload_bytes),
        model_id=MILP_MODEL_ID,
        cost_model_id=PAYLOAD_COST_MODEL_ID,
        release_mode=release_mode,
        time_limit_ms=time_limit_ms,
        relative_gap=relative_gap,
        wire_cost_model=None,
        semantic_phase_ordinal=semantic_phase_ordinal,
    )



def solve_bounded_wire(
    problem: SchedulingProblem,
    *,
    wire_cost_model: RSCFWireCostModel | None = None,
    release_mode: str = "RANK_LOCAL",
    semantic_phase_ordinal: int | None = None,
) -> OracleResult:
    """Return a deterministic feasible reference and a valid analytical bound.

    This path never enters MILP.  For a single phase it uses the stronger
    multistart bipartite edge-colouring reference.  For release-coupled P12 it
    uses the deterministic release-aware feasible constructor.  Every
    canonical task ID is preserved, and only ``certified_optimal=True`` may be
    interpreted as an exact optimum.
    """

    started_ns = time.monotonic_ns()
    cost = wire_cost_model or RSCFWireCostModel()
    release_model_id = _release_model_id(release_mode)
    tasks = _sorted_tasks(problem)
    wire_payload = {
        "wire_cost_model": {
            "default_slope": repr(cost.default_slope),
            "default_intercept": repr(cost.default_intercept),
            "wave_launch_cost": repr(cost.wave_launch_cost),
            "edge_slope": tuple((src, dst, repr(value)) for src, dst, value in cost.edge_slope),
            "edge_intercept": tuple((src, dst, repr(value)) for src, dst, value in cost.edge_intercept),
            "edge_launch": tuple((src, dst, repr(value)) for src, dst, value in cost.edge_launch),
            "source_ready_by_phase_rank": tuple(
                (phase, rank, repr(value))
                for phase, rank, value in cost.source_ready_by_phase_rank
            ),
            "p1_to_p2_delay_by_rank": tuple(
                (rank, repr(value)) for rank, value in cost.p1_to_p2_delay_by_rank
            ),
            "p2_completion_tail_by_rank": tuple(
                (rank, repr(value)) for rank, value in cost.p2_completion_tail_by_rank
            ),
        }
    }
    if not tasks:
        payload = {
            "problem_digest": problem.problem_digest,
            "objective_units": 0,
            "best_bound": 0,
            "waves": (),
            "model_id": BOUNDED_REFERENCE_MODEL_ID,
            "cost_model_id": WIRE_COST_MODEL_ID,
            "release_model_id": release_model_id,
            "solver_backend": BOUNDED_SOLVER_BACKEND,
            **wire_payload,
        }
        return OracleResult(
            supported=True,
            solver_status="optimal_by_analytical_bound",
            certified_optimal=True,
            objective_units=0,
            best_bound=0,
            optimality_gap=0.0,
            search_nodes=0,
            waves=(),
            model_id=BOUNDED_REFERENCE_MODEL_ID,
            cost_model_id=WIRE_COST_MODEL_ID,
            release_model_id=release_model_id,
            result_digest=stable_digest(payload),
            solver_backend=BOUNDED_SOLVER_BACKEND,
            solve_time_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
            canonical_task_count=0,
            incumbent_source="EMPTY_SCHEDULE",
        )

    durations = tuple(
        max(
            0,
            int(
                round(
                    cost.launch(task.src_rank, task.dst_rank)
                    + cost.duration(task.src_rank, task.dst_rank, task.payload_bytes)
                )
            ),
        )
        for task in tasks
    )
    phases = _semantic_phases(tasks, semantic_phase_ordinal)
    source_ready = tuple(
        max(
            0,
            int(tasks[i].ready_at_ns or 0),
            int(round(cost.source_ready(phases[i], tasks[i].src_rank))),
        )
        for i in range(len(tasks))
    )
    final_phase = max(phases)
    terminal_tail = tuple(
        max(0, int(round(cost.p2_completion_tail(tasks[i].dst_rank))))
        if phases[i] == final_phase and final_phase == 2
        else 0
        for i in range(len(tasks))
    )
    groups = _build_groups(tasks, phases, durations, source_ready, terminal_tail)

    if len(set(phases)) == 1:
        return _single_phase_bounded_reference(
            problem=problem,
            tasks=tasks,
            groups=groups,
            model_id=BOUNDED_REFERENCE_MODEL_ID,
            cost_model_id=WIRE_COST_MODEL_ID,
            release_model_id=release_model_id,
            release_mode=release_mode,
            wire_cost_model=cost,
            started_ns=started_ns,
            extra_digest_payload=wire_payload,
        )

    waves, objective = _greedy_feasible_waves(
        groups,
        rank_count=problem.rank_count,
        release_mode=release_mode,
        wire_cost_model=cost,
    )
    lower_bound = _release_aware_analytical_lower_bound(
        groups,
        rank_count=problem.rank_count,
        release_mode=release_mode,
        wire_cost_model=cost,
    )
    best_bound = min(int(objective), int(lower_bound))
    certified = int(objective) == best_bound
    gap = 0.0 if certified else max(
        0.0,
        (int(objective) - best_bound) / max(1, int(objective)),
    )
    status = (
        "optimal_by_analytical_bound"
        if certified
        else "bounded_release_reference"
    )
    payload = {
        "problem_digest": problem.problem_digest,
        "objective_units": int(objective),
        "best_bound": best_bound,
        "optimality_gap": repr(gap),
        "waves": waves,
        "model_id": BOUNDED_REFERENCE_MODEL_ID,
        "cost_model_id": WIRE_COST_MODEL_ID,
        "release_model_id": release_model_id,
        "solver_backend": BOUNDED_SOLVER_BACKEND,
        "canonical_task_count": len(tasks),
        "symmetry_group_count": len(groups),
        "incumbent_source": "DETERMINISTIC_RELEASE_AWARE_GREEDY",
        **wire_payload,
    }
    return OracleResult(
        supported=True,
        solver_status=status,
        certified_optimal=certified,
        objective_units=int(objective),
        best_bound=best_bound,
        optimality_gap=gap,
        search_nodes=0,
        waves=waves,
        model_id=BOUNDED_REFERENCE_MODEL_ID,
        cost_model_id=WIRE_COST_MODEL_ID,
        release_model_id=release_model_id,
        result_digest=stable_digest(payload),
        solver_backend=BOUNDED_SOLVER_BACKEND,
        solve_time_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
        variable_count=0,
        constraint_count=0,
        canonical_task_count=len(tasks),
        symmetry_group_count=len(groups),
        candidate_slot_count=len(waves),
        incumbent_source="DETERMINISTIC_RELEASE_AWARE_GREEDY",
    )


def solve_exact_wire(
    problem: SchedulingProblem,
    *,
    wire_cost_model: RSCFWireCostModel | None = None,
    time_limit_ms: int = 30_000,
    relative_gap: float = 0.0,
    release_mode: str = "RANK_LOCAL",
    semantic_phase_ordinal: int | None = None,
) -> OracleResult:
    """Solve canonical tasks using the same deterministic wire model as RSCF."""

    cost = wire_cost_model or RSCFWireCostModel()
    wire_payload = {
        "wire_cost_model": {
            "default_slope": repr(cost.default_slope),
            "default_intercept": repr(cost.default_intercept),
            "wave_launch_cost": repr(cost.wave_launch_cost),
            "edge_slope": tuple((src, dst, repr(value)) for src, dst, value in cost.edge_slope),
            "edge_intercept": tuple((src, dst, repr(value)) for src, dst, value in cost.edge_intercept),
            "edge_launch": tuple((src, dst, repr(value)) for src, dst, value in cost.edge_launch),
            "source_ready_by_phase_rank": tuple(
                (phase, rank, repr(value))
                for phase, rank, value in cost.source_ready_by_phase_rank
            ),
            "p1_to_p2_delay_by_rank": tuple(
                (rank, repr(value)) for rank, value in cost.p1_to_p2_delay_by_rank
            ),
            "p2_completion_tail_by_rank": tuple(
                (rank, repr(value)) for rank, value in cost.p2_completion_tail_by_rank
            ),
        }
    }
    return _solve_milp(
        problem,
        task_duration=lambda task: max(
            0,
            int(
                round(
                    cost.launch(task.src_rank, task.dst_rank)
                    + cost.duration(task.src_rank, task.dst_rank, task.payload_bytes)
                )
            ),
        ),
        model_id=MILP_MODEL_ID,
        cost_model_id=WIRE_COST_MODEL_ID,
        release_mode=release_mode,
        time_limit_ms=time_limit_ms,
        relative_gap=relative_gap,
        wire_cost_model=cost,
        semantic_phase_ordinal=semantic_phase_ordinal,
        extra_digest_payload=wire_payload,
    )


__all__ = [
    "BOUNDED_REFERENCE_MODEL_ID",
    "BOUNDED_SOLVER_BACKEND",
    "MILP_MODEL_ID",
    "OracleResult",
    "OracleWave",
    "PAYLOAD_COST_MODEL_ID",
    "SOLVER_BACKEND",
    "WIRE_COST_MODEL_ID",
    "solve_exact",
    "solve_bounded_wire",
    "solve_exact_wire",
]
