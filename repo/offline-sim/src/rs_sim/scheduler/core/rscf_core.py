from __future__ import annotations

"""RouterSense Critical Frontier ordering kernel.

The kernel has one formal route.  It orders immutable canonical communication
Tasks and receives generic phase-release dependencies from outer scheduling
decorators.  Local/Joint, Event/Global, prediction, binding, and Safe are not
algorithm variants and do not select alternative scoring implementations.
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from .digest import stable_digest
from .matching import maximum_weight_bipartite_matching


@dataclass(frozen=True, slots=True)
class RSCFTask:
    task_id: str
    phase: int
    src_rank: int
    dst_rank: int
    payload_units: int
    chunk_index: int = 0
    byte_offset: int = 0
    ready_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("RSCF task_id must be non-empty")
        for name in ("phase", "src_rank", "dst_rank", "payload_units", "chunk_index", "byte_offset"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"RSCF {name} must be a non-negative int")
        if self.src_rank == self.dst_rank:
            raise ValueError("RSCF local transfers must not enter the data-plane task set")
        if self.payload_units <= 0:
            raise ValueError("RSCF payload_units must be positive")
        if float(self.ready_at) < 0.0:
            raise ValueError("RSCF ready_at must be non-negative")


@dataclass(frozen=True, slots=True)
class RSCFReleaseDependency:
    """Dependency between two communication phases.

    ``release_scope`` is authoritative and must match the runtime release
    contract.  ``RANK_LOCAL`` releases one downstream source after that
    rank's upstream inbound set completes.  ``PHASE_BARRIER`` releases every
    downstream source only after the entire upstream phase completes.
    """

    upstream_phase: int
    downstream_phase: int
    delay_by_rank: tuple[tuple[int, float], ...] = ()
    release_scope: str = "RANK_LOCAL"

    def __post_init__(self) -> None:
        if self.upstream_phase < 0 or self.downstream_phase < 0:
            raise ValueError("release dependency phases must be non-negative")
        if self.upstream_phase == self.downstream_phase:
            raise ValueError("release dependency must cross phases")
        normalized_scope = str(self.release_scope).upper()
        if normalized_scope not in {"RANK_LOCAL", "PHASE_BARRIER"}:
            raise ValueError("release_scope must be RANK_LOCAL or PHASE_BARRIER")
        object.__setattr__(self, "release_scope", normalized_scope)
        ranks: set[int] = set()
        for rank, delay in self.delay_by_rank:
            if int(rank) < 0 or int(rank) in ranks:
                raise ValueError("release dependency ranks must be unique and non-negative")
            if float(delay) < 0.0:
                raise ValueError("release dependency delays must be non-negative")
            ranks.add(int(rank))

    def delay(self, rank: int) -> float:
        for item_rank, value in self.delay_by_rank:
            if int(item_rank) == int(rank):
                return max(0.0, float(value))
        return 0.0


@dataclass(frozen=True, slots=True)
class RSCFParameters:
    """Frozen parameters for the single formal rank-release objective."""

    path_weight: float = 20.0
    release_weight: float = 8.0
    endpoint_weight: float = 1.0
    work_weight: float = 0.5
    softmax_temperature: float = 0.35
    downstream_max_weight: float = 0.5
    downstream_mean_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "path_weight",
            "release_weight",
            "endpoint_weight",
            "work_weight",
            "softmax_temperature",
            "downstream_max_weight",
            "downstream_mean_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"RSCF parameter {name} must be finite and non-negative")
        if self.softmax_temperature <= 0.0:
            raise ValueError("RSCF softmax_temperature must be positive")


RSCF_PARAMETERS = RSCFParameters()


@dataclass(frozen=True, slots=True)
class RSCFScoredTask:
    task_id: str
    phase: int
    src_rank: int
    dst_rank: int
    score: float
    critical_path_dual: float
    transitive_unlock: float
    endpoint_dual: float
    residual_component: float
    barrier_component: float
    release_gain_component: float
    age_component: float
    barrier_urgency: float = 0.0
    release_gain: float = 0.0


@dataclass(frozen=True, slots=True)
class RSCFServiceSegment:
    task_id: str
    phase: int
    src_rank: int
    dst_rank: int
    byte_offset: int
    service_units: int


@dataclass(frozen=True, slots=True)
class RSCFWireCostModel:
    default_slope: float = 1.0
    default_intercept: float = 0.0
    wave_launch_cost: float = 0.0
    edge_slope: tuple[tuple[int, int, float], ...] = ()
    edge_intercept: tuple[tuple[int, int, float], ...] = ()
    edge_launch: tuple[tuple[int, int, float], ...] = ()
    source_ready_by_phase_rank: tuple[tuple[int, int, float], ...] = ()
    p1_to_p2_delay_by_rank: tuple[tuple[int, float], ...] = ()
    p2_completion_tail_by_rank: tuple[tuple[int, float], ...] = ()

    def _lookup(
        self,
        rows: tuple[tuple[int, int, float], ...],
        src: int,
        dst: int,
        default: float,
    ) -> float:
        for row_src, row_dst, value in rows:
            if int(row_src) == int(src) and int(row_dst) == int(dst):
                return float(value)
        return float(default)

    def duration(self, src: int, dst: int, payload_units: float) -> float:
        slope = self._lookup(self.edge_slope, src, dst, self.default_slope)
        intercept = self._lookup(self.edge_intercept, src, dst, self.default_intercept)
        return max(0.0, intercept + slope * float(payload_units))

    def launch(self, src: int, dst: int) -> float:
        return max(0.0, self._lookup(self.edge_launch, src, dst, self.wave_launch_cost))

    @staticmethod
    def _rank_lookup(rows: tuple[tuple[int, float], ...], rank: int) -> float:
        for row_rank, value in rows:
            if int(row_rank) == int(rank):
                return max(0.0, float(value))
        return 0.0

    def source_ready(self, phase: int, rank: int) -> float:
        for row_phase, row_rank, value in self.source_ready_by_phase_rank:
            if int(row_phase) == int(phase) and int(row_rank) == int(rank):
                return max(0.0, float(value))
        return 0.0

    def p1_to_p2_delay(self, rank: int) -> float:
        return self._rank_lookup(self.p1_to_p2_delay_by_rank, rank)

    def p2_completion_tail(self, rank: int) -> float:
        return self._rank_lookup(self.p2_completion_tail_by_rank, rank)


@dataclass(frozen=True, slots=True)
class RSCFWave:
    wave_id: int
    task_ids: tuple[str, ...]
    service_units: int
    start_time: float
    finish_time: float
    segments: tuple[RSCFServiceSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class RSCFPlan:
    ordered_task_ids: tuple[str, ...]
    waves: tuple[RSCFWave, ...]
    scored_tasks: tuple[RSCFScoredTask, ...]
    plan_digest: str
    tie_break_rule: str


def rank_local_release_dependency(
    *,
    upstream_phase: int,
    downstream_phase: int,
    rank_count: int,
    delay_provider,
) -> RSCFReleaseDependency:
    """Build the generic dependency used by the Joint scope decorator."""

    return RSCFReleaseDependency(
        upstream_phase=int(upstream_phase),
        downstream_phase=int(downstream_phase),
        delay_by_rank=tuple(
            (rank, max(0.0, float(delay_provider(rank))))
            for rank in range(int(rank_count))
        ),
        release_scope="RANK_LOCAL",
    )


def phase_barrier_release_dependency(
    *,
    upstream_phase: int,
    downstream_phase: int,
    rank_count: int,
    delay_provider,
) -> RSCFReleaseDependency:
    """Build the global phase-barrier dependency used by formal P12 runs."""

    return RSCFReleaseDependency(
        upstream_phase=int(upstream_phase),
        downstream_phase=int(downstream_phase),
        delay_by_rank=tuple(
            (rank, max(0.0, float(delay_provider(rank))))
            for rank in range(int(rank_count))
        ),
        release_scope="PHASE_BARRIER",
    )


def _softmax_prices(
    values: dict[int, float], *, temperature: float
) -> dict[int, float]:
    if not values:
        return {}
    scale = max(1.0, max(values.values()))
    logits = {
        key: float(value) / scale / max(float(temperature), 1e-9)
        for key, value in values.items()
    }
    maximum = max(logits.values())
    exponentials = {
        key: math.exp(max(-60.0, min(60.0, value - maximum)))
        for key, value in logits.items()
    }
    total = sum(exponentials.values()) or 1.0
    return {key: value / total for key, value in exponentials.items()}


def order_rscf(
    tasks: Iterable[RSCFTask],
    *,
    rank_count: int,
    wire_cost_model: RSCFWireCostModel | None = None,
    release_dependencies: Iterable[RSCFReleaseDependency] = (),
    parameters: RSCFParameters = RSCF_PARAMETERS,
) -> RSCFPlan:
    """Order canonical tasks using the complete rank-release chain.

    The planner serves exactly one canonical task on each selected physical
    edge.  After every non-preemptive matching wave it recomputes:

    * remaining inbound service for each rank;
    * rank-local compute between dependent phases;
    * remaining downstream communication;
    * destination-local compute before the next communication phase;
    * the largest unfinished source edge into each destination.

    This keeps the planning unit identical to the runtime execution unit and
    directly targets the rank that determines the P1-to-following-P1 window.
    """

    items = tuple(tasks)
    if int(rank_count) <= 0:
        raise ValueError("rank_count must be positive")
    if len({task.task_id for task in items}) != len(items):
        raise ValueError("RSCF tasks must have unique IDs")
    if any(
        task.src_rank >= int(rank_count) or task.dst_rank >= int(rank_count)
        for task in items
    ):
        raise ValueError("RSCF task endpoint exceeds rank_count")
    if not items:
        return RSCFPlan(
            ordered_task_ids=(),
            waves=(),
            scored_tasks=(),
            plan_digest=stable_digest({"algorithm": "rscf", "tasks": ()}),
            tie_break_rule="rank-release path; empty task set",
        )

    model = wire_cost_model or RSCFWireCostModel()
    dependencies = tuple(release_dependencies)
    by_id = {task.task_id: task for task in items}
    queues: dict[tuple[int, int, int], deque[str]] = {}
    for task in sorted(
        items,
        key=lambda item: (
            item.phase,
            item.src_rank,
            item.dst_rank,
            item.chunk_index,
            item.byte_offset,
            item.task_id,
        ),
    ):
        queues.setdefault(
            (task.phase, task.src_rank, task.dst_rank), deque()
        ).append(task.task_id)

    phases = tuple(sorted({task.phase for task in items}))
    pending_inbound_count = {
        (phase, rank): sum(
            1
            for task in items
            if task.phase == phase and task.dst_rank == rank
        )
        for phase in phases
        for rank in range(int(rank_count))
    }
    dependency_by_upstream = {
        dependency.upstream_phase: dependency for dependency in dependencies
    }
    release_time = {
        (phase, rank): 0.0
        for phase in phases
        for rank in range(int(rank_count))
    }
    for dependency in dependencies:
        if dependency.upstream_phase not in phases or dependency.downstream_phase not in phases:
            continue
        upstream_pending_global = sum(
            pending_inbound_count.get((dependency.upstream_phase, rank), 0)
            for rank in range(int(rank_count))
        )
        for rank in range(int(rank_count)):
            blocked = (
                upstream_pending_global > 0
                if dependency.release_scope == "PHASE_BARRIER"
                else pending_inbound_count.get((dependency.upstream_phase, rank), 0) > 0
            )
            release_time[(dependency.downstream_phase, rank)] = (math.inf if blocked else 0.0)

    remaining = set(by_id)
    current_time = 0.0
    waves: list[RSCFWave] = []
    scored_history: list[RSCFScoredTask] = []

    while remaining:
        edge_wire: dict[tuple[int, int, int], float] = {}
        incoming: dict[tuple[int, int], float] = defaultdict(float)
        outgoing: dict[tuple[int, int], float] = defaultdict(float)
        send_total: dict[int, float] = defaultdict(float)
        recv_total: dict[int, float] = defaultdict(float)
        for edge, task_ids in queues.items():
            if not task_ids:
                continue
            phase, src, dst = edge
            value = sum(
                model.duration(src, dst, by_id[task_id].payload_units)
                for task_id in task_ids
            )
            edge_wire[edge] = value
            incoming[(phase, dst)] += value
            outgoing[(phase, src)] += value
            send_total[src] += value
            recv_total[dst] += value

        final_phase = max(phases)
        terminal_tail = (
            model.p2_completion_tail
            if int(final_phase) == 2
            else (lambda _rank: 0.0)
        )
        final_destination_path = {
            rank: incoming.get((final_phase, rank), 0.0)
            + terminal_tail(rank)
            for rank in range(int(rank_count))
        }
        phase_path_by_rank: dict[int, dict[int, float]] = {
            final_phase: final_destination_path
        }
        # Propagate downstream path costs backwards through generic release
        # dependencies.  For the Current-P12 composition this is P1 -> P2.
        for phase in reversed(phases[:-1]):
            dependency = dependency_by_upstream.get(phase)
            if dependency is None:
                phase_path_by_rank[phase] = {
                    rank: incoming.get((phase, rank), 0.0)
                    for rank in range(int(rank_count))
                }
                continue
            downstream_phase = dependency.downstream_phase
            downstream_destination_path = phase_path_by_rank.get(
                downstream_phase,
                {
                    rank: incoming.get((downstream_phase, rank), 0.0)
                    for rank in range(int(rank_count))
                },
            )
            source_tail: dict[int, float] = {}
            for rank in range(int(rank_count)):
                downstream_edges = tuple(
                    (dst, value)
                    for (edge_phase, src, dst), value in edge_wire.items()
                    if edge_phase == downstream_phase and src == rank and value > 0.0
                )
                total = sum(value for _dst, value in downstream_edges)
                maximum = max(
                    (
                        downstream_destination_path.get(dst, 0.0)
                        for dst, _value in downstream_edges
                    ),
                    default=0.0,
                )
                weighted = (
                    sum(
                        value * downstream_destination_path.get(dst, 0.0)
                        for dst, value in downstream_edges
                    )
                    / total
                    if total > 0.0
                    else 0.0
                )
                source_tail[rank] = (
                    outgoing.get((downstream_phase, rank), 0.0)
                    + parameters.downstream_max_weight * maximum
                    + parameters.downstream_mean_weight * weighted
                )
            phase_path_by_rank[phase] = {
                rank: incoming.get((phase, rank), 0.0)
                + dependency.delay(rank)
                + source_tail.get(rank, 0.0)
                for rank in range(int(rank_count))
            }

        path_price = {
            phase: _softmax_prices(
                paths,
                temperature=parameters.softmax_temperature,
            )
            for phase, paths in phase_path_by_rank.items()
        }
        send_price = _softmax_prices(
            dict(send_total), temperature=parameters.softmax_temperature
        )
        recv_price = _softmax_prices(
            dict(recv_total), temperature=parameters.softmax_temperature
        )
        max_edge_wire = max([1.0, *edge_wire.values()])
        max_tail_by_phase = {
            phase: max([1.0, *paths.values()])
            for phase, paths in phase_path_by_rank.items()
        }

        candidates: dict[
            tuple[int, int],
            tuple[float, tuple[int, int, int], str, float, float, float, float],
        ] = {}
        next_ready = math.inf
        for edge, task_ids in sorted(queues.items()):
            if not task_ids:
                continue
            phase, src, dst = edge
            task_id = task_ids[0]
            task = by_id[task_id]
            ready = max(
                float(task.ready_at),
                float(model.source_ready(phase, src)),
                float(release_time.get((phase, src), 0.0)),
            )
            if ready > current_time + 1e-9:
                next_ready = min(next_ready, ready)
                continue

            task_wire = model.duration(src, dst, task.payload_units)
            destination_remaining = max(1.0, incoming.get((phase, dst), task_wire))
            completion_fraction = min(1.0, task_wire / destination_remaining)
            rank_path = phase_path_by_rank.get(phase, {}).get(dst, destination_remaining)
            rank_tail = max(0.0, rank_path - destination_remaining)
            normalized_tail = rank_tail / max_tail_by_phase.get(phase, 1.0)
            criticality = path_price.get(phase, {}).get(dst, 0.0)
            endpoint = send_price.get(src, 0.0) + recv_price.get(dst, 0.0)
            residual = edge_wire[edge] / max_edge_wire
            release_gain = completion_fraction * normalized_tail
            score = (
                1.0
                + parameters.path_weight * criticality
                + parameters.release_weight * release_gain
                + parameters.endpoint_weight * endpoint
                + parameters.work_weight * residual
                + 1e-9 * (max(phases) + 1 - phase)
                + 1e-12 * (int(rank_count) - src)
                + 1e-15 * (int(rank_count) - dst)
            )
            candidate = (
                score,
                edge,
                task_id,
                criticality,
                release_gain,
                endpoint,
                residual,
            )
            previous = candidates.get((src, dst))
            if previous is None or (
                float(candidate[0]),
                -int(phase),
                task_id,
            ) > (
                float(previous[0]),
                -int(previous[1][0]),
                previous[2],
            ):
                candidates[(src, dst)] = candidate

        if not candidates:
            if not math.isfinite(next_ready):
                raise RuntimeError("RSCF release graph deadlocked")
            current_time = next_ready
            continue

        sources = tuple(sorted({src for src, _dst in candidates}))
        destinations = tuple(sorted({dst for _src, dst in candidates}))
        matching = maximum_weight_bipartite_matching(
            sources=sources,
            destinations=destinations,
            edge_weight=lambda src, dst: candidates.get((src, dst), (0.0,))[0],
        )
        if not matching:
            raise RuntimeError("RSCF produced an empty ready matching")

        selected: list[tuple[tuple[int, int, int], str]] = []
        for src, dst in matching:
            score, edge, task_id, criticality, release_gain, endpoint, residual = candidates[(src, dst)]
            task = by_id[task_id]
            selected.append((edge, task_id))
            scored_history.append(
                RSCFScoredTask(
                    task_id=task_id,
                    phase=task.phase,
                    src_rank=task.src_rank,
                    dst_rank=task.dst_rank,
                    score=float(score),
                    critical_path_dual=float(criticality),
                    transitive_unlock=float(release_gain),
                    endpoint_dual=float(endpoint),
                    residual_component=float(residual),
                    barrier_component=float(criticality),
                    release_gain_component=float(release_gain),
                    age_component=0.0,
                    barrier_urgency=float(criticality),
                    release_gain=float(release_gain),
                )
            )

        start = current_time
        finish = current_time + max(
            model.launch(by_id[task_id].src_rank, by_id[task_id].dst_rank)
            + model.duration(
                by_id[task_id].src_rank,
                by_id[task_id].dst_rank,
                by_id[task_id].payload_units,
            )
            for _edge, task_id in selected
        )
        segments: list[RSCFServiceSegment] = []
        for edge, task_id in selected:
            task = by_id[task_id]
            queues[edge].popleft()
            remaining.remove(task_id)
            pending_inbound_count[(task.phase, task.dst_rank)] -= 1
            segments.append(
                RSCFServiceSegment(
                    task_id=task_id,
                    phase=task.phase,
                    src_rank=task.src_rank,
                    dst_rank=task.dst_rank,
                    byte_offset=task.byte_offset,
                    service_units=task.payload_units,
                )
            )
            dependency = dependency_by_upstream.get(task.phase)
            if dependency is not None and dependency.downstream_phase in phases:
                if dependency.release_scope == "PHASE_BARRIER":
                    upstream_remaining = sum(
                        pending_inbound_count.get((dependency.upstream_phase, rank), 0)
                        for rank in range(int(rank_count))
                    )
                    if upstream_remaining == 0:
                        for rank in range(int(rank_count)):
                            release_time[(dependency.downstream_phase, rank)] = min(
                                release_time[(dependency.downstream_phase, rank)],
                                finish + dependency.delay(rank),
                            )
                elif pending_inbound_count[(task.phase, task.dst_rank)] == 0:
                    release_time[(dependency.downstream_phase, task.dst_rank)] = min(
                        release_time[(dependency.downstream_phase, task.dst_rank)],
                        finish + dependency.delay(task.dst_rank),
                    )

        waves.append(
            RSCFWave(
                wave_id=len(waves),
                task_ids=tuple(task_id for _edge, task_id in selected),
                service_units=max(by_id[task_id].payload_units for _edge, task_id in selected),
                start_time=start,
                finish_time=finish,
                segments=tuple(segments),
            )
        )
        current_time = finish

    ordered = tuple(task_id for wave in waves for task_id in wave.task_ids)
    if len(ordered) != len(items) or set(ordered) != set(by_id):
        raise ValueError("RSCF did not cover every canonical task exactly once")
    payload = {
        "algorithm": "rscf",
        "ordered_task_ids": ordered,
        "waves": tuple(
            (wave.task_ids, wave.start_time, wave.finish_time)
            for wave in waves
        ),
        "release_dependencies": dependencies,
        "wire_cost_model": model,
        "parameters": parameters,
    }
    return RSCFPlan(
        ordered_task_ids=ordered,
        waves=tuple(waves),
        scored_tasks=tuple(scored_history),
        plan_digest=stable_digest(payload),
        tie_break_rule=(
            "complete rank-release path, endpoint pressure, residual work, "
            "deterministic canonical tie-break"
        ),
    )


__all__ = [
    "RSCF_PARAMETERS",
    "RSCFParameters",
    "RSCFPlan",
    "RSCFReleaseDependency",
    "RSCFScoredTask",
    "RSCFServiceSegment",
    "RSCFTask",
    "RSCFWave",
    "RSCFWireCostModel",
    "order_rscf",
    "phase_barrier_release_dependency",
    "rank_local_release_dependency",
]
