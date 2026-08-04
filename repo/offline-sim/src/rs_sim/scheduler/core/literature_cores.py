from __future__ import annotations

"""Pure fixed-placement literature baseline implementations.

These ports operate only on canonical remote tasks and return deterministic
conflict-free logical waves.  They intentionally omit mechanisms that would
change endpoints, placement, or the common transport/cost model.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

from .digest import stable_digest
from .matching import maximum_weight_bipartite_matching


@dataclass(frozen=True, slots=True)
class LiteratureTask:
    task_id: str
    phase: int
    src_rank: int
    dst_rank: int
    payload_units: int
    chunk_index: int = 0
    byte_offset: int = 0
    ready_at: float = 0.0


@dataclass(frozen=True, slots=True)
class LiteratureWave:
    wave_id: int
    task_ids: tuple[str, ...]
    duration_units: int
    phase: int


@dataclass(frozen=True, slots=True)
class LiteraturePlan:
    algorithm_id: str
    ordered_task_ids: tuple[str, ...]
    waves: tuple[LiteratureWave, ...]
    plan_digest: str
    diagnostics: tuple[tuple[str, Any], ...]


def _task_key(task: LiteratureTask) -> tuple[int, int, int, int, int, str]:
    return (
        int(task.phase), int(task.src_rank), int(task.dst_rank),
        int(task.chunk_index), int(task.byte_offset), str(task.task_id),
    )


def _phase_groups(tasks: Iterable[LiteratureTask]) -> tuple[tuple[LiteratureTask, ...], ...]:
    items = tuple(tasks)
    phases = sorted({int(item.phase) for item in items})
    return tuple(
        tuple(sorted((item for item in items if item.phase == phase), key=_task_key))
        for phase in phases
    )


def _task_group(tasks: Iterable[LiteratureTask]) -> tuple[LiteratureTask, ...]:
    """Return the exact task set supplied by the outer scope decorator."""

    return tuple(sorted(tuple(tasks), key=_task_key))


def _make_plan(algorithm_id: str, waves: list[LiteratureWave], diagnostics: dict[str, Any]) -> LiteraturePlan:
    order = tuple(task_id for wave in waves for task_id in wave.task_ids)
    payload = {"algorithm_id": algorithm_id, "waves": tuple(waves), "diagnostics": diagnostics}
    return LiteraturePlan(
        algorithm_id=algorithm_id,
        ordered_task_ids=order,
        waves=tuple(waves),
        plan_digest=stable_digest(payload),
        diagnostics=tuple(sorted(diagnostics.items())),
    )


def _rotate(values: tuple[int, ...], pointer: int) -> tuple[int, ...]:
    if not values:
        return ()
    point = int(pointer) % len(values)
    return values[point:] + values[:point]


def order_islip(
    tasks: Iterable[LiteratureTask],
    *,
    rank_count: int,
    max_rounds: int | None = None,
) -> LiteraturePlan:
    ranks = tuple(range(int(rank_count)))
    default_rounds = min(4, max(1, int(rank_count)))
    rounds = max(1, min(int(max_rounds or default_rounds), max(1, int(rank_count))))
    input_pointer = {rank: 0 for rank in ranks}
    output_pointer = {rank: 0 for rank in ranks}
    waves: list[LiteratureWave] = []
    queues: dict[tuple[int, int], deque[LiteratureTask]] = defaultdict(deque)
    for item in _task_group(tasks):
        queues[(item.src_rank, item.dst_rank)].append(item)
    while queues:
        selected_edges: list[tuple[int, int]] = []
        selected_src: set[int] = set()
        selected_dst: set[int] = set()
        for round_id in range(rounds):
            requests: dict[int, list[int]] = defaultdict(list)
            for src, dst in sorted(queues):
                if src not in selected_src and dst not in selected_dst:
                    requests[dst].append(src)
            grants: dict[int, list[int]] = defaultdict(list)
            for dst, srcs in sorted(requests.items()):
                for src in _rotate(ranks, output_pointer.get(dst, 0)):
                    if src in srcs:
                        grants[src].append(dst)
                        break
            accepted: list[tuple[int, int]] = []
            for src, dsts in sorted(grants.items()):
                for dst in _rotate(ranks, input_pointer.get(src, 0)):
                    if dst in dsts:
                        accepted.append((src, dst))
                        if round_id == 0:
                            input_pointer[src] = (ranks.index(dst) + 1) % len(ranks)
                            output_pointer[dst] = (ranks.index(src) + 1) % len(ranks)
                        break
            for edge in accepted:
                if edge[0] not in selected_src and edge[1] not in selected_dst:
                    selected_edges.append(edge)
                    selected_src.add(edge[0])
                    selected_dst.add(edge[1])
        if not selected_edges:
            for edge in sorted(queues):
                if edge[0] not in selected_src and edge[1] not in selected_dst:
                    selected_edges.append(edge)
                    selected_src.add(edge[0])
                    selected_dst.add(edge[1])
        chosen: list[LiteratureTask] = []
        for edge in selected_edges:
            queue = queues.get(edge)
            if not queue:
                continue
            chosen.append(queue.popleft())
            if not queue:
                del queues[edge]
        if not chosen:
            raise RuntimeError("islip_style_reference made no progress")
        waves.append(LiteratureWave(
            len(waves),
            tuple(item.task_id for item in chosen),
            max(item.payload_units for item in chosen),
            chosen[0].phase,
        ))
    return _make_plan("islip", waves, {
        "rounds": rounds,
        "pointer_initialization": "ZERO",
        "pointer_state_scope": "WITHIN_SUPPLIED_PROBLEM",
        "pointer_state_persistent": True,
        "pointer_update_first_iteration_only": True,
        "cell_model": "CANONICAL_BUCKET_AS_CELL",
        "paper_label": "ISLIP_STYLE_4",
        "literature_mapping": "PAPER_DERIVED_STYLE",
        "source": "McKeown_iSLIP_1999",
    })


def order_residual_mwm(
    tasks: Iterable[LiteratureTask], *, rank_count: int
) -> LiteraturePlan:
    ranks = tuple(range(int(rank_count)))
    waves: list[LiteratureWave] = []
    for phase_tasks in (_task_group(tasks),):
        raw_queues: dict[tuple[int, int], list[LiteratureTask]] = defaultdict(list)
        for item in phase_tasks:
            raw_queues[(item.src_rank, item.dst_rank)].append(item)
        queues: dict[tuple[int, int], deque[LiteratureTask]] = {
            edge: deque(sorted(queue, key=_task_key))
            for edge, queue in raw_queues.items()
        }
        residual = {
            edge: sum(item.payload_units for item in queue)
            for edge, queue in queues.items() if queue
        }
        while residual:
            edges = tuple(edge for edge in maximum_weight_bipartite_matching(
                sources=ranks, destinations=ranks,
                edge_weight=lambda src, dst: float(residual.get((src, dst), 0)),
            ) if edge in residual)
            chosen: list[LiteratureTask] = []
            for edge in edges:
                queue = queues.get(edge)
                if not queue:
                    continue
                item = queue.popleft()
                chosen.append(item)
                residual[edge] -= item.payload_units
                if not queue:
                    queues.pop(edge, None)
                    residual.pop(edge, None)
            if not chosen:
                raise RuntimeError("gmwd_style_reference made no progress")
            waves.append(LiteratureWave(len(waves), tuple(item.task_id for item in chosen), max(item.payload_units for item in chosen), chosen[0].phase))
    return _make_plan("residual_mwm", waves, {
        "selection_rule": "maximum residual matching over canonical buckets",
        "service_rule": "one canonical bucket per selected edge in formal execution",
        "paper_claim_allowed": True,
        "paper_label": "RESIDUAL_MWM",
        "literature_mapping": "DESCRIPTIVE_HEURISTIC",
        "source": "RouterSense residual-MWM implementation",
    })


def order_aurora(
    tasks: Iterable[LiteratureTask], *, rank_count: int
) -> LiteraturePlan:
    """Fixed-placement Aurora-style communication ordering.

    Aurora's communication theorem is symmetric in maximum row and column
    load.  The old port only prioritized a heavy sender.  This version plans
    on the heavier orientation: native traffic when a row is the bottleneck,
    or the transposed traffic matrix when a receiver is the bottleneck.  It
    then keeps the bottleneck endpoint first and places every remaining source
    in descending residual-load order while preserving conflict-free waves.

    Canonical tasks remain indivisible, so this is intentionally a ``-style``
    adaptation rather than a strict token-fluid Aurora implementation.
    """

    waves: list[LiteratureWave] = []
    traces = 0
    orientation_rows = 0
    orientation_columns = 0
    bottlenecks: list[tuple[int, int, str]] = []
    for phase_tasks in (_task_group(tasks),):
        if not phase_tasks:
            continue
        row_total: dict[int, int] = defaultdict(int)
        col_total: dict[int, int] = defaultdict(int)
        for item in phase_tasks:
            row_total[item.src_rank] += item.payload_units
            col_total[item.dst_rank] += item.payload_units
        max_row = max(row_total.values(), default=0)
        max_col = max(col_total.values(), default=0)
        transpose = max_col > max_row
        if transpose:
            orientation_columns += 1
        else:
            orientation_rows += 1

        def oriented_src(item: LiteratureTask) -> int:
            return item.dst_rank if transpose else item.src_rank

        def oriented_dst(item: LiteratureTask) -> int:
            return item.src_rank if transpose else item.dst_rank

        source_total: dict[int, int] = defaultdict(int)
        by_source: dict[int, list[LiteratureTask]] = defaultdict(list)
        for item in phase_tasks:
            source_total[oriented_src(item)] += item.payload_units
            by_source[oriented_src(item)].append(item)
        bottleneck = min(source_total, key=lambda rank: (-source_total[rank], rank))
        bottlenecks.append((int(phase_tasks[0].phase), int(bottleneck), "COLUMN" if transpose else "ROW"))
        source_order = [bottleneck] + [
            src for src in sorted(source_total, key=lambda rank: (-source_total[rank], rank))
            if src != bottleneck
        ]

        local_waves: list[list[LiteratureTask]] = []
        durations: list[int] = []
        source_masks: dict[int, int] = defaultdict(int)
        destination_masks: dict[int, int] = defaultdict(int)
        duration_masks: dict[int, int] = defaultdict(int)

        def first_wave(mask: int) -> int:
            return (mask & -mask).bit_length() - 1

        for src in source_order:
            # Aurora's bottleneck order may be arbitrary.  Stable descending
            # payload is used here to avoid a trace-dependent random seed.
            items = sorted(
                by_source[src],
                key=lambda item: (-item.payload_units, oriented_dst(item), item.chunk_index, item.byte_offset, item.task_id),
            )
            for item in items:
                osrc = oriented_src(item)
                odst = oriented_dst(item)
                existing_mask = (1 << len(local_waves)) - 1
                available = existing_mask & ~(
                    source_masks.get(osrc, 0) | destination_masks.get(odst, 0)
                )
                selected = len(local_waves)
                if available:
                    payload = int(item.payload_units)
                    # This is exactly the old tuple ordering without scanning
                    # every wave: first prefer zero extension with the smallest
                    # existing duration, otherwise the largest duration below
                    # the payload, then the earliest wave id.
                    zero_extension = [
                        duration for duration, mask in duration_masks.items()
                        if duration >= payload and (mask & available)
                    ]
                    if zero_extension:
                        duration = min(zero_extension)
                        selected = first_wave(duration_masks[duration] & available)
                    else:
                        extending = [
                            duration for duration, mask in duration_masks.items()
                            if duration < payload and (mask & available)
                        ]
                        if extending:
                            duration = max(extending)
                            selected = first_wave(duration_masks[duration] & available)
                if selected == len(local_waves):
                    local_waves.append([])
                    durations.append(0)
                    duration_masks[0] |= 1 << selected
                bit = 1 << selected
                old_duration = durations[selected]
                local_waves[selected].append(item)
                source_masks[osrc] |= bit
                destination_masks[odst] |= bit
                new_duration = max(old_duration, int(item.payload_units))
                if new_duration != old_duration:
                    duration_masks[old_duration] &= ~bit
                    if not duration_masks[old_duration]:
                        del duration_masks[old_duration]
                    duration_masks[new_duration] |= bit
                    durations[selected] = new_duration
                traces += 1
        for members, duration in zip(local_waves, durations, strict=True):
            waves.append(
                LiteratureWave(
                    len(waves),
                    tuple(item.task_id for item in members),
                    duration,
                    members[0].phase,
                )
            )
    return _make_plan("aurora", waves, {
        "placement": "fixed",
        "trace_rows": traces,
        "orientation_rule": "HEAVIER_OF_MAX_ROW_OR_MAX_COLUMN",
        "row_oriented_phases": orientation_rows,
        "column_oriented_phases": orientation_columns,
        "bottleneck_endpoints": tuple(bottlenecks),
        "paper_claim_allowed": True,
        "paper_label": "AURORA_STYLE",
        "literature_mapping": "FIXED_PLACEMENT_COMMUNICATION_STYLE",
        "missing_mechanisms": (
            "expert_colocation",
            "heterogeneous_gpu_assignment",
            "multi_model_overlap",
            "fluid_token_preemption",
        ),
        "source": "Aurora_arXiv_2410.17043_fixed_placement_style",
    })


def _rank_to_node_from_gpus_per_server(
    rank_count: int, gpus_per_server: int
) -> tuple[int, ...]:
    per_server = int(gpus_per_server)
    if per_server <= 0:
        raise ValueError("gpus_per_server must be positive")
    return tuple(rank // per_server for rank in range(int(rank_count)))


def order_fast(
    tasks: Iterable[LiteratureTask],
    *,
    rank_count: int,
    rank_to_node: tuple[int, ...] | None = None,
    gpus_per_server: int | None = None,
) -> LiteraturePlan:
    """Fixed-endpoint FAST-style two-tier matching on an explicit topology.

    The implementation never guesses node boundaries from EP. Callers must
    provide ``rank_to_node`` or an explicit synthetic ``gpus_per_server`` in a
    unit test. This remains FAST-style because traffic redistribution and
    intra-node rebalancing are intentionally outside the common task contract.
    """

    ranks = tuple(range(int(rank_count)))
    if rank_to_node is None:
        if gpus_per_server is None:
            raise ValueError(
                "FAST-style requires explicit rank_to_node topology; EP-based inference is forbidden"
            )
        rank_to_node = _rank_to_node_from_gpus_per_server(rank_count, gpus_per_server)
    topology = tuple(int(value) for value in rank_to_node)
    if len(topology) != int(rank_count):
        raise ValueError("rank_to_node length must equal rank_count")
    if any(value < 0 for value in topology):
        raise ValueError("rank_to_node values must be non-negative")
    server_ids = tuple(sorted(set(topology)))
    ranks_by_server = {
        server: tuple(rank for rank in ranks if topology[rank] == server)
        for server in server_ids
    }

    waves: list[LiteratureWave] = []
    for phase_tasks in (_task_group(tasks),):
        raw_queues: dict[tuple[int, int], list[LiteratureTask]] = defaultdict(list)
        for item in phase_tasks:
            raw_queues[(item.src_rank, item.dst_rank)].append(item)
        queues: dict[tuple[int, int], deque[LiteratureTask]] = {
            edge: deque(sorted(queue, key=_task_key))
            for edge, queue in raw_queues.items()
        }
        residual = {
            edge: sum(item.payload_units for item in queue)
            for edge, queue in queues.items() if queue
        }
        while residual:
            server_residual: dict[tuple[int, int], int] = defaultdict(int)
            for (src, dst), value in residual.items():
                src_server = topology[src]
                dst_server = topology[dst]
                if src_server != dst_server:
                    server_residual[(src_server, dst_server)] += value
            server_pairs = maximum_weight_bipartite_matching(
                sources=server_ids,
                destinations=server_ids,
                edge_weight=lambda src, dst: (
                    float(server_residual.get((src, dst), 0)) if src != dst else 0.0
                ),
            )
            selected: list[tuple[int, int]] = []
            for src_server, dst_server in server_pairs:
                selected.extend(
                    edge
                    for edge in maximum_weight_bipartite_matching(
                        sources=ranks_by_server[src_server],
                        destinations=ranks_by_server[dst_server],
                        edge_weight=lambda src, dst: float(residual.get((src, dst), 0)),
                    )
                    if edge in residual
                )
            used_src = {src for src, _ in selected}
            used_dst = {dst for _, dst in selected}
            local_edges = sorted(
                (
                    edge
                    for edge in residual
                    if topology[edge[0]] == topology[edge[1]]
                    and edge[0] not in used_src
                    and edge[1] not in used_dst
                ),
                key=lambda edge: (-residual[edge], edge),
            )
            for edge in local_edges:
                if edge[0] not in used_src and edge[1] not in used_dst:
                    selected.append(edge)
                    used_src.add(edge[0])
                    used_dst.add(edge[1])
            if not selected:
                selected = [
                    max(residual, key=lambda edge: (residual[edge], -edge[0], -edge[1]))
                ]
            chosen: list[LiteratureTask] = []
            for edge in selected:
                queue = queues.get(edge)
                if not queue:
                    continue
                item = queue.popleft()
                chosen.append(item)
                residual[edge] -= item.payload_units
                if not queue:
                    queues.pop(edge, None)
                    residual.pop(edge, None)
            if not chosen:
                raise RuntimeError("fast_style_reference made no progress")
            waves.append(
                LiteratureWave(
                    len(waves),
                    tuple(item.task_id for item in chosen),
                    max(item.payload_units for item in chosen),
                    chosen[0].phase,
                )
            )
    return _make_plan(
        "fast",
        waves,
        {
            "rank_to_node": topology,
            "node_count": len(server_ids),
            "topology_source": "EXPLICIT_CONTRACT",
            "fixed_endpoints": True,
            "paper_claim_allowed": True,
            "paper_label": "FAST_STYLE",
            "literature_mapping": "TWO_TIER_FIXED_ENDPOINT_STYLE",
            "missing_mechanisms": (
                "intra_server_rebalancing",
                "traffic_redistribution",
                "balanced_one_to_one_scale_out",
                "redistribution_scale_out_pipeline",
            ),
            "source": "FAST_NSDI_2026_fixed_endpoint_mapping",
        },
    )


__all__ = [
    "LiteraturePlan",
    "LiteratureTask",
    "LiteratureWave",
    "order_aurora",
    "order_fast",
    "order_residual_mwm",
    "order_islip",
]
