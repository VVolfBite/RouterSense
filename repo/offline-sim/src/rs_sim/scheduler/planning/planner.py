from __future__ import annotations

"""Canonical planner for the unified scheduler.

The planner has one job: apply one registered ordering core to the exact task
set selected by the Local or Joint decorator.  It never selects another core,
changes task boundaries, or changes the runtime/transport contract.
"""

import dataclasses
import enum
from dataclasses import dataclass
from typing import Any, Iterable

from rs_sim.contracts.paper_defaults import PAPER_RELEASE_MODE
from rs_sim.scheduler.core.birkhoff_core import BirkhoffTask, decompose_integer_matrix, order_birkhoff
from rs_sim.scheduler.core.literature_cores import (
    LiteratureTask,
    order_aurora,
    order_fast,
    order_residual_mwm,
    order_islip,
)
from rs_sim.scheduler.core.rscf_core import (
    RSCFPlan,
    RSCFTask,
    RSCFWireCostModel,
    order_rscf,
    phase_barrier_release_dependency,
    rank_local_release_dependency,
)
from rs_sim.scheduler.metrics.priority_replay import ready_aware_completion_objective
from rs_sim.scheduler.stable import stable_digest, stable_json


class ExecutionMode(str, enum.Enum):
    ORDER_ONLY = "ORDER_ONLY"


class PlannerScope(str, enum.Enum):
    PHASE_LOCAL = "PHASE_LOCAL"
    WINDOW_JOINT = "WINDOW_JOINT"


@dataclass(frozen=True, slots=True)
class FairnessContract:
    task_catalogue_digest: str
    task_boundary_digest: str
    taskization_digest: str
    receiver_contract_rule_digest: str
    buffer_profile_digest: str
    compiler_digest: str
    transport_digest: str
    release_model_digest: str
    information_digest: str
    cost_model_digest: str

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field.name} must be non-empty")

    @property
    def fairness_digest(self) -> str:
        return stable_digest(self)


@dataclass(frozen=True, slots=True)
class SchedulingTask:
    task_id: str
    phase_token: str
    phase_ordinal: int
    src_rank: int
    dst_rank: int
    payload_bytes: int
    chunk_index: int
    byte_offset: int
    ready_at_ns: int | None

    def __post_init__(self) -> None:
        if not self.task_id or not self.phase_token:
            raise ValueError("task_id and phase_token must be non-empty")
        for name in ("phase_ordinal", "src_rank", "dst_rank", "chunk_index", "byte_offset"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.src_rank == self.dst_rank:
            raise ValueError("local assembly must not enter the remote task catalogue")
        if not isinstance(self.payload_bytes, int) or isinstance(self.payload_bytes, bool) or self.payload_bytes <= 0:
            raise ValueError("payload_bytes must be a positive int")
        if self.ready_at_ns is not None and self.ready_at_ns < 0:
            raise ValueError("ready_at_ns must be non-negative")


@dataclass(frozen=True, slots=True)
class SchedulingProblem:
    rank_count: int
    tasks: tuple[SchedulingTask, ...]
    phase_tokens: tuple[str, ...]
    fairness: FairnessContract

    def __post_init__(self) -> None:
        if not isinstance(self.rank_count, int) or self.rank_count <= 0:
            raise ValueError("rank_count must be positive")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("SchedulingProblem contains duplicate task IDs")
        if any(task.src_rank >= self.rank_count or task.dst_rank >= self.rank_count for task in self.tasks):
            raise ValueError("task endpoint exceeds rank_count")
        if tuple(dict.fromkeys(task.phase_token for task in self.tasks)) != self.phase_tokens:
            raise ValueError("phase_tokens must equal first-occurrence canonical phase order")
        expected_catalogue = stable_digest(tuple(task.task_id for task in self.tasks))
        expected_boundaries = stable_digest(
            tuple(
                (
                    task.task_id,
                    task.phase_token,
                    task.src_rank,
                    task.dst_rank,
                    task.chunk_index,
                    task.byte_offset,
                    task.payload_bytes,
                )
                for task in self.tasks
            )
        )
        if self.fairness.task_catalogue_digest != expected_catalogue:
            raise ValueError("fairness task_catalogue_digest does not match canonical task IDs")
        if self.fairness.task_boundary_digest != expected_boundaries:
            raise ValueError("fairness task_boundary_digest does not match canonical task boundaries")

    @property
    def problem_digest(self) -> str:
        return stable_digest(
            {
                "rank_count": self.rank_count,
                "tasks": self.tasks,
                "phase_tokens": self.phase_tokens,
                "fairness": self.fairness,
            }
        )


@dataclass(frozen=True, slots=True)
class AlgorithmWave:
    wave_id: int
    task_ids: tuple[str, ...]
    phase_tokens: tuple[str, ...]
    logical_duration_units: int = 0
    service_segments: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class AlgorithmPlan:
    algorithm_id: str
    scope: PlannerScope
    execution_mode: ExecutionMode
    ordered_task_ids: tuple[str, ...]
    waves: tuple[AlgorithmWave, ...]
    fairness: FairnessContract
    task_catalogue_digest: str
    task_boundary_digest: str
    plan_digest: str
    diagnostics: tuple[tuple[str, Any], ...] = ()

    def validate_against(self, problem: SchedulingProblem) -> None:
        expected = tuple(task.task_id for task in problem.tasks)
        if len(self.ordered_task_ids) != len(expected) or set(self.ordered_task_ids) != set(expected):
            raise ValueError("algorithm plan must cover every canonical task exactly once")
        if len(set(self.ordered_task_ids)) != len(self.ordered_task_ids):
            raise ValueError("algorithm plan contains duplicate task IDs")
        flattened = tuple(task_id for wave in self.waves for task_id in wave.task_ids)
        if flattened != self.ordered_task_ids:
            raise ValueError("wave flattening must equal ordered_task_ids")
        if self.fairness != problem.fairness:
            raise ValueError("algorithm changed the fairness contract")


def build_problem_from_catalogue(
    *,
    adapter: Any,
    catalogue: Any,
    runtime: Any,
    phase_keys: Iterable[Any],
    rank_count: int,
    receiver_contract_rule_digest: str,
    buffer_profile_digest: str,
    compiler_digest: str,
    transport_digest: str,
    release_model_digest: str,
    information_digest: str,
    cost_model_digest: str,
    eligible_states: tuple[str, ...] | None = None,
) -> SchedulingProblem:
    keys = tuple(phase_keys)
    requested_tokens = tuple(stable_json(adapter.phase_payload(key)) for key in keys)
    if len(set(requested_tokens)) != len(requested_tokens):
        raise ValueError("phase_keys contains aliases of the same phase")
    allowed_states = None if eligible_states is None else frozenset(str(item) for item in eligible_states)
    tasks: list[SchedulingTask] = []
    included_tokens: list[str] = []
    taskization_digests: set[str] = set()
    for phase_key, phase_token in zip(keys, requested_tokens, strict=True):
        selected_ids = tuple(
            task_id
            for task_id in catalogue.task_ids_for_phase(phase_key)
            if allowed_states is None or runtime.facts(task_id).state in allowed_states
        )
        if not selected_ids:
            continue
        phase_ordinal = len(included_tokens)
        included_tokens.append(phase_token)
        for task_id in selected_ids:
            view = catalogue.view(task_id)
            facts = runtime.facts(task_id)
            taskization_digests.add(str(view.taskization_digest))
            tasks.append(
                SchedulingTask(
                    task_id=str(view.task_id),
                    phase_token=phase_token,
                    phase_ordinal=phase_ordinal,
                    src_rank=int(view.src_rank),
                    dst_rank=int(view.dst_rank),
                    payload_bytes=int(view.payload_bytes),
                    chunk_index=int(view.chunk_index),
                    byte_offset=int(view.byte_offset),
                    ready_at_ns=facts.ready_at_ns,
                )
            )
    canonical_tasks = tuple(tasks)
    fairness = FairnessContract(
        task_catalogue_digest=stable_digest(tuple(task.task_id for task in canonical_tasks)),
        task_boundary_digest=stable_digest(
            tuple(
                (
                    task.task_id,
                    task.phase_token,
                    task.src_rank,
                    task.dst_rank,
                    task.chunk_index,
                    task.byte_offset,
                    task.payload_bytes,
                )
                for task in canonical_tasks
            )
        ),
        taskization_digest=(
            next(iter(taskization_digests))
            if len(taskization_digests) == 1
            else stable_digest(tuple(sorted(taskization_digests)))
        ),
        receiver_contract_rule_digest=str(receiver_contract_rule_digest),
        buffer_profile_digest=str(buffer_profile_digest),
        compiler_digest=str(compiler_digest),
        transport_digest=str(transport_digest),
        release_model_digest=str(release_model_digest),
        information_digest=str(information_digest),
        cost_model_digest=str(cost_model_digest),
    )
    return SchedulingProblem(
        rank_count=int(rank_count),
        tasks=canonical_tasks,
        phase_tokens=tuple(included_tokens),
        fairness=fairness,
    )


def _phase_groups(problem: SchedulingProblem) -> tuple[tuple[SchedulingTask, ...], ...]:
    """Group by the authoritative phase tokens, not ordinal numbering.

    Public SchedulingProblem instances are allowed to carry sparse or semantic
    ordinals (for example a P2-only problem with ordinal 2).  phase_tokens is
    the ordered contract; relying on range(len(phase_tokens)) silently drops
    such tasks.
    """

    return tuple(
        tuple(task for task in problem.tasks if task.phase_token == phase_token)
        for phase_token in problem.phase_tokens
    )


def _scope_groups(problem: SchedulingProblem, scope: PlannerScope) -> tuple[tuple[SchedulingTask, ...], ...]:
    return _phase_groups(problem) if scope is PlannerScope.PHASE_LOCAL else (tuple(problem.tasks),)


def _pack_tasks(tasks: Iterable[SchedulingTask], *, wave_offset: int = 0) -> tuple[AlgorithmWave, ...]:
    items = tuple(tasks)
    waves: list[AlgorithmWave] = []
    current: list[SchedulingTask] = []
    srcs: set[int] = set()
    dsts: set[int] = set()
    for task in items:
        if current and (task.src_rank in srcs or task.dst_rank in dsts):
            waves.append(
                AlgorithmWave(
                    wave_id=wave_offset + len(waves),
                    task_ids=tuple(item.task_id for item in current),
                    phase_tokens=tuple(dict.fromkeys(item.phase_token for item in current)),
                    logical_duration_units=max(item.payload_bytes for item in current),
                )
            )
            current, srcs, dsts = [], set(), set()
        current.append(task)
        srcs.add(task.src_rank)
        dsts.add(task.dst_rank)
    if current:
        waves.append(
            AlgorithmWave(
                wave_id=wave_offset + len(waves),
                task_ids=tuple(item.task_id for item in current),
                phase_tokens=tuple(dict.fromkeys(item.phase_token for item in current)),
                logical_duration_units=max(item.payload_bytes for item in current),
            )
        )
    return tuple(waves)


def _renumber_waves(
    waves: Iterable[AlgorithmWave],
    *,
    offset: int,
) -> tuple[AlgorithmWave, ...]:
    return tuple(
        AlgorithmWave(
            wave_id=int(offset) + index,
            task_ids=tuple(wave.task_ids),
            phase_tokens=tuple(wave.phase_tokens),
            logical_duration_units=int(wave.logical_duration_units),
            service_segments=tuple(wave.service_segments),
        )
        for index, wave in enumerate(waves)
    )


def _oracle_ready_aware_portfolio(
    problem: SchedulingProblem,
    solver_waves: tuple[AlgorithmWave, ...],
    *,
    scope: PlannerScope,
    release_mode: str,
    wire_cost_model: RSCFWireCostModel | None,
    semantic_phase_ordinal: int | None,
) -> tuple[tuple[AlgorithmWave, ...], str, dict[str, int]]:
    """Select a best-found candidate under the live ready-aware priority contract.

    Solver certification remains certification of the solver's own mathematical
    model.  This portfolio is only a runtime-aligned best-found selector; it
    never promotes a heuristic candidate to certified optimum.
    """

    model = wire_cost_model or RSCFWireCostModel()
    by_id = {task.task_id: task for task in problem.tasks}
    candidates: dict[str, tuple[AlgorithmWave, ...]] = {
        "solver": _renumber_waves(solver_waves, offset=0),
    }

    fifo_order = tuple(sorted(problem.tasks, key=lambda task: (
        int(task.ready_at_ns or 0), task.phase_ordinal, task.src_rank,
        task.dst_rank, task.chunk_index, task.byte_offset, task.task_id,
    )))
    candidates["fifo"] = _pack_tasks(fifo_order, wave_offset=0)

    try:
        _order, logical_waves, _certificate = order_birkhoff(
            (
                BirkhoffTask(
                    task_id=task.task_id,
                    src_rank=task.src_rank,
                    dst_rank=task.dst_rank,
                    payload_units=task.payload_bytes,
                    chunk_index=task.chunk_index,
                    byte_offset=task.byte_offset,
                )
                for task in problem.tasks
            ),
            rank_count=problem.rank_count,
        )
        candidates["birkhoff"] = tuple(
            AlgorithmWave(
                wave_id=index,
                task_ids=tuple(task_ids),
                phase_tokens=tuple(dict.fromkeys(by_id[item].phase_token for item in task_ids)),
                logical_duration_units=max(by_id[item].payload_bytes for item in task_ids),
            )
            for index, task_ids in enumerate(logical_waves)
        )
    except Exception:
        pass

    try:
        rscf_tasks = _rscf_tasks(
            problem.tasks,
            semantic_phase_ordinal=semantic_phase_ordinal,
        )
        dependencies = ()
        semantic_phases = tuple(sorted({task.phase for task in rscf_tasks}))
        if scope is PlannerScope.WINDOW_JOINT and len(semantic_phases) >= 2:
            normalized_release = str(release_mode).upper()
            if normalized_release == "PHASE_BARRIER":
                factory = phase_barrier_release_dependency
            elif normalized_release == "RANK_LOCAL":
                factory = rank_local_release_dependency
            else:
                raise ValueError("release_mode must be PHASE_BARRIER or RANK_LOCAL")
            dependencies = (
                factory(
                    upstream_phase=semantic_phases[0],
                    downstream_phase=semantic_phases[1],
                    rank_count=problem.rank_count,
                    delay_provider=model.p1_to_p2_delay,
                ),
            )
        rscf = order_rscf(
            rscf_tasks,
            rank_count=problem.rank_count,
            wire_cost_model=model,
            release_dependencies=dependencies,
        )
        candidates["rscf"] = tuple(
            AlgorithmWave(
                wave_id=index,
                task_ids=tuple(wave.task_ids),
                phase_tokens=tuple(dict.fromkeys(by_id[item].phase_token for item in wave.task_ids)),
                logical_duration_units=max(0, int(round(wave.finish_time - wave.start_time))),
                service_segments=tuple(
                    (segment.task_id, int(segment.byte_offset), int(segment.service_units))
                    for segment in wave.segments
                ),
            )
            for index, wave in enumerate(rscf.waves)
            if wave.task_ids
        )
    except Exception:
        pass

    scores: dict[str, int] = {}
    for name, waves in candidates.items():
        objective = ready_aware_completion_objective(
            problem,
            waves,
            model,
            release_mode=str(release_mode),
            semantic_phase_ordinal=semantic_phase_ordinal,
        )
        scores[name] = int(objective)

    preference = {"solver": 0, "rscf": 1, "birkhoff": 2, "fifo": 3}
    selected_name = min(
        scores,
        key=lambda name: (scores[name], preference.get(name, 99), name),
    )
    return candidates[selected_name], selected_name, scores


def _greedy_order(
    tasks: Iterable[SchedulingTask],
    *,
    scope: PlannerScope,
) -> tuple[SchedulingTask, ...]:
    """Return the canonical Greedy order for Local or Joint planning.

    Local Greedy remains the original longest-canonical-task-first policy.
    Joint Greedy receives one deliberately small P2 extension that follows the
    same longest-work-first idea without importing RSCF critical-path terms:

    * predicted P2 outgoing bytes of rank ``r`` are the downstream work
      released when P1 inbound to destination ``r`` completes;
    * that downstream work is distributed proportionally over the exact P1
      inbound canonical tasks of ``r``;
    * each P1 task is ranked by ``own payload + downstream P2 share``;
    * P2 tasks themselves retain their own payload as their Greedy length.

    When P2 is masked to zero (or absent), every downstream share is exactly
    zero, so WINDOW_JOINT Greedy reduces byte-for-byte to the original P1-only
    Greedy ordering.  The extension changes only ordering evidence; canonical
    task identities, boundaries, readiness and transport legality are untouched.
    """

    items = tuple(tasks)
    if scope is not PlannerScope.WINDOW_JOINT or not items:
        return tuple(sorted(items, key=lambda task: (
            -task.payload_bytes, task.phase_ordinal, task.src_rank,
            task.dst_rank, task.chunk_index, task.byte_offset, task.task_id,
        )))

    phases = tuple(sorted({int(task.phase_ordinal) for task in items}))
    if len(phases) < 2:
        return tuple(sorted(items, key=lambda task: (
            -task.payload_bytes, task.phase_ordinal, task.src_rank,
            task.dst_rank, task.chunk_index, task.byte_offset, task.task_id,
        )))

    p1_phase, p2_phase = phases[0], phases[1]
    p1_inbound_bytes: dict[int, int] = {}
    p2_outgoing_bytes: dict[int, int] = {}
    for task in items:
        if int(task.phase_ordinal) == p1_phase:
            p1_inbound_bytes[task.dst_rank] = (
                p1_inbound_bytes.get(task.dst_rank, 0) + int(task.payload_bytes)
            )
        elif int(task.phase_ordinal) == p2_phase:
            p2_outgoing_bytes[task.src_rank] = (
                p2_outgoing_bytes.get(task.src_rank, 0) + int(task.payload_bytes)
            )

    def downstream_share(task: SchedulingTask) -> int:
        if int(task.phase_ordinal) != p1_phase:
            return 0
        inbound = int(p1_inbound_bytes.get(task.dst_rank, 0))
        future = int(p2_outgoing_bytes.get(task.dst_rank, 0))
        if inbound <= 0 or future <= 0:
            return 0
        # Integer proportional allocation is deterministic and preserves the
        # exact zero-P2 reduction contract.  It is only a priority hint, not a
        # synthetic task or an execution-time charge.
        return (future * int(task.payload_bytes)) // inbound

    return tuple(sorted(items, key=lambda task: (
        -(int(task.payload_bytes) + downstream_share(task)),
        task.phase_ordinal,
        task.src_rank,
        task.dst_rank,
        task.chunk_index,
        task.byte_offset,
        task.task_id,
    )))


_JOINT_BASELINE_ADAPTATIONS = {
    "greedy": "LONGEST_WORK_PLUS_PROPORTIONAL_DOWNSTREAM_P2_SHARE",
    "birkhoff": "COMBINED_P1_P2_CANONICAL_BUCKET_MATRIX",
    "islip": "COMBINED_P1_P2_VIRTUAL_OUTPUT_QUEUES",
    "residual_mwm": "COMBINED_P1_P2_EDGE_RESIDUAL_MATCHING",
    "fast": "COMBINED_P1_P2_TWO_TIER_RESIDUAL_MATCHING",
    "aurora": "COMBINED_P1_P2_ENDPOINT_LOAD_PLACEMENT",
}


def _rscf_tasks(tasks: Iterable[SchedulingTask], *, semantic_phase_ordinal: int | None = None) -> tuple[RSCFTask, ...]:
    items = tuple(tasks)
    phase_values = {int(task.phase_ordinal) for task in items}
    phase_offset = 1 if phase_values and max(phase_values) <= 1 else 0
    semantic = -1 if semantic_phase_ordinal is None else int(semantic_phase_ordinal)
    return tuple(
        RSCFTask(
            task_id=task.task_id,
            phase=(semantic if semantic >= 0 else task.phase_ordinal + phase_offset),
            src_rank=task.src_rank,
            dst_rank=task.dst_rank,
            payload_units=task.payload_bytes,
            chunk_index=task.chunk_index,
            byte_offset=task.byte_offset,
            ready_at=float(task.ready_at_ns or 0),
        )
        for task in items
    )


def _literature_tasks(tasks: Iterable[SchedulingTask]) -> tuple[LiteratureTask, ...]:
    return tuple(
        LiteratureTask(
            task_id=task.task_id,
            phase=task.phase_ordinal,
            src_rank=task.src_rank,
            dst_rank=task.dst_rank,
            payload_units=task.payload_bytes,
            chunk_index=task.chunk_index,
            byte_offset=task.byte_offset,
            ready_at=float(task.ready_at_ns or 0),
        )
        for task in tasks
    )


def _waves_from_rscf(problem: SchedulingProblem, plans: Iterable[RSCFPlan]) -> tuple[AlgorithmWave, ...]:
    by_id = {task.task_id: task for task in problem.tasks}
    output: list[AlgorithmWave] = []
    for plan in plans:
        for wave in plan.waves:
            if not wave.task_ids:
                continue
            output.append(
                AlgorithmWave(
                    wave_id=len(output),
                    task_ids=tuple(wave.task_ids),
                    phase_tokens=tuple(dict.fromkeys(by_id[item].phase_token for item in wave.task_ids)),
                    logical_duration_units=max(0, int(round(wave.finish_time - wave.start_time))),
                    service_segments=tuple(
                        (segment.task_id, int(segment.byte_offset), int(segment.service_units))
                        for segment in wave.segments
                    ),
                )
            )
    return tuple(output)


def _critical_completion_objective_for_waves(
    problem: SchedulingProblem,
    waves: Iterable[AlgorithmWave],
    model: RSCFWireCostModel | None,
    *,
    release_mode: str = PAPER_RELEASE_MODE,
    semantic_phase_ordinal: int | None = None,
) -> int:
    """Evaluate one frozen wave order under the formal P12 timing model.

    ``SchedulingProblem.phase_ordinal`` is dense within the selected problem,
    so a standalone P2 suffix may carry ordinal zero.  The optional semantic
    ordinal preserves the P2 completion tail in that case.
    """

    cost = model or RSCFWireCostModel()
    by_id = {task.task_id: task for task in problem.tasks}
    if not by_id:
        return 0
    ordinals = sorted({task.phase_ordinal for task in problem.tasks})
    if len(ordinals) == 1:
        only = ordinals[0]
        semantic = (
            int(semantic_phase_ordinal)
            if semantic_phase_ordinal is not None
            else (only + 1 if only <= 1 else only)
        )
        p1_ordinal = only if semantic == 1 else None
        p2_ordinal = only if semantic == 2 else None
    else:
        p1_ordinal = ordinals[0]
        p2_ordinal = ordinals[1]

    p1_pending = {rank: set() for rank in range(problem.rank_count)}
    p2_pending = {rank: set() for rank in range(problem.rank_count)}
    for task in problem.tasks:
        if p1_ordinal is not None and task.phase_ordinal == p1_ordinal:
            p1_pending[task.dst_rank].add(task.task_id)
        elif p2_ordinal is not None and task.phase_ordinal == p2_ordinal:
            p2_pending[task.dst_rank].add(task.task_id)

    normalized_release = str(release_mode).upper()
    if normalized_release not in {"PHASE_BARRIER", "RANK_LOCAL"}:
        raise ValueError("release_mode must be PHASE_BARRIER or RANK_LOCAL")
    p2_source_ready = {
        rank: (
            0.0
            if p1_ordinal is None or not p1_pending[rank]
            else float("inf")
        )
        for rank in range(problem.rank_count)
    }
    p2_destination_finish: dict[int, float] = {}
    now = 0.0
    for wave in waves:
        members = tuple(by_id[item] for item in wave.task_ids if item in by_id)
        if not members:
            continue
        required = now
        for item in members:
            required = max(required, float(item.ready_at_ns or 0))
            if p2_ordinal is not None and item.phase_ordinal == p2_ordinal:
                required = max(required, p2_source_ready[item.src_rank])
        if required == float("inf"):
            return 2**63 - 1
        now = required
        now += max(
            cost.launch(item.src_rank, item.dst_rank)
            + cost.duration(item.src_rank, item.dst_rank, item.payload_bytes)
            for item in members
        )
        for item in members:
            if p1_ordinal is not None and item.phase_ordinal == p1_ordinal:
                p1_pending[item.dst_rank].discard(item.task_id)
                if normalized_release == "RANK_LOCAL" and not p1_pending[item.dst_rank]:
                    p2_source_ready[item.dst_rank] = min(
                        p2_source_ready[item.dst_rank],
                        now + cost.p1_to_p2_delay(item.dst_rank),
                    )
            elif p2_ordinal is not None and item.phase_ordinal == p2_ordinal:
                p2_pending[item.dst_rank].discard(item.task_id)
                if not p2_pending[item.dst_rank]:
                    p2_destination_finish[item.dst_rank] = now
        if (
            normalized_release == "PHASE_BARRIER"
            and p1_ordinal is not None
            and p2_ordinal is not None
            and all(not pending for pending in p1_pending.values())
            and any(value == float("inf") for value in p2_source_ready.values())
        ):
            for rank in range(problem.rank_count):
                p2_source_ready[rank] = now + cost.p1_to_p2_delay(rank)
    if any(pending for pending in p1_pending.values()) or any(
        pending for pending in p2_pending.values()
    ):
        return 2**63 - 1
    completion = now
    for rank, finish in p2_destination_finish.items():
        completion = max(completion, finish + cost.p2_completion_tail(rank))
    return max(0, int(round(completion)))


def _estimated_p1_release_tail_for_waves(
    problem: SchedulingProblem,
    waves: Iterable[AlgorithmWave],
    model: RSCFWireCostModel | None,
    *,
    release_mode: str = PAPER_RELEASE_MODE,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    cost = model or RSCFWireCostModel()
    by_id = {task.task_id: task for task in problem.tasks}
    if not by_id:
        return 0, ()
    p1_ordinal = min(task.phase_ordinal for task in problem.tasks)
    remaining: dict[int, set[str]] = {}
    for task in problem.tasks:
        if task.phase_ordinal == p1_ordinal:
            remaining.setdefault(task.dst_rank, set()).add(task.task_id)
    released_at: dict[int, int] = {}
    now = 0.0
    for wave in waves:
        members = tuple(by_id[item] for item in wave.task_ids if item in by_id)
        if not members:
            continue
        now += max(
            cost.launch(item.src_rank, item.dst_rank)
            + cost.duration(item.src_rank, item.dst_rank, item.payload_bytes)
            for item in members
        )
        for item in members:
            if item.phase_ordinal != p1_ordinal:
                continue
            remaining[item.dst_rank].discard(item.task_id)
            if not remaining[item.dst_rank] and item.dst_rank not in released_at:
                released_at[item.dst_rank] = int(round(now))
    normalized_release = str(release_mode).upper()
    if normalized_release not in {"PHASE_BARRIER", "RANK_LOCAL"}:
        raise ValueError("release_mode must be PHASE_BARRIER or RANK_LOCAL")
    release_tail = max(released_at.values(), default=0)
    if normalized_release == "PHASE_BARRIER" and released_at:
        pairs = tuple((rank, release_tail) for rank in sorted(released_at))
    else:
        pairs = tuple(sorted(released_at.items()))
    return release_tail, pairs


class OrderOnlyPlanner:
    SUPPORTED = frozenset({
        "null", "fifo", "greedy", "birkhoff", "islip",
        "residual_mwm", "fast", "aurora", "rscf", "oracle",
    })

    def plan(
        self,
        problem: SchedulingProblem,
        *,
        algorithm_id: str,
        planner_scope: PlannerScope | str,
        rscf_wire_cost_model: RSCFWireCostModel | None = None,
        rank_to_node: tuple[int, ...] | None = None,
        rscf_semantic_phase_ordinal: int | None = None,
        release_mode: str = PAPER_RELEASE_MODE,
        oracle_time_limit_ms: int = 30_000,
        oracle_relative_gap: float = 0.0,
        oracle_require_certified: bool = True,
    ) -> AlgorithmPlan:
        core_id = str(algorithm_id)
        if core_id not in self.SUPPORTED:
            raise ValueError(f"unregistered algorithm core {core_id!r}")
        scope = PlannerScope(planner_scope)
        groups = tuple(group for group in _scope_groups(problem, scope) if group)
        by_id = {task.task_id: task for task in problem.tasks}
        waves: list[AlgorithmWave] = []
        diagnostics: dict[str, Any] = {"core_id": core_id, "scope": scope.value}

        for group in groups:
            if core_id == "null":
                group_order = tuple(task.task_id for task in group)
                group_waves = _pack_tasks(group, wave_offset=len(waves))
            elif core_id == "fifo":
                ordered = tuple(sorted(group, key=lambda task: (
                    int(task.ready_at_ns or 0), task.phase_ordinal, task.src_rank,
                    task.dst_rank, task.chunk_index, task.byte_offset, task.task_id,
                )))
                group_order = tuple(task.task_id for task in ordered)
                group_waves = _pack_tasks(ordered, wave_offset=len(waves))
            elif core_id == "greedy":
                ordered = _greedy_order(group, scope=scope)
                group_order = tuple(task.task_id for task in ordered)
                group_waves = _pack_tasks(ordered, wave_offset=len(waves))
            elif core_id == "birkhoff":
                group_order, logical_waves, cert = order_birkhoff(
                    (
                        BirkhoffTask(
                            task_id=task.task_id,
                            src_rank=task.src_rank,
                            dst_rank=task.dst_rank,
                            payload_units=task.payload_bytes,
                            chunk_index=task.chunk_index,
                            byte_offset=task.byte_offset,
                        )
                        for task in group
                    ),
                    rank_count=problem.rank_count,
                )
                group_waves = tuple(
                    AlgorithmWave(
                        wave_id=len(waves) + index,
                        task_ids=tuple(task_ids),
                        phase_tokens=tuple(dict.fromkeys(by_id[item].phase_token for item in task_ids)),
                        logical_duration_units=max(by_id[item].payload_bytes for item in task_ids),
                    )
                    for index, task_ids in enumerate(logical_waves)
                )
                matrix = [[0 for _ in range(problem.rank_count)] for _ in range(problem.rank_count)]
                for task in group:
                    matrix[task.src_rank][task.dst_rank] += task.payload_bytes
                byte_cert = decompose_integer_matrix(matrix)
                diagnostics.setdefault("birkhoff_bucket_certificates", []).append(cert.certificate_digest)
                diagnostics.setdefault("birkhoff_byte_lower_bounds", []).append(byte_cert.delta)
            elif core_id in {"islip", "residual_mwm", "fast", "aurora"}:
                literature_tasks = _literature_tasks(group)
                if core_id == "islip":
                    result = order_islip(literature_tasks, rank_count=problem.rank_count)
                elif core_id == "residual_mwm":
                    result = order_residual_mwm(literature_tasks, rank_count=problem.rank_count)
                elif core_id == "fast":
                    result = order_fast(
                        literature_tasks,
                        rank_count=problem.rank_count,
                        rank_to_node=rank_to_node,
                    )
                else:
                    result = order_aurora(literature_tasks, rank_count=problem.rank_count)
                group_order = result.ordered_task_ids
                group_waves = tuple(
                    AlgorithmWave(
                        wave_id=len(waves) + index,
                        task_ids=tuple(wave.task_ids),
                        phase_tokens=tuple(dict.fromkeys(by_id[item].phase_token for item in wave.task_ids)),
                        logical_duration_units=int(wave.duration_units),
                    )
                    for index, wave in enumerate(result.waves)
                )
                diagnostics.setdefault("core_plan_digests", []).append(result.plan_digest)
            elif core_id == "rscf":
                rscf_tasks = _rscf_tasks(
                    group,
                    semantic_phase_ordinal=(
                        rscf_semantic_phase_ordinal if len(groups) == 1 and len(problem.phase_tokens) == 1 else None
                    ),
                )
                dependencies = ()
                semantic_phases = tuple(sorted({task.phase for task in rscf_tasks}))
                if scope is PlannerScope.WINDOW_JOINT and len(semantic_phases) >= 2:
                    model = rscf_wire_cost_model or RSCFWireCostModel()
                    normalized_release = str(release_mode).upper()
                    if normalized_release == "PHASE_BARRIER":
                        dependency_factory = phase_barrier_release_dependency
                    elif normalized_release == "RANK_LOCAL":
                        dependency_factory = rank_local_release_dependency
                    else:
                        raise ValueError("release_mode must be PHASE_BARRIER or RANK_LOCAL")
                    dependencies = (
                        dependency_factory(
                            upstream_phase=semantic_phases[0],
                            downstream_phase=semantic_phases[1],
                            rank_count=problem.rank_count,
                            delay_provider=model.p1_to_p2_delay,
                        ),
                    )
                    diagnostics["rscf_release_scope"] = normalized_release
                result = order_rscf(
                    rscf_tasks,
                    rank_count=problem.rank_count,
                    wire_cost_model=rscf_wire_cost_model,
                    release_dependencies=dependencies,
                )
                group_order = result.ordered_task_ids
                group_waves = tuple(
                    AlgorithmWave(
                        wave_id=len(waves) + index,
                        task_ids=wave.task_ids,
                        phase_tokens=tuple(dict.fromkeys(by_id[item].phase_token for item in wave.task_ids)),
                        logical_duration_units=max(0, int(round(wave.finish_time - wave.start_time))),
                        service_segments=tuple(
                            (segment.task_id, int(segment.byte_offset), int(segment.service_units))
                            for segment in wave.segments
                        ),
                    )
                    for index, wave in enumerate(result.waves)
                    if wave.task_ids
                )
                diagnostics.setdefault("rscf_plan_digests", []).append(result.plan_digest)
            else:
                from rs_sim.scheduler.core.oracle import solve_exact_wire

                semantic_phase = (
                    rscf_semantic_phase_ordinal
                    if len(groups) == 1 and len(problem.phase_tokens) == 1
                    else None
                )
                oracle_problem = SchedulingProblem(
                    rank_count=problem.rank_count,
                    tasks=tuple(group),
                    phase_tokens=tuple(dict.fromkeys(task.phase_token for task in group)),
                    fairness=FairnessContract(
                        task_catalogue_digest=stable_digest(tuple(task.task_id for task in group)),
                        task_boundary_digest=stable_digest(tuple(
                            (task.task_id, task.phase_token, task.src_rank, task.dst_rank,
                             task.chunk_index, task.byte_offset, task.payload_bytes)
                            for task in group
                        )),
                        taskization_digest=problem.fairness.taskization_digest,
                        receiver_contract_rule_digest=problem.fairness.receiver_contract_rule_digest,
                        buffer_profile_digest=problem.fairness.buffer_profile_digest,
                        compiler_digest=problem.fairness.compiler_digest,
                        transport_digest=problem.fairness.transport_digest,
                        release_model_digest=problem.fairness.release_model_digest,
                        information_digest=problem.fairness.information_digest,
                        cost_model_digest=problem.fairness.cost_model_digest,
                    ),
                )
                oracle = solve_exact_wire(
                    oracle_problem,
                    wire_cost_model=rscf_wire_cost_model,
                    time_limit_ms=int(oracle_time_limit_ms),
                    relative_gap=float(oracle_relative_gap),
                    release_mode=str(release_mode),
                    semantic_phase_ordinal=semantic_phase,
                )
                if not oracle.has_feasible_schedule:
                    raise ValueError(
                        f"oracle unavailable: {oracle.solver_status}: {oracle.failure_reason}"
                    )
                if bool(oracle_require_certified) and not oracle.certified_optimal:
                    raise ValueError(
                        "oracle did not certify optimality: "
                        f"status={oracle.solver_status}, gap={oracle.optimality_gap}, "
                        f"bound={oracle.best_bound}, incumbent={oracle.objective_units}"
                    )
                solver_waves = tuple(
                    AlgorithmWave(
                        wave_id=index,
                        task_ids=tuple(wave.task_ids),
                        phase_tokens=tuple(dict.fromkeys(by_id[item].phase_token for item in wave.task_ids)),
                        logical_duration_units=int(wave.duration_units),
                    )
                    for index, wave in enumerate(oracle.waves)
                )
                selected_waves, selected_source, replay_scores = _oracle_ready_aware_portfolio(
                    oracle_problem,
                    solver_waves,
                    scope=scope,
                    release_mode=str(release_mode),
                    wire_cost_model=rscf_wire_cost_model,
                    semantic_phase_ordinal=semantic_phase,
                )
                group_waves = _renumber_waves(selected_waves, offset=len(waves))
                group_order = tuple(item for wave in group_waves for item in wave.task_ids)
                diagnostics.setdefault("oracle_result_digests", []).append(oracle.result_digest)
                diagnostics.setdefault("oracle_solver_status", []).append(oracle.solver_status)
                diagnostics.setdefault("oracle_certified_optimal", []).append(oracle.certified_optimal)
                diagnostics.setdefault("oracle_solver_model_certified", []).append(
                    oracle.certified_optimal
                )
                diagnostics.setdefault("oracle_objective_units", []).append(oracle.objective_units)
                diagnostics.setdefault("oracle_best_bound", []).append(oracle.best_bound)
                diagnostics.setdefault("oracle_optimality_gap_ppm", []).append(
                    None
                    if oracle.optimality_gap is None
                    else int(round(float(oracle.optimality_gap) * 1_000_000.0))
                )
                diagnostics.setdefault("oracle_solve_time_us", []).append(
                    None
                    if oracle.solve_time_ms is None
                    else int(round(float(oracle.solve_time_ms) * 1_000.0))
                )
                diagnostics.setdefault("oracle_variable_count", []).append(oracle.variable_count)
                diagnostics.setdefault("oracle_constraint_count", []).append(oracle.constraint_count)
                diagnostics.setdefault("oracle_canonical_task_count", []).append(
                    oracle.canonical_task_count
                )
                diagnostics.setdefault("oracle_symmetry_group_count", []).append(
                    oracle.symmetry_group_count
                )
                diagnostics.setdefault("oracle_candidate_slot_count", []).append(
                    oracle.candidate_slot_count
                )
                diagnostics.setdefault("oracle_incumbent_source", []).append(
                    oracle.incumbent_source
                )
                diagnostics.setdefault("oracle_selected_candidate_source", []).append(
                    selected_source
                )
                diagnostics.setdefault("oracle_selected_matches_solver", []).append(
                    selected_source == "solver"
                )
                diagnostics.setdefault("oracle_ready_aware_objective_ns", []).append(
                    int(replay_scores[selected_source])
                )
                diagnostics.setdefault("oracle_solver_ready_aware_objective_ns", []).append(
                    int(replay_scores["solver"])
                )
                diagnostics.setdefault("oracle_ready_aware_candidate_scores", []).append(
                    tuple(sorted((name, int(value)) for name, value in replay_scores.items()))
                )
                diagnostics.setdefault("oracle_runtime_replay_scope", []).append(
                    "TRANSPORT_PRIORITY_DIAGNOSTIC_NOT_FULL_BACKEND"
                )
            if tuple(item for wave in group_waves for item in wave.task_ids) != tuple(group_order):
                raise ValueError(f"{core_id} wave order is inconsistent")
            waves.extend(group_waves)

        if scope is PlannerScope.WINDOW_JOINT and core_id in _JOINT_BASELINE_ADAPTATIONS:
            phases = tuple(sorted({int(task.phase_ordinal) for task in problem.tasks}))
            p2_phase = phases[1] if len(phases) >= 2 else None
            p2_tasks = tuple(
                task
                for task in problem.tasks
                if p2_phase is not None and int(task.phase_ordinal) == p2_phase
            )
            diagnostics["joint_p2_adaptation"] = _JOINT_BASELINE_ADAPTATIONS[core_id]
            diagnostics["joint_p2_task_count"] = len(p2_tasks)
            diagnostics["joint_p2_payload_units"] = sum(
                int(task.payload_bytes) for task in p2_tasks
            )
            diagnostics["zero_p2_reduction_contract"] = "EXACT_P1_ONLY_CORE_ORDER"

        order = tuple(item for wave in waves for item in wave.task_ids)
        diagnostics = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in diagnostics.items()
        }
        payload = {
            "algorithm_id": core_id,
            "scope": scope.value,
            "order": order,
            "waves": tuple(waves),
            "fairness_digest": problem.fairness.fairness_digest,
            "diagnostics": diagnostics,
        }
        plan = AlgorithmPlan(
            algorithm_id=core_id,
            scope=scope,
            execution_mode=ExecutionMode.ORDER_ONLY,
            ordered_task_ids=order,
            waves=tuple(waves),
            fairness=problem.fairness,
            task_catalogue_digest=problem.fairness.task_catalogue_digest,
            task_boundary_digest=problem.fairness.task_boundary_digest,
            plan_digest=stable_digest(payload),
            diagnostics=tuple(sorted(diagnostics.items())),
        )
        plan.validate_against(problem)
        return plan


def validate_order_only_pair(plans: Iterable[AlgorithmPlan]) -> None:
    items = tuple(plans)
    if not items:
        raise ValueError("at least one plan is required")
    reference = items[0].fairness
    for plan in items:
        if plan.execution_mode is not ExecutionMode.ORDER_ONLY:
            raise ValueError("comparison contains a non-canonical execution mode")
        if plan.fairness != reference:
            raise ValueError("comparison changed the fairness contract")


__all__ = [
    "AlgorithmPlan",
    "AlgorithmWave",
    "ExecutionMode",
    "FairnessContract",
    "OrderOnlyPlanner",
    "PlannerScope",
    "SchedulingProblem",
    "SchedulingTask",
    "_critical_completion_objective_for_waves",
    "_estimated_p1_release_tail_for_waves",
    "build_problem_from_catalogue",
    "validate_order_only_pair",
]
