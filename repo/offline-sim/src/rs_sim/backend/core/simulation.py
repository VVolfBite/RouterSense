from __future__ import annotations

import heapq
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

from rs_sim.contracts.schema import GoldenTimelineRow, KernelPhase, SimulationEvent, TimeNs
from rs_sim.contracts.digest import stable_digest, stable_json_dumps


class KernelFaultCode(str, Enum):
    CAUSAL_CYCLE = "CAUSAL_CYCLE"
    DEADLOCK_NO_PROGRESS = "DEADLOCK_NO_PROGRESS"


class KernelContractError(RuntimeError):
    """Base class for fail-closed kernel contract violations."""


class PastEventError(KernelContractError):
    pass


class RecursiveExecutionError(KernelContractError):
    pass


class DuplicateStableEventIdError(KernelContractError):
    pass


class UnknownEventTypeError(KernelContractError):
    pass


@dataclass(frozen=True, slots=True)
class ProgressSignal:
    authoritative_state_updates: int = 0
    successful_commits: int = 0
    rank_releases: int = 0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "authoritative_state_updates",
            "successful_commits",
            "rank_releases",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if not isinstance(self.notes, tuple) or not all(isinstance(x, str) for x in self.notes):
            raise TypeError("notes must be a tuple[str, ...]")

    @property
    def made_progress(self) -> bool:
        return bool(
            self.authoritative_state_updates
            or self.successful_commits
            or self.rank_releases
        )

    def merged(self, other: ProgressSignal | None) -> ProgressSignal:
        if other is None:
            return self
        if not isinstance(other, ProgressSignal):
            raise TypeError("kernel handlers and phase callbacks must return ProgressSignal or None")
        return ProgressSignal(
            authoritative_state_updates=(
                self.authoritative_state_updates + other.authoritative_state_updates
            ),
            successful_commits=self.successful_commits + other.successful_commits,
            rank_releases=self.rank_releases + other.rank_releases,
            notes=self.notes + other.notes,
        )


@dataclass(frozen=True, slots=True)
class RoundSummary:
    time_ns: TimeNs
    round_index: int
    events_processed: int
    authoritative_state_updates: int
    successful_commits: int
    rank_releases: int
    same_time_events_pending: int


@dataclass(frozen=True, slots=True)
class KernelFaultEvidence:
    fault_code: KernelFaultCode
    time_ns: TimeNs
    message: str
    max_stabilization_rounds: int
    pending_event_count: int
    next_event_time_ns: TimeNs | None
    round_summaries: tuple[RoundSummary, ...]
    timeline_tail: tuple[GoldenTimelineRow, ...]
    external_evidence: tuple[tuple[str, str], ...]


class KernelFault(KernelContractError):
    def __init__(self, evidence: KernelFaultEvidence):
        super().__init__(f"{evidence.fault_code.value}: {evidence.message}")
        self.evidence = evidence


EventHandler: TypeAlias = Callable[["SimulationKernel", SimulationEvent], ProgressSignal | None]
PhaseCallback: TypeAlias = Callable[["SimulationKernel"], ProgressSignal | None]
EvidenceProvider: TypeAlias = Callable[[], Mapping[str, Any]]


def make_stable_event_id(
    *,
    producer: str,
    time_ns: TimeNs,
    round_index: int,
    phase_priority: KernelPhase,
    event_type: str,
    ordinal: int,
    subject_id: str = "",
) -> str:
    """Create a semantic event identity independent of object address/hash order."""

    semantic_identity = (
        producer,
        time_ns,
        round_index,
        int(phase_priority),
        event_type,
        ordinal,
        subject_id,
    )
    digest = stable_digest(semantic_identity, domain="RS-SIM-STABLE-EVENT-ID")[:32]
    return f"evt-{ordinal:020d}-{digest}"


class SimulationKernel:
    """Integer-nanosecond, round-based, eight-phase fixed-point kernel."""

    def __init__(
        self,
        *,
        max_stabilization_rounds: int = 1024,
        timeline_tail_limit: int = 64,
        work_remaining_predicate: Callable[[], bool] | None = None,
    ) -> None:
        if not isinstance(max_stabilization_rounds, int) or max_stabilization_rounds <= 0:
            raise ValueError("max_stabilization_rounds must be a positive int")
        if not isinstance(timeline_tail_limit, int) or timeline_tail_limit <= 0:
            raise ValueError("timeline_tail_limit must be a positive int")
        self.max_stabilization_rounds = max_stabilization_rounds
        self._timeline_tail_limit = timeline_tail_limit
        self._queue: list[tuple[tuple[int, int, int, str], SimulationEvent]] = []
        self._queued_ids: set[str] = set()
        self._processed_ids: set[str] = set()
        self._handlers: dict[str, EventHandler] = {}
        self._phase_callbacks: dict[KernelPhase, list[tuple[str, PhaseCallback]]] = defaultdict(list)
        self._evidence_providers: dict[str, EvidenceProvider] = {}
        self._timeline: list[GoldenTimelineRow] = []
        self._timeline_tail: deque[GoldenTimelineRow] = deque(maxlen=timeline_tail_limit)
        self._now_ns: TimeNs = 0
        self._started = False
        self._processing = False
        self._current_round: int | None = None
        self._current_phase: KernelPhase | None = None
        self._round_history: deque[RoundSummary] = deque(maxlen=timeline_tail_limit)
        self._work_remaining_predicate = work_remaining_predicate

    @property
    def now_ns(self) -> TimeNs:
        return self._now_ns

    @property
    def current_round(self) -> int | None:
        return self._current_round

    @property
    def current_phase(self) -> KernelPhase | None:
        return self._current_phase

    def has_events(self) -> bool:
        return bool(self._queue)

    def pending_event_count(self) -> int:
        return len(self._queue)

    def dispose(self) -> None:
        """Release callback graphs after a terminal run.

        The simulator is intentionally single-process and its handlers are bound
        methods that form large reference cycles across transport/backend/scheduler.  Audit runners
        may execute many independent runs in one process; clearing the kernel-owned
        callback registry after terminal evidence is frozen prevents interpreter
        shutdown and cumulative-suite cleanup from traversing the entire completed
        runtime graph.  A disposed kernel is not reusable.
        """

        if self._processing:
            raise KernelContractError("cannot dispose SimulationKernel during execution")
        self._queue.clear()
        self._queued_ids.clear()
        self._processed_ids.clear()
        self._handlers.clear()
        self._phase_callbacks.clear()
        self._evidence_providers.clear()
        self._timeline.clear()
        self._timeline_tail.clear()
        self._round_history.clear()
        self._work_remaining_predicate = None

    def next_event_time_ns(self) -> TimeNs | None:
        return self._queue[0][1].time_ns if self._queue else None

    def register_event_handler(self, event_type: str, handler: EventHandler) -> None:
        if not event_type:
            raise ValueError("event_type must be non-empty")
        if event_type in self._handlers:
            raise ValueError(f"handler already registered for {event_type}")
        self._handlers[event_type] = handler

    def register_phase_callback(
        self,
        phase: KernelPhase,
        callback_name: str,
        callback: PhaseCallback,
    ) -> None:
        if not isinstance(phase, KernelPhase):
            raise TypeError("phase must be KernelPhase")
        if not callback_name:
            raise ValueError("callback_name must be non-empty")
        callbacks = self._phase_callbacks[phase]
        if any(name == callback_name for name, _ in callbacks):
            raise ValueError(f"duplicate phase callback name: {callback_name}")
        callbacks.append((callback_name, callback))
        callbacks.sort(key=lambda item: item[0])

    def set_work_remaining_predicate(
        self, predicate: Callable[[], bool] | None
    ) -> None:
        if self._processing:
            raise KernelContractError(
                "work_remaining_predicate cannot be changed during execution"
            )
        self._work_remaining_predicate = predicate

    def register_evidence_provider(self, name: str, provider: EvidenceProvider) -> None:
        if not name:
            raise ValueError("evidence provider name must be non-empty")
        if name in self._evidence_providers:
            raise ValueError(f"duplicate evidence provider: {name}")
        self._evidence_providers[name] = provider

    def schedule(
        self,
        *,
        time_ns: TimeNs,
        phase_priority: KernelPhase,
        producer: str,
        event_type: str,
        ordinal: int,
        subject_id: str = "",
        attributes: tuple[tuple[str, str], ...] = (),
    ) -> SimulationEvent:
        if not isinstance(time_ns, int) or isinstance(time_ns, bool) or time_ns < 0:
            raise ValueError("time_ns must be a non-negative int")
        if not isinstance(phase_priority, KernelPhase):
            raise TypeError("phase_priority must be KernelPhase")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative int")

        if self._processing:
            if time_ns < self._now_ns:
                raise PastEventError(f"cannot schedule event in the past: {time_ns} < {self._now_ns}")
            if time_ns == self._now_ns:
                assert self._current_round is not None and self._current_phase is not None
                round_index = (
                    self._current_round
                    if phase_priority > self._current_phase
                    else self._current_round + 1
                )
            else:
                round_index = 0
        else:
            if self._started and time_ns <= self._now_ns:
                raise PastEventError(
                    "after a timestamp reaches quiescence, scheduling at or before now is illegal"
                )
            round_index = 0

        stable_event_id = make_stable_event_id(
            producer=producer,
            time_ns=time_ns,
            round_index=round_index,
            phase_priority=phase_priority,
            event_type=event_type,
            ordinal=ordinal,
            subject_id=subject_id,
        )
        if stable_event_id in self._queued_ids or stable_event_id in self._processed_ids:
            raise DuplicateStableEventIdError(stable_event_id)
        event = SimulationEvent(
            time_ns=time_ns,
            round_index=round_index,
            phase_priority=phase_priority,
            stable_event_id=stable_event_id,
            producer=producer,
            event_type=event_type,
            subject_id=subject_id,
            ordinal=ordinal,
            attributes=attributes,
        )
        heapq.heappush(self._queue, (event.sort_key, event))
        self._queued_ids.add(stable_event_id)
        return event

    def timeline(self) -> tuple[GoldenTimelineRow, ...]:
        return tuple(self._timeline)

    def timeline_digest(self) -> str:
        return stable_digest(self.timeline(), domain="RS_SIM_SIMULATION_TIMELINE")

    def event_digest(self) -> str:
        semantic_rows = tuple(
            (
                row.time_ns,
                row.round_index,
                int(row.phase_priority),
                row.stable_event_id,
                row.producer,
                row.event_type,
                row.subject_id,
                row.outcome,
                row.details_digest,
            )
            for row in self._timeline
        )
        return stable_digest(semantic_rows, domain="RS_SIM_SIMULATION_EVENTS")

    def run_next_timestamp(self) -> tuple[SimulationEvent, ...]:
        if self._processing:
            raise RecursiveExecutionError("SimulationKernel execution is not recursive")
        if not self._queue:
            return ()

        timestamp = self._queue[0][1].time_ns
        self._now_ns = timestamp
        self._started = True
        self._processing = True
        processed_at_timestamp: list[SimulationEvent] = []
        local_round_summaries: list[RoundSummary] = []

        try:
            round_index = 0
            while True:
                if round_index >= self.max_stabilization_rounds:
                    raise KernelFault(
                        self._fault_evidence(
                            KernelFaultCode.CAUSAL_CYCLE,
                            f"timestamp {timestamp} exceeded fixed-point round limit",
                            local_round_summaries,
                        )
                    )

                round_signal = ProgressSignal()
                events_processed_this_round = 0
                self._current_round = round_index

                for phase in KernelPhase:
                    self._current_phase = phase
                    while self._slot_available(timestamp, round_index, phase):
                        _, event = heapq.heappop(self._queue)
                        self._queued_ids.remove(event.stable_event_id)
                        handler = self._handlers.get(event.event_type)
                        if handler is None:
                            raise UnknownEventTypeError(event.event_type)
                        signal = handler(self, event)
                        round_signal = round_signal.merged(signal)
                        self._processed_ids.add(event.stable_event_id)
                        processed_at_timestamp.append(event)
                        events_processed_this_round += 1
                        self._record_timeline(event, signal)

                    for _, callback in self._phase_callbacks.get(phase, ()):
                        round_signal = round_signal.merged(callback(self))

                same_time_pending = sum(
                    1 for _, event in self._queue if event.time_ns == timestamp
                )
                summary = RoundSummary(
                    time_ns=timestamp,
                    round_index=round_index,
                    events_processed=events_processed_this_round,
                    authoritative_state_updates=round_signal.authoritative_state_updates,
                    successful_commits=round_signal.successful_commits,
                    rank_releases=round_signal.rank_releases,
                    same_time_events_pending=same_time_pending,
                )
                local_round_summaries.append(summary)
                self._round_history.append(summary)

                if same_time_pending or round_signal.made_progress:
                    round_index += 1
                    continue
                break
        finally:
            self._processing = False
            self._current_round = None
            self._current_phase = None

        if (
            not self._queue
            and self._work_remaining_predicate is not None
            and self._work_remaining_predicate()
        ):
            raise KernelFault(
                self._fault_evidence(
                    KernelFaultCode.DEADLOCK_NO_PROGRESS,
                    "timestamp reached quiescence while registered work remains",
                    local_round_summaries,
                )
            )

        return tuple(processed_at_timestamp)

    def run_until_complete(
        self,
        completion_predicate: Callable[[], bool],
        *,
        max_timestamps: int | None = None,
    ) -> tuple[GoldenTimelineRow, ...]:
        timestamps_run = 0
        while not completion_predicate():
            if not self._queue:
                raise KernelFault(
                    self._fault_evidence(
                        KernelFaultCode.DEADLOCK_NO_PROGRESS,
                        "work remains but no future event can make progress",
                        tuple(self._round_history),
                    )
                )
            self.run_next_timestamp()
            timestamps_run += 1
            if max_timestamps is not None and timestamps_run >= max_timestamps:
                if not completion_predicate():
                    raise KernelContractError("max_timestamps reached before completion")
        return self.timeline()

    def _slot_available(
        self,
        time_ns: TimeNs,
        round_index: int,
        phase: KernelPhase,
    ) -> bool:
        if not self._queue:
            return False
        event = self._queue[0][1]
        return (
            event.time_ns == time_ns
            and event.round_index == round_index
            and event.phase_priority == phase
        )

    def _record_timeline(
        self,
        event: SimulationEvent,
        signal: ProgressSignal | None,
    ) -> None:
        details = signal if signal is not None else ProgressSignal()
        row = GoldenTimelineRow(
            timeline_index=len(self._timeline),
            time_ns=event.time_ns,
            round_index=event.round_index,
            phase_priority=event.phase_priority,
            stable_event_id=event.stable_event_id,
            producer=event.producer,
            event_type=event.event_type,
            subject_id=event.subject_id,
            outcome="HANDLED",
            details_digest=stable_digest(details, domain="RS-SIM-EVENT-OUTCOME"),
        )
        self._timeline.append(row)
        self._timeline_tail.append(row)

    def _fault_evidence(
        self,
        fault_code: KernelFaultCode,
        message: str,
        round_summaries: list[RoundSummary] | tuple[RoundSummary, ...],
    ) -> KernelFaultEvidence:
        external: list[tuple[str, str]] = []
        for name in sorted(self._evidence_providers):
            try:
                evidence = self._evidence_providers[name]()
                encoded = stable_json_dumps(evidence)
            except Exception as exc:  # evidence collection must not hide root fault
                encoded = stable_json_dumps({"provider_error": type(exc).__name__, "message": str(exc)})
            external.append((name, encoded))
        return KernelFaultEvidence(
            fault_code=fault_code,
            time_ns=self._now_ns,
            message=message,
            max_stabilization_rounds=self.max_stabilization_rounds,
            pending_event_count=len(self._queue),
            next_event_time_ns=self.next_event_time_ns(),
            round_summaries=tuple(round_summaries)[-self._timeline_tail_limit :],
            timeline_tail=tuple(self._timeline_tail),
            external_evidence=tuple(external),
        )
