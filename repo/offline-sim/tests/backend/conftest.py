from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from rs_sim.backend import (
    AttributeSharedObjectAdapter,
    CallablePhaseSemantics,
    LinearReceiverCostModel,
    ReceiverService,
    SimulationBackend,
)


@dataclass(frozen=True)
class Phase:
    window: int
    layer: int
    kind: str


@dataclass(frozen=True)
class Expectation:
    edge_key: Any
    phase_key: Any
    src_rank: int
    dst_rank: int
    total_expected_payload_bytes: int
    expectation_digest: str
    origin: str
    created_at_ns: int
    zero_edge: bool
    descriptor_digest_or_none: str | None = None


@dataclass(frozen=True)
class Task:
    task_id: str
    edge_key: Any
    phase_key: Any
    src_rank: int
    dst_rank: int
    chunk_index: int
    byte_offset: int
    payload_bytes: int
    taskization_digest: str
    registered_at_ns: int


@dataclass(frozen=True)
class Permit:
    permit_id: str
    task_id: Any
    edge_key: Any
    chunk_index: int
    byte_offset: int
    task_bytes: int
    credit_reservation_id: str
    expectation_digest: str
    descriptor_digest_or_none: str | None
    posted_at_ns: int


class EdgeFactory:
    def make_edge_key(self, *, phase_key: Any, src_rank: int, dst_rank: int) -> Any:
        return (phase_key, src_rank, dst_rank)


class ExpectationObjects:
    def create_receive_expectation(self, **kwargs: Any) -> Expectation:
        return Expectation(**kwargs)


class PermitObjects:
    def create_receive_permit(self, **kwargs: Any) -> Permit:
        return Permit(**kwargs)


class Observer:
    def __init__(self) -> None:
        self.rows: list[tuple[str, int, dict[str, Any]]] = []

    def emit(self, *, kind: str, at_ns: int, payload: Mapping[str, Any]) -> None:
        self.rows.append((kind, at_ns, dict(payload)))

    def times(self, kind: str) -> list[int]:
        return [at for row_kind, at, _ in self.rows if row_kind == kind]

    def payloads(self, kind: str) -> list[dict[str, Any]]:
        return [payload for row_kind, _, payload in self.rows if row_kind == kind]


class KernelHarness:
    def __init__(self) -> None:
        self._events: list[tuple[int, int, str, int, str, dict[str, Any]]] = []
        self._ordinal = 0
        self.backend: SimulationBackend | None = None

    def schedule_backend_event(
        self,
        *,
        time_ns: int,
        phase_priority: int,
        stable_event_id: str,
        event_kind: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._ordinal += 1
        heapq.heappush(
            self._events,
            (time_ns, phase_priority, stable_event_id, self._ordinal, event_kind, dict(payload)),
        )

    def run_until(self, limit_ns: int | None = None) -> None:
        assert self.backend is not None
        while self._events and (limit_ns is None or self._events[0][0] <= limit_ns):
            at_ns, _, _, _, kind, payload = heapq.heappop(self._events)
            self.backend.handle_event(event_kind=kind, payload=payload, at_ns=at_ns)

    def next_time(self) -> int | None:
        return self._events[0][0] if self._events else None


@dataclass
class System:
    backend: SimulationBackend
    receiver: ReceiverService
    kernel: KernelHarness
    observer: Observer
    adapter: AttributeSharedObjectAdapter
    edge_factory: EdgeFactory


def make_system(
    *,
    world_size: int,
    capacity: int | None,
    posting_fixed_ns: int = 0,
    drain_fixed_ns: int = 0,
    release_mode: str = "RANK_LOCAL",
    p0_p1_compute_end_barrier: bool = False,
) -> System:
    adapter = AttributeSharedObjectAdapter()
    phase_semantics = CallablePhaseSemantics(
        phase_kind=lambda phase: phase.kind,
        phase_sort_key=lambda phase: f"w{phase.window}:l{phase.layer}:{phase.kind}",
    )
    kernel = KernelHarness()
    observer = Observer()
    receiver = ReceiverService(
        world_size=world_size,
        staging_capacity_bytes_by_rank={rank: capacity for rank in range(world_size)},
        kernel=kernel,
        observer=observer,
        adapter=adapter,
        phase_semantics=phase_semantics,
        permit_factory=PermitObjects(),
        cost_model=LinearReceiverCostModel(
            posting_fixed_ns=posting_fixed_ns,
            posting_bytes_per_ns=10**9,
            drain_fixed_ns=drain_fixed_ns,
            drain_bytes_per_ns=10**9,
        ),
        allow_legacy_local_network_tasks_for_unit_tests=True,
    )
    edge_factory = EdgeFactory()
    backend = SimulationBackend(
        world_size=world_size,
        kernel=kernel,
        observer=observer,
        adapter=adapter,
        phase_semantics=phase_semantics,
        edge_key_factory=edge_factory,
        expectation_factory=ExpectationObjects(),
        receiver=receiver,
        release_mode=release_mode,
        p0_p1_compute_end_barrier=p0_p1_compute_end_barrier,
    )
    kernel.backend = backend
    return System(backend, receiver, kernel, observer, adapter, edge_factory)


def make_task(
    *,
    phase: Phase,
    src: int,
    dst: int,
    chunk: int,
    offset: int,
    size: int,
    registered: int = 0,
) -> Task:
    return Task(
        task_id=f"{phase.window}:{phase.layer}:{phase.kind}:{src}->{dst}:{chunk}",
        edge_key=(phase, src, dst),
        phase_key=phase,
        src_rank=src,
        dst_rank=dst,
        chunk_index=chunk,
        byte_offset=offset,
        payload_bytes=size,
        taskization_digest="fixture-taskization-v1",
        registered_at_ns=registered,
    )


@pytest.fixture

def phases() -> dict[str, Phase]:
    return {
        "d0": Phase(0, 0, "DISPATCH"),
        "c0": Phase(0, 0, "COMBINE"),
        "d1": Phase(0, 1, "DISPATCH"),
        "c1": Phase(0, 1, "COMBINE"),
    }
