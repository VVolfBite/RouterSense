from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pytest

from rs_sim import ControlPlaneDelivery, stable_digest
from rs_sim.backend import (
    AttributeSharedObjectAdapter,
    BackendTraceFixtureBuilder,
    LinearReceiverCostModel,
    ReceiverJobStatus,
    ReceiverService,
    SimulationBackend,
    compute_fixture_staging_capacity_bytes_by_rank,
)
from rs_sim.runtime.adapters.backend import BackendControlPlaneAdapter
from rs_sim.runtime.assembly.bindings import (
    SchemaEdgeKeyFactory,
    SchemaExpectationFactory,
    SchemaPermitFactory,
    build_scheduling_stack,
    make_phase_semantics,
)
from rs_sim.scheduler import TaskizationSpec
from rs_sim.trace.schema.fixtures import build_builtin_fixtures


class Observer:
    def __init__(self) -> None:
        self.rows: list[tuple[str, int, dict[str, Any]]] = []

    def emit(self, *, kind: str, at_ns: int, payload: Mapping[str, Any]) -> None:
        self.rows.append((kind, int(at_ns), dict(payload)))

    def digest(self) -> str:
        normalized = []
        for kind, at_ns, payload in self.rows:
            normalized.append(
                {
                    "kind": kind,
                    "at_ns": at_ns,
                    "phase": repr(payload.get("phase_key")),
                    "rank": payload.get("rank_id", payload.get("dst_rank", payload.get("src_rank"))),
                    "task": str(payload.get("task_id", "")),
                }
            )
        return stable_digest(tuple(normalized), domain="BACKEND_EP4_EVENT_EXPECTATION")


class KernelHarness:
    def __init__(self) -> None:
        self._events: list[tuple[int, int, str, int, str, Any]] = []
        self._ordinal = 0
        self.backend: SimulationBackend | None = None
        self.now_ns = 0

    def schedule_backend_event(
        self,
        *,
        time_ns: int,
        phase_priority: int,
        stable_event_id: str,
        event_kind: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._push(time_ns, phase_priority, stable_event_id, "backend", (event_kind, dict(payload)))

    def schedule_external(
        self, *, time_ns: int, phase_priority: int, stable_event_id: str, callback: Callable[[], None]
    ) -> None:
        self._push(time_ns, phase_priority, stable_event_id, "external", callback)

    def _push(self, time_ns: int, priority: int, stable_id: str, kind: str, payload: Any) -> None:
        self._ordinal += 1
        heapq.heappush(
            self._events,
            (int(time_ns), int(priority), str(stable_id), self._ordinal, kind, payload),
        )

    def run_one(self) -> None:
        assert self.backend is not None
        time_ns, _, _, _, kind, payload = heapq.heappop(self._events)
        self.now_ns = time_ns
        if kind == "backend":
            event_kind, event_payload = payload
            self.backend.handle_event(event_kind=event_kind, payload=event_payload, at_ns=time_ns)
        else:
            payload()

    @property
    def has_events(self) -> bool:
        return bool(self._events)


class ScheduledControlPlane:
    def __init__(self, *, kernel: KernelHarness, latency_ns: int) -> None:
        self.kernel = kernel
        self.latency_ns = int(latency_ns)
        self.sink: Any | None = None
        self.requests: list[Any] = []

    def attach_delivery_sink(self, sink: Any) -> None:
        self.sink = sink

    def publish_row(self, request: Any) -> str:
        assert self.sink is not None
        request_digest = stable_digest(request, domain="ROW_BROADCAST_REQUEST")
        delivered = int(request.published_at_ns) + self.latency_ns
        delivery = ControlPlaneDelivery(
            request_digest=request_digest,
            phase_key=request.phase_key,
            src_rank=request.src_rank,
            delivery_start_ns=request.published_at_ns,
            delivered_at_ns=delivered,
            control_channel_id="backend-test-control",
        )
        self.requests.append(request)
        self.kernel.schedule_external(
            time_ns=delivered,
            phase_priority=2,
            stable_event_id=f"cp:{request_digest}",
            callback=lambda delivery=delivery: self.sink.on_control_plane_delivery(delivery),
        )
        return request_digest


@dataclass
class RunResult:
    backend: SimulationBackend
    receiver: ReceiverService
    observer: Observer
    registration: Any
    digest: str


def run_two_layer_ep4(*, sensitivity: str, release_mode: str = "RANK_LOCAL") -> RunResult:
    fixture = build_builtin_fixtures()[0]
    max_task_bytes = 8192
    capacities = compute_fixture_staging_capacity_bytes_by_rank(
        fixture_input=fixture,
        sensitivity=sensitivity,
        alignment_bytes=256,
        max_canonical_task_payload_bytes=max_task_bytes,
    )
    kernel = KernelHarness()
    observer = Observer()
    adapter = AttributeSharedObjectAdapter()
    semantics = make_phase_semantics()
    receiver = ReceiverService(
        world_size=fixture.world_size,
        staging_capacity_bytes_by_rank=capacities,
        kernel=kernel,
        observer=observer,
        adapter=adapter,
        phase_semantics=semantics,
        permit_factory=SchemaPermitFactory(),
        cost_model=LinearReceiverCostModel(
            posting_fixed_ns=3,
            posting_bytes_per_ns=4096,
            drain_fixed_ns=2,
            drain_bytes_per_ns=4096,
        ),
        local_assembly_cost_ns=5,
    )
    backend = SimulationBackend(
        world_size=fixture.world_size,
        kernel=kernel,
        observer=observer,
        adapter=adapter,
        phase_semantics=semantics,
        edge_key_factory=SchemaEdgeKeyFactory(),
        expectation_factory=SchemaExpectationFactory(),
        receiver=receiver,
        release_mode=release_mode,
        node_id_by_rank={rank: fixture.windows[0].mapping.rank_to_node[rank] for rank in range(fixture.world_size)},
    )
    kernel.backend = backend
    control = ScheduledControlPlane(kernel=kernel, latency_ns=11)
    control_adapter = BackendControlPlaneAdapter(backend=backend, control_plane=control)
    backend.attach_exact_row_publisher(control_adapter)
    registration = BackendTraceFixtureBuilder(backend=backend).register_fixture(
        fixture_input=fixture,
        run_id=f"backend-{sensitivity}-{release_mode}",
    )

    scheduling = build_scheduling_stack(
        taskization_spec=TaskizationSpec(chunk_bytes=max_task_bytes, alignment_bytes=256)
    )
    taskized_edges: set[str] = set()
    completion_scheduled: set[str] = set()

    def register_new_catalogues() -> None:
        for edge_key, edge in sorted(receiver.edges_by_key.items()):
            if edge_key in taskized_edges or edge.zero_edge or edge.src_rank == edge.dst_rank:
                taskized_edges.add(edge_key)
                continue
            tasks = scheduling.taskizer.taskize(
                edge.expectation_object,
                registered_at_ns=edge.expectation_available_at_ns,
            )
            backend.register_canonical_task_catalogue(tasks)
            taskized_edges.add(edge_key)

    def schedule_new_completions() -> None:
        for task_key, task in sorted(receiver.tasks_by_key.items()):
            if task_key in completion_scheduled or task.status is not ReceiverJobStatus.POSTED:
                continue
            assert task.receive_posted_at_ns is not None
            completion_at = task.receive_posted_at_ns + 7 + task.payload_bytes // 4096
            kernel.schedule_external(
                time_ns=completion_at,
                phase_priority=3,
                stable_event_id=f"net:{task_key}",
                callback=lambda task_id=task.task_id, at=completion_at: backend.on_transfer_completed(
                    task_id=task_id, at_ns=at
                ),
            )
            completion_scheduled.add(task_key)

    for _ in range(200000):
        register_new_catalogues()
        schedule_new_completions()
        terminal = backend.phase_terminal_snapshot(
            phase_key=registration.terminal_combine_phase_key
        )
        if terminal["closed"]:
            break
        assert kernel.has_events, terminal
        kernel.run_one()
    else:
        raise AssertionError("EP4 fixture did not terminate")

    for phase_key in registration.all_phase_keys:
        backend.assert_phase_closed(phase_key=phase_key)
    assert all(receiver.current_memory(rank)["total_receiver_bytes"] == 0 for rank in range(fixture.world_size))
    return RunResult(backend, receiver, observer, registration, observer.digest())


@pytest.mark.parametrize("sensitivity", ["UNBOUNDED", "1.0X", "0.5X", "0.25X"])
def test_two_layer_ep4_closes_for_all_staging_sensitivities(sensitivity: str) -> None:
    result = run_two_layer_ep4(sensitivity=sensitivity)
    metrics = result.backend.metrics_snapshot()
    assert set(metrics.peak_staging_bytes_per_rank) == {0, 1, 2, 3}
    assert all(value >= 0 for value in metrics.receiver_posting_service_ns.values())
    assert all(value >= 0 for value in metrics.receiver_posting_queue_wait_ns.values())
    assert all(value >= 0 for value in metrics.receiver_drain_service_ns.values())
    assert result.observer.rows


def test_two_layer_ep4_digest_is_deterministic() -> None:
    digests = {run_two_layer_ep4(sensitivity="0.25X").digest for _ in range(5)}
    assert len(digests) == 1


def test_rank_local_fast_rank_advances_before_slow_rank() -> None:
    result = run_two_layer_ep4(sensitivity="0.5X", release_mode="RANK_LOCAL")
    first_dispatch = result.registration.windows[0].keys.dispatch_phase_key
    releases = [
        result.backend.rank_release_at(phase_key=first_dispatch, rank_id=rank)
        for rank in range(4)
    ]
    assert all(value is not None for value in releases)
    assert len(set(releases)) > 1


def test_phase_barrier_equalizes_dispatch_release() -> None:
    result = run_two_layer_ep4(sensitivity="0.5X", release_mode="PHASE_BARRIER")
    first_dispatch = result.registration.windows[0].keys.dispatch_phase_key
    releases = {
        result.backend.rank_release_at(phase_key=first_dispatch, rank_id=rank)
        for rank in range(4)
    }
    assert len(releases) == 1
