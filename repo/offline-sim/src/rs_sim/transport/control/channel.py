from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

from rs_sim.contracts.factories import (
    ceil_transfer_time_ns,
    control_plane_profile_digest,
)
from rs_sim.backend.core.simulation import ProgressSignal, SimulationKernel
from rs_sim.contracts.schema import (
    ControlPlaneDelivery,
    ControlPlaneProfile,
    KernelPhase,
    PhaseKind,
    RowBroadcastRequest,
    SimulationEvent,
    PhaseKey,
    WindowKey,
)
from rs_sim.contracts.digest import stable_digest
from rs_sim.transport.api.ports import ControlPlaneDeliverySink

from ..core.ordering import semantic_ordinal
from ..core.lifecycle import make_process_lifecycle_diagnostics


@dataclass(frozen=True, slots=True)
class _QueuedRequest:
    arrival_ns: int
    publish_sequence: int
    request_digest: str
    request: RowBroadcastRequest


@dataclass(frozen=True, slots=True)
class _ControlMechanismEvent:
    phase_key: PhaseKey
    kind: str
    payload_bytes: int
    queue_wait_ns: int
    service_time_ns: int
    end_to_end_ns: int
    at_ns: int


class FormalControlPlaneTransport:
    """One deterministic FIFO, single-server, non-preemptive row broadcast channel.

    Arrival events are collected before a same-timestamp service-kick event. This
    prevents kernel event-ID order from changing FIFO order among rows with the
    same authoritative arrival timestamp. Zero-duration deliveries are legal and
    advance through fixed-point rounds without adding fictitious time.
    """

    ARRIVAL_EVENT = "TRANSPORT_CONTROL_ARRIVAL"
    SERVICE_EVENT = "TRANSPORT_CONTROL_SERVICE"
    SERVICE_COMPLETE_EVENT = "TRANSPORT_CONTROL_SERVICE_COMPLETE"
    DELIVERY_EVENT = "TRANSPORT_CONTROL_DELIVERY"
    PRODUCER = "RS_SIM_TRANSPORT_CONTROL_PLANE"
    CHANNEL_ID = "control-fifo-0"

    def __init__(
        self,
        *,
        kernel: SimulationKernel,
        profile: ControlPlaneProfile,
        delivery_sink: ControlPlaneDeliverySink | None = None,
    ) -> None:
        if not isinstance(kernel, SimulationKernel):
            raise TypeError("kernel must be SimulationKernel")
        if not isinstance(profile, ControlPlaneProfile):
            raise TypeError("profile must be ControlPlaneProfile")
        if control_plane_profile_digest(profile) != profile.profile_digest:
            raise ValueError("control profile digest does not match its semantic fields")
        self.kernel = kernel
        self.profile = profile
        self._delivery_sink = delivery_sink
        self._publish_sequence = 0
        self._server_busy = False
        self._fifo: list[tuple[tuple[int, int], _QueuedRequest]] = []
        self._published_request_digests: set[str] = set()
        self._request_by_arrival_event: dict[str, _QueuedRequest] = {}
        self._service_event_ids: set[str] = set()
        self._service_complete_event_ids: set[str] = set()
        self._pending_delivery_by_event: dict[str, ControlPlaneDelivery] = {}
        self._service_complete_ns_by_request: dict[str, int] = {}
        self._deliveries_by_request: dict[str, ControlPlaneDelivery] = {}
        self._delivery_order: list[str] = []
        self._delivered_request_digests: set[str] = set()
        self._publish_sequence_by_request: dict[str, int] = {}
        self._request_by_digest: dict[str, RowBroadcastRequest] = {}
        self._mechanism_events: list[_ControlMechanismEvent] = []

        self._payload_bytes_by_request: dict[str, int] = {}
        self._arrival_ns_by_request: dict[str, int] = {}
        self._published_request_count = 0
        self._published_payload_bytes = 0
        self._delivered_request_count = 0
        self._delivered_payload_bytes = 0
        self._total_queue_wait_ns = 0
        self._max_queue_wait_ns = 0
        self._total_service_time_ns = 0
        self._max_service_time_ns = 0
        self._total_end_to_end_latency_ns = 0
        self._max_end_to_end_latency_ns = 0
        self._first_arrival_ns: int | None = None
        self._last_delivery_ns: int | None = None
        self._closed = False
        self._disposed = False
        self._kernel_callbacks_disposed = False
        self._final_lifecycle_evidence_digest: str | None = None
        self._closed_lifecycle_diagnostics: dict[str, Any] | None = None
        self._disposed_lifecycle_diagnostics: dict[str, Any] | None = None

        kernel.register_event_handler(self.ARRIVAL_EVENT, self._handle_arrival)
        kernel.register_event_handler(self.SERVICE_EVENT, self._handle_service)
        kernel.register_event_handler(
            self.SERVICE_COMPLETE_EVENT, self._handle_service_complete
        )
        kernel.register_event_handler(self.DELIVERY_EVENT, self._handle_delivery)
        kernel.register_evidence_provider("transport_control_plane", self.evidence)

    def _ensure_open(self) -> None:
        if self._closed or self._disposed:
            raise RuntimeError("Formal transport ControlPlane is closed")

    @staticmethod
    def _window_for_phase(phase_key: PhaseKey) -> WindowKey:
        return WindowKey(
            run_id=phase_key.run_id,
            sample_id=phase_key.sample_id,
            window_index=phase_key.layer_index,
        )

    @staticmethod
    def _phase_sort_key(phase_key: PhaseKey) -> tuple[str, str, int, str]:
        return (
            phase_key.run_id,
            phase_key.sample_id,
            phase_key.layer_index,
            phase_key.phase_kind.value,
        )

    @staticmethod
    def _window_sort_key(window_key: WindowKey) -> tuple[str, str, int]:
        return (window_key.run_id, window_key.sample_id, window_key.window_index)

    def _record_mechanism_event(
        self,
        *,
        phase_key: PhaseKey,
        kind: str,
        payload_bytes: int,
        queue_wait_ns: int = 0,
        service_time_ns: int = 0,
        end_to_end_ns: int = 0,
        at_ns: int,
    ) -> None:
        self._mechanism_events.append(
            _ControlMechanismEvent(
                phase_key=phase_key,
                kind=str(kind),
                payload_bytes=int(payload_bytes),
                queue_wait_ns=int(queue_wait_ns),
                service_time_ns=int(service_time_ns),
                end_to_end_ns=int(end_to_end_ns),
                at_ns=int(at_ns),
            )
        )

    def _scoped_statistics(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        phase_rows: dict[PhaseKey, dict[str, int]] = {}
        window_rows: dict[WindowKey, dict[str, int]] = {}

        def bump(values: dict[str, int], key: str, amount: int = 1) -> None:
            values[key] = int(values.get(key, 0)) + int(amount)

        def update(scope: dict[str, int], event: _ControlMechanismEvent) -> None:
            bump(scope, "event_count")
            if event.kind == "PUBLISHED":
                bump(scope, "published_request_count")
                bump(scope, "published_payload_bytes", event.payload_bytes)
            elif event.kind == "DELIVERED":
                bump(scope, "delivered_request_count")
                bump(scope, "delivered_payload_bytes", event.payload_bytes)
                bump(scope, "total_queue_wait_ns", event.queue_wait_ns)
                bump(scope, "total_service_time_ns", event.service_time_ns)
                bump(scope, "total_end_to_end_latency_ns", event.end_to_end_ns)
                scope["max_queue_wait_ns"] = max(
                    int(scope.get("max_queue_wait_ns", 0)), event.queue_wait_ns
                )
                scope["max_service_time_ns"] = max(
                    int(scope.get("max_service_time_ns", 0)), event.service_time_ns
                )
                scope["max_end_to_end_latency_ns"] = max(
                    int(scope.get("max_end_to_end_latency_ns", 0)),
                    event.end_to_end_ns,
                )
            scope["first_event_ns"] = min(
                int(scope.get("first_event_ns", event.at_ns)), event.at_ns
            )
            scope["last_event_ns"] = max(
                int(scope.get("last_event_ns", event.at_ns)), event.at_ns
            )

        for event in self._mechanism_events:
            update(phase_rows.setdefault(event.phase_key, {}), event)
            update(
                window_rows.setdefault(self._window_for_phase(event.phase_key), {}),
                event,
            )

        phases = tuple(
            (key, tuple(sorted(phase_rows[key].items())))
            for key in sorted(phase_rows, key=self._phase_sort_key)
        )
        windows = tuple(
            (key, tuple(sorted(window_rows[key].items())))
            for key in sorted(window_rows, key=self._window_sort_key)
        )
        return phases, windows

    def fifo_evidence(self) -> tuple[tuple[Any, ...], ...]:
        rows = []
        for request_digest in self._delivery_order:
            request = self._request_by_digest[request_digest]
            delivery = self._deliveries_by_request[request_digest]
            rows.append(
                (
                    self._publish_sequence_by_request[request_digest],
                    request_digest,
                    request.phase_key,
                    request.src_rank,
                    self._arrival_ns_by_request[request_digest],
                    delivery.delivery_start_ns,
                    delivery.delivered_at_ns,
                )
            )
        return tuple(rows)

    def attach_delivery_sink(self, sink: ControlPlaneDeliverySink) -> None:
        self._ensure_open()
        if self._delivery_sink is not None and self._delivery_sink is not sink:
            raise RuntimeError("ControlPlane delivery sink already attached")
        self._delivery_sink = sink

    def publish_row(self, request: RowBroadcastRequest) -> str:
        self._ensure_open()
        if self._delivery_sink is None:
            raise RuntimeError("ControlPlane delivery sink is not attached")
        if not isinstance(request, RowBroadcastRequest):
            raise TypeError("request must be RowBroadcastRequest")
        if request.phase_key.phase_kind is not PhaseKind.DISPATCH:
            raise ValueError("ControlPlane accepts Dispatch row broadcasts only")
        request_digest = stable_digest(request, domain="ROW_BROADCAST_REQUEST")
        if request_digest in self._published_request_digests:
            raise RuntimeError("duplicate ControlPlane request digest")

        sequence = self._publish_sequence
        self._publish_sequence += 1
        if int(request.published_at_ns) < int(self.kernel.now_ns):
            raise ValueError(
                "ControlPlane publication timestamp cannot precede kernel.now_ns"
            )
        arrival_ns = int(request.published_at_ns)
        queued = _QueuedRequest(arrival_ns, sequence, request_digest, request)
        arrival = self.kernel.schedule(
            time_ns=arrival_ns,
            phase_priority=KernelPhase.AUTHORITATIVE_STATE_UPDATES,
            producer=self.PRODUCER,
            event_type=self.ARRIVAL_EVENT,
            ordinal=semantic_ordinal("arrival", sequence, request_digest, arrival_ns),
            subject_id=request_digest,
        )
        self._published_request_digests.add(request_digest)
        self._publish_sequence_by_request[request_digest] = sequence
        self._request_by_digest[request_digest] = request
        self._request_by_arrival_event[arrival.stable_event_id] = queued
        payload_bytes = int(request.descriptor_payload_bytes)
        self._payload_bytes_by_request[request_digest] = payload_bytes
        self._arrival_ns_by_request[request_digest] = arrival_ns
        self._published_request_count += 1
        self._published_payload_bytes += payload_bytes
        self._first_arrival_ns = (
            arrival_ns
            if self._first_arrival_ns is None
            else min(self._first_arrival_ns, arrival_ns)
        )
        self._record_mechanism_event(
            phase_key=request.phase_key,
            kind="PUBLISHED",
            payload_bytes=payload_bytes,
            at_ns=arrival_ns,
        )
        return request_digest

    def _handle_arrival(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        queued = self._request_by_arrival_event.pop(event.stable_event_id, None)
        if queued is None:
            raise RuntimeError(
                f"duplicate or unknown ControlPlane arrival event: {event.stable_event_id}"
            )
        heapq.heappush(
            self._fifo,
            ((queued.arrival_ns, queued.publish_sequence), queued),
        )
        if not self._server_busy:
            self._schedule_service(kernel, at_ns=event.time_ns)
        return ProgressSignal(
            authoritative_state_updates=1,
            notes=("transport_control_arrival",),
        )

    def _schedule_service(self, kernel: SimulationKernel, *, at_ns: int) -> None:
        if self._server_busy or not self._fifo or self._service_event_ids:
            return
        next_item = self._fifo[0][1]
        event = kernel.schedule(
            time_ns=int(at_ns),
            phase_priority=KernelPhase.EXECUTION_STABILIZATION_SUBMIT,
            producer=self.PRODUCER,
            event_type=self.SERVICE_EVENT,
            ordinal=semantic_ordinal(
                "service",
                next_item.arrival_ns,
                next_item.publish_sequence,
                next_item.request_digest,
                at_ns,
            ),
            subject_id=next_item.request_digest,
        )
        self._service_event_ids.add(event.stable_event_id)

    def _handle_service(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        if event.stable_event_id not in self._service_event_ids:
            raise RuntimeError(
                f"duplicate or unknown ControlPlane service event: {event.stable_event_id}"
            )
        self._service_event_ids.remove(event.stable_event_id)
        if self._server_busy or not self._fifo:
            return ProgressSignal(notes=("transport_control_service_noop",))
        _, queued = heapq.heappop(self._fifo)
        start_ns = max(int(event.time_ns), int(queued.arrival_ns))
        serialization_ns = ceil_transfer_time_ns(
            int(queued.request.descriptor_payload_bytes),
            int(self.profile.bandwidth_bytes_per_second),
        )
        service_complete_ns = start_ns + int(serialization_ns)
        delivered_at_ns = service_complete_ns + int(self.profile.fixed_latency_ns)
        delivery = ControlPlaneDelivery(
            request_digest=queued.request_digest,
            phase_key=queued.request.phase_key,
            src_rank=queued.request.src_rank,
            delivery_start_ns=start_ns,
            delivered_at_ns=delivered_at_ns,
            control_channel_id=self.CHANNEL_ID,
        )
        service_complete_event = kernel.schedule(
            time_ns=service_complete_ns,
            phase_priority=KernelPhase.EXECUTION_STABILIZATION_SUBMIT,
            producer=self.PRODUCER,
            event_type=self.SERVICE_COMPLETE_EVENT,
            ordinal=int(queued.publish_sequence),
            subject_id=queued.request_digest,
        )
        self._service_complete_event_ids.add(service_complete_event.stable_event_id)
        self._service_complete_ns_by_request[queued.request_digest] = service_complete_ns
        delivery_event = kernel.schedule(
            time_ns=delivered_at_ns,
            phase_priority=KernelPhase.DESCRIPTOR_OBSERVATION_DELIVERY,
            producer=self.PRODUCER,
            event_type=self.DELIVERY_EVENT,
            ordinal=int(queued.publish_sequence),
            subject_id=queued.request_digest,
        )
        self._server_busy = True
        self._deliveries_by_request[queued.request_digest] = delivery
        self._delivery_order.append(queued.request_digest)
        self._pending_delivery_by_event[delivery_event.stable_event_id] = delivery
        return ProgressSignal(
            authoritative_state_updates=1,
            notes=("transport_control_service_start",),
        )

    def _handle_service_complete(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        if event.stable_event_id not in self._service_complete_event_ids:
            raise RuntimeError(
                f"duplicate or unknown ControlPlane service completion: {event.stable_event_id}"
            )
        self._service_complete_event_ids.remove(event.stable_event_id)
        if not self._server_busy:
            raise RuntimeError("ControlPlane service completed while server was idle")
        self._server_busy = False
        self._schedule_service(kernel, at_ns=event.time_ns)
        return ProgressSignal(
            authoritative_state_updates=1,
            notes=("transport_control_service_complete",),
        )

    def _handle_delivery(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        delivery = self._pending_delivery_by_event.pop(event.stable_event_id, None)
        if delivery is None:
            raise RuntimeError(
                f"duplicate or unknown ControlPlane delivery event: {event.stable_event_id}"
            )
        assert self._delivery_sink is not None
        self._delivery_sink.on_control_plane_delivery(delivery)

        arrival_ns = self._arrival_ns_by_request[delivery.request_digest]
        payload_bytes = self._payload_bytes_by_request[delivery.request_digest]
        queue_wait_ns = int(delivery.delivery_start_ns) - int(arrival_ns)
        service_complete_ns = self._service_complete_ns_by_request[delivery.request_digest]
        service_time_ns = int(service_complete_ns) - int(delivery.delivery_start_ns)
        end_to_end_ns = int(delivery.delivered_at_ns) - int(arrival_ns)
        self._delivered_request_count += 1
        self._delivered_payload_bytes += payload_bytes
        self._total_queue_wait_ns += queue_wait_ns
        self._max_queue_wait_ns = max(self._max_queue_wait_ns, queue_wait_ns)
        self._total_service_time_ns += service_time_ns
        self._max_service_time_ns = max(self._max_service_time_ns, service_time_ns)
        self._total_end_to_end_latency_ns += end_to_end_ns
        self._max_end_to_end_latency_ns = max(
            self._max_end_to_end_latency_ns, end_to_end_ns
        )
        self._last_delivery_ns = (
            int(delivery.delivered_at_ns)
            if self._last_delivery_ns is None
            else max(self._last_delivery_ns, int(delivery.delivered_at_ns))
        )

        self._record_mechanism_event(
            phase_key=delivery.phase_key,
            kind="DELIVERED",
            payload_bytes=payload_bytes,
            queue_wait_ns=queue_wait_ns,
            service_time_ns=service_time_ns,
            end_to_end_ns=end_to_end_ns,
            at_ns=delivery.delivered_at_ns,
        )
        self._delivered_request_digests.add(delivery.request_digest)
        return ProgressSignal(
            authoritative_state_updates=1,
            notes=("transport_control_delivery",),
        )

    def deliveries(self) -> tuple[ControlPlaneDelivery, ...]:
        return tuple(self._deliveries_by_request[key] for key in self._delivery_order)

    def delivery_digest(self) -> str:
        return stable_digest(self.deliveries(), domain="TRANSPORT_CONTROL_DELIVERIES")

    def _delivery_timeline(self) -> tuple[tuple[Any, ...], ...]:
        rows: list[tuple[Any, ...]] = []
        for request_digest in self._delivery_order:
            request = self._request_by_digest[request_digest]
            delivery = self._deliveries_by_request[request_digest]
            arrival_ns = self._arrival_ns_by_request[request_digest]
            payload_bytes = self._payload_bytes_by_request[request_digest]
            rows.append(
                (
                    self._publish_sequence_by_request[request_digest],
                    request_digest,
                    request.phase_key,
                    request.src_rank,
                    payload_bytes,
                    arrival_ns,
                    delivery.delivery_start_ns,
                    self._service_complete_ns_by_request[request_digest],
                    delivery.delivered_at_ns,
                    delivery.delivery_start_ns - arrival_ns,
                    self._service_complete_ns_by_request[request_digest]
                    - delivery.delivery_start_ns,
                    delivery.delivered_at_ns - arrival_ns,
                )
            )
        return tuple(rows)

    def _statistics_reconciliation(self) -> dict[str, Any]:
        delivered_digests = tuple(self._delivery_order)
        delivered_bytes = sum(
            self._payload_bytes_by_request[digest] for digest in delivered_digests
        )
        reconciled = bool(
            len(delivered_digests) == self._delivered_request_count
            and delivered_bytes == self._delivered_payload_bytes
            and set(delivered_digests) == self._delivered_request_digests
            and len(set(delivered_digests)) == len(delivered_digests)
        )
        return {
            "reconciled": reconciled,
            "delivery_record_count": len(delivered_digests),
            "delivery_record_bytes": delivered_bytes,
            "counter_delivered_request_count": self._delivered_request_count,
            "counter_delivered_payload_bytes": self._delivered_payload_bytes,
        }

    def assert_statistics_reconcile(self) -> None:
        reconciliation = self._statistics_reconciliation()
        if not reconciliation["reconciled"]:
            raise RuntimeError(
                f"Formal transport ControlPlane statistics do not reconcile: {reconciliation}"
            )

    def statistics(self) -> dict[str, Any]:
        phase_statistics, window_statistics = self._scoped_statistics()
        if self._first_arrival_ns is None:
            start_ns = end_ns = int(self.kernel.now_ns)
        else:
            start_ns = int(self._first_arrival_ns)
            end_ns = max(
                int(self.kernel.now_ns),
                int(self._last_delivery_ns or start_ns),
            )
        observation_window_ns = max(0, end_ns - start_ns)
        payload: dict[str, Any] = {
            "schema_version": "TRANSPORT_CONTROL_PLANE_STATISTICS",
            "profile_provenance": self.profile.profile_provenance,
            "performance_eligible": self.profile.performance_eligible,
            "observation_start_ns": start_ns,
            "observation_end_ns": end_ns,
            "observation_window_ns": observation_window_ns,
            "published_request_count": self._published_request_count,
            "published_payload_bytes": self._published_payload_bytes,
            "delivered_request_count": self._delivered_request_count,
            "delivered_payload_bytes": self._delivered_payload_bytes,
            "total_queue_wait_ns": self._total_queue_wait_ns,
            "max_queue_wait_ns": self._max_queue_wait_ns,
            "total_service_time_ns": self._total_service_time_ns,
            "max_service_time_ns": self._max_service_time_ns,
            "total_end_to_end_latency_ns": self._total_end_to_end_latency_ns,
            "max_end_to_end_latency_ns": self._max_end_to_end_latency_ns,
            "channel_busy_time_ns": self._total_service_time_ns,
            "channel_utilization_rational": (
                self._total_service_time_ns,
                observation_window_ns,
            ),
            "delivery_timeline": self._delivery_timeline(),
            "statistics_reconciliation": self._statistics_reconciliation(),
            "phase_mechanism_statistics": phase_statistics,
            "window_mechanism_statistics": window_statistics,
            "mechanism_statistics_digest": stable_digest(
                (phase_statistics, window_statistics),
                domain="TRANSPORT_CONTROL_SCOPED_STATISTICS",
            ),
        }
        payload["statistics_digest"] = stable_digest(
            payload, domain="TRANSPORT_CONTROL_PLANE_STATISTICS"
        )
        return payload

    def formal_runtime_metrics(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "TRANSPORT_CONTROL_RUNTIME_METRICS",
            "control_profile_id": self.profile.profile_id,
            "control_profile_digest": self.profile.profile_digest,
            "profile_provenance": self.profile.profile_provenance,
            "performance_eligible": self.profile.performance_eligible,
            "channel_id": self.CHANNEL_ID,
            "statistics": self.statistics(),
            "delivery_timeline": self._delivery_timeline(),
            "delivery_digest": self.delivery_digest(),
            "fifo_evidence_digest": stable_digest(
                self.fifo_evidence(), domain="TRANSPORT_CONTROL_FIFO_EVIDENCE"
            ),
            "terminal_resource_evidence": self.terminal_state(),
            "policy_wait_queue_depth": 0,
        }
        payload["runtime_metrics_digest"] = stable_digest(
            payload, domain="TRANSPORT_CONTROL_RUNTIME_METRICS"
        )
        return payload

    def terminal_state(self) -> dict[str, Any]:
        pending_arrival_count = len(self._request_by_arrival_event)
        queued_count = len(self._fifo)
        pending_service_count = len(self._service_event_ids)
        pending_service_complete_count = len(self._service_complete_event_ids)
        pending_delivery_count = len(self._pending_delivery_by_event)
        terminal = bool(
            not self._server_busy
            and pending_arrival_count == 0
            and queued_count == 0
            and pending_service_count == 0
            and pending_service_complete_count == 0
            and pending_delivery_count == 0
        )
        return {
            "terminal": terminal,
            "closed": self._closed,
            "disposed": self._disposed,
            "server_busy": self._server_busy,
            "pending_arrival_count": pending_arrival_count,
            "queued_request_count": queued_count,
            "pending_service_count": pending_service_count,
            "pending_service_complete_count": pending_service_complete_count,
            "pending_delivery_count": pending_delivery_count,
            "delivered_request_count": self._delivered_request_count,
            "delivered_payload_bytes": self._delivered_payload_bytes,
            "policy_wait_queue_depth": 0,
            "channel_fifo_depth": queued_count,
        }

    def assert_terminal(self) -> None:
        state = self.terminal_state()
        if not state["terminal"]:
            raise RuntimeError(f"Formal transport ControlPlane is not terminal: {state}")

    def _build_lifecycle_diagnostics(self) -> dict[str, Any]:
        state = self.terminal_state()
        return make_process_lifecycle_diagnostics(
            component="TRANSPORT_CONTROL_PLANE",
            closed=self._closed,
            disposed=self._disposed,
            kernel_pending_event_count=int(self.kernel.pending_event_count()),
            live_receipt_count=0,
            live_transfer_or_request_count=(
                int(state["pending_arrival_count"])
                + int(state["queued_request_count"])
                + int(state["pending_service_count"])
                + int(state["pending_service_complete_count"])
                + int(state["pending_delivery_count"])
            ),
            all_resources_free=not bool(state["server_busy"]),
            final_evidence_digest=self._final_lifecycle_evidence_digest,
            kernel_callback_registry_disposed=self._kernel_callbacks_disposed,
        )

    def lifecycle_diagnostics(self) -> dict[str, Any]:
        if self._disposed:
            if self._disposed_lifecycle_diagnostics is None:
                self._disposed_lifecycle_diagnostics = (
                    self._build_lifecycle_diagnostics()
                )
            return dict(self._disposed_lifecycle_diagnostics)
        if self._closed:
            if self._closed_lifecycle_diagnostics is None:
                self._closed_lifecycle_diagnostics = (
                    self._build_lifecycle_diagnostics()
                )
            return dict(self._closed_lifecycle_diagnostics)
        return self._build_lifecycle_diagnostics()

    def close(self) -> dict[str, Any]:
        if not self._closed:
            self.assert_terminal()
            self.assert_statistics_reconcile()
            self._final_lifecycle_evidence_digest = stable_digest(
                (
                    self.deliveries(),
                    self.statistics(),
                    self.terminal_state(),
                ),
                domain="TRANSPORT_CONTROL_PLANE_FINAL_LIFECYCLE_EVIDENCE",
            )
            self._closed = True
            self._closed_lifecycle_diagnostics = (
                self._build_lifecycle_diagnostics()
            )
        return self.lifecycle_diagnostics()

    def _reset_to_disposed_baseline(self) -> None:
        self._publish_sequence = 0
        self._server_busy = False
        self._fifo.clear()
        self._published_request_digests.clear()
        self._request_by_arrival_event.clear()
        self._service_event_ids.clear()
        self._service_complete_event_ids.clear()
        self._pending_delivery_by_event.clear()
        self._service_complete_ns_by_request.clear()
        self._deliveries_by_request.clear()
        self._delivery_order.clear()
        self._delivered_request_digests.clear()
        self._publish_sequence_by_request.clear()
        self._request_by_digest.clear()
        self._mechanism_events.clear()
        self._payload_bytes_by_request.clear()
        self._arrival_ns_by_request.clear()
        self._published_request_count = 0
        self._published_payload_bytes = 0
        self._delivered_request_count = 0
        self._delivered_payload_bytes = 0
        self._total_queue_wait_ns = 0
        self._max_queue_wait_ns = 0
        self._total_service_time_ns = 0
        self._max_service_time_ns = 0
        self._total_end_to_end_latency_ns = 0
        self._max_end_to_end_latency_ns = 0
        self._first_arrival_ns = None
        self._last_delivery_ns = None

    def _mark_kernel_callbacks_disposed(self) -> None:
        self._kernel_callbacks_disposed = True
        self._disposed_lifecycle_diagnostics = None

    def dispose(self, *, dispose_kernel: bool = False) -> dict[str, Any]:
        if self._disposed:
            return self.lifecycle_diagnostics()
        self.close()
        self._delivery_sink = None
        self._reset_to_disposed_baseline()
        if dispose_kernel:
            self.kernel.dispose()
            self._mark_kernel_callbacks_disposed()
        self._disposed = True
        self._disposed_lifecycle_diagnostics = (
            self._build_lifecycle_diagnostics()
        )
        return self.lifecycle_diagnostics()

    def evidence(self) -> dict[str, Any]:
        return {
            "channel_id": self.CHANNEL_ID,
            "server_busy": self._server_busy,
            "pending_arrival_request_digests": tuple(
                sorted(
                    item.request_digest
                    for item in self._request_by_arrival_event.values()
                )
            ),
            "queued_request_digests": tuple(
                item.request_digest for _, item in sorted(self._fifo)
            ),
            "pending_service_event_ids": tuple(sorted(self._service_event_ids)),
            "pending_service_complete_event_ids": tuple(
                sorted(self._service_complete_event_ids)
            ),
            "pending_delivery_request_digests": tuple(
                sorted(
                    delivery.request_digest
                    for delivery in self._pending_delivery_by_event.values()
                )
            ),
            "delivered_request_digests": tuple(sorted(self._delivered_request_digests)),
            "delivery_digest": self.delivery_digest(),
            "fifo_evidence": self.fifo_evidence(),
            "fifo_evidence_digest": stable_digest(
                self.fifo_evidence(), domain="TRANSPORT_CONTROL_FIFO_EVIDENCE"
            ),
            "terminal_state": self.terminal_state(),
            "statistics": self.statistics(),
            "formal_runtime_metrics": self.formal_runtime_metrics(),
        }

    def manifest_fragment(self) -> dict[str, Any]:
        return {
            "control_plane_mode": "ROW_BROADCAST",
            "control_plane_profile_id": self.profile.profile_id,
            "control_plane_profile_digest": self.profile.profile_digest,
            "control_plane_profile_provenance": self.profile.profile_provenance,
            "control_plane_performance_eligible": self.profile.performance_eligible,
            "control_plane_channel_count": self.profile.channel_count,
            "control_plane_fifo": self.profile.fifo,
            "control_plane_non_preemptive": self.profile.non_preemptive,
            "control_plane_shares_data_nic": self.profile.shares_data_nic,
            "control_plane_evidence_schema": "TRANSPORT_CONTROL_PLANE_STATISTICS",
            "control_plane_runtime_metrics_schema": "TRANSPORT_CONTROL_RUNTIME_METRICS",
            "control_plane_terminal_check_supported": True,
            "control_plane_profile_sensitivity_input_supported": True,
            "control_plane_phase_window_statistics_supported": True,
            "control_plane_exact_fifo_evidence_supported": True,
            "control_plane_duplicate_event_fail_closed": True,
            "control_plane_idempotent_terminal_close_supported": True,
            "control_plane_idempotent_terminal_dispose_supported": True,
            "control_plane_owned_threads": 0,
            "control_plane_owned_executors": 0,
            "control_plane_owned_file_handles": 0,
            "control_plane_owned_child_processes": 0,
        }


__all__ = ["FormalControlPlaneTransport"]
