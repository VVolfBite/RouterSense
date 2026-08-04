"""backend adapter protocols.

shared-schema owns all shared immutable schemas.  backend therefore consumes foreign objects
through attribute adapters and emits requests through narrow ports.  No type in
this module is a replacement for an shared dataclass.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class KernelPort(Protocol):
    """Thin adapter implemented by the shared-schema SimulationKernel integration."""

    def schedule_backend_event(
        self,
        *,
        time_ns: int,
        phase_priority: int,
        stable_event_id: str,
        event_kind: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Schedule an backend event without advancing time."""


@runtime_checkable
class BackendObserverPort(Protocol):
    """Receives causally visible Backend observations and ownership events."""

    def emit(self, *, kind: str, at_ns: int, payload: Mapping[str, Any]) -> None:
        """Publish a stable observation to Scheduler/Control/Integration."""


@runtime_checkable
class ExactRowPublisherPort(Protocol):
    """Publishes one Backend-owned exact Dispatch row through transport ControlPlane."""

    def publish_exact_descriptor(self, descriptor: Any) -> str:
        """Return the canonical RowBroadcastRequest digest."""


@runtime_checkable
class BackendSealReadinessPort(Protocol):
    """Read-only backend closure facts consumed by GLOBAL catalogue finalization."""

    def phase_closure_summary(self, *, phase_key: Any) -> Any | None:
        """Return the immutable closure summary, or ``None`` before closure."""

    def require_phase_closure_summary(self, *, phase_key: Any) -> Any:
        """Return closure facts or fail closed when the phase is not ready."""


@runtime_checkable
class BackendCausalTimingPort(Protocol):
    """Read-only causal timing and terminal evidence for rolling windows."""

    def phase_causal_timing_observation(self, *, phase_key: Any) -> Any:
        """Return stable phase timing facts without creating planning state."""

    def future_prepare_trigger_candidate(
        self, *, source_phase_key: Any, target_phase_key: Any
    ) -> Any | None:
        """Return a closure-grounded earlier-window trigger, or ``None``."""

    def future_overlap_deadlines(
        self, *, source_phase_key: Any, target_phase_key: Any
    ) -> Any | None:
        """Return actual compute/local-path deadlines, or ``None``."""

    def window_terminal_evidence(
        self, *, phase_keys: Sequence[Any], require_ranks_done: bool = False
    ) -> Any:
        """Return phase-scoped wait, memory, release and disposal evidence."""


@runtime_checkable
class SharedObjectAdapter(Protocol):
    """Extracts frozen shared-schema fields from foreign shared objects."""

    def get(self, obj: Any, field: str) -> Any:
        """Return one named field or raise a deterministic contract error."""

    def stable_key(self, value: Any) -> str:
        """Return a deterministic sortable/serializable key."""


@runtime_checkable
class ExpectationFactory(Protocol):
    """Creates the shared-schema ReceiveExpectation object."""

    def create_receive_expectation(
        self,
        *,
        edge_key: Any,
        phase_key: Any,
        src_rank: int,
        dst_rank: int,
        total_expected_payload_bytes: int,
        expectation_digest: str,
        origin: str,
        created_at_ns: int,
        zero_edge: bool,
        descriptor_digest_or_none: str | None,
    ) -> Any:
        """Return an immutable shared-schema ReceiveExpectation."""


@runtime_checkable
class PermitFactory(Protocol):
    """Creates the shared-schema task-level ReceivePermit object."""

    def create_receive_permit(
        self,
        *,
        permit_id: str,
        task_id: Any,
        edge_key: Any,
        chunk_index: int,
        byte_offset: int,
        task_bytes: int,
        credit_reservation_id: str,
        expectation_digest: str,
        descriptor_digest_or_none: str | None,
        posted_at_ns: int,
    ) -> Any:
        """Return an immutable ReceivePermit."""


@runtime_checkable
class CostModel(Protocol):
    """Backend receiver service/drain costs in integer nanoseconds."""

    def receiver_service_cost_ns(self, task_bytes: int) -> int:
        """Posting service duration."""

    def receiver_drain_cost_ns(self, task_bytes: int) -> int:
        """Drain service duration."""


@runtime_checkable
class EdgeKeyFactory(Protocol):
    """Builds the shared-schema EdgeKey without backend defining its schema."""

    def make_edge_key(self, *, phase_key: Any, src_rank: int, dst_rank: int) -> Any:
        """Return the immutable shared EdgeKey."""


@runtime_checkable
class PhaseSemantics(Protocol):
    """Maps an PhaseKey to stable backend semantics."""

    def phase_kind(self, phase_key: Any) -> str:
        """Return exactly 'DISPATCH' or 'COMBINE'."""

    def phase_sort_key(self, phase_key: Any) -> str:
        """Return a stable key used in Receiver FIFO ordering."""


@runtime_checkable
class TaskCataloguePort(Protocol):
    """Optional bulk source of immutable canonical tasks."""

    def tasks_for_edge(self, edge_key: Any) -> Sequence[Any]:
        """Return the complete canonical task catalogue for an edge."""
