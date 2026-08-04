"""backend receiver, closure and terminal evidence records.

The dataclasses in this module are Backend-owned read-only evidence.  They do
not replace any shared schema and do not create Scheduler catalogue state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rs_sim.backend.core.internal import DestinationMemory


@dataclass(frozen=True, slots=True)
class RemoteCanonicalTaskExpectationInput:
    """One remote expectation that must be represented by canonical tasks.

    backend reports only immutable expectation facts.  Taskization boundaries and
    canonical task creation remain exclusively owned by scheduler.
    """

    edge_stable_key: str
    src_rank: int
    dst_rank: int
    expected_payload_bytes: int
    expectation_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_stable_key": self.edge_stable_key,
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "expected_payload_bytes": self.expected_payload_bytes,
            "expectation_digest": self.expectation_digest,
        }


@dataclass(frozen=True, slots=True)
class PhaseClosureSummary:
    """Authoritative backend phase closure facts used by GLOBAL catalogue sealing.

    ``closure_generation`` is immutable for a phase.  A summary becomes
    available only after the complete descriptor/expectation truth is known.
    It does not claim that scheduler has sealed a catalogue or created a PlanVersion.
    """

    phase_key: Any
    phase_stable_key: str
    phase_kind: str
    closure_generation: int
    finalized_at_ns: int
    expected_descriptor_count: int
    delivered_descriptor_count: int
    expected_expectation_count: int
    expectation_count: int
    zero_expectation_count: int
    local_nonzero_expectation_count: int
    remote_task_expectation_inputs: tuple[RemoteCanonicalTaskExpectationInput, ...]
    all_expectations_digest: str
    remote_task_inputs_digest: str
    closure_digest: str

    @property
    def seal_ready(self) -> bool:
        return (
            self.delivered_descriptor_count == self.expected_descriptor_count
            and self.expectation_count == self.expected_expectation_count
        )

    @property
    def remote_task_expectation_count(self) -> int:
        return len(self.remote_task_expectation_inputs)

    @property
    def remote_task_expected_payload_bytes(self) -> int:
        return sum(
            item.expected_payload_bytes
            for item in self.remote_task_expectation_inputs
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase_key": self.phase_key,
            "phase_stable_key": self.phase_stable_key,
            "phase_kind": self.phase_kind,
            "closure_generation": self.closure_generation,
            "finalized_at_ns": self.finalized_at_ns,
            "expected_descriptor_count": self.expected_descriptor_count,
            "delivered_descriptor_count": self.delivered_descriptor_count,
            "expected_expectation_count": self.expected_expectation_count,
            "expectation_count": self.expectation_count,
            "zero_expectation_count": self.zero_expectation_count,
            "local_nonzero_expectation_count": (
                self.local_nonzero_expectation_count
            ),
            "remote_task_expectation_count": (
                self.remote_task_expectation_count
            ),
            "remote_task_expected_payload_bytes": (
                self.remote_task_expected_payload_bytes
            ),
            "remote_task_expectation_inputs": tuple(
                item.as_dict() for item in self.remote_task_expectation_inputs
            ),
            "all_expectations_digest": self.all_expectations_digest,
            "remote_task_inputs_digest": self.remote_task_inputs_digest,
            "closure_digest": self.closure_digest,
            "seal_ready": self.seal_ready,
        }


@dataclass(frozen=True, slots=True)
class ReceiverMetricsSnapshot:
    peak_staging_bytes_per_rank: Mapping[int, int]
    peak_final_assembly_bytes_per_rank: Mapping[int, int]
    peak_total_receiver_bytes_per_rank: Mapping[int, int]
    dispatch_closure_wait_ns: Mapping[int, int]
    receiver_posting_service_ns: Mapping[int, int]
    receiver_posting_queue_wait_ns: Mapping[int, int]
    receiver_buffer_stall_ns: Mapping[int, int]
    receiver_drain_queue_wait_ns: Mapping[int, int]
    receiver_drain_service_ns: Mapping[int, int]

    def as_dict(self) -> dict[str, dict[int, int]]:
        return {
            "peak_staging_bytes_per_rank": dict(self.peak_staging_bytes_per_rank),
            "peak_final_assembly_bytes_per_rank": dict(
                self.peak_final_assembly_bytes_per_rank
            ),
            "peak_total_receiver_bytes_per_rank": dict(
                self.peak_total_receiver_bytes_per_rank
            ),
            "dispatch_closure_wait_ns": dict(self.dispatch_closure_wait_ns),
            "receiver_posting_service_ns": dict(self.receiver_posting_service_ns),
            "receiver_posting_queue_wait_ns": dict(self.receiver_posting_queue_wait_ns),
            "receiver_buffer_stall_ns": dict(self.receiver_buffer_stall_ns),
            "receiver_drain_queue_wait_ns": dict(self.receiver_drain_queue_wait_ns),
            "receiver_drain_service_ns": dict(self.receiver_drain_service_ns),
        }


@dataclass(frozen=True, slots=True)
class PhaseRankMetricsSnapshot:
    """One phase/rank attribution row from the production Backend state."""

    phase_key: Any
    rank_id: int
    closure_wait_ns: int
    data_wait_ns: int
    receiver_posting_service_ns: int
    receiver_posting_queue_wait_ns: int
    receiver_buffer_stall_ns: int
    receiver_drain_queue_wait_ns: int
    receiver_drain_service_ns: int
    peak_staging_bytes: int
    peak_final_assembly_bytes: int
    peak_total_receiver_bytes: int
    rank_release_at_ns: int | None = None
    phase_close_at_ns: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase_key": self.phase_key,
            "rank_id": self.rank_id,
            "closure_wait_ns": self.closure_wait_ns,
            "data_wait_ns": self.data_wait_ns,
            "receiver_posting_service_ns": self.receiver_posting_service_ns,
            "receiver_posting_queue_wait_ns": self.receiver_posting_queue_wait_ns,
            "receiver_buffer_stall_ns": self.receiver_buffer_stall_ns,
            "receiver_drain_queue_wait_ns": self.receiver_drain_queue_wait_ns,
            "receiver_drain_service_ns": self.receiver_drain_service_ns,
            "peak_staging_bytes": self.peak_staging_bytes,
            "peak_final_assembly_bytes": self.peak_final_assembly_bytes,
            "peak_total_receiver_bytes": self.peak_total_receiver_bytes,
            "rank_release_at_ns": self.rank_release_at_ns,
            "phase_close_at_ns": self.phase_close_at_ns,
        }


@dataclass(frozen=True, slots=True)
class PhaseCausalTimingObservation:
    """Authoritative read-only causal timing for one physical phase.

    All vectors are rank-major and contain integer nanoseconds or ``None``
    while the corresponding causal fact has not yet occurred.  The record is
    evidence only: it does not create planning windows, tasks or plan state.
    """

    phase_key: Any
    phase_stable_key: str
    phase_kind: str
    phase_truth_first_observable_at_ns: int | None
    source_local_path_start_at_ns_by_rank: tuple[int | None, ...]
    source_local_path_complete_at_ns_by_rank: tuple[int | None, ...]
    target_first_executable_truth_observation_at_ns: int | None
    destination_compute_ready_at_ns_by_rank: tuple[int | None, ...]
    rank_release_at_ns_by_rank: tuple[int | None, ...]
    phase_terminal_at_ns: int | None
    causal_timing_digest: str

    @property
    def monotonic(self) -> bool:
        scalars = [
            self.phase_truth_first_observable_at_ns,
            self.target_first_executable_truth_observation_at_ns,
            self.phase_terminal_at_ns,
            *self.source_local_path_start_at_ns_by_rank,
            *self.source_local_path_complete_at_ns_by_rank,
            *self.destination_compute_ready_at_ns_by_rank,
            *self.rank_release_at_ns_by_rank,
        ]
        if any(value is not None and (not isinstance(value, int) or value < 0) for value in scalars):
            return False
        for start, finish in zip(
            self.source_local_path_start_at_ns_by_rank,
            self.source_local_path_complete_at_ns_by_rank,
            strict=True,
        ):
            if start is not None and finish is not None and start > finish:
                return False
        for ready, release in zip(
            self.destination_compute_ready_at_ns_by_rank,
            self.rank_release_at_ns_by_rank,
            strict=True,
        ):
            if ready is not None and release is not None and ready > release:
                return False
        truth = self.phase_truth_first_observable_at_ns
        executable = self.target_first_executable_truth_observation_at_ns
        terminal = self.phase_terminal_at_ns
        if truth is not None and executable is not None and truth > executable:
            return False
        if executable is not None and terminal is not None and executable > terminal:
            return False
        if terminal is not None and any(
            value is not None and value > terminal
            for value in (
                *self.source_local_path_complete_at_ns_by_rank,
                *self.destination_compute_ready_at_ns_by_rank,
                *self.rank_release_at_ns_by_rank,
            )
        ):
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase_key": self.phase_key,
            "phase_stable_key": self.phase_stable_key,
            "phase_kind": self.phase_kind,
            "phase_truth_first_observable_at_ns": self.phase_truth_first_observable_at_ns,
            "source_local_path_start_at_ns_by_rank": self.source_local_path_start_at_ns_by_rank,
            "source_local_path_complete_at_ns_by_rank": self.source_local_path_complete_at_ns_by_rank,
            "target_first_executable_truth_observation_at_ns": (
                self.target_first_executable_truth_observation_at_ns
            ),
            "destination_compute_ready_at_ns_by_rank": self.destination_compute_ready_at_ns_by_rank,
            "rank_release_at_ns_by_rank": self.rank_release_at_ns_by_rank,
            "phase_terminal_at_ns": self.phase_terminal_at_ns,
            "causal_timing_digest": self.causal_timing_digest,
            "monotonic": self.monotonic,
        }


@dataclass(frozen=True, slots=True)
class FuturePrepareTriggerCandidate:
    """A real Backend-progress trigger that may start earlier-window Future work."""

    source_phase_key: Any
    source_phase_stable_key: str
    target_phase_key: Any
    target_phase_stable_key: str
    available_at_ns: int
    source_closure_digest: str
    target_first_truth_observation_at_ns: int | None
    template_ready_deadline_at_ns: int | None
    trigger_digest: str

    @property
    def precedes_target_truth(self) -> bool | None:
        if self.target_first_truth_observation_at_ns is None:
            return None
        return self.available_at_ns < self.target_first_truth_observation_at_ns

    @property
    def target_truth_margin_ns(self) -> int | None:
        if self.target_first_truth_observation_at_ns is None:
            return None
        return self.target_first_truth_observation_at_ns - self.available_at_ns

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_phase_key": self.source_phase_key,
            "source_phase_stable_key": self.source_phase_stable_key,
            "target_phase_key": self.target_phase_key,
            "target_phase_stable_key": self.target_phase_stable_key,
            "available_at_ns": self.available_at_ns,
            "source_closure_digest": self.source_closure_digest,
            "target_first_truth_observation_at_ns": self.target_first_truth_observation_at_ns,
            "template_ready_deadline_at_ns": self.template_ready_deadline_at_ns,
            "precedes_target_truth": self.precedes_target_truth,
            "target_truth_margin_ns": self.target_truth_margin_ns,
            "trigger_digest": self.trigger_digest,
        }


@dataclass(frozen=True, slots=True)
class FutureOverlapDeadlines:
    """Grounded deadlines for Prediction, Control and Binding service hiding.

    Prediction is bounded by the first actual target source-local-path start;
    Control by the first actual target source-local-path completion; Binding by
    the first actual target destination compute-ready time.  No constant slack
    is introduced by the backend.
    """

    source_phase_key: Any
    target_phase_key: Any
    trigger_available_at_ns: int
    prediction_hide_deadline_at_ns: int | None
    control_hide_deadline_at_ns: int | None
    binding_hide_deadline_at_ns: int | None
    target_source_local_path_start_at_ns_by_rank: tuple[int | None, ...]
    target_source_local_path_complete_at_ns_by_rank: tuple[int | None, ...]
    target_destination_compute_ready_at_ns_by_rank: tuple[int | None, ...]
    deadlines_digest: str

    @property
    def fully_observed(self) -> bool:
        return all(
            value is not None
            for value in (
                self.prediction_hide_deadline_at_ns,
                self.control_hide_deadline_at_ns,
                self.binding_hide_deadline_at_ns,
            )
        )

    @property
    def monotonic(self) -> bool:
        if not self.fully_observed:
            return False
        assert self.prediction_hide_deadline_at_ns is not None
        assert self.control_hide_deadline_at_ns is not None
        assert self.binding_hide_deadline_at_ns is not None
        return (
            self.trigger_available_at_ns
            <= self.prediction_hide_deadline_at_ns
            <= self.control_hide_deadline_at_ns
            <= self.binding_hide_deadline_at_ns
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_phase_key": self.source_phase_key,
            "target_phase_key": self.target_phase_key,
            "trigger_available_at_ns": self.trigger_available_at_ns,
            "prediction_hide_deadline_at_ns": self.prediction_hide_deadline_at_ns,
            "control_hide_deadline_at_ns": self.control_hide_deadline_at_ns,
            "binding_hide_deadline_at_ns": self.binding_hide_deadline_at_ns,
            "target_source_local_path_start_at_ns_by_rank": (
                self.target_source_local_path_start_at_ns_by_rank
            ),
            "target_source_local_path_complete_at_ns_by_rank": (
                self.target_source_local_path_complete_at_ns_by_rank
            ),
            "target_destination_compute_ready_at_ns_by_rank": (
                self.target_destination_compute_ready_at_ns_by_rank
            ),
            "fully_observed": self.fully_observed,
            "monotonic": self.monotonic,
            "deadlines_digest": self.deadlines_digest,
        }


@dataclass(frozen=True, slots=True)
class WindowTerminalEvidence:
    """Phase-scoped terminal and resource evidence for an anchor/window slice."""

    phase_keys: tuple[Any, ...]
    phase_stable_keys: tuple[str, ...]
    referenced_phase_count: int
    unique_phase_count: int
    duplicate_phase_reference_count: int
    total_expected_payload_bytes: int
    total_assembled_payload_bytes: int
    receiver_posting_service_ns_by_rank: tuple[int, ...]
    receiver_posting_queue_wait_ns_by_rank: tuple[int, ...]
    receiver_buffer_stall_ns_by_rank: tuple[int, ...]
    receiver_drain_queue_wait_ns_by_rank: tuple[int, ...]
    receiver_drain_service_ns_by_rank: tuple[int, ...]
    peak_staging_bytes_by_rank: tuple[int, ...]
    peak_final_assembly_bytes_by_rank: tuple[int, ...]
    peak_total_receiver_bytes_by_rank: tuple[int, ...]
    phase_rank_release_at_ns: tuple[tuple[int | None, ...], ...]
    phase_terminal_at_ns: tuple[int | None, ...]
    window_terminal_at_ns: int | None
    no_residual_receiver_jobs: bool
    no_residual_permits: bool
    selected_phase_receiver_memory_zero: bool
    global_receiver_memory_zero: bool
    ranks_done: bool
    pending_backend_stabilization_count: int
    closed: bool
    backend_disposed: bool
    evidence_digest: str

    @property
    def bytes_reconciled(self) -> bool:
        return self.total_expected_payload_bytes == self.total_assembled_payload_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase_keys": self.phase_keys,
            "phase_stable_keys": self.phase_stable_keys,
            "referenced_phase_count": self.referenced_phase_count,
            "unique_phase_count": self.unique_phase_count,
            "duplicate_phase_reference_count": self.duplicate_phase_reference_count,
            "total_expected_payload_bytes": self.total_expected_payload_bytes,
            "total_assembled_payload_bytes": self.total_assembled_payload_bytes,
            "bytes_reconciled": self.bytes_reconciled,
            "receiver_posting_service_ns_by_rank": self.receiver_posting_service_ns_by_rank,
            "receiver_posting_queue_wait_ns_by_rank": self.receiver_posting_queue_wait_ns_by_rank,
            "receiver_buffer_stall_ns_by_rank": self.receiver_buffer_stall_ns_by_rank,
            "receiver_drain_queue_wait_ns_by_rank": self.receiver_drain_queue_wait_ns_by_rank,
            "receiver_drain_service_ns_by_rank": self.receiver_drain_service_ns_by_rank,
            "peak_staging_bytes_by_rank": self.peak_staging_bytes_by_rank,
            "peak_final_assembly_bytes_by_rank": self.peak_final_assembly_bytes_by_rank,
            "peak_total_receiver_bytes_by_rank": self.peak_total_receiver_bytes_by_rank,
            "phase_rank_release_at_ns": self.phase_rank_release_at_ns,
            "phase_terminal_at_ns": self.phase_terminal_at_ns,
            "window_terminal_at_ns": self.window_terminal_at_ns,
            "no_residual_receiver_jobs": self.no_residual_receiver_jobs,
            "no_residual_permits": self.no_residual_permits,
            "selected_phase_receiver_memory_zero": self.selected_phase_receiver_memory_zero,
            "global_receiver_memory_zero": self.global_receiver_memory_zero,
            "ranks_done": self.ranks_done,
            "pending_backend_stabilization_count": self.pending_backend_stabilization_count,
            "closed": self.closed,
            "backend_disposed": self.backend_disposed,
            "evidence_digest": self.evidence_digest,
        }


def snapshot_memory(
    memory_by_rank: Mapping[int, DestinationMemory],
    *,
    dispatch_closure_wait_ns: Mapping[int, int] | None = None,
) -> ReceiverMetricsSnapshot:
    ordered = sorted(memory_by_rank)
    closure = dispatch_closure_wait_ns or {}
    return ReceiverMetricsSnapshot(
        peak_staging_bytes_per_rank={
            rank: memory_by_rank[rank].peak_staging_bytes for rank in ordered
        },
        peak_final_assembly_bytes_per_rank={
            rank: memory_by_rank[rank].peak_final_assembly_bytes for rank in ordered
        },
        peak_total_receiver_bytes_per_rank={
            rank: memory_by_rank[rank].peak_total_receiver_bytes for rank in ordered
        },
        dispatch_closure_wait_ns={rank: int(closure.get(rank, 0)) for rank in ordered},
        receiver_posting_service_ns={
            rank: memory_by_rank[rank].receiver_posting_service_ns for rank in ordered
        },
        receiver_posting_queue_wait_ns={
            rank: memory_by_rank[rank].receiver_posting_queue_wait_ns for rank in ordered
        },
        receiver_buffer_stall_ns={
            rank: memory_by_rank[rank].receiver_buffer_stall_ns for rank in ordered
        },
        receiver_drain_queue_wait_ns={
            rank: memory_by_rank[rank].receiver_drain_queue_wait_ns for rank in ordered
        },
        receiver_drain_service_ns={
            rank: memory_by_rank[rank].receiver_drain_service_ns for rank in ordered
        },
    )
