"""RS-SIM backend SimulationBackend.

The backend owns receiver expectations, closure, local-path progression, final
assembly lifetime and rank release.  It never taskizes, submits transport work,
or advances simulation time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rs_sim.contracts.factories import make_exact_dispatch_row_truth, make_exact_row_descriptor
from rs_sim.contracts.paper_defaults import (
    PAPER_P0_P1_COMPUTE_END_BARRIER, PAPER_RELEASE_MODE,
)
from rs_sim.contracts.schema import ExactDispatchRowTruth, ExactRowDescriptor
from typing import Any

from rs_sim.backend.core.errors import (
    BackendContractError,
    DuplicateRegistrationError,
    IllegalTransitionError,
    UnknownObjectError,
)
from rs_sim.backend.core.internal import (
    CombineDestinationState,
    DispatchComputeSpec,
    DispatchDestinationState,
    LocalPathSpec,
    RankState,
)
from rs_sim.backend.core.ports import (
    BackendObserverPort,
    EdgeKeyFactory,
    ExactRowPublisherPort,
    ExpectationFactory,
    KernelPort,
    PhaseSemantics,
    SharedObjectAdapter,
)
from rs_sim.backend.observability.metrics import (
    FutureOverlapDeadlines,
    FuturePrepareTriggerCandidate,
    PhaseCausalTimingObservation,
    PhaseClosureSummary,
    PhaseRankMetricsSnapshot,
    ReceiverMetricsSnapshot,
    RemoteCanonicalTaskExpectationInput,
    WindowTerminalEvidence,
    snapshot_memory,
)
from rs_sim.backend.resources.receiver import BACKEND_PHASE_PRIORITY, ReceiverService
from rs_sim.backend.resources.rank_actor import RankActor
from rs_sim.backend.core.util import (
    require_nonnegative_int,
    require_time_ns,
    stable_digest,
    stable_semantic_event_id,
)

_EVENT_POST_COMBINE_COMPLETE = "BACKEND_POST_COMBINE_LOCAL_PATH_COMPLETE"
_EVENT_LOCAL_PATH_COMPLETE = "BACKEND_LOCAL_PATH_COMPLETE"
_EVENT_DISPATCH_POSTPROCESS_COMPLETE = "BACKEND_DISPATCH_POSTPROCESS_COMPLETE"
_EVENT_COMBINE_SOURCE_READY = "BACKEND_COMBINE_SOURCE_READY"
_EVENT_P0_P1_LOCAL_COMPUTE_COMPLETE = "BACKEND_P0_P1_LOCAL_COMPUTE_COMPLETE"
_EVENT_BACKEND_STABILIZE = "BACKEND_STABILIZE"
_OBSERVATION_PHASE_CLOSURE_READY = "PHASE_CLOSURE_SUMMARY_READY"


class SimulationBackend:
    """Receiver-decoupled multi-rank inference lifecycle controller."""

    def __init__(
        self,
        *,
        world_size: int,
        kernel: KernelPort,
        observer: BackendObserverPort,
        adapter: SharedObjectAdapter,
        phase_semantics: PhaseSemantics,
        edge_key_factory: EdgeKeyFactory,
        expectation_factory: ExpectationFactory,
        receiver: ReceiverService,
        release_mode: str = PAPER_RELEASE_MODE,
        p0_p1_compute_end_barrier: bool = PAPER_P0_P1_COMPUTE_END_BARRIER,
        node_id_by_rank: Mapping[int, int] | None = None,
        exact_row_publisher: ExactRowPublisherPort | None = None,
    ) -> None:
        if not isinstance(world_size, int) or world_size <= 0:
            raise BackendContractError("world_size must be a positive int")
        if receiver.world_size != world_size:
            raise BackendContractError("ReceiverService world_size mismatch")
        self.world_size = world_size
        self.kernel = kernel
        self.observer = observer
        self.adapter = adapter
        self.phase_semantics = phase_semantics
        self.edge_key_factory = edge_key_factory
        self.expectation_factory = expectation_factory
        self.receiver = receiver
        self._exact_row_publisher = exact_row_publisher
        normalized_release_mode = str(release_mode).upper()
        if normalized_release_mode not in {"RANK_LOCAL", "PHASE_BARRIER"}:
            raise BackendContractError(
                "release_mode must be RANK_LOCAL or PHASE_BARRIER"
            )
        self.release_mode = normalized_release_mode
        if not isinstance(p0_p1_compute_end_barrier, bool):
            raise BackendContractError("p0_p1_compute_end_barrier must be bool")
        self.p0_p1_compute_end_barrier = p0_p1_compute_end_barrier
        if node_id_by_rank is not None and set(node_id_by_rank) != set(range(world_size)):
            raise BackendContractError("node_id_by_rank must cover every logical rank")

        self._dispatch_destinations: dict[
            tuple[str, int], DispatchDestinationState
        ] = {}
        self._combine_destinations: dict[
            tuple[str, int], CombineDestinationState
        ] = {}
        self._local_path_specs: dict[tuple[str, int], LocalPathSpec] = {}
        self._dispatch_compute_specs: dict[
            tuple[str, int], DispatchComputeSpec
        ] = {}
        self._stabilize_destination_signature: tuple[int, int, int] | None = None
        self._stabilize_destinations_cache: tuple[tuple[str, int, Any], ...] = ()
        self._rank_actors: dict[int, RankActor] = {
            rank: RankActor(
                rank_id=rank,
                node_id=(node_id_by_rank[rank] if node_id_by_rank is not None else None),
            )
            for rank in range(world_size)
        }
        self._combine_barrier_opened: set[str] = set()
        self._dispatch_barrier_released: set[str] = set()
        self._p0_p1_compute_complete_at: dict[tuple[str, int], int] = {}
        self._p0_p1_compute_barrier_opened: set[str] = set()
        self._p0_p1_compute_barrier_release_at: dict[str, int] = {}
        self._rank_release_at: dict[tuple[str, int], int] = {}
        self._source_descriptor_ready_at: dict[tuple[str, int], int] = {}
        self._dispatch_row_truth: dict[
            tuple[str, int], ExactDispatchRowTruth
        ] = {}
        self._exact_descriptor_by_source: dict[
            tuple[str, int], ExactRowDescriptor
        ] = {}
        self._control_request_digest_by_source: dict[tuple[str, int], str] = {}
        self._pending_stabilization_times: set[int] = set()
        self._stabilization_ordinal_by_time: dict[int, int] = {}
        self._dispatch_closure_wait_by_rank: dict[int, int] = {
            rank: 0 for rank in range(world_size)
        }
        self._terminal_combine_ranks: set[tuple[str, int]] = set()
        self._phase_closure_summary_by_phase: dict[
            str, PhaseClosureSummary
        ] = {}
        self._source_local_path_start_at: dict[tuple[str, int], int] = {}
        self._source_local_path_complete_at: dict[tuple[str, int], int] = {}
        self._source_local_path_origin: dict[tuple[str, int], str] = {}

    def attach_exact_row_publisher(self, publisher: ExactRowPublisherPort) -> None:
        """Attach the formal backend-to-transport ControlPlane publisher before row creation.

        Wave B owns runtime wiring.  backend only freezes the one-way publication
        port and rejects late attachment after any exact descriptor exists.
        """
        if self._exact_descriptor_by_source:
            raise IllegalTransitionError(
                "exact row publisher cannot be attached after descriptor creation"
            )
        if self._exact_row_publisher is not None and self._exact_row_publisher is not publisher:
            raise DuplicateRegistrationError("exact row publisher already attached")
        self._exact_row_publisher = publisher

    # ------------------------------------------------------------------
    # Authoritative phase closure / GLOBAL seal-readiness facts
    # ------------------------------------------------------------------
    def phase_closure_summary(
        self, *, phase_key: Any
    ) -> PhaseClosureSummary | None:
        """Return immutable backend closure facts, or ``None`` before closure.

        This is a read-only seal-readiness port.  It does not seal the scheduler
        catalogue, create a plan, or assert that a GLOBAL plan exists.
        """

        return self._phase_closure_summary_by_phase.get(
            self.phase_semantics.phase_sort_key(phase_key)
        )

    def require_phase_closure_summary(
        self, *, phase_key: Any
    ) -> PhaseClosureSummary:
        summary = self.phase_closure_summary(phase_key=phase_key)
        if summary is None:
            raise IllegalTransitionError(
                "phase closure facts are not yet complete"
            )
        return summary

    def closure_summaries(self) -> tuple[PhaseClosureSummary, ...]:
        """Return all finalized summaries in deterministic phase order."""

        return tuple(
            self._phase_closure_summary_by_phase[key]
            for key in sorted(self._phase_closure_summary_by_phase)
        )

    def _maybe_finalize_phase_closure(
        self, *, phase_key: Any
    ) -> PhaseClosureSummary | None:
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        phase_kind = self.phase_semantics.phase_kind(phase_key)
        edges = sorted(
            (
                edge
                for edge in self.receiver.edges_by_key.values()
                if edge.phase_stable_key == phase_stable_key
            ),
            key=lambda edge: (
                edge.src_rank,
                edge.dst_rank,
                edge.edge_stable_key,
            ),
        )
        expected_expectation_count = self.world_size * self.world_size
        if len(edges) != expected_expectation_count:
            return None
        edge_pairs = {(edge.src_rank, edge.dst_rank) for edge in edges}
        if len(edge_pairs) != expected_expectation_count:
            raise IllegalTransitionError(
                "phase expectation set contains duplicate source/destination pairs"
            )

        expected_descriptor_count = self.world_size if phase_kind == "DISPATCH" else 0
        delivered_descriptor_count = 0
        closure_times: list[int] = []
        if phase_kind == "DISPATCH":
            descriptor_digest_by_source: dict[int, str] = {}
            for edge in edges:
                if not edge.descriptor_digest_or_none:
                    raise IllegalTransitionError(
                        "Dispatch closure contains an expectation without descriptor digest"
                    )
                existing = descriptor_digest_by_source.get(edge.src_rank)
                if existing is not None and existing != edge.descriptor_digest_or_none:
                    raise IllegalTransitionError(
                        "one Dispatch source produced conflicting descriptor digests"
                    )
                descriptor_digest_by_source[edge.src_rank] = str(
                    edge.descriptor_digest_or_none
                )
            delivered_descriptor_count = len(descriptor_digest_by_source)
            if delivered_descriptor_count != expected_descriptor_count:
                return None
            for dst_rank in range(self.world_size):
                state = self._dispatch_destinations.get(
                    (phase_stable_key, dst_rank)
                )
                if (
                    state is None
                    or len(state.descriptor_sources_delivered) != self.world_size
                    or state.descriptor_closure_at_ns is None
                ):
                    return None
                closure_times.append(int(state.descriptor_closure_at_ns))
        else:
            for dst_rank in range(self.world_size):
                closure_at = self.receiver.expectation_closure_at_ns(
                    phase_key=phase_key, dst_rank=dst_rank
                )
                if closure_at is None:
                    return None
                closure_times.append(int(closure_at))

        expectation_payloads = tuple(
            (
                edge.edge_stable_key,
                edge.src_rank,
                edge.dst_rank,
                edge.expected_bytes,
                edge.expectation_digest,
                edge.zero_edge,
                edge.descriptor_digest_or_none,
            )
            for edge in edges
        )
        remote_inputs = tuple(
            RemoteCanonicalTaskExpectationInput(
                edge_stable_key=edge.edge_stable_key,
                src_rank=edge.src_rank,
                dst_rank=edge.dst_rank,
                expected_payload_bytes=edge.expected_bytes,
                expectation_digest=edge.expectation_digest,
            )
            for edge in edges
            if edge.src_rank != edge.dst_rank and not edge.zero_edge
        )
        remote_payloads = tuple(
            (
                item.edge_stable_key,
                item.src_rank,
                item.dst_rank,
                item.expected_payload_bytes,
                item.expectation_digest,
            )
            for item in remote_inputs
        )
        all_expectations_digest = stable_digest(
            expectation_payloads, prefix="phase-expectations"
        )
        remote_task_inputs_digest = stable_digest(
            remote_payloads, prefix="remote-task-inputs"
        )
        zero_expectation_count = sum(1 for edge in edges if edge.zero_edge)
        local_nonzero_expectation_count = sum(
            1
            for edge in edges
            if edge.src_rank == edge.dst_rank and not edge.zero_edge
        )
        finalized_at_ns = max(closure_times)
        closure_digest = stable_digest(
            [
                phase_stable_key,
                phase_kind,
                1,
                finalized_at_ns,
                expected_descriptor_count,
                delivered_descriptor_count,
                expected_expectation_count,
                len(edges),
                zero_expectation_count,
                local_nonzero_expectation_count,
                all_expectations_digest,
                remote_task_inputs_digest,
            ],
            prefix="phase-closure",
        )
        candidate = PhaseClosureSummary(
            phase_key=phase_key,
            phase_stable_key=phase_stable_key,
            phase_kind=phase_kind,
            closure_generation=1,
            finalized_at_ns=finalized_at_ns,
            expected_descriptor_count=expected_descriptor_count,
            delivered_descriptor_count=delivered_descriptor_count,
            expected_expectation_count=expected_expectation_count,
            expectation_count=len(edges),
            zero_expectation_count=zero_expectation_count,
            local_nonzero_expectation_count=local_nonzero_expectation_count,
            remote_task_expectation_inputs=remote_inputs,
            all_expectations_digest=all_expectations_digest,
            remote_task_inputs_digest=remote_task_inputs_digest,
            closure_digest=closure_digest,
        )
        existing = self._phase_closure_summary_by_phase.get(phase_stable_key)
        if existing is not None:
            if existing.closure_digest != candidate.closure_digest:
                raise IllegalTransitionError(
                    "same phase attempted a conflicting closure generation"
                )
            return existing

        self.receiver.finalize_expectation_closure(
            phase_key=phase_key, closure_digest=closure_digest
        )
        self._phase_closure_summary_by_phase[phase_stable_key] = candidate
        self.observer.emit(
            kind=_OBSERVATION_PHASE_CLOSURE_READY,
            at_ns=finalized_at_ns,
            payload={
                "phase_key": phase_key,
                "phase_kind": phase_kind,
                "closure_generation": candidate.closure_generation,
                "closure_digest": candidate.closure_digest,
                "expected_descriptor_count": expected_descriptor_count,
                "expected_expectation_count": expected_expectation_count,
                "remote_task_expectation_count": (
                    candidate.remote_task_expectation_count
                ),
                "remote_task_expected_payload_bytes": (
                    candidate.remote_task_expected_payload_bytes
                ),
                "summary": candidate,
            },
        )
        return candidate

    # ------------------------------------------------------------------
    # Authoritative causal timing registration / read-only evidence
    # ------------------------------------------------------------------
    def _record_source_local_path_start(
        self,
        *,
        phase_key: Any,
        rank_id: int,
        at_ns: int,
        origin: str,
    ) -> None:
        self._validate_rank(rank_id)
        at_ns = require_time_ns(at_ns, field="source_local_path_start.at_ns")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        key = (phase_stable_key, rank_id)
        existing = self._source_local_path_start_at.get(key)
        if existing is not None:
            if existing != at_ns:
                raise DuplicateRegistrationError(
                    "source/local path start time changed after observation"
                )
            return
        complete = self._source_local_path_complete_at.get(key)
        if complete is not None and at_ns > complete:
            raise IllegalTransitionError(
                "source/local path start cannot follow its completion"
            )
        self._source_local_path_start_at[key] = at_ns
        self._source_local_path_origin[key] = str(origin)

    def _record_source_local_path_complete(
        self,
        *,
        phase_key: Any,
        rank_id: int,
        at_ns: int,
        origin: str,
    ) -> None:
        self._validate_rank(rank_id)
        at_ns = require_time_ns(at_ns, field="source_local_path_complete.at_ns")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        key = (phase_stable_key, rank_id)
        existing = self._source_local_path_complete_at.get(key)
        if existing is not None:
            if existing != at_ns:
                raise DuplicateRegistrationError(
                    "source/local path completion time changed after observation"
                )
            return
        start = self._source_local_path_start_at.get(key)
        if start is None:
            # Contract/unit fixtures may inject a ready boundary directly.  A
            # zero-width interval preserves causality without inventing slack;
            # production Trace registration provides the real start boundary.
            self._record_source_local_path_start(
                phase_key=phase_key,
                rank_id=rank_id,
                at_ns=at_ns,
                origin=f"{origin}:DIRECT_READY_BOUNDARY",
            )
            start = at_ns
        if start > at_ns:
            raise IllegalTransitionError(
                "source/local path completion cannot precede its start"
            )
        self._source_local_path_complete_at[key] = at_ns

    def register_bootstrap_source_local_path_start(
        self, *, phase_key: Any, rank_id: int, at_ns: int
    ) -> None:
        """Register the actual Bootstrap-P0 local path start from Trace truth."""

        if self.phase_semantics.phase_kind(phase_key) != "DISPATCH":
            raise BackendContractError(
                "bootstrap source/local path must target a Dispatch phase"
            )
        self._record_source_local_path_start(
            phase_key=phase_key,
            rank_id=rank_id,
            at_ns=at_ns,
            origin="BOOTSTRAP_P0_LOCAL_PATH",
        )

    def phase_causal_timing_observation(
        self, *, phase_key: Any
    ) -> PhaseCausalTimingObservation:
        """Return deterministic integer-ns causal timing for one phase.

        ``phase_truth_first_observable_at_ns`` is the first exact expectation
        made visible to Backend.  ``target_first_executable_truth`` additionally
        requires the corresponding source payload to be real and ready.  The
        two are intentionally distinct for pre-registered Combine truth.
        """

        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        phase_kind = self.phase_semantics.phase_kind(phase_key)
        edges = tuple(
            sorted(
                (
                    edge
                    for edge in self.receiver.edges_by_key.values()
                    if edge.phase_stable_key == phase_stable_key
                ),
                key=lambda edge: edge.edge_stable_key,
            )
        )
        truth_first = (
            min(int(edge.expectation_available_at_ns) for edge in edges)
            if edges
            else None
        )
        executable_candidates: list[int] = []
        for edge in edges:
            if edge.zero_edge:
                continue
            source_ready = self._source_local_path_complete_at.get(
                (phase_stable_key, edge.src_rank)
            )
            if source_ready is None:
                continue
            executable_candidates.append(
                max(int(edge.expectation_available_at_ns), int(source_ready))
            )
        if executable_candidates:
            executable_first: int | None = min(executable_candidates)
        elif edges and all(edge.zero_edge for edge in edges):
            executable_first = truth_first
        else:
            executable_first = None

        starts = tuple(
            self._source_local_path_start_at.get((phase_stable_key, rank))
            for rank in range(self.world_size)
        )
        completes = tuple(
            self._source_local_path_complete_at.get((phase_stable_key, rank))
            for rank in range(self.world_size)
        )
        if phase_kind == "DISPATCH":
            destination_ready = tuple(
                (
                    None
                    if (phase_stable_key, rank) not in self._dispatch_destinations
                    else self._dispatch_destinations[
                        (phase_stable_key, rank)
                    ].compute_ready_at_ns
                )
                for rank in range(self.world_size)
            )
        else:
            destination_ready = tuple(
                (
                    None
                    if (phase_stable_key, rank) not in self._combine_destinations
                    else self._combine_destinations[
                        (phase_stable_key, rank)
                    ].data_ready_at_ns
                )
                for rank in range(self.world_size)
            )
        releases = tuple(
            self.rank_release_at(phase_key=phase_key, rank_id=rank)
            for rank in range(self.world_size)
        )
        terminal = self.phase_close_at(phase_key=phase_key)
        digest = stable_digest(
            [
                phase_stable_key,
                phase_kind,
                truth_first,
                starts,
                completes,
                executable_first,
                destination_ready,
                releases,
                terminal,
            ],
            prefix="causal-timing",
        )
        observation = PhaseCausalTimingObservation(
            phase_key=phase_key,
            phase_stable_key=phase_stable_key,
            phase_kind=phase_kind,
            phase_truth_first_observable_at_ns=truth_first,
            source_local_path_start_at_ns_by_rank=starts,
            source_local_path_complete_at_ns_by_rank=completes,
            target_first_executable_truth_observation_at_ns=executable_first,
            destination_compute_ready_at_ns_by_rank=destination_ready,
            rank_release_at_ns_by_rank=releases,
            phase_terminal_at_ns=terminal,
            causal_timing_digest=digest,
        )
        if not observation.monotonic:
            raise IllegalTransitionError(
                f"non-monotonic Backend causal timing: {observation.as_dict()}"
            )
        return observation

    def future_prepare_trigger_candidate(
        self, *, source_phase_key: Any, target_phase_key: Any
    ) -> FuturePrepareTriggerCandidate | None:
        """Return a source-closure-grounded Future trigger candidate.

        The caller owns PlanningWindow identity.  backend only states when the full
        source phase truth became causally available and how that time compares
        with the target phase's first real truth observation.
        """

        source_stable_key = self.phase_semantics.phase_sort_key(source_phase_key)
        target_stable_key = self.phase_semantics.phase_sort_key(target_phase_key)
        if source_stable_key == target_stable_key:
            raise BackendContractError(
                "Future trigger source and target phases must be distinct"
            )
        source_closure = self.phase_closure_summary(phase_key=source_phase_key)
        if source_closure is None:
            return None
        source_timing = self.phase_causal_timing_observation(
            phase_key=source_phase_key
        )
        source_executable = (
            source_timing.target_first_executable_truth_observation_at_ns
        )
        available_at_ns = max(
            int(source_closure.finalized_at_ns),
            int(source_executable) if source_executable is not None else 0,
        )
        target_timing = self.phase_causal_timing_observation(
            phase_key=target_phase_key
        )
        target_truth = target_timing.phase_truth_first_observable_at_ns
        trigger_digest = stable_digest(
            [
                source_stable_key,
                target_stable_key,
                available_at_ns,
                source_closure.finalized_at_ns,
                source_closure.closure_digest,
                source_executable,
                target_truth,
            ],
            prefix="future-trigger",
        )
        return FuturePrepareTriggerCandidate(
            source_phase_key=source_phase_key,
            source_phase_stable_key=source_stable_key,
            target_phase_key=target_phase_key,
            target_phase_stable_key=target_stable_key,
            available_at_ns=available_at_ns,
            source_closure_digest=source_closure.closure_digest,
            target_first_truth_observation_at_ns=target_truth,
            template_ready_deadline_at_ns=target_truth,
            trigger_digest=trigger_digest,
        )

    @staticmethod
    def _minimum_observed_time(
        values: Sequence[int | None],
    ) -> int | None:
        observed = tuple(int(value) for value in values if value is not None)
        return min(observed) if observed else None

    def future_overlap_deadlines(
        self, *, source_phase_key: Any, target_phase_key: Any
    ) -> FutureOverlapDeadlines | None:
        """Return actual Prediction/Control/Binding hiding deadlines.

        Prediction ends at the first target source-local-path start, Control at
        its first completion, and Binding at the first destination
        compute-ready boundary.  These are observed Backend intervals, not
        configured slack constants.
        """

        trigger = self.future_prepare_trigger_candidate(
            source_phase_key=source_phase_key,
            target_phase_key=target_phase_key,
        )
        if trigger is None:
            return None
        target = self.phase_causal_timing_observation(phase_key=target_phase_key)
        prediction_deadline = self._minimum_observed_time(
            target.source_local_path_start_at_ns_by_rank
        )
        control_deadline = self._minimum_observed_time(
            target.source_local_path_complete_at_ns_by_rank
        )
        binding_deadline = self._minimum_observed_time(
            target.destination_compute_ready_at_ns_by_rank
        )
        digest = stable_digest(
            [
                trigger.trigger_digest,
                prediction_deadline,
                control_deadline,
                binding_deadline,
                target.source_local_path_start_at_ns_by_rank,
                target.source_local_path_complete_at_ns_by_rank,
                target.destination_compute_ready_at_ns_by_rank,
            ],
            prefix="future-deadlines",
        )
        return FutureOverlapDeadlines(
            source_phase_key=source_phase_key,
            target_phase_key=target_phase_key,
            trigger_available_at_ns=trigger.available_at_ns,
            prediction_hide_deadline_at_ns=prediction_deadline,
            control_hide_deadline_at_ns=control_deadline,
            binding_hide_deadline_at_ns=binding_deadline,
            target_source_local_path_start_at_ns_by_rank=(
                target.source_local_path_start_at_ns_by_rank
            ),
            target_source_local_path_complete_at_ns_by_rank=(
                target.source_local_path_complete_at_ns_by_rank
            ),
            target_destination_compute_ready_at_ns_by_rank=(
                target.destination_compute_ready_at_ns_by_rank
            ),
            deadlines_digest=digest,
        )

    def window_terminal_evidence(
        self,
        *,
        phase_keys: Sequence[Any],
        require_ranks_done: bool = False,
    ) -> WindowTerminalEvidence:
        """Return phase-scoped terminal evidence without duplicate byte sums."""

        referenced = tuple(phase_keys)
        if not referenced:
            raise BackendContractError("phase_keys must be non-empty")
        unique_by_stable_key: dict[str, Any] = {}
        for phase_key in referenced:
            stable_key = self.phase_semantics.phase_sort_key(phase_key)
            unique_by_stable_key.setdefault(stable_key, phase_key)
        # Preserve the caller's exact phase order while de-duplicating repeated
        # references.  This is the anchor/window order; byte totals still count
        # each unique PhaseAuthority exactly once.
        ordered_stable_keys = tuple(unique_by_stable_key)
        unique_phase_keys = tuple(
            unique_by_stable_key[key] for key in ordered_stable_keys
        )
        phase_states = tuple(
            self.phase_terminal_snapshot(phase_key=phase_key)
            for phase_key in unique_phase_keys
        )
        metric_rows = tuple(
            self.phase_rank_metrics_snapshot(
                phase_key=phase_key, rank_id=rank
            )
            for phase_key in unique_phase_keys
            for rank in range(self.world_size)
        )

        def sum_by_rank(field: str) -> tuple[int, ...]:
            return tuple(
                sum(
                    int(getattr(row, field))
                    for row in metric_rows
                    if row.rank_id == rank
                )
                for rank in range(self.world_size)
            )

        def peak_by_rank(field: str) -> tuple[int, ...]:
            return tuple(
                max(
                    (
                        int(getattr(row, field))
                        for row in metric_rows
                        if row.rank_id == rank
                    ),
                    default=0,
                )
                for rank in range(self.world_size)
            )

        release_vectors = tuple(
            tuple(
                self.rank_release_at(phase_key=phase_key, rank_id=rank)
                for rank in range(self.world_size)
            )
            for phase_key in unique_phase_keys
        )
        terminal_times = tuple(
            self.phase_close_at(phase_key=phase_key)
            for phase_key in unique_phase_keys
        )
        current_memory = tuple(
            self.receiver.current_memory(rank)
            for rank in range(self.world_size)
        )
        global_memory_zero = all(
            int(row["total_receiver_bytes"]) == 0 for row in current_memory
        )
        ranks_done = all(
            self.rank_state(rank) is RankState.DONE
            for rank in range(self.world_size)
        )
        no_jobs = all(not state["outstanding_receiver_jobs"] for state in phase_states)
        no_permits = all(not state["unconsumed_permits"] for state in phase_states)
        selected_memory_zero = all(
            not any(int(value) for value in state["staging_bytes_per_rank"].values())
            and not any(
                int(value) for value in state["final_assembly_bytes_per_rank"].values()
            )
            for state in phase_states
        )
        closed = bool(
            all(bool(state["closed"]) for state in phase_states)
            and no_jobs
            and no_permits
            and selected_memory_zero
            and (ranks_done or not require_ranks_done)
        )
        pending_stabilization_count = len(self._pending_stabilization_times)
        disposed = bool(
            closed
            and global_memory_zero
            and ranks_done
            and pending_stabilization_count == 0
        )
        terminal_observed = tuple(
            int(value) for value in terminal_times if value is not None
        )
        window_terminal_at_ns = (
            max(terminal_observed) if terminal_observed else None
        )
        total_expected = sum(
            int(state["total_expected_bytes"]) for state in phase_states
        )
        total_assembled = sum(
            int(state["total_assembled_bytes"]) for state in phase_states
        )
        post_wait = sum_by_rank("receiver_posting_service_ns")
        posting_queue_wait = sum_by_rank("receiver_posting_queue_wait_ns")
        buffer_stall = sum_by_rank("receiver_buffer_stall_ns")
        drain_queue_wait = sum_by_rank("receiver_drain_queue_wait_ns")
        drain_service = sum_by_rank("receiver_drain_service_ns")
        peak_staging = peak_by_rank("peak_staging_bytes")
        peak_final = peak_by_rank("peak_final_assembly_bytes")
        peak_total = peak_by_rank("peak_total_receiver_bytes")
        evidence_digest = stable_digest(
            [
                ordered_stable_keys,
                len(referenced),
                total_expected,
                total_assembled,
                post_wait,
                posting_queue_wait,
                buffer_stall,
                drain_queue_wait,
                drain_service,
                peak_staging,
                peak_final,
                peak_total,
                release_vectors,
                terminal_times,
                no_jobs,
                no_permits,
                selected_memory_zero,
                global_memory_zero,
                ranks_done,
                pending_stabilization_count,
                closed,
                disposed,
            ],
            prefix="window-terminal",
        )
        return WindowTerminalEvidence(
            phase_keys=unique_phase_keys,
            phase_stable_keys=ordered_stable_keys,
            referenced_phase_count=len(referenced),
            unique_phase_count=len(unique_phase_keys),
            duplicate_phase_reference_count=(
                len(referenced) - len(unique_phase_keys)
            ),
            total_expected_payload_bytes=total_expected,
            total_assembled_payload_bytes=total_assembled,
            receiver_posting_service_ns_by_rank=post_wait,
            receiver_posting_queue_wait_ns_by_rank=posting_queue_wait,
            receiver_buffer_stall_ns_by_rank=buffer_stall,
            receiver_drain_queue_wait_ns_by_rank=drain_queue_wait,
            receiver_drain_service_ns_by_rank=drain_service,
            peak_staging_bytes_by_rank=peak_staging,
            peak_final_assembly_bytes_by_rank=peak_final,
            peak_total_receiver_bytes_by_rank=peak_total,
            phase_rank_release_at_ns=release_vectors,
            phase_terminal_at_ns=terminal_times,
            window_terminal_at_ns=window_terminal_at_ns,
            no_residual_receiver_jobs=no_jobs,
            no_residual_permits=no_permits,
            selected_phase_receiver_memory_zero=selected_memory_zero,
            global_receiver_memory_zero=global_memory_zero,
            ranks_done=ranks_done,
            pending_backend_stabilization_count=pending_stabilization_count,
            closed=closed,
            backend_disposed=disposed,
            evidence_digest=evidence_digest,
        )

    # ------------------------------------------------------------------
    # Static trace-derived local path registration
    # ------------------------------------------------------------------
    def register_local_path_spec(
        self,
        *,
        combine_phase_key: Any,
        next_dispatch_phase_key: Any,
        rank_id: int,
        combine_release_to_router_ready_ns: int,
        router_and_pack_ns: int,
    ) -> None:
        self._validate_rank(rank_id)
        if self.phase_semantics.phase_kind(combine_phase_key) != "COMBINE":
            raise BackendContractError("local path source phase must be COMBINE")
        if self.phase_semantics.phase_kind(next_dispatch_phase_key) != "DISPATCH":
            raise BackendContractError("local path target phase must be DISPATCH")
        combine_release_to_router_ready_ns = require_nonnegative_int(
            combine_release_to_router_ready_ns,
            field="combine_release_to_router_ready_ns",
        )
        router_and_pack_ns = require_nonnegative_int(
            router_and_pack_ns, field="router_and_pack_ns"
        )
        key = (self.phase_semantics.phase_sort_key(combine_phase_key), rank_id)
        candidate = LocalPathSpec(
            combine_phase_key=combine_phase_key,
            next_dispatch_phase_key=next_dispatch_phase_key,
            rank_id=rank_id,
            combine_release_to_router_ready_ns=combine_release_to_router_ready_ns,
            router_and_pack_ns=router_and_pack_ns,
        )
        existing = self._local_path_specs.get(key)
        if existing is not None and existing != candidate:
            raise DuplicateRegistrationError("local path spec changed")
        self._local_path_specs[key] = candidate

    def register_terminal_local_path_spec(
        self,
        *,
        combine_phase_key: Any,
        rank_id: int,
        combine_release_to_router_ready_ns: int,
        terminal_local_compute_ns: int,
    ) -> None:
        """Register the final Combine local path without fabricating a next P0."""

        self._validate_rank(rank_id)
        if self.phase_semantics.phase_kind(combine_phase_key) != "COMBINE":
            raise BackendContractError("terminal local path source phase must be COMBINE")
        combine_release_to_router_ready_ns = require_nonnegative_int(
            combine_release_to_router_ready_ns,
            field="combine_release_to_router_ready_ns",
        )
        terminal_local_compute_ns = require_nonnegative_int(
            terminal_local_compute_ns, field="terminal_local_compute_ns"
        )
        key = (self.phase_semantics.phase_sort_key(combine_phase_key), rank_id)
        candidate = LocalPathSpec(
            combine_phase_key=combine_phase_key,
            next_dispatch_phase_key=None,
            rank_id=rank_id,
            combine_release_to_router_ready_ns=combine_release_to_router_ready_ns,
            router_and_pack_ns=terminal_local_compute_ns,
        )
        existing = self._local_path_specs.get(key)
        if existing is not None and existing != candidate:
            raise DuplicateRegistrationError("terminal local path spec changed")
        self._local_path_specs[key] = candidate

    def register_dispatch_compute_spec(
        self,
        *,
        dispatch_phase_key: Any,
        next_combine_phase_key: Any,
        rank_id: int,
        dispatch_local_postprocess_ns: int,
        dispatch_release_to_combine_source_ready_ns: int,
    ) -> None:
        self._validate_rank(rank_id)
        if self.phase_semantics.phase_kind(dispatch_phase_key) != "DISPATCH":
            raise BackendContractError("dispatch compute source phase must be DISPATCH")
        if self.phase_semantics.phase_kind(next_combine_phase_key) != "COMBINE":
            raise BackendContractError("dispatch compute target phase must be COMBINE")
        dispatch_local_postprocess_ns = require_nonnegative_int(
            dispatch_local_postprocess_ns,
            field="dispatch_local_postprocess_ns",
        )
        dispatch_release_to_combine_source_ready_ns = require_nonnegative_int(
            dispatch_release_to_combine_source_ready_ns,
            field="dispatch_release_to_combine_source_ready_ns",
        )
        key = (self.phase_semantics.phase_sort_key(dispatch_phase_key), rank_id)
        candidate = DispatchComputeSpec(
            dispatch_phase_key=dispatch_phase_key,
            next_combine_phase_key=next_combine_phase_key,
            rank_id=rank_id,
            dispatch_local_postprocess_ns=dispatch_local_postprocess_ns,
            dispatch_release_to_combine_source_ready_ns=(
                dispatch_release_to_combine_source_ready_ns
            ),
        )
        existing = self._dispatch_compute_specs.get(key)
        if existing is not None and existing != candidate:
            raise DuplicateRegistrationError("dispatch compute spec changed")
        self._dispatch_compute_specs[key] = candidate

    def register_exact_dispatch_row_truth(
        self,
        *,
        phase_key: Any,
        src_rank: int,
        payload_bytes_by_destination: Sequence[int] | None = None,
        payload_spec_digest: str | None = None,
        descriptor_payload_bytes: int,
        realized_rows_by_destination: Sequence[int] | None = None,
        dispatch_payload_bytes_by_destination: Sequence[int] | None = None,
        combine_return_payload_bytes_by_expert: Sequence[int] | None = None,
        dispatch_payload_spec_digest: str | None = None,
        combine_payload_spec_digest: str | None = None,
        truth_digest: str | None = None,
    ) -> None:
        """Register immutable rank-local Dispatch truth without publishing it.

        The formal runtime supplies realized rows plus independently derived
        Dispatch and Combine payload bytes.  The legacy ``payload_*`` aliases
        remain only for contract fixtures; they deliberately use equal phase
        bytes and are not performance-eligible.
        """
        self._validate_rank(src_rank)
        if self.phase_semantics.phase_kind(phase_key) != "DISPATCH":
            raise BackendContractError("exact row truth is Dispatch-only")
        raw_dispatch = (
            dispatch_payload_bytes_by_destination
            if dispatch_payload_bytes_by_destination is not None
            else payload_bytes_by_destination
        )
        if raw_dispatch is None:
            raise BackendContractError("dispatch payload bytes are required")
        dispatch_bytes = tuple(
            require_nonnegative_int(value, field="dispatch_payload_bytes_by_destination")
            for value in raw_dispatch
        )
        if len(dispatch_bytes) != self.world_size:
            raise BackendContractError("exact dispatch row truth must cover world_size destinations")
        if realized_rows_by_destination is None:
            # Legacy fixture-only compatibility.  It preserves zero/nonzero
            # closure but does not claim row-accurate phase conversion.
            realized_rows = tuple(0 if value == 0 else 1 for value in dispatch_bytes)
        else:
            realized_rows = tuple(
                require_nonnegative_int(value, field="realized_rows_by_destination")
                for value in realized_rows_by_destination
            )
        if len(realized_rows) != self.world_size:
            raise BackendContractError("realized row truth must cover world_size destinations")
        combine_bytes = tuple(
            require_nonnegative_int(value, field="combine_return_payload_bytes_by_expert")
            for value in (
                combine_return_payload_bytes_by_expert
                if combine_return_payload_bytes_by_expert is not None
                else dispatch_bytes
            )
        )
        if len(combine_bytes) != self.world_size:
            raise BackendContractError("combine return truth must cover world_size destinations")
        dispatch_spec_digest = str(dispatch_payload_spec_digest or payload_spec_digest or "")
        combine_spec_digest = str(combine_payload_spec_digest or dispatch_spec_digest)
        if not dispatch_spec_digest or not combine_spec_digest:
            raise BackendContractError("both Dispatch and Combine PayloadSpec digests are required")
        descriptor_payload_bytes = require_nonnegative_int(
            descriptor_payload_bytes, field="descriptor_payload_bytes"
        )
        candidate = make_exact_dispatch_row_truth(
            phase_key=phase_key,
            src_rank=src_rank,
            realized_rows_by_destination=realized_rows,
            dispatch_payload_bytes_by_destination=dispatch_bytes,
            combine_return_payload_bytes_by_expert=combine_bytes,
            dispatch_payload_spec_digest=dispatch_spec_digest,
            combine_payload_spec_digest=combine_spec_digest,
            descriptor_payload_bytes=descriptor_payload_bytes,
        )
        if truth_digest is not None and str(truth_digest) != candidate.truth_digest:
            raise BackendContractError("provided exact dispatch truth digest mismatch")
        key = (self.phase_semantics.phase_sort_key(phase_key), src_rank)
        existing = self._dispatch_row_truth.get(key)
        if existing is not None and existing != candidate:
            raise DuplicateRegistrationError("exact dispatch row truth changed")
        self._dispatch_row_truth[key] = candidate

    def register_combine_expectations_from_dispatch_truth(
        self,
        *,
        dispatch_phase_key: Any,
        combine_phase_key: Any,
        original_rank: int,
        created_at_ns: int,
    ) -> tuple[Any, ...]:
        """Create Combine expectations using row-derived Combine PayloadSpec bytes."""
        key = (self.phase_semantics.phase_sort_key(dispatch_phase_key), int(original_rank))
        try:
            truth = self._dispatch_row_truth[key]
        except KeyError as exc:
            raise UnknownObjectError("exact Dispatch row truth is not registered") from exc
        return self.register_combine_expectations_from_realized_dispatch(
            combine_phase_key=combine_phase_key,
            original_rank=original_rank,
            realized_dispatch_payload_bytes_by_expert=(
                truth.combine_return_payload_bytes_by_expert
            ),
            created_at_ns=created_at_ns,
        )

    def on_exact_row_descriptor_delivered(
        self, *, descriptor: ExactRowDescriptor, delivered_at_ns: int
    ) -> tuple[Any, ...]:
        if not isinstance(descriptor, ExactRowDescriptor):
            raise BackendContractError("descriptor must be ExactRowDescriptor")
        return self.on_dispatch_descriptor_delivered(
            phase_key=descriptor.phase_key,
            src_rank=descriptor.src_rank,
            payload_bytes_by_destination=descriptor.payload_bytes_by_destination,
            descriptor_digest=descriptor.descriptor_digest,
            delivered_at_ns=delivered_at_ns,
        )

    # ------------------------------------------------------------------
    # Exact descriptor and realized-dispatch expectation construction
    # ------------------------------------------------------------------
    def on_dispatch_descriptor_delivered(
        self,
        *,
        phase_key: Any,
        src_rank: int,
        payload_bytes_by_destination: Sequence[int],
        descriptor_digest: str,
        delivered_at_ns: int,
    ) -> tuple[Any, ...]:
        """Apply one delivered exact dispatch row, including every zero edge."""
        self._validate_rank(src_rank)
        delivered_at_ns = require_time_ns(
            delivered_at_ns, field="descriptor.delivered_at_ns"
        )
        if self.phase_semantics.phase_kind(phase_key) != "DISPATCH":
            raise BackendContractError("descriptor delivery is Dispatch-only")
        if len(payload_bytes_by_destination) != self.world_size:
            raise BackendContractError(
                "dispatch descriptor must contain a complete world-size row"
            )
        if not descriptor_digest:
            raise BackendContractError("descriptor digest must be non-empty")

        expectations: list[Any] = []
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        for dst_rank, raw_bytes in enumerate(payload_bytes_by_destination):
            payload_bytes = require_nonnegative_int(
                raw_bytes, field=f"descriptor.payload_bytes[{dst_rank}]"
            )
            edge_key = self.edge_key_factory.make_edge_key(
                phase_key=phase_key, src_rank=src_rank, dst_rank=dst_rank
            )
            expectation_digest = stable_digest(
                [
                    phase_stable_key,
                    src_rank,
                    dst_rank,
                    payload_bytes,
                    descriptor_digest,
                    "DISPATCH",
                ],
                prefix="expectation",
            )
            expectation = self.expectation_factory.create_receive_expectation(
                edge_key=edge_key,
                phase_key=phase_key,
                src_rank=src_rank,
                dst_rank=dst_rank,
                total_expected_payload_bytes=payload_bytes,
                expectation_digest=expectation_digest,
                origin="DELIVERED_DISPATCH_DESCRIPTOR",
                created_at_ns=delivered_at_ns,
                zero_edge=payload_bytes == 0,
                descriptor_digest_or_none=descriptor_digest,
            )
            self.receiver.register_expectation(
                expectation,
                descriptor_digest_or_none=descriptor_digest,
            )
            expectations.append(expectation)
            state = self._dispatch_state(phase_key=phase_key, dst_rank=dst_rank)
            existing = state.descriptor_sources_delivered.get(src_rank)
            if existing is not None and existing != delivered_at_ns:
                raise DuplicateRegistrationError(
                    "descriptor delivery timestamp changed for source/destination"
                )
            state.descriptor_sources_delivered[src_rank] = delivered_at_ns
            if len(state.descriptor_sources_delivered) == self.world_size:
                closure = max(state.descriptor_sources_delivered.values())
                if state.descriptor_closure_at_ns is None:
                    state.descriptor_closure_at_ns = closure
                    self.observer.emit(
                        kind="DISPATCH_DESCRIPTOR_CLOSED",
                        at_ns=closure,
                        payload={
                            "phase_key": phase_key,
                            "dst_rank": dst_rank,
                            "descriptor_sources": self.world_size,
                        },
                    )
                elif state.descriptor_closure_at_ns != closure:
                    raise IllegalTransitionError("descriptor closure changed")
        self._maybe_finalize_phase_closure(phase_key=phase_key)
        self._request_stabilization(at_ns=delivered_at_ns)
        return tuple(expectations)

    def register_combine_expectations_from_realized_dispatch(
        self,
        *,
        combine_phase_key: Any,
        original_rank: int,
        realized_dispatch_payload_bytes_by_expert: Sequence[int],
        created_at_ns: int,
    ) -> tuple[Any, ...]:
        """Pre-register the exact Combine transpose for one original rank."""
        self._validate_rank(original_rank)
        created_at_ns = require_time_ns(
            created_at_ns, field="combine_expectation.created_at_ns"
        )
        if self.phase_semantics.phase_kind(combine_phase_key) != "COMBINE":
            raise BackendContractError("Combine expectation requires COMBINE phase")
        if len(realized_dispatch_payload_bytes_by_expert) != self.world_size:
            raise BackendContractError(
                "realized dispatch row must cover every expert/source rank"
            )
        phase_stable_key = self.phase_semantics.phase_sort_key(combine_phase_key)
        expectations: list[Any] = []
        for expert_rank, raw_bytes in enumerate(
            realized_dispatch_payload_bytes_by_expert
        ):
            payload_bytes = require_nonnegative_int(
                raw_bytes,
                field=f"realized_dispatch_payload_bytes[{expert_rank}]",
            )
            edge_key = self.edge_key_factory.make_edge_key(
                phase_key=combine_phase_key,
                src_rank=expert_rank,
                dst_rank=original_rank,
            )
            expectation_digest = stable_digest(
                [
                    phase_stable_key,
                    expert_rank,
                    original_rank,
                    payload_bytes,
                    "COMBINE_TRANSPOSE",
                ],
                prefix="expectation",
            )
            expectation = self.expectation_factory.create_receive_expectation(
                edge_key=edge_key,
                phase_key=combine_phase_key,
                src_rank=expert_rank,
                dst_rank=original_rank,
                total_expected_payload_bytes=payload_bytes,
                expectation_digest=expectation_digest,
                origin="REALIZED_DISPATCH_TRANSPOSE",
                created_at_ns=created_at_ns,
                zero_edge=payload_bytes == 0,
                descriptor_digest_or_none=None,
            )
            self.receiver.register_expectation(
                expectation, descriptor_digest_or_none=None
            )
            expectations.append(expectation)
        self.receiver.seal_combine_expectations(
            phase_key=combine_phase_key,
            dst_rank=original_rank,
            at_ns=created_at_ns,
        )
        self._maybe_finalize_phase_closure(phase_key=combine_phase_key)
        self._request_stabilization(at_ns=created_at_ns)
        return tuple(expectations)

    # ------------------------------------------------------------------
    # Source hooks and destination model thread readiness
    # ------------------------------------------------------------------
    def on_bootstrap_dispatch_local_path_complete(
        self, *, phase_key: Any, rank_id: int, at_ns: int
    ) -> None:
        """Publish initial P0 descriptor/payload/thread readiness."""
        self._publish_dispatch_source_and_thread_ready(
            phase_key=phase_key,
            rank_id=rank_id,
            at_ns=at_ns,
            origin="BOOTSTRAP_P0_LOCAL_PATH",
        )

    def mark_dispatch_model_thread_ready(
        self, *, phase_key: Any, dst_rank: int, at_ns: int
    ) -> None:
        self._validate_rank(dst_rank)
        at_ns = require_time_ns(at_ns, field="dispatch_thread_ready.at_ns")
        state = self._dispatch_state(phase_key=phase_key, dst_rank=dst_rank)
        existing = state.model_thread_ready_at_ns
        if existing is not None:
            if existing != at_ns:
                raise DuplicateRegistrationError(
                    "destination dispatch model-thread readiness changed"
                )
            return
        state.model_thread_ready_at_ns = at_ns
        self.observer.emit(
            kind="DESTINATION_DISPATCH_THREAD_READY",
            at_ns=at_ns,
            payload={"phase_key": phase_key, "dst_rank": dst_rank},
        )
        self._request_stabilization(at_ns=at_ns)

    # ------------------------------------------------------------------
    # Scheduler/Transport-facing events
    # ------------------------------------------------------------------
    def register_canonical_task_catalogue(self, tasks: Sequence[Any]) -> None:
        self.receiver.register_task_catalogue(tasks)
        registration_times = sorted(
            {
                require_time_ns(
                    self.adapter.get(task, "registered_at_ns"),
                    field="task.registered_at_ns",
                )
                for task in tasks
            }
        )
        for at_ns in registration_times:
            self._request_stabilization(at_ns=at_ns)

    def on_source_payload_ready(
        self, *, phase_key: Any, src_rank: int, at_ns: int
    ) -> None:
        self.receiver.mark_source_payload_ready(
            phase_key=phase_key, src_rank=src_rank, at_ns=at_ns
        )
        self._record_source_local_path_complete(
            phase_key=phase_key,
            rank_id=src_rank,
            at_ns=at_ns,
            origin="SOURCE_PAYLOAD_READY",
        )
        self._request_stabilization(at_ns=at_ns)

    def on_transfer_completed(self, *, task_id: Any, at_ns: int) -> None:
        self.receiver.on_transfer_completed(task_id=task_id, at_ns=at_ns)
        self._request_stabilization(at_ns=at_ns)

    def handle_event(
        self, *, event_kind: str, payload: Mapping[str, Any], at_ns: int
    ) -> None:
        """Handle an event dispatched by the shared-schema kernel adapter."""
        at_ns = require_time_ns(at_ns, field="backend_event.at_ns")
        if event_kind == _EVENT_BACKEND_STABILIZE:
            self._pending_stabilization_times.discard(at_ns)
            self.stabilize(at_ns=at_ns)
            return
        if event_kind == "BACKEND_RECEIVER_POST_COMPLETE":
            self.receiver.on_receiver_post_complete(
                task_key=str(payload["task_key"]), at_ns=at_ns
            )
            self._request_stabilization(at_ns=at_ns)
            return
        if event_kind == "BACKEND_RECEIVER_DRAIN_FINISH":
            self.receiver.on_receiver_drain_finish(
                task_key=str(payload["task_key"]), at_ns=at_ns
            )
            self._request_stabilization(at_ns=at_ns)
            return
        if event_kind == "BACKEND_LOCAL_ASSEMBLY_FINISH":
            self.receiver.on_local_assembly_finish(
                edge_key=str(payload["edge_key"]), at_ns=at_ns
            )
            self._request_stabilization(at_ns=at_ns)
            return
        if event_kind == _EVENT_POST_COMBINE_COMPLETE:
            self._on_post_combine_local_path_complete(
                phase_key=payload["phase_key"],
                rank_id=int(payload["rank_id"]),
                at_ns=at_ns,
            )
            return
        if event_kind == _EVENT_LOCAL_PATH_COMPLETE:
            self._on_local_path_complete(
                combine_phase_key=payload["combine_phase_key"],
                rank_id=int(payload["rank_id"]),
                at_ns=at_ns,
            )
            return
        if event_kind == _EVENT_DISPATCH_POSTPROCESS_COMPLETE:
            self._on_dispatch_postprocess_complete(
                phase_key=payload["phase_key"],
                rank_id=int(payload["rank_id"]),
                at_ns=at_ns,
            )
            return
        if event_kind == _EVENT_P0_P1_LOCAL_COMPUTE_COMPLETE:
            self._on_p0_p1_local_compute_complete(
                dispatch_phase_key=payload["dispatch_phase_key"],
                combine_phase_key=payload["combine_phase_key"],
                rank_id=int(payload["rank_id"]),
                at_ns=at_ns,
            )
            return
        if event_kind == _EVENT_COMBINE_SOURCE_READY:
            self._on_combine_source_ready(
                phase_key=payload["phase_key"],
                rank_id=int(payload["rank_id"]),
                at_ns=at_ns,
            )
            return
        raise UnknownObjectError(f"unknown backend event kind {event_kind!r}")

    def _request_stabilization(self, *, at_ns: int) -> None:
        at_ns = require_time_ns(at_ns, field="backend_stabilization.at_ns")
        if at_ns in self._pending_stabilization_times:
            return
        ordinal = self._stabilization_ordinal_by_time.get(at_ns, 0)
        self._stabilization_ordinal_by_time[at_ns] = ordinal + 1
        self._pending_stabilization_times.add(at_ns)
        self.kernel.schedule_backend_event(
            time_ns=at_ns,
            phase_priority=BACKEND_PHASE_PRIORITY,
            stable_event_id=stable_semantic_event_id(
                event_kind=_EVENT_BACKEND_STABILIZE,
                time_ns=at_ns,
                semantic_parts=[ordinal],
            ),
            event_kind=_EVENT_BACKEND_STABILIZE,
            payload={"ordinal": ordinal},
        )

    def stabilize(self, *, at_ns: int) -> bool:
        """Run one phase-4 fixed-point pass after same-time inputs are collected."""
        at_ns = require_time_ns(at_ns, field="backend_stabilize.at_ns")
        progressed = self.receiver.stabilize(now_ns=at_ns)
        destination_signature = (
            len(self.receiver.edges_by_key),
            len(self._dispatch_destinations),
            len(self._combine_destinations),
        )
        if destination_signature != self._stabilize_destination_signature:
            destinations = {
                (edge.phase_stable_key, edge.dst_rank, edge.phase_key)
                for edge in self.receiver.edges_by_key.values()
            }
            destinations.update(
                (state.phase_stable_key, state.dst_rank, state.phase_key)
                for state in self._dispatch_destinations.values()
            )
            destinations.update(
                (state.phase_stable_key, state.dst_rank, state.phase_key)
                for state in self._combine_destinations.values()
            )
            self._stabilize_destinations_cache = tuple(
                sorted(destinations, key=lambda row: (row[0], row[1]))
            )
            self._stabilize_destination_signature = destination_signature
        before_releases = len(self._rank_release_at)
        before_scheduled = sum(
            1 for state in self._dispatch_destinations.values()
            if state.postprocess_scheduled
        ) + sum(
            1 for state in self._combine_destinations.values()
            if state.post_combine_scheduled
        )
        for _, dst_rank, phase_key in self._stabilize_destinations_cache:
            self._evaluate_destination(
                phase_key=phase_key, dst_rank=dst_rank, now_ns=at_ns
            )
        after_scheduled = sum(
            1 for state in self._dispatch_destinations.values()
            if state.postprocess_scheduled
        ) + sum(
            1 for state in self._combine_destinations.values()
            if state.post_combine_scheduled
        )
        return (
            progressed
            or len(self._rank_release_at) > before_releases
            or after_scheduled > before_scheduled
        )

    # ------------------------------------------------------------------
    # Receiver completion -> local path -> next hook
    # ------------------------------------------------------------------
    def _evaluate_destination(
        self, *, phase_key: Any, dst_rank: int, now_ns: int
    ) -> None:
        kind = self.phase_semantics.phase_kind(phase_key)
        if kind == "DISPATCH":
            self._evaluate_dispatch_destination(
                phase_key=phase_key, dst_rank=dst_rank, now_ns=now_ns
            )
        else:
            self._evaluate_combine_destination(
                phase_key=phase_key, dst_rank=dst_rank, now_ns=now_ns
            )

    def _evaluate_combine_destination(
        self, *, phase_key: Any, dst_rank: int, now_ns: int
    ) -> None:
        state = self._combine_state(phase_key=phase_key, dst_rank=dst_rank)
        complete, data_ready_at = self.receiver.all_nonzero_inbound_assembled(
            phase_key=phase_key, dst_rank=dst_rank
        )
        if not complete or data_ready_at is None:
            return
        if state.data_ready_at_ns is None:
            state.data_ready_at_ns = data_ready_at
            self.observer.emit(
                kind="P1_DATA_READY",
                at_ns=data_ready_at,
                payload={
                    "phase_key": phase_key,
                    "rank_id": dst_rank,
                    "p1_local_complete_at_ns": data_ready_at,
                },
            )
            if self.release_mode == "PHASE_BARRIER":
                self._rank_actors[dst_rank].transition(
                    state=RankState.WAIT_PHASE_BARRIER,
                    phase_key=phase_key,
                    at_ns=data_ready_at,
                    reason="P1_DATA_READY_WAIT_PHASE_BARRIER",
                )
        elif state.data_ready_at_ns != data_ready_at:
            raise IllegalTransitionError("Combine P1 data-ready time changed")

        # Destination-side zero/inbound closure is not enough to advance a
        # model rank.  The same rank must also have reached the source-side
        # Combine hook for this layer (its local expert work / local
        # contribution is complete).  Without this gate, a zero-inbound rank
        # can run post-Combine paths for future layers at catalogue-close time
        # and even enter DONE before earlier Dispatch/Compute events occur.
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        source_ready_at = self._source_local_path_complete_at.get(
            (phase_stable_key, dst_rank)
        )
        if source_ready_at is None:
            return
        causal_ready_at = max(int(data_ready_at), int(source_ready_at))

        if self.release_mode == "RANK_LOCAL":
            self._schedule_post_combine_local_path(
                phase_key=phase_key,
                rank_id=dst_rank,
                start_at_ns=causal_ready_at,
                observed_at_ns=now_ns,
            )
        else:
            self._maybe_open_combine_phase_barrier(
                phase_key=phase_key, now_ns=now_ns
            )

    def _schedule_post_combine_local_path(
        self,
        *,
        phase_key: Any,
        rank_id: int,
        start_at_ns: int,
        observed_at_ns: int,
    ) -> None:
        state = self._combine_state(phase_key=phase_key, dst_rank=rank_id)
        if state.post_combine_scheduled:
            return
        spec = self._local_path_specs.get(
            (self.phase_semantics.phase_sort_key(phase_key), rank_id)
        )
        if spec is None:
            raise BackendContractError(
                "missing LocalPathSpec for completed Combine destination"
            )
        # A zero/sparse inbound destination can become semantically data-ready
        # before catalogue closure causes the backend to observe that fact.  Do
        # not retroactively move the rank actor or schedule events into the
        # past; preserve ``data_ready_at_ns`` as evidence, while starting the
        # causal local path at the first time it can actually be committed.
        effective_start_at_ns = max(
            int(start_at_ns),
            int(observed_at_ns),
            int(self._rank_actors[rank_id].last_transition_at_ns),
        )
        self._rank_actors[rank_id].transition(
            state=RankState.POST_COMBINE_LOCAL_PATH,
            phase_key=phase_key,
            at_ns=effective_start_at_ns,
            reason=(
                "P1_DATA_READY"
                if self.release_mode == "RANK_LOCAL"
                else "COMBINE_PHASE_BARRIER_OPEN"
            ),
        )
        state.local_path_start_at_ns = effective_start_at_ns
        if spec.next_dispatch_phase_key is not None:
            self._record_source_local_path_start(
                phase_key=spec.next_dispatch_phase_key,
                rank_id=rank_id,
                at_ns=effective_start_at_ns,
                origin="POST_COMBINE_LOCAL_PATH",
            )
        completion = effective_start_at_ns + spec.combine_release_to_router_ready_ns
        state.post_combine_complete_at_ns = completion
        state.post_combine_scheduled = True
        self.kernel.schedule_backend_event(
            time_ns=completion,
            phase_priority=BACKEND_PHASE_PRIORITY,
            stable_event_id=stable_semantic_event_id(
                event_kind=_EVENT_POST_COMBINE_COMPLETE,
                time_ns=completion,
                semantic_parts=[
                    self.phase_semantics.phase_sort_key(phase_key), rank_id
                ],
            ),
            event_kind=_EVENT_POST_COMBINE_COMPLETE,
            payload={"phase_key": phase_key, "rank_id": rank_id},
        )

    def _maybe_open_combine_phase_barrier(
        self, *, phase_key: Any, now_ns: int
    ) -> None:
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        if phase_stable_key in self._combine_barrier_opened:
            return
        states = [
            self._combine_state(phase_key=phase_key, dst_rank=rank)
            for rank in range(self.world_size)
        ]
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        source_ready = tuple(
            self._source_local_path_complete_at.get((phase_stable_key, rank))
            for rank in range(self.world_size)
        )
        if any(state.data_ready_at_ns is None for state in states) or any(
            value is None for value in source_ready
        ):
            return
        barrier_at = max(
            max(int(state.data_ready_at_ns or 0), int(source_ready[rank] or 0))
            for rank, state in enumerate(states)
        )
        if barrier_at > now_ns:
            return
        self._combine_barrier_opened.add(phase_stable_key)
        self.observer.emit(
            kind="COMBINE_PHASE_BARRIER_OPEN",
            at_ns=barrier_at,
            payload={"phase_key": phase_key, "world_size": self.world_size},
        )
        for rank_id in range(self.world_size):
            self._schedule_post_combine_local_path(
                phase_key=phase_key,
                rank_id=rank_id,
                start_at_ns=barrier_at,
                observed_at_ns=now_ns,
            )

    def _on_post_combine_local_path_complete(
        self, *, phase_key: Any, rank_id: int, at_ns: int
    ) -> None:
        state = self._combine_state(phase_key=phase_key, dst_rank=rank_id)
        if state.post_combine_complete_at_ns != at_ns:
            raise IllegalTransitionError("post-combine completion timestamp mismatch")
        self.receiver.release_final_assembly(
            phase_key=phase_key, dst_rank=rank_id, at_ns=at_ns
        )
        self._rank_actors[rank_id].transition(
            state=RankState.ROUTER_AND_PACK,
            phase_key=phase_key,
            at_ns=at_ns,
            reason="POST_COMBINE_LOCAL_PATH_COMPLETE",
        )
        self.observer.emit(
            kind="POST_COMBINE_LOCAL_PATH_COMPLETE",
            at_ns=at_ns,
            payload={"phase_key": phase_key, "rank_id": rank_id},
        )
        spec = self._local_path_specs[
            (self.phase_semantics.phase_sort_key(phase_key), rank_id)
        ]
        completion = at_ns + spec.router_and_pack_ns
        state.local_path_complete_at_ns = completion
        state.local_path_scheduled = True
        self.kernel.schedule_backend_event(
            time_ns=completion,
            phase_priority=BACKEND_PHASE_PRIORITY,
            stable_event_id=stable_semantic_event_id(
                event_kind=_EVENT_LOCAL_PATH_COMPLETE,
                time_ns=completion,
                semantic_parts=[
                    self.phase_semantics.phase_sort_key(phase_key), rank_id
                ],
            ),
            event_kind=_EVENT_LOCAL_PATH_COMPLETE,
            payload={"combine_phase_key": phase_key, "rank_id": rank_id},
        )

    def _on_local_path_complete(
        self, *, combine_phase_key: Any, rank_id: int, at_ns: int
    ) -> None:
        state = self._combine_state(
            phase_key=combine_phase_key, dst_rank=rank_id
        )
        if state.local_path_complete_at_ns != at_ns:
            raise IllegalTransitionError("local path completion timestamp mismatch")
        spec = self._local_path_specs[
            (self.phase_semantics.phase_sort_key(combine_phase_key), rank_id)
        ]
        if spec.next_dispatch_phase_key is None:
            terminal_key = (
                self.phase_semantics.phase_sort_key(combine_phase_key), rank_id
            )
            if terminal_key in self._terminal_combine_ranks:
                raise IllegalTransitionError("terminal Combine rank completed twice")
            self._terminal_combine_ranks.add(terminal_key)
            self._rank_actors[rank_id].transition(
                state=RankState.DONE,
                phase_key=combine_phase_key,
                at_ns=at_ns,
                reason="TERMINAL_LOCAL_PATH_COMPLETE",
            )
            self.observer.emit(
                kind="BACKEND_RANK_TERMINAL",
                at_ns=at_ns,
                payload={"phase_key": combine_phase_key, "rank_id": rank_id},
            )
            if all(
                (self.phase_semantics.phase_sort_key(combine_phase_key), rank)
                in self._terminal_combine_ranks
                for rank in range(self.world_size)
            ):
                self.observer.emit(
                    kind="BACKEND_TERMINAL_WINDOW_CLOSED",
                    at_ns=at_ns,
                    payload={
                        "phase_key": combine_phase_key,
                        "world_size": self.world_size,
                    },
                )
            return
        dispatch_key = spec.next_dispatch_phase_key
        source_key = (self.phase_semantics.phase_sort_key(dispatch_key), rank_id)
        if source_key in self._source_descriptor_ready_at:
            self._publish_dispatch_thread_ready_only(
                phase_key=dispatch_key,
                rank_id=rank_id,
                at_ns=at_ns,
                origin="POST_COMBINE_LOCAL_PATH",
            )
        else:
            self._publish_dispatch_source_and_thread_ready(
                phase_key=dispatch_key,
                rank_id=rank_id,
                at_ns=at_ns,
                origin="POST_COMBINE_LOCAL_PATH",
            )

    def _publish_dispatch_source_ready_only(
        self, *, phase_key: Any, rank_id: int, at_ns: int, origin: str
    ) -> dict[str, Any]:
        """Publish P2 source truth/payload without advancing the model thread.

        This split is used only by the opt-in streaming diagnostic.  The
        default fraction is 1.0, which preserves the frozen lifecycle.
        """
        self._validate_rank(rank_id)
        at_ns = require_time_ns(at_ns, field="source_descriptor_ready.at_ns")
        if self.phase_semantics.phase_kind(phase_key) != "DISPATCH":
            raise BackendContractError("local path must publish a Dispatch phase")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        key = (phase_stable_key, rank_id)
        existing = self._source_descriptor_ready_at.get(key)
        if existing is not None:
            return {
                "phase_key": phase_key,
                "rank_id": rank_id,
                "origin": origin,
                "already_published": True,
            }
        truth = self._dispatch_row_truth.get((phase_stable_key, rank_id))
        exact_descriptor = None
        control_request_digest = None
        descriptor_key = (phase_stable_key, rank_id)
        if truth is not None:
            exact_descriptor = make_exact_row_descriptor(
                phase_key=phase_key,
                src_rank=rank_id,
                realized_rows_by_destination=truth.realized_rows_by_destination,
                payload_bytes_by_destination=truth.dispatch_payload_bytes_by_destination,
                payload_spec_digest=truth.dispatch_payload_spec_digest,
                published_at_ns=at_ns,
                descriptor_payload_bytes=truth.descriptor_payload_bytes,
            )
            existing_descriptor = self._exact_descriptor_by_source.get(descriptor_key)
            if existing_descriptor is not None and existing_descriptor != exact_descriptor:
                raise IllegalTransitionError("exact row descriptor changed after creation")
            if self._exact_row_publisher is not None:
                control_request_digest = self._exact_row_publisher.publish_exact_descriptor(
                    exact_descriptor
                )
                if not isinstance(control_request_digest, str) or not control_request_digest:
                    raise BackendContractError(
                        "exact row publisher returned an invalid request digest"
                    )
                existing_request = self._control_request_digest_by_source.get(
                    descriptor_key
                )
                if existing_request is not None and existing_request != control_request_digest:
                    raise IllegalTransitionError(
                        "ControlPlane request digest changed after publication"
                    )

        self._source_descriptor_ready_at[key] = at_ns
        self.on_source_payload_ready(
            phase_key=phase_key, src_rank=rank_id, at_ns=at_ns
        )
        if exact_descriptor is not None:
            self._exact_descriptor_by_source[descriptor_key] = exact_descriptor
        if control_request_digest is not None:
            self._control_request_digest_by_source[descriptor_key] = control_request_digest
        common_payload = {
            "phase_key": phase_key,
            "rank_id": rank_id,
            "origin": origin,
            "exact_row_descriptor": exact_descriptor,
            "formal_descriptor_contract": exact_descriptor is not None,
            "control_plane_published": control_request_digest is not None,
            "control_request_digest": control_request_digest,
        }
        self.observer.emit(
            kind="SOURCE_DESCRIPTOR_READY", at_ns=at_ns, payload=common_payload
        )
        self.observer.emit(
            kind="DISPATCH_HOOK_READY", at_ns=at_ns, payload=common_payload
        )
        return common_payload

    def _publish_dispatch_thread_ready_only(
        self, *, phase_key: Any, rank_id: int, at_ns: int, origin: str
    ) -> None:
        self.mark_dispatch_model_thread_ready(
            phase_key=phase_key, dst_rank=rank_id, at_ns=at_ns
        )
        self._rank_actors[rank_id].transition(
            state=RankState.WAIT_DISPATCH,
            phase_key=phase_key,
            at_ns=at_ns,
            reason="LOCAL_PATH_COMPLETE",
        )
        self.observer.emit(
            kind="LOCAL_PATH_COMPLETE",
            at_ns=at_ns,
            payload={
                "phase_key": phase_key,
                "rank_id": rank_id,
                "origin": origin,
                },
        )

    def _publish_dispatch_source_and_thread_ready(
        self, *, phase_key: Any, rank_id: int, at_ns: int, origin: str
    ) -> None:
        self._validate_rank(rank_id)
        at_ns = require_time_ns(at_ns, field="local_path_complete.at_ns")
        if self.phase_semantics.phase_kind(phase_key) != "DISPATCH":
            raise BackendContractError("local path must publish a Dispatch phase")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        key = (phase_stable_key, rank_id)
        existing = self._source_descriptor_ready_at.get(key)
        if existing is not None:
            if existing != at_ns:
                raise DuplicateRegistrationError(
                    "source descriptor readiness changed after publication"
                )
            return
        truth = self._dispatch_row_truth.get((phase_stable_key, rank_id))
        exact_descriptor = None
        control_request_digest = None
        descriptor_key = (phase_stable_key, rank_id)
        if truth is not None:
            exact_descriptor = make_exact_row_descriptor(
                phase_key=phase_key,
                src_rank=rank_id,
                realized_rows_by_destination=truth.realized_rows_by_destination,
                payload_bytes_by_destination=truth.dispatch_payload_bytes_by_destination,
                payload_spec_digest=truth.dispatch_payload_spec_digest,
                published_at_ns=at_ns,
                descriptor_payload_bytes=truth.descriptor_payload_bytes,
            )
            existing_descriptor = self._exact_descriptor_by_source.get(descriptor_key)
            if existing_descriptor is not None and existing_descriptor != exact_descriptor:
                raise IllegalTransitionError("exact row descriptor changed after creation")
            if self._exact_row_publisher is not None:
                # Publish before mutating source/thread state so a rejected transport
                # publication cannot leave a half-visible local-path completion.
                control_request_digest = self._exact_row_publisher.publish_exact_descriptor(
                    exact_descriptor
                )
                if not isinstance(control_request_digest, str) or not control_request_digest:
                    raise BackendContractError(
                        "exact row publisher returned an invalid request digest"
                    )
                existing_request = self._control_request_digest_by_source.get(
                    descriptor_key
                )
                if existing_request is not None and existing_request != control_request_digest:
                    raise IllegalTransitionError(
                        "ControlPlane request digest changed after publication"
                    )

        self._source_descriptor_ready_at[key] = at_ns
        self.on_source_payload_ready(
            phase_key=phase_key, src_rank=rank_id, at_ns=at_ns
        )
        self.mark_dispatch_model_thread_ready(
            phase_key=phase_key, dst_rank=rank_id, at_ns=at_ns
        )
        self._rank_actors[rank_id].transition(
            state=RankState.WAIT_DISPATCH,
            phase_key=phase_key,
            at_ns=at_ns,
            reason="LOCAL_PATH_COMPLETE",
        )
        if exact_descriptor is not None:
            self._exact_descriptor_by_source[descriptor_key] = exact_descriptor
        if control_request_digest is not None:
            self._control_request_digest_by_source[
                descriptor_key
            ] = control_request_digest
        common_payload = {
            "phase_key": phase_key,
            "rank_id": rank_id,
            "origin": origin,
            "exact_row_descriptor": exact_descriptor,
            "formal_descriptor_contract": exact_descriptor is not None,
            "control_plane_published": control_request_digest is not None,
            "control_request_digest": control_request_digest,
        }
        self.observer.emit(
            kind="SOURCE_DESCRIPTOR_READY", at_ns=at_ns, payload=common_payload
        )
        self.observer.emit(
            kind="LOCAL_PATH_COMPLETE", at_ns=at_ns, payload=common_payload
        )
        self.observer.emit(
            kind="DISPATCH_HOOK_READY", at_ns=at_ns, payload=common_payload
        )

    # ------------------------------------------------------------------
    # Destination Dispatch release and dynamic Combine hook
    # ------------------------------------------------------------------
    def _evaluate_dispatch_destination(
        self, *, phase_key: Any, dst_rank: int, now_ns: int
    ) -> None:
        state = self._dispatch_state(phase_key=phase_key, dst_rank=dst_rank)
        complete, assembled_at = self.receiver.all_nonzero_inbound_assembled(
            phase_key=phase_key, dst_rank=dst_rank
        )
        if complete and assembled_at is not None:
            if state.all_inbound_assembled_at_ns is None:
                state.all_inbound_assembled_at_ns = assembled_at
            elif state.all_inbound_assembled_at_ns != assembled_at:
                raise IllegalTransitionError("dispatch inbound completion changed")
        if (
            state.descriptor_closure_at_ns is None
            or state.all_inbound_assembled_at_ns is None
            or state.model_thread_ready_at_ns is None
            or state.postprocess_scheduled
        ):
            return
        spec = self._dispatch_compute_specs.get(
            (self.phase_semantics.phase_sort_key(phase_key), dst_rank)
        )
        if spec is None:
            raise BackendContractError(
                "missing DispatchComputeSpec for releasable destination"
            )
        inbound_ready = max(
            state.descriptor_closure_at_ns, state.all_inbound_assembled_at_ns
        )
        start = max(state.model_thread_ready_at_ns, inbound_ready)
        closure_baseline = max(
            state.model_thread_ready_at_ns, state.all_inbound_assembled_at_ns
        )
        state.closure_wait_ns = max(0, state.descriptor_closure_at_ns - closure_baseline)
        self._dispatch_closure_wait_by_rank[dst_rank] += state.closure_wait_ns
        completion = start + spec.dispatch_local_postprocess_ns
        state.postprocess_start_at_ns = start
        state.compute_ready_at_ns = completion
        state.postprocess_scheduled = True
        self.observer.emit(
            kind="DISPATCH_POSTPROCESS_STARTED",
            at_ns=start,
            payload={
                "phase_key": phase_key,
                "dst_rank": dst_rank,
                "descriptor_closure_at_ns": state.descriptor_closure_at_ns,
                "all_inbound_assembled_at_ns": state.all_inbound_assembled_at_ns,
                "model_thread_ready_at_ns": state.model_thread_ready_at_ns,
            },
        )
        self.kernel.schedule_backend_event(
            time_ns=completion,
            phase_priority=BACKEND_PHASE_PRIORITY,
            stable_event_id=stable_semantic_event_id(
                event_kind=_EVENT_DISPATCH_POSTPROCESS_COMPLETE,
                time_ns=completion,
                semantic_parts=[
                    self.phase_semantics.phase_sort_key(phase_key), dst_rank
                ],
            ),
            event_kind=_EVENT_DISPATCH_POSTPROCESS_COMPLETE,
            payload={"phase_key": phase_key, "rank_id": dst_rank},
        )

    def _on_dispatch_postprocess_complete(
        self, *, phase_key: Any, rank_id: int, at_ns: int
    ) -> None:
        state = self._dispatch_state(phase_key=phase_key, dst_rank=rank_id)
        if state.compute_ready_at_ns != at_ns:
            raise IllegalTransitionError("dispatch postprocess timestamp mismatch")
        if state.final_assembly_released:
            raise IllegalTransitionError("dispatch postprocess completed twice")
        self.receiver.release_final_assembly(
            phase_key=phase_key, dst_rank=rank_id, at_ns=at_ns
        )
        state.final_assembly_released = True
        self.observer.emit(
            kind="DESTINATION_COMPUTE_READY",
            at_ns=at_ns,
            payload={"phase_key": phase_key, "rank_id": rank_id},
        )
        if self.release_mode == "PHASE_BARRIER":
            self._rank_actors[rank_id].transition(
                state=RankState.WAIT_PHASE_BARRIER,
                phase_key=phase_key,
                at_ns=at_ns,
                reason="DESTINATION_COMPUTE_READY_WAIT_PHASE_BARRIER",
            )
        if self.release_mode == "RANK_LOCAL":
            self._release_dispatch_rank(
                phase_key=phase_key, rank_id=rank_id, release_at_ns=at_ns
            )
        else:
            self._maybe_release_dispatch_phase_barrier(
                phase_key=phase_key, now_ns=at_ns
            )

    def _maybe_release_dispatch_phase_barrier(
        self, *, phase_key: Any, now_ns: int
    ) -> None:
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        if phase_stable_key in self._dispatch_barrier_released:
            return
        states = [
            self._dispatch_state(phase_key=phase_key, dst_rank=rank)
            for rank in range(self.world_size)
        ]
        if any(
            state.compute_ready_at_ns is None or not state.final_assembly_released
            for state in states
        ):
            return
        barrier_at = max(state.compute_ready_at_ns or 0 for state in states)
        if barrier_at > now_ns:
            return
        self._dispatch_barrier_released.add(phase_stable_key)
        self.observer.emit(
            kind="DISPATCH_PHASE_BARRIER_RELEASE",
            at_ns=barrier_at,
            payload={"phase_key": phase_key, "world_size": self.world_size},
        )
        for rank_id in range(self.world_size):
            self._release_dispatch_rank(
                phase_key=phase_key, rank_id=rank_id, release_at_ns=barrier_at
            )

    def _release_dispatch_rank(
        self, *, phase_key: Any, rank_id: int, release_at_ns: int
    ) -> None:
        state = self._dispatch_state(phase_key=phase_key, dst_rank=rank_id)
        if state.released:
            raise IllegalTransitionError("rank already released for dispatch phase")
        if not state.final_assembly_released:
            raise IllegalTransitionError(
                "rank release requires completed dispatch local postprocess"
            )
        release_key = (self.phase_semantics.phase_sort_key(phase_key), rank_id)
        spec = self._dispatch_compute_specs[release_key]
        state.released = True
        self._rank_release_at[release_key] = release_at_ns
        self._record_source_local_path_start(
            phase_key=spec.next_combine_phase_key,
            rank_id=rank_id,
            at_ns=release_at_ns,
            origin="DISPATCH_RELEASE_TO_EXPERT_COMPUTE",
        )
        # Dispatch release starts this rank's source-side expert work.  The
        # destination-side rank progression may already be further ahead under
        # rank-local asynchronous execution, so the single diagnostic actor
        # must never be moved backward from POST_COMBINE/ROUTER/DONE.
        actor = self._rank_actors[rank_id]
        if actor.state in {RankState.WAIT_DISPATCH, RankState.EXPERT_COMPUTE}:
            actor.transition(
                state=RankState.EXPERT_COMPUTE,
                phase_key=phase_key,
                at_ns=release_at_ns,
                reason=(
                    "DESTINATION_COMPUTE_READY"
                    if self.release_mode == "RANK_LOCAL"
                    else "DISPATCH_PHASE_BARRIER_RELEASE"
                ),
            )
        self.observer.emit(
            kind="BACKEND_RANK_RELEASED",
            at_ns=release_at_ns,
            payload={"phase_key": phase_key, "rank_id": rank_id},
        )
        compute_complete_at = (
            release_at_ns + spec.dispatch_release_to_combine_source_ready_ns
        )
        if not self.p0_p1_compute_end_barrier:
            phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
            key = (phase_stable_key, rank_id)
            existing = self._p0_p1_compute_complete_at.get(key)
            if existing is not None and existing != compute_complete_at:
                raise DuplicateRegistrationError(
                    "P0-P1 local compute completion time changed"
                )
            if existing is None:
                self._p0_p1_compute_complete_at[key] = compute_complete_at
                self.observer.emit(
                    kind="P0_P1_LOCAL_COMPUTE_COMPLETE",
                    at_ns=compute_complete_at,
                    payload={
                        "dispatch_phase_key": phase_key,
                        "combine_phase_key": spec.next_combine_phase_key,
                        "rank_id": rank_id,
                    },
                )
            self._schedule_combine_source_ready(
                combine_phase_key=spec.next_combine_phase_key,
                rank_id=rank_id,
                at_ns=compute_complete_at,
            )
            return
        self.kernel.schedule_backend_event(
            time_ns=compute_complete_at,
            phase_priority=BACKEND_PHASE_PRIORITY,
            stable_event_id=stable_semantic_event_id(
                event_kind=_EVENT_P0_P1_LOCAL_COMPUTE_COMPLETE,
                time_ns=compute_complete_at,
                semantic_parts=[
                    self.phase_semantics.phase_sort_key(phase_key),
                    rank_id,
                ],
            ),
            event_kind=_EVENT_P0_P1_LOCAL_COMPUTE_COMPLETE,
            payload={
                "dispatch_phase_key": phase_key,
                "combine_phase_key": spec.next_combine_phase_key,
                "rank_id": rank_id,
            },
        )

    def _schedule_combine_source_ready(
        self, *, combine_phase_key: Any, rank_id: int, at_ns: int
    ) -> None:
        self.kernel.schedule_backend_event(
            time_ns=at_ns,
            phase_priority=BACKEND_PHASE_PRIORITY,
            stable_event_id=stable_semantic_event_id(
                event_kind=_EVENT_COMBINE_SOURCE_READY,
                time_ns=at_ns,
                semantic_parts=[
                    self.phase_semantics.phase_sort_key(combine_phase_key),
                    rank_id,
                ],
            ),
            event_kind=_EVENT_COMBINE_SOURCE_READY,
            payload={"phase_key": combine_phase_key, "rank_id": rank_id},
        )

    def _on_p0_p1_local_compute_complete(
        self,
        *,
        dispatch_phase_key: Any,
        combine_phase_key: Any,
        rank_id: int,
        at_ns: int,
    ) -> None:
        if not self.p0_p1_compute_end_barrier:
            raise BackendContractError(
                "P0-P1 local-compute barrier event requires p0_p1_compute_end_barrier"
            )
        if self.phase_semantics.phase_kind(dispatch_phase_key) != "DISPATCH":
            raise BackendContractError("compute barrier source must be DISPATCH")
        if self.phase_semantics.phase_kind(combine_phase_key) != "COMBINE":
            raise BackendContractError("compute barrier target must be COMBINE")
        phase_stable_key = self.phase_semantics.phase_sort_key(dispatch_phase_key)
        key = (phase_stable_key, rank_id)
        existing = self._p0_p1_compute_complete_at.get(key)
        if existing is not None:
            if existing != at_ns:
                raise DuplicateRegistrationError(
                    "P0-P1 local compute completion time changed"
                )
            return
        self._p0_p1_compute_complete_at[key] = at_ns
        self.observer.emit(
            kind="P0_P1_LOCAL_COMPUTE_COMPLETE",
            at_ns=at_ns,
            payload={
                "dispatch_phase_key": dispatch_phase_key,
                "combine_phase_key": combine_phase_key,
                "rank_id": rank_id,
            },
        )
        if phase_stable_key in self._p0_p1_compute_barrier_opened:
            return
        completions = [
            self._p0_p1_compute_complete_at.get((phase_stable_key, rank))
            for rank in range(self.world_size)
        ]
        if any(value is None for value in completions):
            return
        barrier_at = max(int(value) for value in completions if value is not None)
        if barrier_at > at_ns:
            return
        self._p0_p1_compute_barrier_opened.add(phase_stable_key)
        self._p0_p1_compute_barrier_release_at[phase_stable_key] = barrier_at
        self.observer.emit(
            kind="P0_P1_COMPUTE_END_BARRIER_RELEASE",
            at_ns=barrier_at,
            payload={
                "dispatch_phase_key": dispatch_phase_key,
                "combine_phase_key": combine_phase_key,
                "world_size": self.world_size,
            },
        )
        for rank in range(self.world_size):
            self._schedule_combine_source_ready(
                combine_phase_key=combine_phase_key,
                rank_id=rank,
                at_ns=barrier_at,
            )

    def p0_p1_compute_barrier_release_at(
        self, *, dispatch_phase_key: Any
    ) -> int | None:
        return self._p0_p1_compute_barrier_release_at.get(
            self.phase_semantics.phase_sort_key(dispatch_phase_key)
        )

    def p0_p1_compute_barrier_metrics(
        self, *, dispatch_phase_key: Any
    ) -> dict[str, Any]:
        """Return rank-local completion and barrier wait attribution.

        Local completion times are recorded in both barrier and rank-local
        modes.  When the barrier is disabled, release/wait fields are ``None``
        and zero respectively so the ablation remains directly comparable.
        """
        phase_stable_key = self.phase_semantics.phase_sort_key(dispatch_phase_key)
        local_times = tuple(
            self._p0_p1_compute_complete_at.get((phase_stable_key, rank))
            for rank in range(self.world_size)
        )
        release_at = self._p0_p1_compute_barrier_release_at.get(phase_stable_key)
        waits = tuple(
            0 if release_at is None or local_at is None else int(release_at) - int(local_at)
            for local_at in local_times
        )
        return {
            "enabled": bool(self.p0_p1_compute_end_barrier),
            "local_complete_times_ns": local_times,
            "barrier_release_ns": release_at,
            "barrier_wait_ns_by_rank": waits,
            "barrier_wait_ns_sum": sum(waits),
            "barrier_wait_ns_max": max(waits, default=0),
        }

    def _on_combine_source_ready(
        self, *, phase_key: Any, rank_id: int, at_ns: int
    ) -> None:
        if self.phase_semantics.phase_kind(phase_key) != "COMBINE":
            raise BackendContractError("expert compute must produce Combine source")
        self.on_source_payload_ready(
            phase_key=phase_key, src_rank=rank_id, at_ns=at_ns
        )
        # Source-side expert completion and destination-side Combine progress
        # are independent.  On sufficiently asynchronous/rank-local runs a
        # rank may already have completed its destination Combine/local path
        # before its own outgoing Combine payload becomes ready.  Do not let a
        # late source-ready event regress the single diagnostic RankActor from
        # ROUTER_AND_PACK/DONE back to WAIT_COMBINE.
        actor = self._rank_actors[rank_id]
        if actor.state in {RankState.EXPERT_COMPUTE, RankState.WAIT_COMBINE}:
            actor.transition(
                state=RankState.WAIT_COMBINE,
                phase_key=phase_key,
                at_ns=at_ns,
                reason="COMBINE_SOURCE_READY",
            )
        self.observer.emit(
            kind="COMBINE_HOOK_READY",
            at_ns=at_ns,
            payload={"phase_key": phase_key, "rank_id": rank_id},
        )

    # ------------------------------------------------------------------
    # Inspection/reporting
    # ------------------------------------------------------------------
    def rank_state(self, rank_id: int) -> RankState:
        self._validate_rank(rank_id)
        return self._rank_actors[rank_id].state

    def rank_actor_snapshot(self, rank_id: int) -> dict[str, Any]:
        self._validate_rank(rank_id)
        return self._rank_actors[rank_id].snapshot()

    def rank_release_at(self, *, phase_key: Any, rank_id: int) -> int | None:
        """Return the authoritative rank-local release/completion time."""

        self._validate_rank(rank_id)
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        if self.phase_semantics.phase_kind(phase_key) == "DISPATCH":
            return self._rank_release_at.get((phase_stable_key, rank_id))
        state = self._combine_destinations.get((phase_stable_key, rank_id))
        return None if state is None else state.local_path_complete_at_ns

    def phase_close_at(self, *, phase_key: Any) -> int | None:
        """Return the final rank lifecycle close time for one phase."""

        release_times = tuple(
            self.rank_release_at(phase_key=phase_key, rank_id=rank)
            for rank in range(self.world_size)
        )
        if any(value is None for value in release_times):
            return None
        return max(int(value) for value in release_times if value is not None)

    def rank_release_times(self, *, phase_key: Any) -> Mapping[int, int | None]:
        return {
            rank: self.rank_release_at(phase_key=phase_key, rank_id=rank)
            for rank in range(self.world_size)
        }

    def exact_row_descriptor(
        self, *, phase_key: Any, src_rank: int
    ) -> ExactRowDescriptor | None:
        """Return the immutable descriptor created at source local-path completion."""
        self._validate_rank(src_rank)
        return self._exact_descriptor_by_source.get(
            (self.phase_semantics.phase_sort_key(phase_key), src_rank)
        )

    def control_request_digest(
        self, *, phase_key: Any, src_rank: int
    ) -> str | None:
        """Return the transport ControlPlane request digest for a published exact row."""
        self._validate_rank(src_rank)
        return self._control_request_digest_by_source.get(
            (self.phase_semantics.phase_sort_key(phase_key), src_rank)
        )

    def dispatch_destination_snapshot(
        self, *, phase_key: Any, dst_rank: int
    ) -> dict[str, Any]:
        state = self._dispatch_state(phase_key=phase_key, dst_rank=dst_rank)
        return {
            "descriptor_sources_delivered": dict(
                sorted(state.descriptor_sources_delivered.items())
            ),
            "descriptor_closure_at_ns": state.descriptor_closure_at_ns,
            "model_thread_ready_at_ns": state.model_thread_ready_at_ns,
            "all_inbound_assembled_at_ns": state.all_inbound_assembled_at_ns,
            "postprocess_start_at_ns": state.postprocess_start_at_ns,
            "compute_ready_at_ns": state.compute_ready_at_ns,
            "closure_wait_ns": state.closure_wait_ns,
            "released": state.released,
        }

    def combine_destination_snapshot(
        self, *, phase_key: Any, dst_rank: int
    ) -> dict[str, Any]:
        state = self._combine_state(phase_key=phase_key, dst_rank=dst_rank)
        return {
            "data_ready_at_ns": state.data_ready_at_ns,
            "local_path_start_at_ns": state.local_path_start_at_ns,
            "post_combine_complete_at_ns": state.post_combine_complete_at_ns,
            "local_path_complete_at_ns": state.local_path_complete_at_ns,
        }

    def metrics_snapshot(self) -> ReceiverMetricsSnapshot:
        """Return cross-phase aggregate receiver metrics per rank."""

        return snapshot_memory(
            self.receiver.memory_by_rank,
            dispatch_closure_wait_ns=self._dispatch_closure_wait_by_rank,
        )

    def phase_rank_metrics_snapshot(
        self, *, phase_key: Any, rank_id: int
    ) -> PhaseRankMetricsSnapshot:
        """Return one authoritative per-phase/per-rank attribution row.

        ``closure_wait_ns`` and ``data_wait_ns`` are destination release gates.
        Receiver service, credit, drain and memory fields are attributed from
        the task/assembly records owned by that same phase and destination.
        """

        self._validate_rank(rank_id)
        receiver = self.receiver.phase_rank_receiver_metrics(
            phase_key=phase_key, dst_rank=rank_id
        )
        kind = self.phase_semantics.phase_kind(phase_key)
        closure_wait_ns = 0
        data_wait_ns = 0
        if kind == "DISPATCH":
            state = self._dispatch_state(
                phase_key=phase_key, dst_rank=rank_id
            )
            closure_wait_ns = int(state.closure_wait_ns)
            if (
                state.all_inbound_assembled_at_ns is not None
                and state.descriptor_closure_at_ns is not None
                and state.model_thread_ready_at_ns is not None
            ):
                data_wait_ns = max(
                    0,
                    int(state.all_inbound_assembled_at_ns)
                    - max(
                        int(state.descriptor_closure_at_ns),
                        int(state.model_thread_ready_at_ns),
                    ),
                )
        else:
            state = self._combine_state(
                phase_key=phase_key, dst_rank=rank_id
            )
            closure_at = self.receiver.expectation_closure_at_ns(
                phase_key=phase_key, dst_rank=rank_id
            )
            if state.data_ready_at_ns is not None and closure_at is not None:
                data_wait_ns = max(
                    0, int(state.data_ready_at_ns) - int(closure_at)
                )
        return PhaseRankMetricsSnapshot(
            phase_key=phase_key,
            rank_id=rank_id,
            closure_wait_ns=closure_wait_ns,
            data_wait_ns=data_wait_ns,
            receiver_posting_service_ns=int(receiver["receiver_posting_service_ns"]),
            receiver_posting_queue_wait_ns=int(receiver["receiver_posting_queue_wait_ns"]),
            receiver_buffer_stall_ns=int(receiver["receiver_buffer_stall_ns"]),
            receiver_drain_queue_wait_ns=int(receiver["receiver_drain_queue_wait_ns"]),
            receiver_drain_service_ns=int(receiver["receiver_drain_service_ns"]),
            peak_staging_bytes=int(receiver["peak_staging_bytes"]),
            peak_final_assembly_bytes=int(
                receiver["peak_final_assembly_bytes"]
            ),
            peak_total_receiver_bytes=int(
                receiver["peak_total_receiver_bytes"]
            ),
            rank_release_at_ns=self.rank_release_at(
                phase_key=phase_key, rank_id=rank_id
            ),
            phase_close_at_ns=self.phase_close_at(phase_key=phase_key),
        )

    def phase_metrics_matrix(
        self, *, phase_keys: Sequence[Any]
    ) -> tuple[PhaseRankMetricsSnapshot, ...]:
        """Return stable phase-major/rank-minor Backend metrics."""

        rows: list[PhaseRankMetricsSnapshot] = []
        for phase_key in phase_keys:
            for rank_id in range(self.world_size):
                rows.append(
                    self.phase_rank_metrics_snapshot(
                        phase_key=phase_key, rank_id=rank_id
                    )
                )
        return tuple(rows)

    def phase_terminal_snapshot(self, *, phase_key: Any) -> dict[str, Any]:
        """Return closure, receiver, release and lifecycle terminal evidence."""

        receiver = self.receiver.phase_terminal_snapshot(phase_key=phase_key)
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        kind = self.phase_semantics.phase_kind(phase_key)
        rank_release_at_ns = self.rank_release_times(phase_key=phase_key)
        released = tuple(
            rank
            for rank, at_ns in rank_release_at_ns.items()
            if at_ns is not None
        )
        all_ranks_released = len(released) == self.world_size
        if kind == "DISPATCH":
            lifecycle_closed = all(
                self._dispatch_destinations.get((phase_stable_key, rank))
                is not None
                and self._dispatch_destinations[
                    (phase_stable_key, rank)
                ].released
                for rank in range(self.world_size)
            )
        else:
            lifecycle_closed = all(
                self._combine_destinations.get((phase_stable_key, rank))
                is not None
                and self._combine_destinations[
                    (phase_stable_key, rank)
                ].local_path_complete_at_ns
                is not None
                for rank in range(self.world_size)
            )
        closure_summary = self.phase_closure_summary(phase_key=phase_key)
        memory_reconciliation = self.receiver.metrics_reconciliation_snapshot()
        phase_close_at_ns = self.phase_close_at(phase_key=phase_key)
        snapshot = {
            **receiver,
            "closure_summary": closure_summary,
            "closure_digest": (
                None if closure_summary is None else closure_summary.closure_digest
            ),
            "closure_generation": (
                None
                if closure_summary is None
                else closure_summary.closure_generation
            ),
            "seal_ready": bool(
                closure_summary is not None and closure_summary.seal_ready
            ),
            "released_ranks": released,
            "rank_release_at_ns": rank_release_at_ns,
            "all_ranks_released": all_ranks_released,
            "phase_close_at_ns": phase_close_at_ns,
            "lifecycle_closed": lifecycle_closed,
            "memory_metrics_reconciled": bool(
                memory_reconciliation["reconciled"]
            ),
            "memory_reconciliation": memory_reconciliation,
        }
        snapshot["closed"] = bool(
            receiver["closed"]
            and closure_summary is not None
            and closure_summary.seal_ready
            and all_ranks_released
            and lifecycle_closed
            and phase_close_at_ns is not None
            and memory_reconciliation["reconciled"]
        )
        return snapshot

    def window_terminal_snapshot(
        self,
        *,
        phase_keys: Sequence[Any],
        require_ranks_done: bool = False,
    ) -> dict[str, Any]:
        """Return a stable terminal audit across a phase window/run slice."""

        ordered_phase_keys = tuple(phase_keys)
        if not ordered_phase_keys:
            raise BackendContractError("phase_keys must be non-empty")
        phase_states = tuple(
            self.phase_terminal_snapshot(phase_key=phase_key)
            for phase_key in ordered_phase_keys
        )
        memory_by_rank = {
            rank: self.receiver.current_memory(rank)
            for rank in range(self.world_size)
        }
        receiver_memory_zero = all(
            int(row["total_receiver_bytes"]) == 0
            for row in memory_by_rank.values()
        )
        rank_states = {
            rank: self.rank_state(rank).value
            for rank in range(self.world_size)
        }
        ranks_done = all(state == "DONE" for state in rank_states.values())
        close_times = tuple(
            int(state["phase_close_at_ns"])
            for state in phase_states
            if state["phase_close_at_ns"] is not None
        )
        memory_reconciliation = self.receiver.metrics_reconciliation_snapshot()
        closed = bool(
            all(state["closed"] for state in phase_states)
            and receiver_memory_zero
            and memory_reconciliation["reconciled"]
            and (ranks_done or not require_ranks_done)
        )
        return {
            "phase_states": phase_states,
            "phase_count": len(phase_states),
            "window_close_at_ns": max(close_times) if close_times else None,
            "receiver_memory_by_rank": memory_by_rank,
            "receiver_memory_zero": receiver_memory_zero,
            "memory_reconciliation": memory_reconciliation,
            "rank_states": rank_states,
            "ranks_done": ranks_done,
            "require_ranks_done": bool(require_ranks_done),
            "closed": closed,
        }

    def assert_window_closed(
        self,
        *,
        phase_keys: Sequence[Any],
        require_ranks_done: bool = False,
    ) -> dict[str, Any]:
        snapshot = self.window_terminal_snapshot(
            phase_keys=phase_keys,
            require_ranks_done=require_ranks_done,
        )
        if not snapshot["closed"]:
            raise IllegalTransitionError(
                f"phase window is not closed: {snapshot}"
            )
        return snapshot

    def assert_phase_closed(self, *, phase_key: Any) -> dict[str, Any]:
        snapshot = self.phase_terminal_snapshot(phase_key=phase_key)
        if not snapshot["closed"]:
            raise IllegalTransitionError(
                f"phase is not closed: {snapshot}"
            )
        return snapshot

    def _dispatch_state(
        self, *, phase_key: Any, dst_rank: int
    ) -> DispatchDestinationState:
        self._validate_rank(dst_rank)
        if self.phase_semantics.phase_kind(phase_key) != "DISPATCH":
            raise BackendContractError("expected Dispatch phase")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        key = (phase_stable_key, dst_rank)
        state = self._dispatch_destinations.get(key)
        if state is None:
            state = DispatchDestinationState(
                phase_key=phase_key,
                phase_stable_key=phase_stable_key,
                dst_rank=dst_rank,
            )
            self._dispatch_destinations[key] = state
        return state

    def _combine_state(
        self, *, phase_key: Any, dst_rank: int
    ) -> CombineDestinationState:
        self._validate_rank(dst_rank)
        if self.phase_semantics.phase_kind(phase_key) != "COMBINE":
            raise BackendContractError("expected Combine phase")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        key = (phase_stable_key, dst_rank)
        state = self._combine_destinations.get(key)
        if state is None:
            state = CombineDestinationState(
                phase_key=phase_key,
                phase_stable_key=phase_stable_key,
                dst_rank=dst_rank,
            )
            self._combine_destinations[key] = state
        return state

    def _validate_rank(self, rank_id: int) -> None:
        if (
            not isinstance(rank_id, int)
            or isinstance(rank_id, bool)
            or not 0 <= rank_id < self.world_size
        ):
            raise BackendContractError(f"rank {rank_id!r} is outside world_size")

    @property
    def event_kinds(self) -> frozenset[str]:
        return self.receiver.event_kinds | frozenset(
            {
                _EVENT_POST_COMBINE_COMPLETE,
                _EVENT_LOCAL_PATH_COMPLETE,
                _EVENT_DISPATCH_POSTPROCESS_COMPLETE,
                _EVENT_COMBINE_SOURCE_READY,
                _EVENT_P0_P1_LOCAL_COMPUTE_COMPLETE,
                _EVENT_BACKEND_STABILIZE,
            }
        )
