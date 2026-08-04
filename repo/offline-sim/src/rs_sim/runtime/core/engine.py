from __future__ import annotations

"""Current-P12 runtime assembly.

This module wires the unique backend, scheduler, receiver, and transport path.
"""

import dataclasses
import enum
from dataclasses import dataclass
from typing import Any, Mapping

from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel

from rs_sim import LinkClass, SimulationKernel, WindowKey, make_network_topology, stable_digest
from rs_sim.contracts.paper_defaults import (
    PAPER_ALIGNMENT_BYTES, PAPER_MAX_TASK_BYTES,
    PAPER_P0_P1_COMPUTE_END_BARRIER, PAPER_RELEASE_MODE,
)
from rs_sim.backend import (
    AttributeSharedObjectAdapter,
    LinearReceiverCostModel,
    ReceiverService,
    SimulationBackend,
    compute_fixture_staging_capacity_bytes_by_rank,
)
from rs_sim.scheduler.prediction.timing import (
    P12RankTimingProfile,
    causal_last_observed_timing_estimate,
)
from rs_sim.scheduler import (
    CurrentP12Window,
    FormalRuntimeRecord,
    FormalSchedulingRuntimeAdapter,
    P12InformationMode,
    PairedInstanceKey,
    Provenance,
    RunStatus,
    LiveFairnessInputs,
    LivePolicySession,
    LivePolicySpec,
    PlanningCostModel,
    PlanningMode,
    PlanningTrigger,
    ReleaseMode as SchedulingReleaseMode,
    SchedulerWindow,
    TaskizationSpec,
    ThreeLineServices,
    build_current_p12_windows,
    build_p2_prediction,
    evaluate_p2_prediction,
    make_formal_runtime_record,
    normalize_p12_information_mode,
    parse_algorithm_expression,
)
from rs_sim.scheduler.planning.planner import PlannerScope
from rs_sim.transport.control.channel import FormalControlPlaneTransport
from rs_sim.transport.data.channel import FormalDataPlaneTransport

from ..adapters.backend import BackendRuntimeDriver, build_backend_runtime_driver
from ..metrics.communication import (
    compute_excluded_communication_makespan_ns,
    network_active_union_ns,
    summarize_rank_communication_exposure_ns,
)
from ..assembly.bindings import (
    SchemaEdgeKeyFactory,
    SchemaExpectationFactory,
    SchemaPermitFactory,
    SchedulingStack,
    build_scheduling_stack,
    make_phase_semantics,
)
from ..adapters.kernel import BackendKernelBridge
from ..adapters.live_scheduler import (
    CompositeFormalSchedulingRuntimeAdapter,
    CurrentP12TriggerRoute,
)
from ..adapters.public_ports import ReceiverPermitLookup, SharedTopologyTaskResolver
from ..adapters.scheduler import build_scheduler_port_bundle
from ..config.profiles import (
    RuntimeProfileBundle,
    make_default_synthetic_runtime_profile,
)


def _evidence_tree(value: Any) -> Any:
    """Convert runtime evidence to the immutable stable-JSON value domain."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise TypeError("float is not permitted in formal runtime evidence")
    if isinstance(value, enum.Enum):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "fields": {
                field.name: _evidence_tree(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {str(key): _evidence_tree(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return tuple(_evidence_tree(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_evidence_tree(item) for item in value), key=repr))
    return value


@dataclass(frozen=True, slots=True)
class CurrentP12WindowRecord:
    window_key: WindowKey
    anchor_layer_id: int
    p0_trigger_phase_key: Any
    p1_combine_phase_key: Any
    p2_dispatch_phase_key: Any
    information_mode: str
    prediction_digest: str
    prediction_quality_digest: str | None
    prediction_absolute_error_bytes: int | None
    prediction_relative_absolute_error_ppm: int | None
    prediction_matrix_overlap_ppm: int | None
    prediction_top_destination_accuracy_ppm: int | None
    window_start_ns: int
    window_end_ns: int
    window_makespan_ns: int
    network_transfer_span_ns: int
    compute_excluded_communication_makespan_ns: int
    network_active_union_ns: int
    rank_communication_exposed_p1_ns_by_rank: tuple[int, ...]
    rank_communication_exposed_p2_ns_by_rank: tuple[int, ...]
    rank_descriptor_stall_p2_ns_by_rank: tuple[int, ...]
    rank_data_stall_p2_ns_by_rank: tuple[int, ...]
    rank_communication_exposed_ns_by_rank: tuple[int, ...]
    rank_communication_exposed_ns_sum: int
    rank_communication_exposed_ns_mean: int
    rank_communication_exposed_ns_max: int
    rank_communication_exposed_ns_p95: int
    rank_communication_exposed_ns_p99: int
    rank_communication_critical_rank: int | None
    task_ids: tuple[str, ...]
    task_catalogue_digest: str
    task_boundary_digest: str
    truth_digest: str
    rank_release_times_ns: tuple[tuple[str, int, int], ...]
    p0_p1_local_complete_times_ns: tuple[int | None, ...]
    p0_p1_barrier_release_ns: int | None
    p0_p1_barrier_wait_ns_by_rank: tuple[int, ...]
    p0_p1_barrier_wait_ns_sum: int
    p0_p1_barrier_wait_ns_max: int
    physical_completed_bytes: int
    template_ready_margin_ns: int | None
    target_bind_wait_ns: int | None
    reconciliation_status: str | None
    safe_selector_choice: str | None
    safe_selector_reason: str | None
    safe_selector_local_objective: int | None
    safe_selector_joint_objective: int | None
    template_digest: str | None
    predicted_p2_slot_count: int
    bound_exact_p2_task_count: int
    unmatched_exact_p2_task_count: int
    exact_bind_count: int
    boundary_mismatch_bind_count: int
    overflow_bind_count: int
    unused_slot_count: int
    appended_task_count: int
    repair_task_count: int
    repair_task_bytes: int
    repair_task_ratio_ppm: int
    repair_byte_ratio_ppm: int
    binding_repair_reason: str | None
    prediction_fallback_reason: str | None
    prediction_generated: bool
    prediction_nonempty: bool
    prediction_validated: bool
    prediction_consumed: bool
    prediction_fallback: bool
    algorithm_core_run_count: int
    repair_count: int
    incremental_bind_job_count: int
    prediction_service_ns: int
    prediction_hidden_ns: int
    prediction_exposed_ns: int
    control_service_ns: int
    control_hidden_ns: int
    control_exposed_ns: int
    binding_service_ns: int
    binding_hidden_ns: int
    binding_exposed_ns: int
    terminal: bool
    record_digest: str


@dataclass(slots=True)
class FormalIntegrationRuntime:
    """Unique production-shape runtime for correctness and audit execution."""

    kernel: SimulationKernel
    kernel_bridge: BackendKernelBridge
    scheduling: SchedulingStack
    observer_bridge: Any
    backend_driver: BackendRuntimeDriver
    data_plane: FormalDataPlaneTransport
    control_plane: FormalControlPlaneTransport
    topology: Any
    hardware_profile: Any
    runtime_profile: RuntimeProfileBundle
    registration: Any
    run_id: str
    scheduler_runtime_mode: str = "CURRENT_P12"
    run_axes: Mapping[str, Any] | None = None
    fixture_input: Any | None = None
    current_p12_windows: tuple[CurrentP12Window, ...] = ()
    current_p12_predictions: Mapping[str, Any] | None = None

    @property
    def backend(self) -> SimulationBackend:
        return self.backend_driver.backend

    @property
    def receiver(self) -> ReceiverService:
        return self.backend_driver.receiver

    def run_to_completion(self, *, max_timestamps: int = 100_000) -> int:
        if not isinstance(max_timestamps, int) or isinstance(max_timestamps, bool) or max_timestamps <= 0:
            raise ValueError("max_timestamps must be a positive integer")
        timestamps = 0
        while self.kernel.has_events():
            self.kernel.run_next_timestamp()
            timestamps += 1
            if timestamps > max_timestamps:
                raise RuntimeError("formal runtime exceeded deterministic timestamp bound")
        return timestamps

    def terminal_state(self) -> dict[str, Any]:
        backend_state = self.backend_driver.terminal_state(self.registration)
        data_state = self.data_plane.terminal_state()
        control_state = self.control_plane.terminal_state()
        active_plans = tuple(
            phase_key
            for phase_key in self.registration.all_phase_keys
            if self.scheduling.authority.active_plan(phase_key) is not None
        )
        state = {
            "transport_transport": True,
            "transport_mode": "TRANSPORT_NO_STUB",
            "kernel_pending_events": int(self.kernel.pending_event_count()),
            "backend": backend_state,
            "data_plane": data_state,
            "control_plane": control_state,
            "active_plan_phase_keys": active_plans,
            "terminal": bool(
                backend_state["terminal"]
                and data_state["terminal"]
                and control_state["terminal"]
                and not active_plans
                and self.kernel.pending_event_count() == 0
            ),
        }
        return state

    def assert_terminal(self) -> dict[str, Any]:
        state = self.terminal_state()
        if not state["terminal"]:
            raise RuntimeError(f"formal integration runtime is not terminal: {state}")
        return state

    def current_p12_window_records(self) -> tuple[CurrentP12WindowRecord, ...]:
        if not self.current_p12_windows:
            return ()
        metrics = self.observer_bridge.metrics()
        evidence_by_window = {
            item.planning_window_digest: item
            for item in getattr(metrics, "current_p12_template_evidence", ())
        }
        predictions = dict(self.current_p12_predictions or {})
        records: list[CurrentP12WindowRecord] = []
        for window in self.current_p12_windows:
            evidence = evidence_by_window.get(window.planning_window_digest)
            if evidence is None or evidence.trigger_at_ns is None:
                continue
            phases = window.referenced_phase_keys
            closes = tuple(self.backend.phase_close_at(phase_key=item) for item in phases)
            if any(item is None for item in closes):
                terminal = False
                end_ns = max(int(evidence.trigger_at_ns), *(int(item or 0) for item in closes))
            else:
                terminal = True
                end_ns = max(int(item) for item in closes if item is not None)
            task_ids = tuple(
                sorted(
                    task_id
                    for phase_key in phases
                    for task_id in self.scheduling.catalogue.task_ids_for_phase(phase_key)
                )
            )
            physical = self.data_plane.physical_metrics(
                phase_keys=phases,
                window_task_ids=task_ids,
            )
            starts = tuple(int(item.start_at_ns) for item in physical.task_metrics)
            completes = tuple(int(item.complete_at_ns) for item in physical.task_metrics)
            network_transfer_span = (
                max(completes) - min(starts) if starts and completes else 0
            )
            ready_complete_intervals: list[tuple[int, int]] = []
            start_complete_intervals: list[tuple[int, int]] = []
            for task_metric in physical.task_metrics:
                facts = self.scheduling.runtime.facts(task_metric.task_id)
                # The communication-only clock starts when payload data exists,
                # not when the scheduler/receiver finally grants a permit.
                # Using ``ready_at_ns=max(payload, permit)`` would hide an
                # algorithm-dependent permit delay and unfairly reward a policy
                # that simply postpones admission.
                if facts.source_payload_ready_at_ns is None:
                    raise RuntimeError(
                        "formal communication metric requires source_payload_ready_at_ns "
                        f"for task {task_metric.task_id}"
                    )
                ready_at_ns = int(facts.source_payload_ready_at_ns)
                ready_at_ns = max(int(evidence.trigger_at_ns), ready_at_ns)
                complete_at_ns = int(task_metric.complete_at_ns)
                if complete_at_ns < ready_at_ns:
                    ready_at_ns = int(task_metric.start_at_ns)
                ready_complete_intervals.append((ready_at_ns, complete_at_ns))
                start_complete_intervals.append((
                    int(task_metric.start_at_ns),
                    complete_at_ns,
                ))
            compute_excluded_communication = compute_excluded_communication_makespan_ns(
                task_ready_complete_intervals=ready_complete_intervals
            )
            active_network_union = network_active_union_ns(
                task_start_complete_intervals=start_complete_intervals
            )

            p1_timing = self.backend.phase_causal_timing_observation(
                phase_key=window.p1_combine_phase_key
            )
            p1_exposed_by_rank: list[int] = []
            p2_exposed_by_rank: list[int] = []
            p2_descriptor_stall_by_rank: list[int] = []
            p2_data_stall_by_rank: list[int] = []
            for rank_id in range(int(self.fixture_input.world_size)):
                p1_source_ready = p1_timing.source_local_path_complete_at_ns_by_rank[rank_id]
                p1_data_ready = p1_timing.destination_compute_ready_at_ns_by_rank[rank_id]
                p1_exposed_by_rank.append(
                    0
                    if p1_source_ready is None or p1_data_ready is None
                    else max(0, int(p1_data_ready) - int(p1_source_ready))
                )
                dispatch_snapshot = self.backend.dispatch_destination_snapshot(
                    phase_key=window.p2_dispatch_phase_key,
                    dst_rank=rank_id,
                )
                inbound_ready = dispatch_snapshot["all_inbound_assembled_at_ns"]
                descriptor_ready = dispatch_snapshot["descriptor_closure_at_ns"]
                model_ready = dispatch_snapshot["model_thread_ready_at_ns"]
                if inbound_ready is None or descriptor_ready is None or model_ready is None:
                    p2_descriptor_stall_by_rank.append(0)
                    p2_data_stall_by_rank.append(0)
                    p2_exposed_by_rank.append(0)
                else:
                    descriptor_stall = max(0, int(descriptor_ready) - int(model_ready))
                    data_stall = max(
                        0, int(inbound_ready) - max(int(descriptor_ready), int(model_ready))
                    )
                    p2_descriptor_stall_by_rank.append(descriptor_stall)
                    p2_data_stall_by_rank.append(data_stall)
                    p2_exposed_by_rank.append(
                        max(0, int(inbound_ready) - int(model_ready))
                    )
            rank_exposure = summarize_rank_communication_exposure_ns(tuple(
                int(p1_exposed_by_rank[rank]) + int(p2_exposed_by_rank[rank])
                for rank in range(int(self.fixture_input.world_size))
            ))
            rank_releases = tuple(
                sorted(
                    (
                        stable_digest(phase_key),
                        int(rank),
                        int(at_ns),
                    )
                    for phase_key in phases
                    for rank in range(int(self.fixture_input.world_size))
                    for at_ns in (self.backend.rank_release_at(phase_key=phase_key, rank_id=rank),)
                    if at_ns is not None
                )
            )
            prediction = predictions.get(window.planning_window_digest)
            prediction_digest = (
                evidence.prediction_digest
                if evidence.prediction_digest is not None
                else getattr(prediction, "prediction_digest", "NO_P2_PREDICTION")
            )
            boundary_rows = []
            for role, phase_key in (("P1", window.p1_combine_phase_key), ("P2", window.p2_dispatch_phase_key)):
                for task_id in self.scheduling.catalogue.task_ids_for_phase(phase_key):
                    view = self.scheduling.catalogue.view(task_id)
                    boundary_rows.append((
                        role,
                        int(view.src_rank),
                        int(view.dst_rank),
                        int(view.chunk_index),
                        int(view.byte_offset),
                        int(view.payload_bytes),
                    ))
            boundary_rows = tuple(sorted(boundary_rows))
            current_trace = next(
                item for item in self.fixture_input.windows
                if int(item.layer_id) == int(window.anchor_layer_id)
            )
            next_trace = next(
                item for item in self.fixture_input.windows
                if int(item.layer_id) == int(window.anchor_layer_id) + 1
            )
            actual_p2_matrix = tuple(
                tuple(int(value) for value in row)
                for row in next_trace.payload_matrix("DISPATCH")
            )
            prediction_quality = None
            if prediction is not None:
                prediction_quality = evaluate_p2_prediction(
                    predicted_matrix=getattr(prediction, "matrix"),
                    actual_matrix=actual_p2_matrix,
                )
            truth_payload = {
                "schema_version": "CURRENT_P12_WINDOW_TRUTH",
                "fixture_id": str(self.fixture_input.fixture_id),
                "anchor_layer_id": int(window.anchor_layer_id),
                "p1_source_workload_digest": current_trace.workload_identity_digest(),
                "p2_source_workload_digest": next_trace.workload_identity_digest(),
                "task_boundaries": boundary_rows,
            }
            task_boundary_digest = stable_digest({
                "schema_version": "CURRENT_P12_TASK_BOUNDARIES",
                "task_boundaries": boundary_rows,
            })
            barrier_metrics = self.backend_driver.backend.p0_p1_compute_barrier_metrics(
                dispatch_phase_key=window.p0_trigger_phase_key
            )
            payload = {
                "window_key": window.window_key,
                "anchor_layer_id": int(window.anchor_layer_id),
                "p0_trigger_phase_key": window.p0_trigger_phase_key,
                "p1_combine_phase_key": window.p1_combine_phase_key,
                "p2_dispatch_phase_key": window.p2_dispatch_phase_key,
                "information_mode": str(evidence.information_mode),
                "prediction_digest": str(prediction_digest),
                "prediction_quality_digest": (
                    None if prediction_quality is None else prediction_quality.quality_digest
                ),
                "prediction_absolute_error_bytes": (
                    None if prediction_quality is None else int(prediction_quality.absolute_error_bytes)
                ),
                "prediction_relative_absolute_error_ppm": (
                    None if prediction_quality is None else int(prediction_quality.relative_absolute_error_ppm)
                ),
                "prediction_matrix_overlap_ppm": (
                    None if prediction_quality is None else int(prediction_quality.matrix_overlap_ppm)
                ),
                "prediction_top_destination_accuracy_ppm": (
                    None if prediction_quality is None else int(prediction_quality.top_destination_accuracy_ppm)
                ),
                "window_start_ns": int(evidence.trigger_at_ns),
                "window_end_ns": int(end_ns),
                "window_makespan_ns": int(end_ns) - int(evidence.trigger_at_ns),
                "network_transfer_span_ns": int(network_transfer_span),
                "compute_excluded_communication_makespan_ns": int(compute_excluded_communication),
                "network_active_union_ns": int(active_network_union),
                "rank_communication_exposed_p1_ns_by_rank": tuple(int(value) for value in p1_exposed_by_rank),
                "rank_communication_exposed_p2_ns_by_rank": tuple(int(value) for value in p2_exposed_by_rank),
                "rank_descriptor_stall_p2_ns_by_rank": tuple(
                    int(value) for value in p2_descriptor_stall_by_rank
                ),
                "rank_data_stall_p2_ns_by_rank": tuple(
                    int(value) for value in p2_data_stall_by_rank
                ),
                "rank_communication_exposed_ns_by_rank": rank_exposure.values_ns,
                "rank_communication_exposed_ns_sum": int(rank_exposure.total_ns),
                "rank_communication_exposed_ns_mean": int(rank_exposure.mean_ns),
                "rank_communication_exposed_ns_max": int(rank_exposure.max_ns),
                "rank_communication_exposed_ns_p95": int(rank_exposure.p95_ns),
                "rank_communication_exposed_ns_p99": int(rank_exposure.p99_ns),
                "rank_communication_critical_rank": rank_exposure.critical_rank,
                "task_ids": task_ids,
                "task_catalogue_digest": stable_digest({
                    "schema_version": "CURRENT_P12_CANONICAL_CATALOGUE",
                    "task_ids": task_ids,
                    "task_boundary_digest": task_boundary_digest,
                }),
                "task_boundary_digest": task_boundary_digest,
                "truth_digest": stable_digest(truth_payload),
                "rank_release_times_ns": rank_releases,
                "p0_p1_local_complete_times_ns": tuple(
                    barrier_metrics["local_complete_times_ns"]
                ),
                "p0_p1_barrier_release_ns": barrier_metrics["barrier_release_ns"],
                "p0_p1_barrier_wait_ns_by_rank": tuple(
                    int(value) for value in barrier_metrics["barrier_wait_ns_by_rank"]
                ),
                "p0_p1_barrier_wait_ns_sum": int(barrier_metrics["barrier_wait_ns_sum"]),
                "p0_p1_barrier_wait_ns_max": int(barrier_metrics["barrier_wait_ns_max"]),
                "physical_completed_bytes": int(physical.physical_completed_bytes),
                "template_ready_margin_ns": evidence.template_ready_margin_ns,
                "target_bind_wait_ns": evidence.target_bind_wait_ns,
                "reconciliation_status": evidence.reconciliation_status,
                "safe_selector_choice": evidence.safe_selector_choice,
                "safe_selector_reason": evidence.safe_selector_reason,
                "safe_selector_local_objective": evidence.safe_selector_local_objective,
                "safe_selector_joint_objective": evidence.safe_selector_joint_objective,
                "template_digest": evidence.template_digest,
                "predicted_p2_slot_count": int(evidence.predicted_p2_slot_count),
                "bound_exact_p2_task_count": int(evidence.bound_exact_p2_task_count),
                "unmatched_exact_p2_task_count": int(evidence.unmatched_exact_p2_task_count),
                "exact_bind_count": int(evidence.exact_bind_count),
                "boundary_mismatch_bind_count": int(evidence.boundary_mismatch_bind_count),
                "overflow_bind_count": int(evidence.overflow_bind_count),
                "unused_slot_count": int(evidence.unused_slot_count),
                "appended_task_count": int(evidence.appended_task_count),
                "repair_task_count": int(evidence.repair_task_count),
                "repair_task_bytes": int(evidence.repair_task_bytes),
                "repair_task_ratio_ppm": int(evidence.repair_task_ratio_ppm),
                "repair_byte_ratio_ppm": int(evidence.repair_byte_ratio_ppm),
                "binding_repair_reason": evidence.binding_repair_reason,
                "prediction_fallback_reason": evidence.prediction_fallback_reason,
                "prediction_generated": bool(evidence.prediction_generated),
                "prediction_nonempty": bool(evidence.prediction_nonempty),
                "prediction_validated": bool(evidence.prediction_validated),
                "prediction_consumed": bool(evidence.prediction_consumed),
                "prediction_fallback": bool(evidence.prediction_fallback),
                "algorithm_core_run_count": int(evidence.algorithm_core_run_count),
                "repair_count": int(evidence.repair_count),
                "incremental_bind_job_count": int(evidence.incremental_bind_job_count),
                "prediction_service_ns": int(evidence.prediction_service_ns),
                "prediction_hidden_ns": int(evidence.prediction_hidden_ns),
                "prediction_exposed_ns": int(evidence.prediction_exposed_ns),
                "control_service_ns": int(evidence.control_service_ns),
                "control_hidden_ns": int(evidence.control_hidden_ns),
                "control_exposed_ns": int(evidence.control_exposed_ns),
                "binding_service_ns": int(evidence.binding_service_ns),
                "binding_hidden_ns": int(evidence.binding_hidden_ns),
                "binding_exposed_ns": int(evidence.binding_exposed_ns),
                "terminal": bool(terminal and physical.terminal),
            }
            records.append(
                CurrentP12WindowRecord(
                    **payload,
                    record_digest=stable_digest(payload),
                )
            )
        return tuple(records)

    def formal_current_p12_records(self) -> tuple[FormalRuntimeRecord, ...]:
        """Build paper-facing anchor-local runtime records in nanoseconds.

        Algorithmic oracle records are intentionally not created here because
        their objective unit is logical payload rather than formal runtime time.
        """

        window_records = self.current_p12_window_records()
        if not window_records:
            return ()
        scheduler_metrics = self.observer_bridge.metrics()
        activation_evidence = tuple(getattr(scheduler_metrics, "activation_evidence", ()))
        phase_metrics = tuple(
            self.backend.phase_metrics_matrix(phase_keys=self.registration.all_phase_keys)
        )
        provenance_source = getattr(self.fixture_input, "provenance", None)
        trace_digest = (
            str(self.fixture_input.truth_digest())
            if hasattr(self.fixture_input, "truth_digest")
            else stable_digest(self.fixture_input)
        )
        provenance = Provenance(
            trace_source=str(getattr(provenance_source, "source_kind", "TRACE_FIXTURE")),
            trace_digest=trace_digest,
            payload_profile_source="PHASE_SPECIFIC_PAYLOAD_SPEC",
            compute_profile_source="TRACE_PURE_LOCAL_COMPUTE",
            hardware_profile_source=str(self.runtime_profile.profile_provenance),
            hardware_profile_digest=str(self.runtime_profile.profile_digest),
            calibration_state=str(self.runtime_profile.profile_kind),
            synthetic_components=(
                (
                    "data_plane_profile",
                    "control_plane_profile",
                    "receiver_service_profile",
                    "planning_service_profile",
                )
                if not self.runtime_profile.performance_eligible
                else ()
            ),
            performance_eligible=bool(self.runtime_profile.performance_eligible),
        )
        algorithm_core = str((self.run_axes or {}).get("algorithm_core", (self.run_axes or {}).get("core_id", "unknown")))
        result: list[FormalRuntimeRecord] = []
        for record in window_records:
            phases = (record.p1_combine_phase_key, record.p2_dispatch_phase_key)
            physical = self.data_plane.physical_metrics(
                phase_keys=phases,
                window_task_ids=record.task_ids,
            )
            phase_tokens = {stable_digest(item) for item in phases}
            activated_plan_count = sum(
                1
                for item in activation_evidence
                if getattr(item, "phase_token", None) in phase_tokens
            )
            # Current-P12 GLOBAL prepares one immutable P12 template before
            # phase activation.  That algorithm invocation is a formal plan and
            # must not disappear merely because the legacy counter only saw
            # phase-authority activations.  EVENT may report more than one core
            # run, so preserve the larger authoritative count.
            plan_count = max(
                int(activated_plan_count),
                int(record.algorithm_core_run_count),
            )
            release_by_rank: dict[int, int] = {}
            for _phase_digest, rank, at_ns in record.rank_release_times_ns:
                release_by_rank[int(rank)] = max(release_by_rank.get(int(rank), 0), int(at_ns))
            selected_backend_rows = tuple(
                item for item in phase_metrics if item.phase_key in phases
            )
            # Receiver stages are reported as mutually exclusive intervals:
            # posting queue, buffer HOL stall, posting service, drain queue,
            # and drain service.
            receiver_posting_service_ns = sum(
                int(item.receiver_posting_service_ns) for item in selected_backend_rows
            )
            receiver_posting_queue_wait_ns = sum(
                int(item.receiver_posting_queue_wait_ns) for item in selected_backend_rows
            )
            receiver_buffer_stall_ns = sum(
                int(item.receiver_buffer_stall_ns) for item in selected_backend_rows
            )
            receiver_drain_queue_wait_ns = sum(
                int(item.receiver_drain_queue_wait_ns) for item in selected_backend_rows
            )
            receiver_drain_service_ns = sum(
                int(item.receiver_drain_service_ns) for item in selected_backend_rows
            )
            receiver_total_delay_ns = (
                receiver_posting_queue_wait_ns
                + receiver_buffer_stall_ns
                + receiver_posting_service_ns
                + receiver_drain_queue_wait_ns
                + receiver_drain_service_ns
            )
            window_memory = self.backend.receiver.window_memory_peaks(phase_keys=phases)
            memory_by_rank = {
                int(rank): int(values["total"]) for rank, values in window_memory.items()
            }
            staging_by_rank = {
                int(rank): int(values["staging"]) for rank, values in window_memory.items()
            }
            final_assembly_by_rank = {
                int(rank): int(values["final_assembly"])
                for rank, values in window_memory.items()
            }
            # This is a wall-clock union, not a sum of overlapping task service
            # intervals.
            network_active_union_ns = int(record.network_active_union_ns)
            information_digest = stable_digest({
                "information_mode": record.information_mode,
                "prediction_digest": record.prediction_digest,
                "prediction_quality_digest": record.prediction_quality_digest,
            })
            cost_model_digest = stable_digest({
                "overlap_mode": (self.run_axes or {}).get("overlap_mode"),
                "execution_line": "CURRENT_P12",
                "planning_window": "P12",
                "p0_p1_compute_end_barrier": (self.run_axes or {}).get("p0_p1_compute_end_barrier", False),
                "runtime_profile_digest": self.runtime_profile.profile_digest,
            })
            paired_key = PairedInstanceKey(
                run_id=f"paired:{self.fixture_input.fixture_id}:P12:{record.anchor_layer_id}",
                sample_id=str(record.window_key.sample_id),
                window_index=int(record.window_key.window_index),
                rank_count=int(self.fixture_input.world_size),
                task_catalogue_digest=record.task_catalogue_digest,
                task_boundary_digest=record.task_boundary_digest,
                workload_digest=record.truth_digest,
                topology_digest=str(self.topology.topology_digest),
                hardware_profile_digest=str(self.hardware_profile.profile_digest),
                information_digest=information_digest,
                cost_model_digest=cost_model_digest,
                fixture_id=str(self.fixture_input.fixture_id),
                anchor_layer_id=int(record.anchor_layer_id),
                horizon="P12",
                window_truth_digest=record.truth_digest,
            )
            fairness_digest = stable_digest({
                "task_catalogue_digest": record.task_catalogue_digest,
                "task_boundary_digest": record.task_boundary_digest,
                "truth_digest": record.truth_digest,
                "topology_digest": self.topology.topology_digest,
                "hardware_profile_digest": self.hardware_profile.profile_digest,
                "runtime_profile_digest": self.runtime_profile.profile_digest,
                "release_mode": (self.run_axes or {}).get("release_mode"),
                "p0_p1_compute_end_barrier": (self.run_axes or {}).get("p0_p1_compute_end_barrier", False),
                "planner_scope": (self.run_axes or {}).get("planner_scope"),
                "execution_mode": "ORDER_ONLY",
            })
            result.append(make_formal_runtime_record(
                paired_key=paired_key,
                algorithm_id=algorithm_core,
                status=RunStatus.COMPLETED if record.terminal else RunStatus.FAILED,
                provenance=provenance,
                window_makespan_ns=int(record.window_makespan_ns),
                run_forward_makespan_ns=int(self.kernel.now_ns),
                network_transfer_span_ns=int(record.network_transfer_span_ns),
                rank_release_times_ns=tuple(sorted(release_by_rank.items())),
                control_exposed_ns=int(record.control_exposed_ns),
                prediction_exposed_ns=int(record.prediction_exposed_ns),
                receiver_total_delay_ns=int(receiver_total_delay_ns),
                network_active_union_ns=int(network_active_union_ns),
                memory_peak_bytes_by_rank=tuple(sorted(memory_by_rank.items())),
                plan_count=int(plan_count),
                completed_bytes=int(record.physical_completed_bytes),
                terminal_status="TERMINAL" if record.terminal else "NON_TERMINAL",
                fairness_digest=fairness_digest,
                physical_completion_digest=physical.metrics_digest,
                window_key=record.window_key,
                anchor_layer_id=int(record.anchor_layer_id),
                horizon="P12",
                window_start_ns=int(record.window_start_ns),
                window_end_ns=int(record.window_end_ns),
                window_task_ids=record.task_ids,
                window_task_catalogue_digest=record.task_catalogue_digest,
                window_truth_digest=record.truth_digest,
                is_truncated_tail=False,
                prediction_hidden_ns=int(record.prediction_hidden_ns),
                control_hidden_ns=int(record.control_hidden_ns),
                binding_hidden_ns=int(record.binding_hidden_ns),
                binding_exposed_ns=int(record.binding_exposed_ns),
                target_bind_wait_ns=int(record.target_bind_wait_ns or 0),
                template_ready_margin_ns=record.template_ready_margin_ns,
                reconciliation_status=record.reconciliation_status,
                prediction_quality_digest=record.prediction_quality_digest,
                prediction_absolute_error_bytes=record.prediction_absolute_error_bytes,
                prediction_relative_absolute_error_ppm=record.prediction_relative_absolute_error_ppm,
                prediction_matrix_overlap_ppm=record.prediction_matrix_overlap_ppm,
                prediction_top_destination_accuracy_ppm=record.prediction_top_destination_accuracy_ppm,
                receiver_posting_service_ns=int(receiver_posting_service_ns),
                receiver_posting_queue_wait_ns=int(receiver_posting_queue_wait_ns),
                receiver_buffer_stall_ns=int(receiver_buffer_stall_ns),
                receiver_drain_queue_wait_ns=int(receiver_drain_queue_wait_ns),
                receiver_drain_service_ns=int(receiver_drain_service_ns),
                peak_staging_bytes_by_rank=tuple(sorted(staging_by_rank.items())),
                peak_final_assembly_bytes_by_rank=tuple(sorted(final_assembly_by_rank.items())),
            ))
        return tuple(result)

    def dispose(self) -> None:
        """Break completed runtime callback cycles after evidence is persisted."""

        # The runtime is terminal when public runners call this method.  Detach
        # module-to-module callback references before clearing the Kernel registry.
        adapters = getattr(self.observer_bridge, "adapters", ())
        for adapter in adapters:
            if hasattr(adapter, "backend"):
                adapter.backend = None
            if hasattr(adapter, "transport"):
                adapter.transport = None
            for name in (
                "_payload_by_event_id",
                "_pipeline_jobs",
                "_prepared_by_job",
                "_dirty_phases",
            ):
                value = getattr(adapter, name, None)
                if hasattr(value, "clear"):
                    value.clear()
        if hasattr(self.observer_bridge, "backend"):
            self.observer_bridge.backend = None
        if hasattr(self.observer_bridge, "transport"):
            self.observer_bridge.transport = None

        self.backend.observer = None
        self.receiver.observer = None
        self.kernel_bridge.backend = None
        self.kernel_bridge._payload_by_event_id.clear()
        self.backend_driver.control_adapter.backend = None
        self.backend_driver.control_adapter.control_plane = None
        self.backend_driver.control_adapter._descriptor_by_request_digest.clear()
        self.data_plane.completion_sink = None
        self.data_plane.resource_release_sink = None
        self.data_plane.authority_validation = None
        self.data_plane.task_lookup = None
        self.data_plane.permit_lookup = None
        self.control_plane._delivery_sink = None
        self.kernel.dispose()

        # Completed formal runs form large cross-module callback graphs.  The
        # explicit detach above makes them collectible; collect here so matrix
        # workers and pytest files do not defer that cost until interpreter
        # shutdown or a later unrelated case.
        import gc

        gc.collect()

    def evidence(self) -> dict[str, Any]:
        return _evidence_tree({
            "schema_version": "INTEGRATION_RUNTIME_EVIDENCE",
            "run_id": self.run_id,
            "terminal_state": self.terminal_state(),
            "run_axes": dict(self.run_axes or {}),
            "runtime_profile": self.runtime_profile.manifest_fragment(),
            "data_plane_statistics": self.data_plane.statistics(),
            "data_plane_runtime_metrics": self.data_plane.formal_runtime_metrics(),
            "control_plane_statistics": self.control_plane.statistics(),
            "control_plane_runtime_metrics": self.control_plane.formal_runtime_metrics(),
            "scheduler_runtime_metrics": (
                self.observer_bridge.metrics()
                if hasattr(self.observer_bridge, "metrics")
                else None
            ),
            "scheduler_algorithm_diagnostics": tuple(
                diagnostic
                for adapter in getattr(self.observer_bridge, "adapters", ())
                for diagnostic in (
                    adapter.algorithm_diagnostics()
                    if hasattr(adapter, "algorithm_diagnostics")
                    else ()
                )
            ),
            "backend_phase_metrics": tuple(
                row.as_dict()
                for row in self.backend.phase_metrics_matrix(
                    phase_keys=self.registration.all_phase_keys
                )
            ),
            "current_p12_window_records": self.current_p12_window_records(),
        })




def _default_topology_for_fixture(fixture_input: Any, *, topology_id: str) -> Any:
    world_size = int(fixture_input.world_size)
    rank_to_node = tuple(int(v) for v in fixture_input.windows[0].mapping.rank_to_node)
    return make_network_topology(
        topology_id=str(topology_id),
        rank_to_node=rank_to_node,
        tx_nic_id_by_rank=tuple(f"rank-{rank}-tx" for rank in range(world_size)),
        rx_nic_id_by_rank=tuple(f"rank-{rank}-rx" for rank in range(world_size)),
        lane_ids_by_link_class=(
            (
                LinkClass.INTRA_NODE,
                tuple(f"intra-lane-{index}" for index in range(world_size)),
            ),
            (
                LinkClass.INTER_NODE,
                tuple(f"inter-lane-{index}" for index in range(world_size)),
            ),
        ),
        nic_id_by_lane=tuple(
            [
                (f"intra-lane-{index}", f"intra-fabric-{index}")
                for index in range(world_size)
            ]
            + [
                (f"inter-lane-{index}", f"inter-fabric-{index}")
                for index in range(world_size)
            ]
        ),
    )



def _all_phase_keys_for_fixture(*, fixture_input: Any, run_id: str) -> tuple[Any, ...]:
    from ..adapters.trace import keys_for_trace_window

    result: list[Any] = []
    for window in fixture_input.windows:
        keys = keys_for_trace_window(run_id=str(run_id), trace_window=window)
        result.extend((keys.dispatch_phase_key, keys.combine_phase_key))
    return tuple(result)






def _rscf_wire_cost_model_from_runtime(
    *,
    topology: Any,
    hardware_profile: Any,
    fixture_input: Any | None = None,
    base_layer_index: int | None = None,
    predicted_p2_matrix: tuple[tuple[int, ...], ...] | None = None,
    predicted_p2_confidence_ppm: int = 0,
    timing_profile: P12RankTimingProfile | None = None,
) -> RSCFWireCostModel:
    launch = {link: int(value) for link, value in hardware_profile.launch_delay_ns_by_link_class}
    fixed = {link: int(value) for link, value in hardware_profile.fixed_latency_ns_by_link_class}
    bandwidth = {link: int(value) for link, value in hardware_profile.bandwidth_bytes_per_second_by_link_class}
    slopes: list[tuple[int, int, float]] = []
    intercepts: list[tuple[int, int, float]] = []
    launches: list[tuple[int, int, float]] = []
    for src in range(len(topology.rank_to_node)):
        for dst in range(len(topology.rank_to_node)):
            if src == dst:
                continue
            link = (
                LinkClass.INTRA_NODE
                if int(topology.rank_to_node[src]) == int(topology.rank_to_node[dst])
                else LinkClass.INTER_NODE
            )
            bw = bandwidth[link]
            slopes.append((src, dst, 1_000_000_000.0 / float(bw)))
            intercepts.append((src, dst, float(fixed[link])))
            launches.append((src, dst, float(launch[link])))
    source_ready: tuple[tuple[int, int, float], ...] = ()
    p1_to_p2: tuple[tuple[int, float], ...] = ()
    p2_tail: tuple[tuple[int, float], ...] = ()
    if fixture_input is not None and base_layer_index is not None:
        windows = {int(window.layer_id): window for window in fixture_input.windows}
        current = windows.get(int(base_layer_index))
        previous = windows.get(int(base_layer_index) - 1)
        if current is not None:
            estimate = causal_last_observed_timing_estimate(
                current_window=current,
                previous_window=previous,
                predicted_p2_matrix=predicted_p2_matrix,
                timing_profile=timing_profile,
                following_layer_id=int(base_layer_index) + 1,
                p2_load_confidence_ppm=int(predicted_p2_confidence_ppm),
            )
            source_ready_base = min(estimate.p1_source_ready_ns)
            source_ready = tuple(
                (
                    1,
                    rank,
                    float(int(estimate.p1_source_ready_ns[rank]) - int(source_ready_base)),
                )
                for rank in range(len(topology.rank_to_node))
            )
            p1_to_p2 = tuple(
                (rank, float(estimate.p1_to_p2_delay_ns[rank]))
                for rank in range(len(topology.rank_to_node))
            )
            p2_tail = tuple(
                (rank, float(estimate.p2_completion_tail_ns[rank]))
                for rank in range(len(topology.rank_to_node))
            )
    return RSCFWireCostModel(
        default_slope=1.0,
        default_intercept=0.0,
        wave_launch_cost=0.0,
        edge_slope=tuple(slopes),
        edge_intercept=tuple(intercepts),
        edge_launch=tuple(launches),
        source_ready_by_phase_rank=source_ready,
        p1_to_p2_delay_by_rank=p1_to_p2,
        p2_completion_tail_by_rank=p2_tail,
    )


def _build_live_session(
    *,
    scheduling: SchedulingStack,
    fixture_input: Any,
    run_id: str,
    phase_keys: tuple[Any, ...],
    window_key: WindowKey,
    base_layer_index: int,
    core_id: str,
    planning_mode: str,
    release_mode: str,
    planner_scope: str,
    information_mode: str,
    capacities: Mapping[int, int | None],
    taskization_spec: TaskizationSpec,
    topology: Any,
    hardware_profile: Any,
    planning_cost_model: PlanningCostModel,
    event_triggers: tuple[str, ...],
    predicted_p2_matrix: tuple[tuple[int, ...], ...] | None,
    predicted_p2_confidence_ppm: int,
    safe_scope_selection: bool = False,
    timing_profile: P12RankTimingProfile | None = None,
    oracle_time_limit_ms: int = 30_000,
    oracle_relative_gap: float = 0.0,
    oracle_require_certified: bool = True,
) -> LivePolicySession:
    normalized_scope = PlannerScope(str(planner_scope).upper())
    normalized_planning = PlanningMode(str(planning_mode).upper())
    normalized_release = SchedulingReleaseMode(str(release_mode).upper())
    window = SchedulerWindow(
        window_key=window_key,
        base_layer_index=int(base_layer_index),
        phase_keys=tuple(phase_keys),
        window_digest=stable_digest({
            "window_key": window_key,
            "phase_keys": tuple(phase_keys),
            "fixture_id": str(fixture_input.fixture_id),
        }),
    )
    fairness = LiveFairnessInputs(
        receiver_contract_rule_digest=stable_digest(
            {"receiver_model": "RECEIVER_DECOUPLED_P12"}
        ),
        buffer_profile_digest=stable_digest(
            {"staging_capacity_bytes_by_rank": tuple(sorted(capacities.items()))}
        ),
        compiler_digest=stable_digest(taskization_spec.stable_payload()),
        transport_digest=stable_digest(
            {
                "topology_digest": topology.topology_digest,
                "hardware_profile_digest": hardware_profile.profile_digest,
            }
        ),
        release_model_digest=stable_digest({"release_mode": str(normalized_release)}),
        information_digest=stable_digest(
            {
                "information_mode": str(information_mode),
                "predicted_p2_matrix": predicted_p2_matrix,
                "predicted_p2_confidence_ppm": int(predicted_p2_confidence_ppm),
            }
        ),
        cost_model_digest=planning_cost_model.model_digest,
    )
    spec = LivePolicySpec(
        core_id=str(core_id),
        planning_mode=normalized_planning,
        release_mode=normalized_release,
        scope=normalized_scope,
        rank_count=int(fixture_input.world_size),
        fairness=fairness,
        event_triggers=tuple(PlanningTrigger(str(item).upper()) for item in event_triggers),
        rscf_wire_cost_model=_rscf_wire_cost_model_from_runtime(
            topology=topology,
            hardware_profile=hardware_profile,
            fixture_input=fixture_input,
            base_layer_index=base_layer_index,
            predicted_p2_matrix=predicted_p2_matrix,
            predicted_p2_confidence_ppm=int(predicted_p2_confidence_ppm),
            timing_profile=timing_profile,
        ),
        rank_to_node=tuple(int(value) for value in topology.rank_to_node),
        safe_scope_selection=bool(safe_scope_selection),
        oracle_time_limit_ms=int(oracle_time_limit_ms),
        oracle_relative_gap=float(oracle_relative_gap),
        oracle_require_certified=bool(oracle_require_certified),
    )
    return LivePolicySession(
        controller=scheduling.controller,
        adapter=scheduling.adapter,
        horizon_window=window,
        spec=spec,
    )


def _current_p12_single_phase_session(
    *,
    scheduling: SchedulingStack,
    fixture_input: Any,
    run_id: str,
    phase_key: Any,
    window_index: int,
    core_id: str,
    planning_mode: str,
    release_mode: str,
    capacities: Mapping[int, int | None],
    taskization_spec: TaskizationSpec,
    topology: Any,
    hardware_profile: Any,
    planning_cost_model: PlanningCostModel,
    event_triggers: tuple[str, ...],
    timing_profile: P12RankTimingProfile | None = None,
    oracle_time_limit_ms: int = 30_000,
    oracle_relative_gap: float = 0.0,
    oracle_require_certified: bool = True,
) -> LivePolicySession:
    sample_id = str(phase_key.sample_id)
    return _build_live_session(
        scheduling=scheduling,
        fixture_input=fixture_input,
        run_id=run_id,
        phase_keys=(phase_key,),
        window_key=WindowKey(run_id, sample_id, int(window_index)),
        base_layer_index=int(phase_key.layer_index),
        core_id=core_id,
        planning_mode=planning_mode,
        release_mode=release_mode,
        planner_scope=PlannerScope.PHASE_LOCAL.value,
        information_mode="NO_P2_INFORMATION_PHASE_LOCAL",
        capacities=capacities,
        taskization_spec=taskization_spec,
        topology=topology,
        hardware_profile=hardware_profile,
        planning_cost_model=planning_cost_model,
        event_triggers=event_triggers,
        predicted_p2_matrix=None,
        predicted_p2_confidence_ppm=0,
        timing_profile=timing_profile,
        oracle_time_limit_ms=int(oracle_time_limit_ms),
        oracle_relative_gap=float(oracle_relative_gap),
        oracle_require_certified=bool(oracle_require_certified),
    )


def build_current_p12_integration_runtime(
    *,
    fixture_input: Any,
    run_id: str,
    algorithm: str = "joint(global_(rscf()))",
    staging_sensitivity: str = "1.0X",
    release_mode: str = PAPER_RELEASE_MODE,
    p0_p1_compute_end_barrier: bool = PAPER_P0_P1_COMPUTE_END_BARRIER,
    information_mode: str = "FATE_P2",
    overlap_mode: str = "OVERLAP",
    max_task_bytes: int = PAPER_MAX_TASK_BYTES,
    alignment_bytes: int = PAPER_ALIGNMENT_BYTES,
    topology: Any | None = None,
    runtime_profile: RuntimeProfileBundle | None = None,
    max_event_plans_per_phase: int = 512,
    max_window_prefix_tasks: int = 1,
    paired_instance_id: str | None = None,
    rank_timing_profile: P12RankTimingProfile | None = None,
    oracle_time_limit_ms: int = 30_000,
    oracle_relative_gap: float = 0.0,
    oracle_require_certified: bool = True,
) -> FormalIntegrationRuntime:
    """Build the paper-facing Current P12 runtime.

    The only formal execution line is:
    ``P0_l truth -> P1_l exact + P2_l predicted -> CURRENT_P12``.
    No alternate horizon or compatibility execution path is accepted.
    """

    composed = parse_algorithm_expression(algorithm)
    scope = composed.scope
    info = normalize_p12_information_mode(information_mode)
    if scope is PlannerScope.PHASE_LOCAL:
        info = normalize_p12_information_mode("ZERO_P2")
    core_id = composed.core_id
    planning_mode = composed.planning.value
    safe_scope_selection = composed.safe
    world_size = int(fixture_input.world_size)
    profile = runtime_profile or make_default_synthetic_runtime_profile(
        max_batch_tasks=world_size
    )
    if not isinstance(profile, RuntimeProfileBundle):
        raise TypeError("runtime_profile must be RuntimeProfileBundle")
    hardware = profile.transport_profile.hardware_profile
    control_profile = profile.transport_profile.control_profile
    receiver_cost_model = profile.receiver_cost_model
    cost_model = profile.planning_cost_model
    local_assembly_cost_ns = profile.transport_profile.local_assembly_latency_ns
    capacities = compute_fixture_staging_capacity_bytes_by_rank(
        fixture_input=fixture_input,
        sensitivity=str(staging_sensitivity),
        alignment_bytes=int(alignment_bytes),
        max_canonical_task_payload_bytes=int(max_task_bytes),
    )
    paired_identity = (
        str(paired_instance_id)
        if paired_instance_id is not None
        else stable_digest({
            "schema_version": "CURRENT_P12_PAIRED_INSTANCE",
            "fixture_id": str(fixture_input.fixture_id),
            "max_task_bytes": int(max_task_bytes),
            "alignment_bytes": int(alignment_bytes),
            "runtime_profile_digest": profile.profile_digest,
        })
    )
    taskization_spec = TaskizationSpec(
        chunk_bytes=int(max_task_bytes),
        alignment_bytes=int(alignment_bytes),
        identity_namespace=paired_identity,
    )
    kernel = SimulationKernel()
    kernel_bridge = BackendKernelBridge(kernel)
    scheduling = build_scheduling_stack(taskization_spec=taskization_spec)
    shared_topology = topology or _default_topology_for_fixture(
        fixture_input, topology_id=f"current-p12-{run_id}"
    )
    shared_lines = ThreeLineServices()
    event_triggers = ("TASK_READY",)
    from ..adapters.trace import keys_for_trace_window

    trace_windows = tuple(fixture_input.windows)
    phase_keys_by_layer = {
        int(window.layer_id): keys_for_trace_window(run_id=str(run_id), trace_window=window)
        for window in trace_windows
    }
    p12_windows = build_current_p12_windows(
        fixture_input=fixture_input, run_id=str(run_id)
    )
    prediction_by_window: dict[str, Any] = {}
    adapters: list[FormalSchedulingRuntimeAdapter] = []
    routes: list[CurrentP12TriggerRoute] = []
    next_window_index = 10_000

    def add_normal_phase(
        phase_key: Any, *, core_id_override: str | None = None
    ) -> FormalSchedulingRuntimeAdapter:
        nonlocal next_window_index
        session = _current_p12_single_phase_session(
            scheduling=scheduling,
            fixture_input=fixture_input,
            run_id=str(run_id),
            phase_key=phase_key,
            window_index=next_window_index,
            core_id=(core_id if core_id_override is None else str(core_id_override)),
            planning_mode=str(planning_mode).upper(),
            release_mode=str(release_mode).upper(),
            capacities=capacities,
            taskization_spec=taskization_spec,
            topology=shared_topology,
            hardware_profile=hardware,
            planning_cost_model=cost_model,
            event_triggers=event_triggers,
            timing_profile=rank_timing_profile,
            oracle_time_limit_ms=int(oracle_time_limit_ms),
            oracle_relative_gap=float(oracle_relative_gap),
            oracle_require_certified=bool(oracle_require_certified),
        )
        next_window_index += 1
        adapter = FormalSchedulingRuntimeAdapter(
            kernel=kernel,
            live_session=session,
            cost_model=cost_model,
            prediction_enabled=False,
            max_event_plans_per_phase=int(max_event_plans_per_phase),
            max_window_prefix_tasks=int(max_window_prefix_tasks),
            event_namespace=f"{run_id}:current-p12:phase:{next_window_index}",
            shared_lines=shared_lines,
            overlap_mode=str(overlap_mode),
        )
        adapters.append(adapter)
        return adapter

    if not trace_windows:
        raise ValueError("fixture contains no trace windows")
    first_layer = int(trace_windows[0].layer_id)
    last_layer = int(trace_windows[-1].layer_id)

    if scope is PlannerScope.WINDOW_JOINT:
        add_normal_phase(
            phase_keys_by_layer[first_layer].dispatch_phase_key,
            core_id_override=core_id,
        )
        for index, window in enumerate(p12_windows):
            current_trace = next(item for item in trace_windows if int(item.layer_id) == window.anchor_layer_id)
            next_trace = next(item for item in trace_windows if int(item.layer_id) == window.anchor_layer_id + 1)
            prediction = build_p2_prediction(
                current_trace_window=current_trace,
                next_trace_window=next_trace,
                information_mode=info,
            )
            prediction_by_window[window.planning_window_digest] = prediction
            session = _build_live_session(
                scheduling=scheduling,
                fixture_input=fixture_input,
                run_id=str(run_id),
                phase_keys=window.referenced_phase_keys,
                window_key=window.window_key,
                base_layer_index=int(window.anchor_layer_id),
                core_id=core_id,
                planning_mode=str(planning_mode).upper(),
                release_mode=str(release_mode).upper(),
                planner_scope=PlannerScope.WINDOW_JOINT.value,
                information_mode=info.value,
                capacities=capacities,
                taskization_spec=taskization_spec,
                topology=shared_topology,
                hardware_profile=hardware,
                planning_cost_model=cost_model,
                event_triggers=event_triggers,
                predicted_p2_matrix=prediction.matrix,
                predicted_p2_confidence_ppm=(
                    0
                    if prediction.confidence_ppm is None
                    else int(prediction.confidence_ppm)
                ),
                safe_scope_selection=bool(safe_scope_selection),
                timing_profile=rank_timing_profile,
                oracle_time_limit_ms=int(oracle_time_limit_ms),
                oracle_relative_gap=float(oracle_relative_gap),
                oracle_require_certified=bool(oracle_require_certified),
            )
            adapter = FormalSchedulingRuntimeAdapter(
                kernel=kernel,
                live_session=session,
                cost_model=cost_model,
                prediction_enabled=True,
                max_event_plans_per_phase=int(max_event_plans_per_phase),
                max_window_prefix_tasks=int(max_window_prefix_tasks),
                event_namespace=f"{run_id}:current-p12:window:{index}",
                shared_lines=shared_lines,
                overlap_mode=str(overlap_mode),
                current_p12_window=window,
                current_p12_information_mode=info.value,
                current_p12_prediction_digest=prediction.prediction_digest,
                current_p12_predicted_p2_matrix=prediction.matrix,
                external_current_p12_trigger=True,
            )
            adapters.append(adapter)
            compute = current_trace.local_compute.dispatch_release_to_combine_source_ready_ns
            routes.append(
                CurrentP12TriggerRoute(
                    trigger_phase_key=window.p0_trigger_phase_key,
                    target_adapter=adapter,
                    p1_source_ready_duration_ns_by_rank=tuple(int(value) for value in compute),
                    planning_window_digest=window.planning_window_digest,
                    p0_p1_compute_end_barrier=bool(p0_p1_compute_end_barrier),
                )
            )
        add_normal_phase(
            phase_keys_by_layer[last_layer].combine_phase_key,
            core_id_override=core_id,
        )
    else:
        # Local uses one authority per phase.  P1 planning is still triggered by
        # P0 and hidden in current compute, while P2 remains invisible until its
        # own exact Dispatch observations arrive.
        window_by_anchor = {item.anchor_layer_id: item for item in p12_windows}
        for layer in sorted(phase_keys_by_layer):
            keys = phase_keys_by_layer[layer]
            add_normal_phase(
                keys.dispatch_phase_key,
                core_id_override=(
                    core_id if layer == first_layer else None
                ),
            )
            if layer in window_by_anchor:
                window = window_by_anchor[layer]
                current_trace = next(item for item in trace_windows if int(item.layer_id) == layer)
                session = _current_p12_single_phase_session(
                    scheduling=scheduling,
                    fixture_input=fixture_input,
                    run_id=str(run_id),
                    phase_key=keys.combine_phase_key,
                    window_index=next_window_index,
                    core_id=core_id,
                    planning_mode=str(planning_mode).upper(),
                    release_mode=str(release_mode).upper(),
                    capacities=capacities,
                    taskization_spec=taskization_spec,
                    topology=shared_topology,
                    hardware_profile=hardware,
                    planning_cost_model=cost_model,
                    event_triggers=event_triggers,
                    timing_profile=rank_timing_profile,
                    oracle_time_limit_ms=int(oracle_time_limit_ms),
                    oracle_relative_gap=float(oracle_relative_gap),
                    oracle_require_certified=bool(oracle_require_certified),
                )
                next_window_index += 1
                adapter = FormalSchedulingRuntimeAdapter(
                    kernel=kernel,
                    live_session=session,
                    cost_model=cost_model,
                    prediction_enabled=False,
                    max_event_plans_per_phase=int(max_event_plans_per_phase),
                    max_window_prefix_tasks=int(max_window_prefix_tasks),
                    event_namespace=f"{run_id}:current-p12:local-combine:{layer}",
                    shared_lines=shared_lines,
                    overlap_mode=str(overlap_mode),
                    current_p12_window=window,
                    current_p12_information_mode="NO_P2_INFORMATION_PHASE_LOCAL",
                    current_p12_prediction_digest=None,
                    external_current_p12_trigger=True,
                )
                adapters.append(adapter)
                routes.append(
                    CurrentP12TriggerRoute(
                        trigger_phase_key=window.p0_trigger_phase_key,
                        target_adapter=adapter,
                        p1_source_ready_duration_ns_by_rank=tuple(
                            int(value)
                            for value in current_trace.local_compute.dispatch_release_to_combine_source_ready_ns
                        ),
                        planning_window_digest=window.planning_window_digest,
                        p0_p1_compute_end_barrier=bool(p0_p1_compute_end_barrier),
                    )
                )
            else:
                add_normal_phase(
                keys.combine_phase_key,
                core_id_override=(
                    core_id if layer == last_layer else None
                ),
            )

    observer = CompositeFormalSchedulingRuntimeAdapter(
        adapters=tuple(adapters),
        scheduling_stack=scheduling,
        current_p12_trigger_routes=tuple(routes),
    )
    control_plane = FormalControlPlaneTransport(
        kernel=kernel,
        profile=control_profile,
    )
    rank_to_node = tuple(int(v) for v in trace_windows[0].mapping.rank_to_node)
    backend_driver = build_backend_runtime_driver(
        world_size=world_size,
        staging_capacity_bytes_by_rank=capacities,
        kernel_bridge=kernel_bridge,
        observer=observer,
        control_plane=control_plane,
        release_mode=str(release_mode),
        p0_p1_compute_end_barrier=bool(p0_p1_compute_end_barrier),
        node_id_by_rank={rank: rank_to_node[rank] for rank in range(world_size)},
        receiver_cost_model=receiver_cost_model,
        local_assembly_cost_ns=int(local_assembly_cost_ns),
    )
    observer.attach_backend(backend_driver.backend)
    resolver = SharedTopologyTaskResolver(shared_topology)
    ports = build_scheduler_port_bundle(
        catalogue=scheduling.catalogue,
        authority=scheduling.authority,
        backend=backend_driver.backend,
        scheduler_bridge=observer,
        resource_resolver=resolver,
        expected_hardware_profile_digest=hardware.profile_digest,
    )
    scheduling.compiler.resources = ports.resource_adapter
    scheduling.validator.resources = ports.resource_adapter
    data_plane = FormalDataPlaneTransport(
        kernel=kernel,
        task_lookup=ports.task_lookup,
        permit_lookup=ReceiverPermitLookup(backend_driver.receiver),
        authority_validation=ports.authority_validation,
        resource_resolver=resolver,
        completion_sink=ports.completion_sink,
        resource_release_sink=ports.resource_release_sink,
        hardware_profile=hardware,
        bandwidth_contention=profile.transport_profile.bandwidth_contention,
    )
    observer.attach_transport(data_plane)
    registration = backend_driver.register_fixture(
        fixture_input=fixture_input, run_id=str(run_id)
    )
    return FormalIntegrationRuntime(
        kernel=kernel,
        kernel_bridge=kernel_bridge,
        scheduling=scheduling,
        observer_bridge=observer,
        backend_driver=backend_driver,
        data_plane=data_plane,
        control_plane=control_plane,
        topology=shared_topology,
        hardware_profile=hardware,
        runtime_profile=profile,
        registration=registration,
        run_id=str(run_id),
        scheduler_runtime_mode="CURRENT_P12",
        run_axes={
            "execution_line": "CURRENT_P12",
            "planning_window": "P12",
            "staging_sensitivity": str(staging_sensitivity),
            "release_mode": str(release_mode).upper(),
            "p0_p1_compute_end_barrier": bool(p0_p1_compute_end_barrier),
            "algorithm": composed.expression,
            "algorithm_core": core_id,
            "safe_scope_selection": bool(safe_scope_selection),
            "planning_mode": str(planning_mode).upper(),
            "planner_scope": scope.value,
            "execution_mode": "ORDER_ONLY",
            "information_mode": info.value if scope is PlannerScope.WINDOW_JOINT else "NO_P2_INFORMATION_PHASE_LOCAL",
            "prediction_confidence_policy": (
                "N_A_PHASE_LOCAL"
                if scope is PlannerScope.PHASE_LOCAL
                else (
                    "FATE_ARTIFACT"
                    if info is P12InformationMode.FATE_P2
                    else (
                        "EXACT_UPPER_BOUND"
                        if info is P12InformationMode.PERFECT_P2
                        else "ZERO_ABLATION"
                    )
                )
            ),
            "overlap_mode": str(overlap_mode).upper(),
            "runtime_profile_id": profile.profile_id,
            "runtime_profile_digest": profile.profile_digest,
            "runtime_profile_kind": profile.profile_kind,
            "runtime_profile_provenance": profile.profile_provenance,
            "hardware_profile_calibrated": bool(profile.performance_eligible),
            "performance_claim_allowed": bool(profile.performance_eligible),
            "paired_instance_id": paired_identity,
        },
        fixture_input=fixture_input,
        current_p12_windows=p12_windows,
        current_p12_predictions=prediction_by_window,
    )


__all__ = [
    "CurrentP12WindowRecord",
    "FormalIntegrationRuntime",
    "build_current_p12_integration_runtime",
]
