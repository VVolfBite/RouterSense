from __future__ import annotations

"""Current-P12 scheduling runtime adapter.

This module owns event handling and mutable runtime state.  Immutable evidence
records and observation coalescing live in :mod:`runtime_models` so schema
changes do not force edits across the state machine.
"""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Iterable

from rs_sim import KernelPhase, ProgressSignal, SimulationEvent, SimulationKernel

from rs_sim.scheduler.planning.catalogue import PhaseCatalogueSeal
from rs_sim.scheduler.planning.current_p12 import (
    PreparedP12PlanTemplate,
    build_predicted_p2_slots,
    evaluate_p12_prediction_evidence,
)
from rs_sim.scheduler.errors import CatalogueSealError, FormalRuntimeError
from rs_sim.scheduler.execution.lines import LineReservation, PlanningCostModel, ThreeLineServices
from rs_sim.scheduler.execution.live import LivePolicySession, PreparedLiveActivation, ReleaseMode
from rs_sim.scheduler.planning.planner import PlannerScope
from rs_sim.scheduler.decorators.planning_gate import PlanningDecision, PlanningGate, PlanningMode, PlanningTrigger
from rs_sim.scheduler.execution.runtime_models import (
    CoalescedObservationBatch,
    CurrentP12TemplateEvidence,
    FormalSchedulingRuntimeMetrics,
    GlobalClosureTruth,
    ObservationEnvelope,
    PhaseObservationAccumulator,
    PlanningPipelineJob,
    RuntimeActivationEvidence,
    _ordinal,
    _phase_token,
)
from rs_sim.scheduler.stable import stable_digest

class FormalSchedulingRuntimeAdapter:
    """Drop-in formal scheduler service runtime for one LivePolicySession.

    Observation callbacks only mutate canonical facts and enqueue coalescing.
    Plan computation occurs at ControlLine completion; authority activation
    occurs at ExecutionBindingLine completion.  Formal transport submissions remain
    single-phase and use the existing immediate transaction stabilizer.
    """

    OBSERVATION_EVENT = "SCHEDULER_OBSERVATION_DELIVERY"
    COALESCE_EVENT = "SCHEDULER_OBSERVATION_COALESCE"
    GLOBAL_DEFER_EVENT = "SCHEDULER_GLOBAL_FINALIZE_DEFER"
    GLOBAL_FINALIZE_EVENT = "SCHEDULER_GLOBAL_CATALOGUE_FINALIZE"
    PREDICTION_COMPLETE_EVENT = "SCHEDULER_PREDICTION_COMPLETE"
    CONTROL_COMPLETE_EVENT = "SCHEDULER_CONTROL_COMPLETE"
    BINDING_COMPLETE_EVENT = "SCHEDULER_BINDING_COMPLETE"
    CURRENT_P12_TRIGGER_EVENT = "SCHEDULER_CURRENT_P12_TRIGGER"

    def __init__(
        self,
        *,
        kernel: SimulationKernel,
        live_session: LivePolicySession,
        cost_model: PlanningCostModel,
        prediction_enabled: bool = False,
        max_event_plans_per_phase: int = 256,
        max_window_prefix_tasks: int = 1,
        event_namespace: str = "default",
        shared_lines: ThreeLineServices | None = None,
        overlap_mode: str = "OVERLAP",
        current_p12_window: Any | None = None,
        current_p12_information_mode: str | None = None,
        current_p12_prediction_digest: str | None = None,
        current_p12_predicted_p2_matrix: tuple[tuple[int, ...], ...] | None = None,
        external_current_p12_trigger: bool = False,
    ) -> None:
        if max_event_plans_per_phase <= 0 or max_window_prefix_tasks <= 0:
            raise ValueError("plan/prefix bounds must be positive")
        self.kernel = kernel
        self.session = live_session
        self.stack = live_session.controller
        self.adapter = live_session.adapter
        self.cost_model = cost_model
        self.prediction_enabled = bool(prediction_enabled)
        self.max_event_plans_per_phase = int(max_event_plans_per_phase)
        self.max_window_prefix_tasks = int(max_window_prefix_tasks)
        normalized_overlap_mode = str(overlap_mode).upper()
        if normalized_overlap_mode not in {"OVERLAP", "SERIALIZED"}:
            raise ValueError("overlap_mode must be OVERLAP or SERIALIZED")
        self.overlap_mode = normalized_overlap_mode
        if not isinstance(event_namespace, str) or not event_namespace:
            raise ValueError("event_namespace must be non-empty")
        self.event_namespace = event_namespace
        suffix = stable_digest({"formal_scheduler_namespace": event_namespace})[:16]
        self.OBSERVATION_EVENT = f"{type(self).OBSERVATION_EVENT}:{suffix}"
        self.COALESCE_EVENT = f"{type(self).COALESCE_EVENT}:{suffix}"
        self.GLOBAL_DEFER_EVENT = f"{type(self).GLOBAL_DEFER_EVENT}:{suffix}"
        self.GLOBAL_FINALIZE_EVENT = f"{type(self).GLOBAL_FINALIZE_EVENT}:{suffix}"
        self.PREDICTION_COMPLETE_EVENT = f"{type(self).PREDICTION_COMPLETE_EVENT}:{suffix}"
        self.CONTROL_COMPLETE_EVENT = f"{type(self).CONTROL_COMPLETE_EVENT}:{suffix}"
        self.BINDING_COMPLETE_EVENT = f"{type(self).BINDING_COMPLETE_EVENT}:{suffix}"
        self.CURRENT_P12_TRIGGER_EVENT = f"{type(self).CURRENT_P12_TRIGGER_EVENT}:{suffix}"
        self._stabilization_callback_name = (
            f"rs_sim.scheduler.formal_runtime_adapter.stabilize:{suffix}"
        )
        self.lines = shared_lines or ThreeLineServices()
        self.accumulator = PhaseObservationAccumulator(adapter=self.adapter)
        self.backend: Any | None = None
        self.transport: Any | None = None
        self._payload_by_event_id: dict[str, Any] = {}
        self._gates_by_phase: dict[str, PlanningGate] = {}
        self._closure_truth_by_phase: dict[str, GlobalClosureTruth] = {}
        self._global_seals_by_phase: dict[str, PhaseCatalogueSeal] = {}
        self._source_ready_at: dict[tuple[str, int], int] = {}
        self._permits_by_task_id: dict[str, Any] = {}
        self._dirty_phases: dict[str, Any] = {}
        self._phase_plan_counts: defaultdict[str, int] = defaultdict(int)
        self._pipeline_jobs: dict[str, PlanningPipelineJob] = {}
        self._prepared_by_job: dict[str, PreparedLiveActivation | None] = {}
        self._algorithm_diagnostics_rows: list[dict[str, Any]] = []
        self._stable_event_ids: list[str] = []
        self._activation_evidence: list[RuntimeActivationEvidence] = []
        self._stale_activation_count = 0
        self._global_joint_planned_tokens: set[str] = set()
        self.current_p12_window = current_p12_window
        self.current_p12_information_mode = (
            None if current_p12_information_mode is None else str(current_p12_information_mode)
        )
        self.current_p12_prediction_digest = current_p12_prediction_digest
        self.current_p12_predicted_p2_matrix = (
            None
            if current_p12_predicted_p2_matrix is None
            else tuple(tuple(int(value) for value in row) for row in current_p12_predicted_p2_matrix)
        )
        self.external_current_p12_trigger = bool(external_current_p12_trigger)
        self._current_p12_triggered = False
        self._current_p12_trigger_requested = False
        self._current_p12_serialized_trigger_payload: dict[str, Any] | None = None
        self._current_p12_actual_consume_at_ns: int | None = None
        self._current_p12_trigger_at_ns: int | None = None
        self._current_p12_hide_until_ns: int | None = None
        self._current_p12_template_ready_at_ns: int | None = None
        self._current_p12_target_first_truth_at_ns: int | None = None
        self._current_p12_target_bound_at_ns: int | None = None
        self._current_p12_target_seal: PhaseCatalogueSeal | None = None
        self._current_p12_target_bound = False
        self._current_p12_reconciliation_status: str | None = None
        self._current_p12_template: PreparedP12PlanTemplate | None = None
        self._current_p12_safe_selector_choice: str | None = None
        self._current_p12_safe_selector_reason: str | None = None
        self._current_p12_safe_selector_local_objective: int | None = None
        self._current_p12_safe_selector_joint_objective: int | None = None
        self._current_p12_seen_target_task_ids: set[str] = set()
        self._current_p12_bound_task_by_slot: dict[str, tuple[str, ...]] = {}
        self._current_p12_unmatched_target_task_ids: set[str] = set()
        self._current_p12_exact_bind_count = 0
        self._current_p12_boundary_mismatch_bind_count = 0
        self._current_p12_overflow_bind_count = 0
        self._current_p12_unused_slot_count = 0
        self._current_p12_appended_task_count = 0
        self._current_p12_repair_task_count = 0
        self._current_p12_repair_task_bytes = 0
        self._current_p12_total_target_task_bytes = 0
        self._current_p12_binding_repair_reason: str | None = None
        self._current_p12_prediction_fallback_reason: str | None = None
        self._current_p12_last_planned_target_task_ids: frozenset[str] = frozenset()
        self._current_p12_bind_payload_by_job: dict[str, dict[str, Any]] = {}
        # At most one incremental P2 bind/repair job may be in flight for one
        # Current-P12 window.  Additional descriptor arrivals are coalesced and
        # picked up immediately after the current binding completes.
        self._current_p12_bind_inflight_job_id: str | None = None
        self._current_p12_algorithm_core_runs = 0
        self._current_p12_repair_count = 0
        self._frontier_replan_count = 0
        self._register_handlers()

    def _register_handlers(self) -> None:
        handlers = {
            self.OBSERVATION_EVENT: self._handle_observation,
            self.COALESCE_EVENT: self._handle_coalesce,
            self.GLOBAL_DEFER_EVENT: self._handle_global_defer,
            self.GLOBAL_FINALIZE_EVENT: self._handle_global_finalize,
            self.PREDICTION_COMPLETE_EVENT: self._handle_prediction_complete,
            self.CONTROL_COMPLETE_EVENT: self._handle_control_complete,
            self.BINDING_COMPLETE_EVENT: self._handle_binding_complete,
            self.CURRENT_P12_TRIGGER_EVENT: self._handle_current_p12_trigger,
        }
        for event_type, handler in handlers.items():
            self.kernel.register_event_handler(event_type, handler)
        self.kernel.register_phase_callback(
            KernelPhase.EXECUTION_STABILIZATION_SUBMIT,
            self._stabilization_callback_name,
            self._stabilize,
        )

    def trigger_current_p12(
        self,
        *,
        at_ns: int,
        hide_until_ns: int,
        trigger_phase_key: Any,
    ) -> SimulationEvent | None:
        if self.current_p12_window is None:
            raise FormalRuntimeError("adapter is not configured for Current P12")
        if self._current_p12_triggered or self._current_p12_trigger_requested:
            # A P0 phase has one immutable Current P12 trigger request.
            return self._schedule(
                time_ns=int(at_ns),
                phase=KernelPhase.THREE_LINE_JOB_TRANSITIONS,
                event_type=self.CURRENT_P12_TRIGGER_EVENT,
                subject_id=f"duplicate:{self.current_p12_window.planning_window_digest}",
                payload={
                    "duplicate": True,
                    "trigger_phase_key": trigger_phase_key,
                    "hide_until_ns": int(hide_until_ns),
                },
            )
        self._current_p12_trigger_requested = True
        payload = {
            "duplicate": False,
            "trigger_phase_key": trigger_phase_key,
            "hide_until_ns": int(hide_until_ns),
        }
        if self.overlap_mode == "SERIALIZED":
            # Serialized mode is released by the first real P1 source-ready
            # observation.  No Prediction/Control work exists before that
            # consumer boundary.
            self._current_p12_serialized_trigger_payload = payload
            return None
        return self._schedule(
            time_ns=int(at_ns),
            phase=KernelPhase.THREE_LINE_JOB_TRANSITIONS,
            event_type=self.CURRENT_P12_TRIGGER_EVENT,
            subject_id=self.current_p12_window.planning_window_digest,
            payload=payload,
        )

    def _seal_phase_from_known_truth(self, phase_key: Any, *, at_ns: int) -> PhaseCatalogueSeal:
        token = _phase_token(self.adapter, phase_key)
        existing = self.stack.catalogue.phase_seal(phase_key)
        if existing is not None:
            return existing
        try:
            truth = self._closure_truth_by_phase[token]
        except KeyError as exc:
            raise CatalogueSealError("Current P12 trigger fired before P1 catalogue closure") from exc
        seal = self.stack.catalogue.seal_phase(
            phase_key,
            expected_expectation_count=truth.expected_expectation_count,
            expected_task_count=truth.expected_task_count,
            expected_catalogue_digest=truth.expected_catalogue_digest,
            closure_digest=truth.closure_digest,
            sealed_at_ns=int(at_ns),
        )
        self._global_seals_by_phase[token] = seal
        return seal

    def _current_p12_predicted_task_count(self) -> int:
        matrix = self.current_p12_predicted_p2_matrix
        if matrix is None:
            return 0
        step = int(self.stack.catalogue.taskizer.spec.effective_chunk_bytes)
        return sum(
            (int(value) + step - 1) // step
            for src, row in enumerate(matrix)
            for dst, value in enumerate(row)
            if src != dst and int(value) > 0
        )

    def _handle_current_p12_trigger(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        payload = self._payload_by_event_id.pop(event.stable_event_id)
        if bool(payload.get("duplicate")):
            return ProgressSignal(notes=("scheduler-current-p12-duplicate-trigger",))
        if self._current_p12_triggered:
            raise FormalRuntimeError("Current P12 trigger was delivered twice")
        window = self.current_p12_window
        assert window is not None
        p1 = window.p1_combine_phase_key
        p2 = window.p2_dispatch_phase_key
        token = _phase_token(self.adapter, p1)
        if self.stack.catalogue.phase_seal(p1) is None and token not in self._closure_truth_by_phase:
            retry = int(payload.get("retry", 0))
            if retry >= 32:
                raise CatalogueSealError("Current P12 trigger could not observe P1 closure within 32 fixed-point rounds")
            retry_payload = dict(payload)
            retry_payload["retry"] = retry + 1
            self._schedule(
                time_ns=event.time_ns,
                phase=KernelPhase.DEADLOCK_PROGRESS_CHECK,
                event_type=self.CURRENT_P12_TRIGGER_EVENT,
                subject_id=f"{window.planning_window_digest}:retry:{retry + 1}",
                payload=retry_payload,
            )
            return ProgressSignal(notes=("scheduler-current-p12-trigger-deferred",))
        seal = self._seal_phase_from_known_truth(p1, at_ns=event.time_ns)
        self._current_p12_triggered = True
        self._current_p12_trigger_at_ns = int(event.time_ns)
        self._current_p12_hide_until_ns = int(payload["hide_until_ns"])
        synthetic = CoalescedObservationBatch(
            phase_key=p1,
            at_ns=int(event.time_ns),
            envelopes=(),
            raw_observation_count=1,
            enabled_triggers=(PlanningTrigger.OBSERVATION_CLOSURE,),
            hide_until_ns=int(payload["hide_until_ns"]),
            batch_digest=stable_digest({
                "schema_version": "CURRENT_P12_TRIGGER_BATCH",
                "planning_window_digest": window.planning_window_digest,
                "trigger_phase_key": payload["trigger_phase_key"],
                "p1_seal_digest": seal.seal_digest,
                "prediction_digest": self.current_p12_prediction_digest,
                "at_ns": int(event.time_ns),
            }),
        )
        self._enqueue_pipeline(
            phase_keys=(p1,),
            batch=synthetic,
            global_seal_digests=(seal.seal_digest,),
            prediction_only=(self.session.spec.scope is PlannerScope.WINDOW_JOINT),
            task_count_override=(
                len(self.stack.catalogue.task_ids_for_phase(p1))
                + self._current_p12_predicted_task_count()
            ),
            job_kind="CURRENT_P12_TEMPLATE",
            planning_window_digest=window.planning_window_digest,
        )
        return ProgressSignal(
            authoritative_state_updates=1,
            notes=("scheduler-current-p12-triggered",),
        )

    def _current_p12_target_boundary(self, task_id: str) -> tuple[int, int, int, int, int]:
        view = self.stack.catalogue.view(task_id)
        return (
            int(view.src_rank),
            int(view.dst_rank),
            int(view.chunk_index),
            int(view.byte_offset),
            int(view.payload_bytes),
        )

    def _refresh_current_p12_bindings(self) -> str | None:
        """Bind exact P2 chunks to predicted edge slots without rerunning RSCF.

        Prediction is an edge-residual advisory, not an exact chunk-boundary
        forecast.  Actual canonical chunks are therefore assigned to predicted
        slots by edge and deterministic chunk order.  Only an entirely new edge
        requires repair; payload-boundary drift is handled by binding alone.
        """

        template = self._current_p12_template
        window = self.current_p12_window
        if template is None or window is None:
            return None
        target_ids = tuple(
            self.stack.catalogue.task_ids_for_phase(window.p2_dispatch_phase_key)
        )
        self._current_p12_seen_target_task_ids.update(target_ids)

        total_target_bytes = sum(
            int(self._current_p12_target_boundary(task_id)[4]) for task_id in target_ids
        )
        self._current_p12_total_target_task_bytes = int(total_target_bytes)
        self._current_p12_exact_bind_count = 0
        self._current_p12_boundary_mismatch_bind_count = 0
        self._current_p12_overflow_bind_count = 0
        self._current_p12_unused_slot_count = 0
        self._current_p12_appended_task_count = 0
        self._current_p12_repair_task_count = 0
        self._current_p12_repair_task_bytes = 0
        self._current_p12_binding_repair_reason = None
        self._current_p12_prediction_fallback_reason = None

        slots_by_edge: dict[tuple[int, int], list[PredictedP2Slot]] = defaultdict(list)
        for slot in template.p2_slots:
            slots_by_edge[slot.edge_key].append(slot)
        for slots in slots_by_edge.values():
            slots.sort(key=lambda item: (item.chunk_index, item.byte_offset, item.slot_id))

        tasks_by_edge: dict[tuple[int, int], list[str]] = defaultdict(list)
        for task_id in target_ids:
            boundary = self._current_p12_target_boundary(task_id)
            tasks_by_edge[(boundary[0], boundary[1])].append(task_id)
        for task_ids in tasks_by_edge.values():
            task_ids.sort(key=lambda item: (*self._current_p12_target_boundary(item), item))

        assignments: dict[str, list[str]] = defaultdict(list)
        unmatched: set[str] = set()
        exact_hit = True
        repair_task_ids: set[str] = set()
        exact_bind_count = 0
        boundary_mismatch_bind_count = 0
        overflow_bind_count = 0
        unused_slot_count = 0
        appended_task_count = 0
        all_edges = sorted(set(slots_by_edge) | set(tasks_by_edge))
        for edge in all_edges:
            task_ids = tasks_by_edge.get(edge, [])
            slots = slots_by_edge.get(edge, [])
            if not slots:
                unmatched.update(task_ids)
                appended_task_count += len(task_ids)
                repair_task_ids.update(task_ids)
                if task_ids:
                    exact_hit = False
                continue
            unused_slot_count += max(0, len(slots) - len(task_ids))
            for index, task_id in enumerate(task_ids):
                if index >= len(slots):
                    slot = slots[-1]
                    overflow_bind_count += 1
                    repair_task_ids.add(task_id)
                    exact_hit = False
                else:
                    slot = slots[index]
                assignments[slot.slot_id].append(task_id)
                if slot.exact_boundary_key == self._current_p12_target_boundary(task_id):
                    if index < len(slots):
                        exact_bind_count += 1
                else:
                    boundary_mismatch_bind_count += 1
                    repair_task_ids.add(task_id)
                    exact_hit = False
            if len(task_ids) != len(slots):
                exact_hit = False

        repair_task_bytes = sum(
            int(self._current_p12_target_boundary(task_id)[4])
            for task_id in repair_task_ids
        )
        self._current_p12_exact_bind_count = int(exact_bind_count)
        self._current_p12_boundary_mismatch_bind_count = int(boundary_mismatch_bind_count)
        self._current_p12_overflow_bind_count = int(overflow_bind_count)
        self._current_p12_unused_slot_count = int(unused_slot_count)
        self._current_p12_appended_task_count = int(appended_task_count)
        self._current_p12_repair_task_count = int(len(repair_task_ids))
        self._current_p12_repair_task_bytes = int(repair_task_bytes)

        self._current_p12_bound_task_by_slot = {
            slot_id: tuple(task_ids)
            for slot_id, task_ids in assignments.items()
        }
        self._current_p12_unmatched_target_task_ids = unmatched

        if str(self.current_p12_information_mode) == "ZERO_P2":
            status = "ZERO_HINT_BIND"
        elif unmatched:
            status = "EDGE_SUPPORT_REPAIR"
        elif exact_hit:
            status = "EXACT_TEMPLATE_HIT"
        else:
            status = "EDGE_RESIDUAL_BIND"
        repair_reasons: list[str] = []
        if appended_task_count:
            repair_reasons.append("EDGE_SUPPORT_APPEND")
        if overflow_bind_count:
            repair_reasons.append("OVERFLOW_TO_LAST_SLOT")
        if unused_slot_count:
            repair_reasons.append("UNUSED_PREDICTED_SLOT")
        if boundary_mismatch_bind_count:
            repair_reasons.append("BOUNDARY_MISMATCH_BIND")
        self._current_p12_binding_repair_reason = (
            None if not repair_reasons else "+".join(repair_reasons)
        )
        self._current_p12_reconciliation_status = status
        return status

    def _current_p12_combined_preferred_order(self) -> tuple[str, ...]:
        template = self._current_p12_template
        if template is None:
            return ()
        result: list[str] = []
        p1_ids = set(template.p1_task_ids)
        for token in template.ordered_tokens:
            if token in p1_ids:
                result.append(token)
                continue
            result.extend(self._current_p12_bound_task_by_slot.get(token, ()))
        unmatched = sorted(
            self._current_p12_unmatched_target_task_ids,
            key=lambda task_id: (*self._current_p12_target_boundary(task_id), task_id),
        )
        result.extend(task_id for task_id in unmatched if task_id not in result)
        window = self.current_p12_window
        if window is not None:
            for task_id in sorted(
                self.stack.catalogue.task_ids_for_phase(window.p2_dispatch_phase_key),
                key=lambda item: (*self._current_p12_target_boundary(item), item),
            ):
                if task_id not in result:
                    result.append(task_id)
        return tuple(result)

    def _current_p12_combined_preferred_waves(self) -> tuple[tuple[str, ...], ...]:
        template = self._current_p12_template
        if template is None:
            return ()
        p1_ids = set(template.p1_task_ids)
        seen: set[str] = set()
        waves: list[tuple[str, ...]] = []
        for template_wave in template.ordered_waves:
            assignments: list[tuple[str, ...]] = []
            for token in template_wave:
                if token in p1_ids:
                    assignments.append((token,))
                else:
                    assignments.append(
                        tuple(self._current_p12_bound_task_by_slot.get(token, ()))
                    )
            rounds = max((len(items) for items in assignments), default=0)
            for index in range(rounds):
                current = tuple(
                    items[index]
                    for items in assignments
                    if index < len(items) and items[index] not in seen
                )
                if current:
                    waves.append(current)
                    seen.update(current)
        for task_id in self._current_p12_combined_preferred_order():
            if task_id not in seen:
                waves.append((task_id,))
                seen.add(task_id)
        return tuple(waves)

    def _current_p12_target_unfrozen_order(self) -> tuple[str, ...]:
        window = self.current_p12_window
        if window is None:
            return ()
        phase_key = window.p2_dispatch_phase_key
        record = self.stack.authority.record_view(phase_key)
        frozen = set(record.committed_task_ids) | set(record.running_task_ids) | set(record.completed_task_ids)
        remaining = set(self.stack.catalogue.task_ids_for_phase(phase_key)) - frozen
        preferred = self._current_p12_combined_preferred_order()
        ordered = [task_id for task_id in preferred if task_id in remaining]
        ordered.extend(
            task_id
            for task_id in sorted(remaining, key=lambda item: (*self._current_p12_target_boundary(item), item))
            if task_id not in ordered
        )
        return tuple(ordered)

    def _submit_current_p12_incremental_bind(self, *, at_ns: int) -> None:
        window = self.current_p12_window
        if window is None or self._current_p12_template is None:
            return
        status = self._refresh_current_p12_bindings()
        current_ids = frozenset(self.stack.catalogue.task_ids_for_phase(window.p2_dispatch_phase_key))
        if not current_ids or current_ids == self._current_p12_last_planned_target_task_ids:
            return
        # Do not create overlapping authority supersessions.  Keep the newest
        # catalogue state visible, then schedule the delta after the current
        # bind reaches its ExecutionBindingLine boundary.
        if self._current_p12_bind_inflight_job_id is not None:
            return
        new_ids = current_ids - self._current_p12_last_planned_target_task_ids
        self._current_p12_last_planned_target_task_ids = current_ids
        repair = status in {"EDGE_SUPPORT_REPAIR", "ZERO_HINT_BIND"}
        payload = {
            "schema_version": "CURRENT_P12_INCREMENTAL_BIND",
            "planning_window_digest": window.planning_window_digest,
            "new_task_ids": tuple(sorted(new_ids)),
            "all_task_ids": tuple(sorted(current_ids)),
            "reconciliation_status": status,
            "at_ns": int(at_ns),
        }
        job_kind = "CURRENT_P12_REPAIR" if repair else "CURRENT_P12_INCREMENTAL_BIND"
        job_id = f"p12-bind:{stable_digest(payload)}"
        if job_id in self._pipeline_jobs:
            return
        job = PlanningPipelineJob(
            job_id=job_id,
            phase_keys=(window.p2_dispatch_phase_key,),
            observation_batch_digest=stable_digest(payload),
            observation_count=0,
            task_count=len(new_ids),
            hide_until_ns=int(at_ns),
            prediction_required=False,
            global_seal_digests=(),
            job_digest=stable_digest({**payload, "job_kind": job_kind}),
            job_kind=job_kind,
            planning_window_digest=window.planning_window_digest,
        )
        self._pipeline_jobs[job_id] = job
        self._current_p12_bind_payload_by_job[job_id] = payload
        self._current_p12_bind_inflight_job_id = job_id
        if repair:
            self._current_p12_repair_count += 1
            reservation = self.lines.control.submit(
                job_id=job_id,
                arrival_at_ns=int(at_ns),
                duration_ns=self.cost_model.control_duration_ns(
                    observation_count=0,
                    task_count=len(new_ids),
                    phase_count=1,
                ),
                hide_until_ns=int(at_ns),
                payload={"job": job, "p12_bind": payload},
            )
            self._schedule_line_completion(reservation, self.CONTROL_COMPLETE_EVENT, job)
        else:
            reservation = self.lines.execution_binding.submit(
                job_id=job_id,
                arrival_at_ns=int(at_ns),
                duration_ns=self.cost_model.binding_duration_ns(
                    task_count=len(new_ids), phase_count=1
                ),
                hide_until_ns=int(at_ns),
                payload={"job": job, "prepared": None, "p12_bind": payload},
            )
            self._schedule_line_completion(reservation, self.BINDING_COMPLETE_EVENT, job)

    def _note_current_p12_consumer_ready(self, *, at_ns: int) -> None:
        if self.current_p12_window is None or self._current_p12_actual_consume_at_ns is not None:
            return
        self._current_p12_actual_consume_at_ns = int(at_ns)
        self._current_p12_hide_until_ns = int(at_ns)
        if self.overlap_mode == "SERIALIZED" and not self._current_p12_triggered:
            payload = self._current_p12_serialized_trigger_payload
            if payload is None:
                return
            payload = dict(payload)
            payload["hide_until_ns"] = int(at_ns)
            self._current_p12_serialized_trigger_payload = None
            self._schedule(
                time_ns=int(at_ns),
                phase=KernelPhase.THREE_LINE_JOB_TRANSITIONS,
                event_type=self.CURRENT_P12_TRIGGER_EVENT,
                subject_id=self.current_p12_window.planning_window_digest,
                payload=payload,
            )
            return
        job_ids = {
            job_id
            for job_id, job in self._pipeline_jobs.items()
            if job.planning_window_digest == self.current_p12_window.planning_window_digest
        }
        for line in (self.lines.prediction, self.lines.control, self.lines.execution_binding):
            line.reclassify_hide_until(job_ids=job_ids, hide_until_ns=int(at_ns))

    def attach_backend(self, backend: Any) -> None:
        self.backend = backend

    def attach_transport(self, transport: Any) -> None:
        self.transport = transport

    @property
    def permits_by_task_id(self) -> Mapping[str, Any]:
        """Immutable-view compatible permit lookup for formal transport handoff."""

        return self._permits_by_task_id

    def _gate(self, phase_key: Any) -> PlanningGate:
        token = _phase_token(self.adapter, phase_key)
        gate = self._gates_by_phase.get(token)
        if gate is None:
            triggers = (
                self.session.spec.event_triggers
                if self.session.spec.planning_mode is PlanningMode.EVENT
                else None
            )
            gate = PlanningGate(
                self.session.spec.planning_mode,
                event_triggers=triggers,
                defer_global_finalize=(
                    self.session.spec.planning_mode is PlanningMode.GLOBAL
                ),
                max_event_plans=self.max_event_plans_per_phase,
            )
            self._gates_by_phase[token] = gate
        return gate

    def _schedule(
        self,
        *,
        time_ns: int,
        phase: KernelPhase,
        event_type: str,
        subject_id: str,
        payload: Any,
    ) -> SimulationEvent:
        event = self.kernel.schedule(
            time_ns=int(time_ns),
            phase_priority=phase,
            producer="SCHEDULER",
            event_type=event_type,
            ordinal=_ordinal(event_type, subject_id, payload),
            subject_id=str(subject_id),
            attributes=(("payload_digest", stable_digest(payload)),),
        )
        self._payload_by_event_id[event.stable_event_id] = payload
        self._stable_event_ids.append(event.stable_event_id)
        return event

    def emit(
        self,
        *,
        kind: str,
        at_ns: int,
        payload: Mapping[str, Any],
        hide_until_ns: int | None = None,
    ) -> SimulationEvent:
        semantic = dict(payload)
        semantic["kind"] = str(kind)
        requested_hide_until = int(at_ns if hide_until_ns is None else hide_until_ns)
        semantic["hide_until_ns"] = (
            int(at_ns) if self.overlap_mode == "SERIALIZED" else requested_hide_until
        )
        subject = f"{kind}:{stable_digest(semantic)}"
        return self._schedule(
            time_ns=int(at_ns),
            phase=KernelPhase.DESCRIPTOR_OBSERVATION_DELIVERY,
            event_type=self.OBSERVATION_EVENT,
            subject_id=subject,
            payload=semantic,
        )

    def notify_transport_resource_release(self, phase_key: Any) -> PlanningDecision:
        self._mark_dirty(phase_key)
        if (
            self.external_current_p12_trigger
            and self._current_p12_triggered
            and self.session.spec.planning_mode is PlanningMode.EVENT
        ):
            synthetic = CoalescedObservationBatch(
                phase_key=phase_key,
                at_ns=int(self.kernel.now_ns),
                envelopes=(),
                raw_observation_count=1,
                enabled_triggers=(PlanningTrigger.TASK_READY,),
                hide_until_ns=int(self.kernel.now_ns),
                batch_digest=stable_digest({
                    "schema_version": "CURRENT_P12_FRONTIER_REPLAN",
                    "planning_window_digest": self.current_p12_window.planning_window_digest,
                    "phase_key": phase_key,
                    "at_ns": int(self.kernel.now_ns),
                }),
            )
            self._enqueue_pipeline(
                phase_keys=(
                    self.session.phase_keys
                    if self.session.spec.scope is PlannerScope.WINDOW_JOINT
                    else (phase_key,)
                ),
                batch=synthetic,
                global_seal_digests=(),
                prediction_only=False,
                job_kind="CURRENT_P12_EVENT_FRONTIER",
                planning_window_digest=self.current_p12_window.planning_window_digest,
            )
            self._frontier_replan_count += 1
        if self.session.spec.release_mode == ReleaseMode.PHASE_BARRIER:
            # Barrier progression is execution stabilization, not ControlLine
            # replanning.  All already-activated phase plans are reconsidered,
            # while _stabilize admits only the earliest unfinished phase.
            for candidate in self.session.phase_keys:
                if self.stack.authority.active_plan(candidate) is not None:
                    self._mark_dirty(candidate)
        return self._gate(phase_key).on_resource_release()

    def _mark_dirty(self, phase_key: Any) -> None:
        self._dirty_phases[_phase_token(self.adapter, phase_key)] = phase_key

    def _add_envelope(
        self,
        *,
        phase_key: Any,
        at_ns: int,
        observation_id: str,
        trigger: PlanningTrigger,
        changed: bool,
        payload: Any,
        hide_until_ns: int,
        closure_truth: GlobalClosureTruth | None = None,
    ) -> None:
        envelope = ObservationEnvelope(
            observation_id=str(observation_id),
            phase_key=phase_key,
            trigger=PlanningTrigger(trigger),
            at_ns=int(at_ns),
            changed=bool(changed),
            payload_digest=stable_digest(payload),
            hide_until_ns=int(hide_until_ns),
            closure_truth=closure_truth,
        )
        first = self.accumulator.add(envelope)
        if first:
            token = _phase_token(self.adapter, phase_key)
            self._schedule(
                time_ns=int(at_ns),
                phase=KernelPhase.BACKEND_RECEIVER_CLOSURE_RELEASE,
                event_type=self.COALESCE_EVENT,
                subject_id=f"{token}:{int(at_ns)}",
                payload={"phase_key": phase_key, "at_ns": int(at_ns)},
            )

    def _handle_observation(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        payload = self._payload_by_event_id.pop(event.stable_event_id)
        kind = str(payload.pop("kind"))
        hide_until_ns = int(payload.pop("hide_until_ns"))
        updates = 0

        if kind == "RECEIVE_EXPECTATION_AVAILABLE":
            expectation = payload.get("expectation")
            if expectation is None:
                raise FormalRuntimeError("expectation observation omitted expectation")
            view = self.adapter.expectation_view(expectation)
            if (
                self.current_p12_window is not None
                and view.phase_key == self.current_p12_window.p2_dispatch_phase_key
                and self._current_p12_target_first_truth_at_ns is None
            ):
                self._current_p12_target_first_truth_at_ns = int(event.time_ns)
            tasks = self.stack.register_expectation(
                expectation, registered_at_ns=event.time_ns
            )
            phase_token = _phase_token(self.adapter, view.phase_key)
            known_ready = self._source_ready_at.get((phase_token, int(view.src_rank)))
            if known_ready is not None:
                for task in tasks:
                    task_id = self.adapter.task_view(task).task_id
                    self.stack.note_source_payload_ready(task_id, at_ns=known_ready)
            if self.backend is not None:
                self.backend.register_canonical_task_catalogue(tasks)
            trigger = (
                PlanningTrigger.DESCRIPTOR_DELIVERY
                if str(view.phase_key.phase_kind.value) == "DISPATCH"
                else PlanningTrigger.EXPECTATION_AVAILABLE
            )
            self._add_envelope(
                phase_key=view.phase_key,
                at_ns=event.time_ns,
                observation_id=f"EXPECTATION:{view.expectation_digest}",
                trigger=trigger,
                changed=True,
                payload=payload,
                hide_until_ns=hide_until_ns,
            )
            updates += 1

        elif kind == "SOURCE_PAYLOAD_READY":
            phase_key = payload["phase_key"]
            src_rank = int(payload["src_rank"])
            token = _phase_token(self.adapter, phase_key)
            self._source_ready_at[(token, src_rank)] = event.time_ns
            became_ready: list[str] = []
            for task in self.stack.catalogue.tasks_for_phase(phase_key):
                view = self.adapter.task_view(task)
                if int(view.src_rank) != src_rank:
                    continue
                before = self.stack.runtime.facts(view.task_id).state
                self.stack.note_source_payload_ready(view.task_id, at_ns=event.time_ns)
                after = self.stack.runtime.facts(view.task_id).state
                if before != "READY_UNCOMMITTED" and after == "READY_UNCOMMITTED":
                    became_ready.append(view.task_id)
                updates += 1
            if (
                became_ready
                and self.current_p12_window is not None
                and phase_key == self.current_p12_window.p1_combine_phase_key
            ):
                self._note_current_p12_consumer_ready(at_ns=event.time_ns)
            self._add_envelope(
                phase_key=phase_key,
                at_ns=event.time_ns,
                observation_id=f"SOURCE:{src_rank}:{event.time_ns}",
                trigger=PlanningTrigger.SOURCE_PAYLOAD_READY,
                changed=True,
                payload=payload,
                hide_until_ns=hide_until_ns,
            )
            for task_id in became_ready:
                self._add_envelope(
                    phase_key=phase_key,
                    at_ns=event.time_ns,
                    observation_id=f"TASK_READY:{task_id}:{event.time_ns}",
                    trigger=PlanningTrigger.TASK_READY,
                    changed=True,
                    payload={"task_id": task_id},
                    hide_until_ns=hide_until_ns,
                )
            self._mark_dirty(phase_key)

        elif kind == "RECEIVE_PERMIT_GRANTED":
            task_id = str(payload["task_id"])
            permit = payload.get("permit")
            if permit is not None:
                permit_task_id = str(getattr(permit, "task_id", task_id))
                if permit_task_id != task_id:
                    raise FormalRuntimeError("permit task_id does not match observation task_id")
                existing = self._permits_by_task_id.get(task_id)
                if existing is not None and existing != permit:
                    raise FormalRuntimeError("task received conflicting immutable permits")
                self._permits_by_task_id[task_id] = permit
            phase_key = self.stack.catalogue.view(task_id).phase_key
            before = self.stack.runtime.facts(task_id).state
            self.stack.note_receive_permit(task_id, at_ns=event.time_ns)
            after = self.stack.runtime.facts(task_id).state
            if (
                before != "READY_UNCOMMITTED"
                and after == "READY_UNCOMMITTED"
                and self.current_p12_window is not None
                and phase_key == self.current_p12_window.p1_combine_phase_key
            ):
                self._note_current_p12_consumer_ready(at_ns=event.time_ns)
            self._add_envelope(
                phase_key=phase_key,
                at_ns=event.time_ns,
                observation_id=f"PERMIT:{task_id}:{event.time_ns}",
                trigger=PlanningTrigger.PERMIT_GRANTED,
                changed=True,
                payload=payload,
                hide_until_ns=hide_until_ns,
            )
            if before != "READY_UNCOMMITTED" and after == "READY_UNCOMMITTED":
                self._add_envelope(
                    phase_key=phase_key,
                    at_ns=event.time_ns,
                    observation_id=f"TASK_READY:{task_id}:{event.time_ns}",
                    trigger=PlanningTrigger.TASK_READY,
                    changed=True,
                    payload={"task_id": task_id},
                    hide_until_ns=hide_until_ns,
                )
            self._mark_dirty(phase_key)
            updates += 1

        elif kind in {"DISPATCH_DESCRIPTOR_CLOSED", "COMBINE_EXPECTATION_CLOSED"}:
            # In the formal transport/backend runtime, destination-scoped closure is only
            # diagnostic and GLOBAL is gated by PHASE_CLOSURE_SUMMARY_READY.
            # Protocol-level scheduler tests may provide an explicit phase-wide count
            # tuple through this legacy observation; retain that deterministic
            # compatibility path without using it in the formal Backend wiring.
            phase_key = payload["phase_key"]
            legacy_truth = None
            if all(
                name in payload
                for name in (
                    "expected_expectation_count",
                    "expected_task_count",
                    "closure_digest",
                )
            ):
                legacy_truth = GlobalClosureTruth(
                    phase_key=phase_key,
                    expected_expectation_count=int(payload["expected_expectation_count"]),
                    expected_task_count=int(payload["expected_task_count"]),
                    expected_catalogue_digest=payload.get("expected_catalogue_digest"),
                    closure_digest=str(payload["closure_digest"]),
                )
            self._add_envelope(
                phase_key=phase_key,
                at_ns=event.time_ns,
                observation_id=f"DEST_CLOSURE:{kind}:{payload.get('dst_rank')}:{event.time_ns}",
                trigger=PlanningTrigger.OBSERVATION_CLOSURE,
                changed=False,
                payload=payload,
                hide_until_ns=hide_until_ns,
                closure_truth=legacy_truth,
            )
            updates += 1

        elif kind == "PHASE_CLOSURE_SUMMARY_READY":
            phase_key = payload["phase_key"]
            summary = payload.get("summary")
            if summary is None:
                raise FormalRuntimeError("phase closure summary observation omitted summary")
            if not bool(getattr(summary, "seal_ready", False)):
                raise FormalRuntimeError("backend phase closure summary is not seal-ready")
            step = int(self.stack.catalogue.taskizer.spec.effective_chunk_bytes)
            expected_task_count = sum(
                (int(item.expected_payload_bytes) + step - 1) // step
                for item in summary.remote_task_expectation_inputs
            )
            truth = GlobalClosureTruth(
                phase_key=phase_key,
                expected_expectation_count=int(summary.expectation_count),
                expected_task_count=int(expected_task_count),
                expected_catalogue_digest=None,
                closure_digest=str(summary.closure_digest),
            )
            token = _phase_token(self.adapter, phase_key)
            existing = self._closure_truth_by_phase.get(token)
            if existing is not None and existing != truth:
                raise FormalRuntimeError("phase received conflicting Backend closure truth")
            self._closure_truth_by_phase[token] = truth
            self._add_envelope(
                phase_key=phase_key,
                at_ns=event.time_ns,
                observation_id=f"PHASE_CLOSURE:{truth.closure_digest}",
                trigger=PlanningTrigger.OBSERVATION_CLOSURE,
                changed=False,
                payload=payload,
                hide_until_ns=hide_until_ns,
                closure_truth=truth,
            )
            updates += 1
        else:
            # backend also publishes receiver, memory, release, and terminal audit
            # observations through the same public observer port.  They are
            # authoritative evidence but are not scheduler planning triggers.
            return ProgressSignal(notes=(f"scheduler-observation-ignored:{kind}",))

        return ProgressSignal(
            authoritative_state_updates=updates,
            notes=(f"scheduler-observation:{kind}",),
        )

    def _handle_coalesce(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        payload = self._payload_by_event_id.pop(event.stable_event_id)
        phase_key = payload["phase_key"]
        batch = self.accumulator.drain(
            at_ns=int(payload["at_ns"]),
            phase_key=phase_key,
            enabled_event_triggers=self.session.spec.event_triggers,
        )
        gate = self._gate(phase_key)
        mode = self.session.spec.planning_mode
        decision: PlanningDecision
        if self.external_current_p12_trigger:
            if not self._current_p12_triggered:
                return ProgressSignal(notes=("scheduler-current-p12-await-trigger",))
            window = self.current_p12_window
            assert window is not None
            is_target = phase_key == window.p2_dispatch_phase_key
            if is_target:
                # Exact P2 descriptors bind incrementally as soon as they enter
                # the canonical catalogue.  Full-source closure remains an backend
                # destination-compute gate, not a DataPlane launch gate.
                self._submit_current_p12_incremental_bind(at_ns=event.time_ns)
            if mode is PlanningMode.GLOBAL:
                truth = batch.closure_truth
                if truth is not None and is_target:
                    self._schedule(
                        time_ns=event.time_ns,
                        phase=KernelPhase.DEADLOCK_PROGRESS_CHECK,
                        event_type=self.GLOBAL_DEFER_EVENT,
                        subject_id=truth.closure_digest,
                        payload=truth,
                    )
                return ProgressSignal(notes=("scheduler-current-p12-global-template-owned",))
            if mode is PlanningMode.EVENT:
                truth = batch.closure_truth
                if truth is not None and is_target:
                    # EVENT and GLOBAL must seal the exact P2 catalogue at the
                    # same backend closure phase.  Sealing directly in the
                    # observation-coalescing phase let EVENT bind a partially
                    # different wave frontier even when every later replan was
                    # rejected.  Defer through the shared closure path; EVENT
                    # remains distinct through transport-release frontier
                    # replanning after the immutable template is active.
                    self._schedule(
                        time_ns=event.time_ns,
                        phase=KernelPhase.DEADLOCK_PROGRESS_CHECK,
                        event_type=self.GLOBAL_DEFER_EVENT,
                        subject_id=truth.closure_digest,
                        payload=truth,
                    )
                return ProgressSignal(notes=("scheduler-current-p12-event-frontier-owned",))
        if mode is PlanningMode.EVENT:
            if not batch.enabled_triggers:
                return ProgressSignal(notes=("scheduler-coalesced:no-enabled-trigger",))
            decision = gate.on_observation(
                f"COALESCED:{batch.batch_digest}",
                trigger=batch.enabled_triggers[0],
                changed=True,
            )
            if decision.action == "CREATE_PLAN_VERSION":
                phase_keys = (
                    self.session.phase_keys
                    if self.session.spec.scope is PlannerScope.WINDOW_JOINT
                    else (phase_key,)
                )
                self._enqueue_pipeline(
                    phase_keys=phase_keys,
                    batch=batch,
                    global_seal_digests=(),
                )
        elif mode is PlanningMode.GLOBAL:
            truth = batch.closure_truth
            decision = gate.on_observation(
                f"COALESCED:{batch.batch_digest}",
                trigger=PlanningTrigger.OBSERVATION_CLOSURE,
                changed=any(item.changed for item in batch.envelopes),
                closure_satisfied=truth is not None,
            )
            if decision.action == "SCHEDULE_GLOBAL_CATALOGUE_FINALIZE":
                assert truth is not None
                self._schedule(
                    time_ns=event.time_ns,
                    phase=KernelPhase.DEADLOCK_PROGRESS_CHECK,
                    event_type=self.GLOBAL_DEFER_EVENT,
                    subject_id=truth.closure_digest,
                    payload=truth,
                )
        else:
            raise AssertionError(f"unsupported planning mode {mode}")
        return ProgressSignal(
            authoritative_state_updates=1,
            notes=(f"scheduler-coalesced:{batch.raw_observation_count}",),
        )

    def _handle_global_defer(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        truth = self._payload_by_event_id.pop(event.stable_event_id)
        self._schedule(
            time_ns=event.time_ns,
            phase=KernelPhase.BACKEND_RECEIVER_CLOSURE_RELEASE,
            event_type=self.GLOBAL_FINALIZE_EVENT,
            subject_id=truth.closure_digest,
            payload=truth,
        )
        return ProgressSignal(notes=("scheduler-global-finalize-deferred",))

    def _handle_global_finalize(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        truth: GlobalClosureTruth = self._payload_by_event_id.pop(event.stable_event_id)
        phase_key = truth.phase_key
        seal = self.stack.catalogue.seal_phase(
            phase_key,
            expected_expectation_count=truth.expected_expectation_count,
            expected_task_count=truth.expected_task_count,
            expected_catalogue_digest=truth.expected_catalogue_digest,
            closure_digest=truth.closure_digest,
            sealed_at_ns=event.time_ns,
        )
        token = _phase_token(self.adapter, phase_key)
        self._global_seals_by_phase[token] = seal
        if (
            self.external_current_p12_trigger
            and self.current_p12_window is not None
            and phase_key == self.current_p12_window.p2_dispatch_phase_key
        ):
            self._current_p12_target_seal = seal
            self._submit_current_p12_incremental_bind(at_ns=event.time_ns)
            self._current_p12_target_bound = (
                frozenset(self.stack.catalogue.task_ids_for_phase(phase_key))
                == self._current_p12_last_planned_target_task_ids
            )
            return ProgressSignal(
                authoritative_state_updates=1,
                notes=("scheduler-current-p12-target-sealed-incremental",),
            )
        if (
            self.external_current_p12_trigger
            and self.current_p12_window is not None
            and phase_key == self.current_p12_window.p1_combine_phase_key
        ):
            return ProgressSignal(notes=("scheduler-current-p12-p1-already-template-planned",))
        decision = self._gate(phase_key).on_global_catalogue_finalized(
            seal_digest=seal.seal_digest
        )
        if decision.action != "CREATE_PLAN_VERSION":
            return ProgressSignal(notes=("scheduler-global-already-finalized",))

        if self.session.spec.scope is PlannerScope.PHASE_LOCAL:
            # GLOBAL is phase-complete, not whole-window-complete.  Backend
            # PHASE_BARRIER and source readiness enforce causal phase order;
            # waiting for future dependent phases to seal would deadlock.
            batch = self._synthetic_finalize_batch(phase_key, event.time_ns, seal)
            self._enqueue_pipeline(
                phase_keys=(phase_key,),
                batch=batch,
                global_seal_digests=(seal.seal_digest,),
            )
        else:
            # WINDOW_JOINT uses every causally available sealed frontier while
            # preserving one active authority per phase.  Future dependent
            # phases may join in a later activation; already planned phases are
            # never re-created by GLOBAL.
            newly_sealed = tuple(
                item
                for item in self.session.phase_keys
                if (
                    _phase_token(self.adapter, item) in self._global_seals_by_phase
                    and _phase_token(self.adapter, item)
                    not in self._global_joint_planned_tokens
                )
            )
            if newly_sealed:
                seals = tuple(
                    self._global_seals_by_phase[
                        _phase_token(self.adapter, item)
                    ].seal_digest
                    for item in newly_sealed
                )
                self._global_joint_planned_tokens.update(
                    _phase_token(self.adapter, item) for item in newly_sealed
                )
                batch = self._synthetic_finalize_batch(phase_key, event.time_ns, seal)
                self._enqueue_pipeline(
                    phase_keys=newly_sealed,
                    batch=batch,
                    global_seal_digests=seals,
                )
        return ProgressSignal(
            authoritative_state_updates=1,
            notes=("scheduler-global-catalogue-sealed",),
        )

    def _synthetic_finalize_batch(
        self, phase_key: Any, at_ns: int, seal: PhaseCatalogueSeal
    ) -> CoalescedObservationBatch:
        return CoalescedObservationBatch(
            phase_key=phase_key,
            at_ns=int(at_ns),
            envelopes=(),
            raw_observation_count=0,
            enabled_triggers=(PlanningTrigger.OBSERVATION_CLOSURE,),
            hide_until_ns=int(at_ns),
            batch_digest=stable_digest(
                {"global_seal_digest": seal.seal_digest, "at_ns": int(at_ns)}
            ),
        )

    def _enqueue_pipeline(
        self,
        *,
        phase_keys: Iterable[Any],
        batch: CoalescedObservationBatch,
        global_seal_digests: tuple[str, ...],
        prediction_only: bool = False,
        task_count_override: int | None = None,
        job_kind: str = "OBSERVATION_PLAN",
        planning_window_digest: str | None = None,
    ) -> None:
        keys = tuple(dict.fromkeys(phase_keys))
        task_count = (
            sum(len(self.stack.catalogue.task_ids_for_phase(item)) for item in keys)
            if task_count_override is None
            else int(task_count_override)
        )
        prediction_required = bool(self.prediction_enabled or prediction_only)
        if str(job_kind) in {
            "CURRENT_P12_EVENT_FRONTIER",
            "CURRENT_P12_INCREMENTAL_BIND",
            "CURRENT_P12_REPAIR",
        }:
            prediction_required = False
        payload = {
            "phase_tokens": tuple(_phase_token(self.adapter, item) for item in keys),
            "observation_batch_digest": batch.batch_digest,
            "observation_count": batch.raw_observation_count,
            "task_count": task_count,
            "hide_until_ns": batch.hide_until_ns,
            "prediction_required": prediction_required,
            "global_seal_digests": tuple(global_seal_digests),
            "job_kind": str(job_kind),
            "planning_window_digest": planning_window_digest,
        }
        job_id = f"planning-job:{stable_digest(payload)}"
        job = PlanningPipelineJob(
            job_id=job_id,
            phase_keys=keys,
            observation_batch_digest=batch.batch_digest,
            observation_count=batch.raw_observation_count,
            task_count=task_count,
            hide_until_ns=batch.hide_until_ns,
            prediction_required=prediction_required,
            global_seal_digests=tuple(global_seal_digests),
            job_digest=stable_digest(payload),
            job_kind=str(job_kind),
            planning_window_digest=planning_window_digest,
        )
        if job_id in self._pipeline_jobs:
            return
        self._pipeline_jobs[job_id] = job
        if job.prediction_required:
            reservation = self.lines.prediction.submit(
                job_id=job.job_id,
                arrival_at_ns=batch.at_ns,
                duration_ns=self.cost_model.prediction_duration_ns(
                    observation_count=job.observation_count,
                    task_count=job.task_count,
                ),
                hide_until_ns=job.hide_until_ns,
                payload=job,
            )
            self._schedule_line_completion(
                reservation, self.PREDICTION_COMPLETE_EVENT, job
            )
        else:
            self._enqueue_control(job, arrival_at_ns=batch.at_ns)

    def _schedule_line_completion(
        self,
        reservation: LineReservation,
        event_type: str,
        job: PlanningPipelineJob,
    ) -> None:
        self._schedule(
            time_ns=reservation.finish_at_ns,
            phase=KernelPhase.THREE_LINE_JOB_TRANSITIONS,
            event_type=event_type,
            subject_id=f"{job.job_id}:{reservation.line_name}",
            payload={"job": job, "reservation": reservation},
        )

    def _enqueue_control(self, job: PlanningPipelineJob, *, arrival_at_ns: int) -> None:
        reservation = self.lines.control.submit(
            job_id=job.job_id,
            arrival_at_ns=int(arrival_at_ns),
            duration_ns=self.cost_model.control_duration_ns(
                observation_count=job.observation_count,
                task_count=job.task_count,
                phase_count=len(job.phase_keys),
            ),
            hide_until_ns=job.hide_until_ns,
            payload=job,
        )
        self._schedule_line_completion(reservation, self.CONTROL_COMPLETE_EVENT, job)

    def _handle_prediction_complete(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        payload = self._payload_by_event_id.pop(event.stable_event_id)
        job: PlanningPipelineJob = payload["job"]
        self._enqueue_control(job, arrival_at_ns=event.time_ns)
        return ProgressSignal(notes=("scheduler-prediction-complete",))

    def _handle_control_complete(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        payload = self._payload_by_event_id.pop(event.stable_event_id)
        job: PlanningPipelineJob = payload["job"]
        if job.job_kind == "CURRENT_P12_REPAIR":
            reservation = self.lines.execution_binding.submit(
                job_id=job.job_id,
                arrival_at_ns=event.time_ns,
                duration_ns=self.cost_model.binding_duration_ns(
                    task_count=job.task_count, phase_count=1
                ),
                hide_until_ns=job.hide_until_ns,
                payload={"job": job, "prepared": None, "p12_bind": self._current_p12_bind_payload_by_job[job.job_id]},
            )
            self._schedule_line_completion(reservation, self.BINDING_COMPLETE_EVENT, job)
            return ProgressSignal(notes=("scheduler-current-p12-repair-control-complete",))

        if (
            job.job_kind == "CURRENT_P12_TEMPLATE"
            and self.current_p12_window is not None
            and self.session.spec.scope is PlannerScope.WINDOW_JOINT
        ):
            window = self.current_p12_window
            predicted_slots = build_predicted_p2_slots(
                planning_window_digest=window.planning_window_digest,
                p2_phase_token=_phase_token(self.adapter, window.p2_dispatch_phase_key),
                predicted_matrix=self.current_p12_predicted_p2_matrix or (),
                chunk_bytes=self.stack.catalogue.taskizer.spec.effective_chunk_bytes,
            )
            template, prepared = self.session.prepare_current_p12_template(
                p1_phase_key=window.p1_combine_phase_key,
                p2_phase_key=window.p2_dispatch_phase_key,
                predicted_slots=predicted_slots,
                planning_window_digest=window.planning_window_digest,
                now_ns=event.time_ns,
            )
            self._current_p12_template = template
            diagnostics = (
                {} if prepared is None else dict(prepared.algorithm_plan.diagnostics)
            )
            choice = diagnostics.get("safe_selector_choice")
            self._current_p12_safe_selector_choice = (
                None if choice is None else str(choice)
            )
            reason = diagnostics.get("safe_selector_reason")
            self._current_p12_safe_selector_reason = (
                None if reason is None else str(reason)
            )
            local_objective = diagnostics.get("safe_selector_local_estimated_objective")
            joint_objective = diagnostics.get("safe_selector_joint_estimated_objective")
            self._current_p12_safe_selector_local_objective = (
                None if local_objective is None else int(local_objective)
            )
            self._current_p12_safe_selector_joint_objective = (
                None if joint_objective is None else int(joint_objective)
            )
            self._current_p12_algorithm_core_runs += 1
            self._prepared_by_job[job.job_id] = prepared
            self._current_p12_template_ready_at_ns = int(event.time_ns)
            self._submit_current_p12_incremental_bind(at_ns=event.time_ns)
        else:
            prepared = self.session.prepare_activation(
                now_ns=event.time_ns,
                phase_keys=job.phase_keys,
                respect_release_barrier=not (
                    self.session.spec.planning_mode is PlanningMode.GLOBAL
                    and self.session.spec.release_mode == ReleaseMode.PHASE_BARRIER
                ),
            )
            if (
                prepared is not None
                and job.job_kind == "CURRENT_P12_EVENT_FRONTIER"
                and not self.session.event_replan_improves_current_frontier(prepared)
            ):
                # Reject before ExecutionBindingLine admission.  A no-op EVENT
                # replan must not queue behind or delay exact incremental P2
                # binding merely to discover at activation time that the
                # immutable-window suffix was already better.
                return ProgressSignal(
                    notes=("scheduler-current-p12-event-replan-rejected-before-bind",)
                )
            self._prepared_by_job[job.job_id] = prepared

        if prepared is not None:
            self._algorithm_diagnostics_rows.append(
                {
                    "job_id": str(job.job_id),
                    "job_kind": str(job.job_kind),
                    "planning_window_digest": job.planning_window_digest,
                    "algorithm_id": str(prepared.algorithm_plan.algorithm_id),
                    "plan_digest": str(prepared.algorithm_plan.plan_digest),
                    "diagnostics": dict(prepared.algorithm_plan.diagnostics),
                }
            )

        task_count = 0 if prepared is None else len(prepared.algorithm_plan.ordered_task_ids)
        phase_count = 0 if prepared is None else len(prepared.phase_orders)
        reservation = self.lines.execution_binding.submit(
            job_id=job.job_id,
            arrival_at_ns=event.time_ns,
            duration_ns=self.cost_model.binding_duration_ns(
                task_count=task_count, phase_count=phase_count
            ),
            hide_until_ns=job.hide_until_ns,
            payload={"job": job, "prepared": prepared},
        )
        self._schedule_line_completion(reservation, self.BINDING_COMPLETE_EVENT, job)
        return ProgressSignal(notes=("scheduler-control-complete",))

    def _handle_binding_complete(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        payload = self._payload_by_event_id.pop(event.stable_event_id)
        job: PlanningPipelineJob = payload["job"]
        if job.job_kind in {"CURRENT_P12_INCREMENTAL_BIND", "CURRENT_P12_REPAIR"}:
            window = self.current_p12_window
            assert window is not None
            phase_key = window.p2_dispatch_phase_key
            active = None
            order = self._current_p12_target_unfrozen_order()
            if order:
                active = self.stack.activate_plan(
                    phase_key=phase_key,
                    window_key=window.window_key,
                    ordered_task_ids=order,
                    now_ns=event.time_ns,
                )
                combined_waves = self._current_p12_combined_preferred_waves()
                if combined_waves:
                    self.session.set_external_preferred_waves(combined_waves)
                else:
                    self.session.set_external_preferred_task_ids(
                        self._current_p12_combined_preferred_order()
                    )
            if active is not None:
                token = _phase_token(self.adapter, phase_key)
                self._phase_plan_counts[token] += 1
                self._mark_dirty(phase_key)
                self._current_p12_target_bound_at_ns = int(event.time_ns)
            self._current_p12_bind_inflight_job_id = None
            self._current_p12_bind_payload_by_job.pop(job.job_id, None)
            if self._current_p12_target_seal is not None:
                catalogue_ids = frozenset(self.stack.catalogue.task_ids_for_phase(phase_key))
                self._current_p12_target_bound = (
                    catalogue_ids == self._current_p12_last_planned_target_task_ids
                )
            # Descriptors may have arrived while this bind job was queued.
            # Schedule exactly one follow-up delta without rerunning the core.
            self._submit_current_p12_incremental_bind(at_ns=event.time_ns)
            evidence_payload = {
                "job_id": job.job_id,
                "prepared_digest": None,
                "activation_digest": None if active is None else str(self.adapter.plan_view(active).plan_digest),
                "activated_phase_tokens": (() if active is None else (_phase_token(self.adapter, phase_key),)),
                "activated_at_ns": event.time_ns,
                "stale_skipped": False,
            }
            self._activation_evidence.append(
                RuntimeActivationEvidence(
                    **evidence_payload,
                    evidence_digest=stable_digest(evidence_payload),
                )
            )
            return ProgressSignal(
                authoritative_state_updates=1 if active is not None else 0,
                notes=("scheduler-current-p12-incremental-bind-complete",),
            )
        prepared = self._prepared_by_job.pop(job.job_id, None)
        activation = None
        stale = False
        event_replan_rejected = False
        if (
            prepared is not None
            and job.job_kind == "CURRENT_P12_EVENT_FRONTIER"
            and not self.session.event_replan_improves_current_frontier(prepared)
        ):
            prepared = None
            event_replan_rejected = True
        if prepared is not None:
            activation = self.session.activate_prepared(
                prepared,
                activated_at_ns=event.time_ns,
                skip_if_stale=(
                    self.session.spec.planning_mode is PlanningMode.EVENT
                ),
            )
            stale = activation is None
        if stale:
            self._stale_activation_count += 1
        if job.job_kind == "CURRENT_P12_EVENT_FRONTIER" and activation is not None:
            activation_waves = tuple(
                tuple(wave.task_ids) for wave in activation.algorithm_plan.waves if wave.task_ids
            )
            if activation_waves:
                self.session.set_external_preferred_waves(activation_waves)
            else:
                self.session.set_external_preferred_task_ids(
                    activation.algorithm_plan.ordered_task_ids
                )
        activated_tokens: tuple[str, ...] = ()
        if activation is not None:
            activated_tokens = tuple(
                _phase_token(self.adapter, phase_key)
                for phase_key, _ in activation.phase_plan_ids
            )
            for phase_key, _ in activation.phase_plan_ids:
                token = _phase_token(self.adapter, phase_key)
                self._phase_plan_counts[token] += 1
                self._mark_dirty(phase_key)
            if self.session.spec.planning_mode is PlanningMode.GLOBAL:
                for phase_key, _ in activation.phase_plan_ids:
                    seal = self.stack.catalogue.phase_seal(phase_key)
                    if seal is None:
                        raise CatalogueSealError("GLOBAL activated an unsealed phase")
                    plan = self.stack.authority.active_plan(phase_key)
                    assert plan is not None
                    plan_view = self.adapter.plan_view(plan)
                    if set(plan_view.remaining_task_ids) != set(
                        self.stack.catalogue.task_ids_for_phase(phase_key)
                    ):
                        raise CatalogueSealError(
                            "GLOBAL plan does not cover the complete sealed catalogue"
                        )
        evidence_payload = {
            "job_id": job.job_id,
            "prepared_digest": None if prepared is None else prepared.prepared_digest,
            "activation_digest": None if activation is None else activation.activation_digest,
            "activated_phase_tokens": activated_tokens,
            "activated_at_ns": event.time_ns,
            "stale_skipped": stale,
        }
        self._activation_evidence.append(
            RuntimeActivationEvidence(
                **evidence_payload,
                evidence_digest=stable_digest(evidence_payload),
            )
        )
        return ProgressSignal(
            authoritative_state_updates=1 if activation is not None else 0,
            notes=(
                "scheduler-current-p12-event-replan-rejected"
                if event_replan_rejected
                else "scheduler-binding-complete",
            ),
        )

    def _stabilize(self, kernel: SimulationKernel) -> ProgressSignal:
        if self.transport is None or not self._dirty_phases:
            return ProgressSignal()
        commits = 0
        if self.session.spec.scope is PlannerScope.WINDOW_JOINT:
            snapshot = self.transport.snapshot()
            snapshot_digest = getattr(snapshot, "snapshot_digest", None) or stable_digest(snapshot)
            _, decision = self.session.arbitrate(
                transport_snapshot_digest=str(snapshot_digest),
                observed_at_ns=kernel.now_ns,
                max_prefix_tasks=self.max_window_prefix_tasks,
            )
            if decision.selected_phase_token is None:
                return ProgressSignal()
            phase_key = next(
                item
                for item in self.session.phase_keys
                if _phase_token(self.adapter, item) == decision.selected_phase_token
            )
            result = self.stack.stabilizer.stabilize(
                phase_key=phase_key,
                snapshot_provider=self.transport.snapshot,
                transport=self.transport,
                now_ns=kernel.now_ns,
                allowed_task_ids=decision.selected_task_ids,
            )
            commits += len(result.accepted_task_ids)
            token = _phase_token(self.adapter, phase_key)
            active = self.stack.authority.active_plan(phase_key)
            ready_remaining = False
            if active is not None:
                active_view = self.adapter.plan_view(active)
                ready_remaining = any(
                    self.stack.runtime.facts(task_id).state == "READY_UNCOMMITTED"
                    for task_id in active_view.remaining_task_ids
                )
            if result.terminal_code == "RETRYABLE_RESOURCE_BUSY" or ready_remaining:
                self._dirty_phases[token] = phase_key
            else:
                self._dirty_phases.pop(token, None)
        else:
            remaining: dict[str, Any] = {}
            dirty_items = tuple(
                (token, self._dirty_phases[token]) for token in sorted(self._dirty_phases)
            )
            if self.session.spec.release_mode == ReleaseMode.PHASE_BARRIER:
                eligible_phase = next(
                    (
                        phase_key
                        for phase_key in self.session.phase_keys
                        if self.stack.authority.active_plan(phase_key) is not None
                        and self.session.phase_has_unfinished_tasks(phase_key)
                    ),
                    None,
                )
                dirty_items = tuple(
                    (token, phase_key)
                    for token, phase_key in dirty_items
                    if phase_key == eligible_phase
                )
                for token, phase_key in self._dirty_phases.items():
                    if phase_key != eligible_phase:
                        remaining[token] = phase_key
            for token, phase_key in dirty_items:
                result = self.stack.stabilizer.stabilize(
                    phase_key=phase_key,
                    snapshot_provider=self.transport.snapshot,
                    transport=self.transport,
                    now_ns=kernel.now_ns,
                )
                commits += len(result.accepted_task_ids)
                active = self.stack.authority.active_plan(phase_key)
                ready_remaining = False
                if active is not None:
                    active_view = self.adapter.plan_view(active)
                    ready_remaining = any(
                        self.stack.runtime.facts(task_id).state == "READY_UNCOMMITTED"
                        for task_id in active_view.remaining_task_ids
                    )
                if result.terminal_code == "RETRYABLE_RESOURCE_BUSY" or ready_remaining:
                    remaining[token] = phase_key
            self._dirty_phases = remaining
        return ProgressSignal(
            successful_commits=commits,
            notes=((f"scheduler-stabilized:{commits}",) if commits else ()),
        )

    def plan_count(self, phase_key: Any) -> int:
        return int(self._phase_plan_counts[_phase_token(self.adapter, phase_key)])

    def algorithm_diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._algorithm_diagnostics_rows)

    def metrics(self) -> FormalSchedulingRuntimeMetrics:
        raw = self.accumulator.raw_observation_count
        batches = self.accumulator.coalesced_batch_count
        p12_evidence: tuple[CurrentP12TemplateEvidence, ...] = ()
        if self.current_p12_window is not None and self._current_p12_trigger_at_ns is not None:
            digest = self.current_p12_window.planning_window_digest
            def _line_totals(line: Any) -> tuple[int, int, int]:
                rows = tuple(
                    reservation
                    for reservation in line.reservations
                    if (
                        reservation.job_id in self._pipeline_jobs
                        and self._pipeline_jobs[reservation.job_id].planning_window_digest == digest
                    )
                )
                return (
                    sum(item.duration_ns for item in rows),
                    sum(item.hidden_service_ns for item in rows),
                    sum(item.exposed_delay_ns for item in rows),
                )
            prediction_totals = _line_totals(self.lines.prediction)
            control_totals = _line_totals(self.lines.control)
            binding_totals = _line_totals(self.lines.execution_binding)
            margin = (
                None
                if self._current_p12_template_ready_at_ns is None or self._current_p12_target_first_truth_at_ns is None
                else int(self._current_p12_target_first_truth_at_ns) - int(self._current_p12_template_ready_at_ns)
            )
            bind_wait = (
                None
                if self._current_p12_target_bound_at_ns is None or self._current_p12_target_first_truth_at_ns is None
                else int(self._current_p12_target_bound_at_ns) - int(self._current_p12_target_first_truth_at_ns)
            )
            predicted_matrix = self.current_p12_predicted_p2_matrix
            bound_prediction_task_count = int(
                self._current_p12_exact_bind_count
                + self._current_p12_boundary_mismatch_bind_count
                + self._current_p12_overflow_bind_count
            )
            prediction_state = evaluate_p12_prediction_evidence(
                information_mode=str(self.current_p12_information_mode),
                matrix=predicted_matrix,
                rank_count=int(self.session.spec.rank_count),
                plan_materialized=self._current_p12_template is not None,
                algorithm_core_run_count=int(self._current_p12_algorithm_core_runs),
                bound_task_count=bound_prediction_task_count,
                expected_target_task_count=len(self._current_p12_seen_target_task_ids),
                fallback_reason=self._current_p12_prediction_fallback_reason,
            )
            evidence_payload = {
                "planning_window_digest": digest,
                "trigger_phase_token": _phase_token(self.adapter, self.current_p12_window.p0_trigger_phase_key),
                "p1_phase_token": _phase_token(self.adapter, self.current_p12_window.p1_combine_phase_key),
                "p2_phase_token": _phase_token(self.adapter, self.current_p12_window.p2_dispatch_phase_key),
                "information_mode": str(self.current_p12_information_mode),
                "trigger_at_ns": int(self._current_p12_trigger_at_ns),
                "hide_until_ns": int(self._current_p12_hide_until_ns or self._current_p12_trigger_at_ns),
                "template_ready_at_ns": self._current_p12_template_ready_at_ns,
                "target_first_truth_at_ns": self._current_p12_target_first_truth_at_ns,
                "target_bound_at_ns": self._current_p12_target_bound_at_ns,
                "template_ready_margin_ns": margin,
                "target_bind_wait_ns": bind_wait,
                "reconciliation_status": self._current_p12_reconciliation_status,
                "safe_selector_choice": self._current_p12_safe_selector_choice,
                "safe_selector_reason": self._current_p12_safe_selector_reason,
                "safe_selector_local_objective": self._current_p12_safe_selector_local_objective,
                "safe_selector_joint_objective": self._current_p12_safe_selector_joint_objective,
                "template_digest": (
                    None if self._current_p12_template is None else self._current_p12_template.template_digest
                ),
                "predicted_p2_slot_count": (
                    0 if self._current_p12_template is None else len(self._current_p12_template.p2_slots)
                ),
                "bound_exact_p2_task_count": sum(
                    len(task_ids) for task_ids in self._current_p12_bound_task_by_slot.values()
                ),
                "unmatched_exact_p2_task_count": len(self._current_p12_unmatched_target_task_ids),
                "exact_bind_count": int(self._current_p12_exact_bind_count),
                "boundary_mismatch_bind_count": int(self._current_p12_boundary_mismatch_bind_count),
                "overflow_bind_count": int(self._current_p12_overflow_bind_count),
                "unused_slot_count": int(self._current_p12_unused_slot_count),
                "appended_task_count": int(self._current_p12_appended_task_count),
                "repair_task_count": int(self._current_p12_repair_task_count),
                "repair_task_bytes": int(self._current_p12_repair_task_bytes),
                "repair_task_ratio_ppm": (
                    0
                    if not self._current_p12_seen_target_task_ids
                    else int(round(1_000_000 * self._current_p12_repair_task_count / len(self._current_p12_seen_target_task_ids)))
                ),
                "repair_byte_ratio_ppm": (
                    0
                    if self._current_p12_total_target_task_bytes <= 0
                    else int(round(1_000_000 * self._current_p12_repair_task_bytes / self._current_p12_total_target_task_bytes))
                ),
                "binding_repair_reason": self._current_p12_binding_repair_reason,
                "prediction_fallback_reason": prediction_state.fallback_reason,
                "prediction_generated": bool(prediction_state.generated),
                "prediction_nonempty": bool(prediction_state.nonempty),
                "prediction_validated": bool(prediction_state.validated),
                "prediction_consumed": bool(prediction_state.consumed),
                "prediction_fallback": bool(prediction_state.fallback),
                "algorithm_core_run_count": int(self._current_p12_algorithm_core_runs),
                "repair_count": int(self._current_p12_repair_count),
                "incremental_bind_job_count": sum(
                    1
                    for item in self._pipeline_jobs.values()
                    if item.planning_window_digest == digest
                    and item.job_kind in {"CURRENT_P12_INCREMENTAL_BIND", "CURRENT_P12_REPAIR"}
                ),
                "prediction_service_ns": prediction_totals[0],
                "prediction_hidden_ns": prediction_totals[1],
                "prediction_exposed_ns": prediction_totals[2],
                "control_service_ns": control_totals[0],
                "control_hidden_ns": control_totals[1],
                "control_exposed_ns": control_totals[2],
                "binding_service_ns": binding_totals[0],
                "binding_hidden_ns": binding_totals[1],
                "binding_exposed_ns": binding_totals[2],
                "prediction_digest": self.current_p12_prediction_digest,
            }
            p12_evidence = (CurrentP12TemplateEvidence(
                **evidence_payload,
                evidence_digest=stable_digest(evidence_payload),
            ),)
        payload = {
            "raw_observation_count": raw,
            "coalesced_batch_count": batches,
            "coalesced_observation_savings": max(0, raw - batches),
            "pipeline_job_count": len(self._pipeline_jobs),
            "activated_plan_count": sum(self._phase_plan_counts.values()),
            "stale_activation_count": self._stale_activation_count,
            "global_seal_count": len(self._global_seals_by_phase),
            "phase_plan_counts": tuple(sorted(self._phase_plan_counts.items())),
            "line_metrics": self.lines.metrics(),
            "stable_event_ids": tuple(self._stable_event_ids),
            "activation_evidence": tuple(self._activation_evidence),
            "current_p12_template_evidence": p12_evidence,
            "frontier_replan_count": int(self._frontier_replan_count),
        }
        return FormalSchedulingRuntimeMetrics(
            **payload,
            metrics_digest=stable_digest(payload),
        )


__all__ = [
    "CoalescedObservationBatch",
    "CurrentP12TemplateEvidence",
    "FormalSchedulingRuntimeAdapter",
    "FormalSchedulingRuntimeMetrics",
    "GlobalClosureTruth",
    "ObservationEnvelope",
    "PhaseObservationAccumulator",
    "PlanningPipelineJob",
    "RuntimeActivationEvidence",
]
