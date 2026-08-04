from __future__ import annotations

"""Stable evidence records and observation coalescing for the formal scheduler.

Keeping these immutable records outside the event adapter prevents reporting
schema changes from expanding the already large runtime state machine.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from rs_sim.scheduler.errors import FormalRuntimeError
from rs_sim.scheduler.execution.lines import ServiceLineMetrics
from rs_sim.scheduler.decorators.planning_gate import PlanningTrigger
from rs_sim.scheduler.stable import stable_digest, stable_json

def _ordinal(*parts: Any) -> int:
    digest = hashlib.sha256(stable_json(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _phase_token(adapter: Any, phase_key: Any) -> str:
    return stable_json(adapter.phase_payload(phase_key))


@dataclass(frozen=True, slots=True)
class GlobalClosureTruth:
    phase_key: Any
    expected_expectation_count: int
    expected_task_count: int
    closure_digest: str
    expected_catalogue_digest: str | None = None

    def __post_init__(self) -> None:
        if self.expected_expectation_count < 0 or self.expected_task_count < 0:
            raise ValueError("GLOBAL closure counts must be non-negative")
        if not isinstance(self.closure_digest, str) or not self.closure_digest:
            raise ValueError("closure_digest must be non-empty")
        if self.expected_catalogue_digest is not None and not self.expected_catalogue_digest:
            raise ValueError("expected_catalogue_digest must be non-empty or None")


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    observation_id: str
    phase_key: Any
    trigger: PlanningTrigger
    at_ns: int
    changed: bool
    payload_digest: str
    hide_until_ns: int
    closure_truth: GlobalClosureTruth | None = None

    @property
    def envelope_digest(self) -> str:
        return stable_digest(self)


@dataclass(frozen=True, slots=True)
class CoalescedObservationBatch:
    phase_key: Any
    at_ns: int
    envelopes: tuple[ObservationEnvelope, ...]
    raw_observation_count: int
    enabled_triggers: tuple[PlanningTrigger, ...]
    hide_until_ns: int
    batch_digest: str

    @property
    def closure_truth(self) -> GlobalClosureTruth | None:
        truths = tuple(item.closure_truth for item in self.envelopes if item.closure_truth is not None)
        if not truths:
            return None
        first = truths[0]
        if any(item != first for item in truths[1:]):
            raise FormalRuntimeError("coalesced observations contain conflicting closure truth")
        return first


class PhaseObservationAccumulator:
    """Deterministically coalesce same-time same-phase observation deliveries."""

    def __init__(self, *, adapter: Any) -> None:
        self.adapter = adapter
        self._buckets: dict[tuple[int, str], dict[str, ObservationEnvelope]] = {}
        self.raw_observation_count = 0
        self.coalesced_batch_count = 0

    def add(self, envelope: ObservationEnvelope) -> bool:
        token = _phase_token(self.adapter, envelope.phase_key)
        key = (int(envelope.at_ns), token)
        bucket = self._buckets.setdefault(key, {})
        self.raw_observation_count += 1
        first = not bucket
        existing = bucket.get(envelope.observation_id)
        if existing is not None and existing != envelope:
            raise FormalRuntimeError("observation_id was reused with conflicting semantics")
        bucket[envelope.observation_id] = envelope
        return first

    def drain(
        self,
        *,
        at_ns: int,
        phase_key: Any,
        enabled_event_triggers: Iterable[PlanningTrigger],
    ) -> CoalescedObservationBatch:
        token = _phase_token(self.adapter, phase_key)
        key = (int(at_ns), token)
        try:
            bucket = self._buckets.pop(key)
        except KeyError as exc:
            raise FormalRuntimeError("missing observation coalescing bucket") from exc
        envelopes = tuple(
            sorted(
                bucket.values(),
                key=lambda item: (
                    item.trigger.value,
                    item.observation_id,
                    item.envelope_digest,
                ),
            )
        )
        enabled = frozenset(PlanningTrigger(item) for item in enabled_event_triggers)
        enabled_triggers = tuple(
            sorted(
                {
                    item.trigger
                    for item in envelopes
                    if item.changed and item.trigger in enabled
                },
                key=lambda item: item.value,
            )
        )
        hide_until = max((item.hide_until_ns for item in envelopes), default=int(at_ns))
        payload = {
            "phase_token": token,
            "at_ns": int(at_ns),
            "envelope_digests": tuple(item.envelope_digest for item in envelopes),
            "enabled_triggers": tuple(item.value for item in enabled_triggers),
            "hide_until_ns": hide_until,
        }
        self.coalesced_batch_count += 1
        return CoalescedObservationBatch(
            phase_key=phase_key,
            at_ns=int(at_ns),
            envelopes=envelopes,
            raw_observation_count=len(envelopes),
            enabled_triggers=enabled_triggers,
            hide_until_ns=hide_until,
            batch_digest=stable_digest(payload),
        )


@dataclass(frozen=True, slots=True)
class PlanningPipelineJob:
    job_id: str
    phase_keys: tuple[Any, ...]
    observation_batch_digest: str
    observation_count: int
    task_count: int
    hide_until_ns: int
    prediction_required: bool
    global_seal_digests: tuple[str, ...]
    job_digest: str
    job_kind: str = "OBSERVATION_PLAN"
    planning_window_digest: str | None = None


@dataclass(frozen=True, slots=True)
class CurrentP12TemplateEvidence:
    planning_window_digest: str
    trigger_phase_token: str
    p1_phase_token: str
    p2_phase_token: str | None
    information_mode: str
    trigger_at_ns: int
    hide_until_ns: int
    template_ready_at_ns: int | None
    target_first_truth_at_ns: int | None
    target_bound_at_ns: int | None
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
    prediction_digest: str | None
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class RuntimeActivationEvidence:
    job_id: str
    prepared_digest: str | None
    activation_digest: str | None
    activated_phase_tokens: tuple[str, ...]
    activated_at_ns: int
    stale_skipped: bool
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class FormalSchedulingRuntimeMetrics:
    raw_observation_count: int
    coalesced_batch_count: int
    coalesced_observation_savings: int
    pipeline_job_count: int
    activated_plan_count: int
    stale_activation_count: int
    global_seal_count: int
    phase_plan_counts: tuple[tuple[str, int], ...]
    line_metrics: tuple[ServiceLineMetrics, ...]
    stable_event_ids: tuple[str, ...]
    activation_evidence: tuple[RuntimeActivationEvidence, ...]
    current_p12_template_evidence: tuple[CurrentP12TemplateEvidence, ...]
    frontier_replan_count: int
    metrics_digest: str

# Preserve the established public import/pickle path.  runtime_adapter re-exports
# these records while this module owns their implementation.
for _public_type in (
    GlobalClosureTruth,
    ObservationEnvelope,
    CoalescedObservationBatch,
    PhaseObservationAccumulator,
    PlanningPipelineJob,
    CurrentP12TemplateEvidence,
    RuntimeActivationEvidence,
    FormalSchedulingRuntimeMetrics,
):
    _public_type.__module__ = "rs_sim.scheduler.execution.runtime_adapter"
del _public_type
