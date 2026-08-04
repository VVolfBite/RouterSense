from __future__ import annotations

"""Scheduler result records, paired identities, and deterministic statistics."""

import enum
import hashlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from rs_sim.scheduler.planning.planner import (
    AlgorithmPlan,
    OrderOnlyPlanner,
    SchedulingProblem,
    validate_order_only_pair,
)
from rs_sim.scheduler.stable import stable_digest


class RunStatus(str, enum.Enum):
    COMPLETED = "COMPLETED"
    UNSUPPORTED = "UNSUPPORTED"
    TIME_LIMIT = "TIME_LIMIT"
    CONTRACT_ERROR = "CONTRACT_ERROR"
    FAILED = "FAILED"


def _digest_safe(value: Any) -> Any:
    """Encode non-authoritative numeric diagnostics without float serialization."""

    if isinstance(value, float):
        return {"decimal": format(value, ".17g")}
    if isinstance(value, dict):
        return {str(key): _digest_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_digest_safe(item) for item in value)
    if isinstance(value, list):
        return tuple(_digest_safe(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PairedInstanceKey:
    run_id: str
    sample_id: str
    window_index: int
    rank_count: int
    task_catalogue_digest: str
    task_boundary_digest: str
    workload_digest: str
    topology_digest: str
    hardware_profile_digest: str
    information_digest: str
    cost_model_digest: str
    fixture_id: str = ""
    anchor_layer_id: int | None = None
    horizon: str | None = None
    window_truth_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "sample_id",
            "task_catalogue_digest",
            "task_boundary_digest",
            "workload_digest",
            "topology_digest",
            "hardware_profile_digest",
            "information_digest",
            "cost_model_digest",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.window_index < 0 or self.rank_count <= 0:
            raise ValueError("window_index/rank_count invalid")
        anchor_fields = (self.anchor_layer_id, self.horizon, self.window_truth_digest)
        if any(value is not None for value in anchor_fields):
            if self.anchor_layer_id is None or self.anchor_layer_id < 0:
                raise ValueError("anchor_layer_id must be non-negative for anchor-local keys")
            if self.horizon != "P12":
                raise ValueError("horizon must be P12 for anchor-local keys")
            if not isinstance(self.window_truth_digest, str) or not self.window_truth_digest:
                raise ValueError("window_truth_digest must be non-empty for anchor-local keys")
            if not isinstance(self.fixture_id, str) or not self.fixture_id:
                raise ValueError("fixture_id must be non-empty for anchor-local keys")

    @property
    def paired_key_digest(self) -> str:
        return stable_digest(self)


@dataclass(frozen=True, slots=True)
class Provenance:
    trace_source: str
    trace_digest: str
    payload_profile_source: str
    compute_profile_source: str
    hardware_profile_source: str
    hardware_profile_digest: str
    calibration_state: str
    synthetic_components: tuple[str, ...]
    performance_eligible: bool

    def __post_init__(self) -> None:
        for name in (
            "trace_source",
            "trace_digest",
            "payload_profile_source",
            "compute_profile_source",
            "hardware_profile_source",
            "hardware_profile_digest",
            "calibration_state",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.synthetic_components, tuple):
            raise TypeError("synthetic_components must be tuple")
        if self.synthetic_components and self.performance_eligible:
            raise ValueError("synthetic provenance cannot be performance eligible")

    @property
    def provenance_digest(self) -> str:
        return stable_digest(self)


@dataclass(frozen=True, slots=True)
class AlgorithmRunRecord:
    paired_key: PairedInstanceKey
    algorithm_id: str
    status: RunStatus
    provenance: Provenance
    plan_digest: str | None
    objective_value: int | float | None
    objective_unit: str | None
    failure_code: str | None
    failure_message: str | None
    result_payload: tuple[tuple[str, Any], ...]
    record_digest: str


def make_paired_key(
    *,
    problem: SchedulingProblem,
    run_id: str,
    sample_id: str,
    window_index: int,
    workload_digest: str,
    topology_digest: str,
    hardware_profile_digest: str,
    fixture_id: str = "",
    anchor_layer_id: int | None = None,
    horizon: str | None = None,
    window_truth_digest: str | None = None,
) -> PairedInstanceKey:
    return PairedInstanceKey(
        run_id=str(run_id),
        sample_id=str(sample_id),
        window_index=int(window_index),
        rank_count=problem.rank_count,
        task_catalogue_digest=problem.fairness.task_catalogue_digest,
        task_boundary_digest=problem.fairness.task_boundary_digest,
        workload_digest=str(workload_digest),
        topology_digest=str(topology_digest),
        hardware_profile_digest=str(hardware_profile_digest),
        information_digest=problem.fairness.information_digest,
        cost_model_digest=problem.fairness.cost_model_digest,
        fixture_id=str(fixture_id),
        anchor_layer_id=anchor_layer_id,
        horizon=horizon,
        window_truth_digest=window_truth_digest,
    )


def _record(
    *,
    key: PairedInstanceKey,
    algorithm_id: str,
    status: RunStatus,
    provenance: Provenance,
    plan_digest: str | None,
    objective_value: int | float | None,
    objective_unit: str | None,
    failure_code: str | None,
    failure_message: str | None,
    payload: dict[str, Any],
) -> AlgorithmRunRecord:
    semantic = {
        "paired_key": key,
        "algorithm_id": algorithm_id,
        "status": status.value,
        "provenance": provenance,
        "plan_digest": plan_digest,
        "objective_value": objective_value,
        "objective_unit": objective_unit,
        "failure_code": failure_code,
        "failure_message": failure_message,
        "payload": payload,
    }
    return AlgorithmRunRecord(
        paired_key=key,
        algorithm_id=algorithm_id,
        status=status,
        provenance=provenance,
        plan_digest=plan_digest,
        objective_value=objective_value,
        objective_unit=objective_unit,
        failure_code=failure_code,
        failure_message=failure_message,
        result_payload=tuple(sorted(payload.items())),
        record_digest=stable_digest(_digest_safe(semantic)),
    )


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    schema_version: str
    baseline_algorithm_id: str
    treatment_algorithm_id: str
    objective_unit: str
    total_candidate_pairs: int
    completed_pair_count: int
    excluded_pair_count: int
    excluded_pair_digests: tuple[str, ...]
    bootstrap_replicates: int
    confidence_bp: int
    mean_delta_numerator: int
    mean_delta_denominator: int
    lower_delta_numerator: int
    lower_delta_denominator: int
    upper_delta_numerator: int
    upper_delta_denominator: int
    seed_digest: str
    performance_claim_allowed: bool
    result_digest: str

    @property
    def mean_delta(self) -> Fraction:
        return Fraction(self.mean_delta_numerator, self.mean_delta_denominator)

    @property
    def lower_delta(self) -> Fraction:
        return Fraction(self.lower_delta_numerator, self.lower_delta_denominator)

    @property
    def upper_delta(self) -> Fraction:
        return Fraction(self.upper_delta_numerator, self.upper_delta_denominator)


def _objective_fraction(value: int | float) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("boolean objective is invalid")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    raise TypeError("objective must be int or float")


def _deterministic_index(seed_digest: str, replicate: int, draw: int, size: int) -> int:
    material = f"{seed_digest}:{replicate}:{draw}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % size


def paired_bootstrap_completed_records(
    records: Iterable[AlgorithmRunRecord],
    *,
    baseline_algorithm_id: str,
    treatment_algorithm_id: str,
    bootstrap_replicates: int = 1000,
    confidence_bp: int = 9500,
    seed_digest: str | None = None,
) -> PairedBootstrapResult:
    """Deterministic paired bootstrap over completed paired records only.

    Delta is ``treatment - baseline``; negative values favor the treatment for
    lower-is-better objectives.  Failed, unsupported or incomplete pairs are
    preserved in the excluded-pair audit instead of silently converted into
    successful observations.
    """

    baseline = str(baseline_algorithm_id)
    treatment = str(treatment_algorithm_id)
    if not baseline or not treatment or baseline == treatment:
        raise ValueError("baseline and treatment IDs must be distinct and non-empty")
    if not isinstance(bootstrap_replicates, int) or isinstance(bootstrap_replicates, bool) or bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not isinstance(confidence_bp, int) or not 1 <= confidence_bp < 10_000:
        raise ValueError("confidence_bp must be in [1, 9999]")

    grouped: dict[str, dict[str, AlgorithmRunRecord]] = {}
    for record in tuple(records):
        pair_digest = record.paired_key.paired_key_digest
        bucket = grouped.setdefault(pair_digest, {})
        if record.algorithm_id in (baseline, treatment):
            if record.algorithm_id in bucket:
                raise ValueError(
                    f"duplicate {record.algorithm_id} record for paired instance {pair_digest}"
                )
            bucket[record.algorithm_id] = record
    if not grouped:
        raise ValueError("no candidate paired records for requested algorithms")

    completed: list[tuple[str, Fraction, bool]] = []
    excluded: list[str] = []
    objective_unit: str | None = None
    for pair_digest in sorted(grouped):
        bucket = grouped[pair_digest]
        left = bucket.get(baseline)
        right = bucket.get(treatment)
        if (
            left is None
            or right is None
            or left.status is not RunStatus.COMPLETED
            or right.status is not RunStatus.COMPLETED
            or left.objective_value is None
            or right.objective_value is None
            or left.objective_unit is None
            or right.objective_unit is None
            or left.objective_unit != right.objective_unit
        ):
            excluded.append(pair_digest)
            continue
        if objective_unit is None:
            objective_unit = left.objective_unit
        elif objective_unit != left.objective_unit:
            excluded.append(pair_digest)
            continue
        delta = _objective_fraction(right.objective_value) - _objective_fraction(left.objective_value)
        eligible = bool(
            left.provenance.performance_eligible
            and right.provenance.performance_eligible
            and not left.provenance.synthetic_components
            and not right.provenance.synthetic_components
        )
        completed.append((pair_digest, delta, eligible))

    if not completed or objective_unit is None:
        raise ValueError("no completed compatible paired records")
    deltas = tuple(item[1] for item in completed)
    mean_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
    seed = seed_digest or stable_digest(
        {
            "baseline": baseline,
            "treatment": treatment,
            "pair_digests": tuple(item[0] for item in completed),
            "record_digests": tuple(
                record.record_digest
                for pair_digest in sorted(grouped)
                for record in grouped[pair_digest].values()
            ),
        }
    )
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed_digest must be non-empty")
    bootstrap_means: list[Fraction] = []
    for replicate in range(bootstrap_replicates):
        sample = [
            deltas[_deterministic_index(seed, replicate, draw, len(deltas))]
            for draw in range(len(deltas))
        ]
        bootstrap_means.append(sum(sample, Fraction(0, 1)) / len(sample))
    bootstrap_means.sort()
    tail_bp = (10_000 - confidence_bp) // 2
    lower_index = (tail_bp * (bootstrap_replicates - 1)) // 10_000
    upper_tail_bp = 10_000 - tail_bp
    upper_index = (upper_tail_bp * (bootstrap_replicates - 1) + 9_999) // 10_000
    upper_index = min(bootstrap_replicates - 1, upper_index)
    lower = bootstrap_means[lower_index]
    upper = bootstrap_means[upper_index]
    performance_allowed = all(item[2] for item in completed) and not excluded
    payload = {
        "schema_version": "PAIRED_BOOTSTRAP",
        "baseline_algorithm_id": baseline,
        "treatment_algorithm_id": treatment,
        "objective_unit": objective_unit,
        "total_candidate_pairs": len(grouped),
        "completed_pair_count": len(completed),
        "excluded_pair_digests": tuple(excluded),
        "bootstrap_replicates": bootstrap_replicates,
        "confidence_bp": confidence_bp,
        "mean_delta": (mean_delta.numerator, mean_delta.denominator),
        "lower_delta": (lower.numerator, lower.denominator),
        "upper_delta": (upper.numerator, upper.denominator),
        "seed_digest": seed,
        "performance_claim_allowed": performance_allowed,
    }
    return PairedBootstrapResult(
        schema_version="PAIRED_BOOTSTRAP",
        baseline_algorithm_id=baseline,
        treatment_algorithm_id=treatment,
        objective_unit=objective_unit,
        total_candidate_pairs=len(grouped),
        completed_pair_count=len(completed),
        excluded_pair_count=len(excluded),
        excluded_pair_digests=tuple(excluded),
        bootstrap_replicates=bootstrap_replicates,
        confidence_bp=confidence_bp,
        mean_delta_numerator=mean_delta.numerator,
        mean_delta_denominator=mean_delta.denominator,
        lower_delta_numerator=lower.numerator,
        lower_delta_denominator=lower.denominator,
        upper_delta_numerator=upper.numerator,
        upper_delta_denominator=upper.denominator,
        seed_digest=seed,
        performance_claim_allowed=performance_allowed,
        result_digest=stable_digest(payload),
    )


@dataclass(frozen=True, slots=True)
class FormalRuntimeRecord:
    """Formal transport/backend/scheduler runtime record; every duration is integer nanoseconds."""

    paired_key: PairedInstanceKey
    algorithm_id: str
    status: RunStatus
    provenance: Provenance
    window_makespan_ns: int
    run_forward_makespan_ns: int
    network_transfer_span_ns: int
    rank_release_times_ns: tuple[tuple[int, int], ...]
    control_exposed_ns: int
    prediction_exposed_ns: int
    receiver_total_delay_ns: int
    network_active_union_ns: int
    memory_peak_bytes_by_rank: tuple[tuple[int, int], ...]
    plan_count: int
    completed_bytes: int
    terminal_status: str
    fairness_digest: str
    physical_completion_digest: str
    objective_unit: str
    record_digest: str
    window_key: Any | None = None
    anchor_layer_id: int | None = None
    horizon: str | None = None
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    window_network_transfer_span_ns: int | None = None
    window_rank_release_times: tuple[tuple[int, int], ...] = ()
    window_task_ids: tuple[str, ...] = ()
    window_task_catalogue_digest: str | None = None
    window_truth_digest: str | None = None
    is_truncated_tail: bool = False
    prediction_hidden_ns: int = 0
    control_hidden_ns: int = 0
    binding_hidden_ns: int = 0
    binding_exposed_ns: int = 0
    target_bind_wait_ns: int = 0
    template_ready_margin_ns: int | None = None
    reconciliation_status: str | None = None
    prediction_quality_digest: str | None = None
    prediction_absolute_error_bytes: int | None = None
    prediction_relative_absolute_error_ppm: int | None = None
    prediction_matrix_overlap_ppm: int | None = None
    prediction_top_destination_accuracy_ppm: int | None = None
    receiver_posting_service_ns: int = 0
    receiver_posting_queue_wait_ns: int = 0
    receiver_buffer_stall_ns: int = 0
    receiver_drain_queue_wait_ns: int = 0
    receiver_drain_service_ns: int = 0
    peak_staging_bytes_by_rank: tuple[tuple[int, int], ...] = ()
    peak_final_assembly_bytes_by_rank: tuple[tuple[int, int], ...] = ()


def make_formal_runtime_record(
    *,
    paired_key: PairedInstanceKey,
    algorithm_id: str,
    status: RunStatus | str,
    provenance: Provenance,
    window_makespan_ns: int,
    run_forward_makespan_ns: int,
    network_transfer_span_ns: int,
    rank_release_times_ns: Iterable[tuple[int, int]],
    control_exposed_ns: int,
    prediction_exposed_ns: int,
    receiver_total_delay_ns: int,
    network_active_union_ns: int,
    memory_peak_bytes_by_rank: Iterable[tuple[int, int]],
    plan_count: int,
    completed_bytes: int,
    terminal_status: str,
    fairness_digest: str,
    physical_completion_digest: str,
    window_key: Any | None = None,
    anchor_layer_id: int | None = None,
    horizon: str | None = None,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
    window_task_ids: Iterable[str] = (),
    window_task_catalogue_digest: str | None = None,
    window_truth_digest: str | None = None,
    is_truncated_tail: bool = False,
    prediction_hidden_ns: int = 0,
    control_hidden_ns: int = 0,
    binding_hidden_ns: int = 0,
    binding_exposed_ns: int = 0,
    target_bind_wait_ns: int = 0,
    template_ready_margin_ns: int | None = None,
    reconciliation_status: str | None = None,
    prediction_quality_digest: str | None = None,
    prediction_absolute_error_bytes: int | None = None,
    prediction_relative_absolute_error_ppm: int | None = None,
    prediction_matrix_overlap_ppm: int | None = None,
    prediction_top_destination_accuracy_ppm: int | None = None,
    receiver_posting_service_ns: int = 0,
    receiver_posting_queue_wait_ns: int = 0,
    receiver_buffer_stall_ns: int = 0,
    receiver_drain_queue_wait_ns: int = 0,
    receiver_drain_service_ns: int = 0,
    peak_staging_bytes_by_rank: Iterable[tuple[int, int]] = (),
    peak_final_assembly_bytes_by_rank: Iterable[tuple[int, int]] = (),
) -> FormalRuntimeRecord:
    normalized_status = RunStatus(status)
    numeric = {
        "window_makespan_ns": window_makespan_ns,
        "run_forward_makespan_ns": run_forward_makespan_ns,
        "network_transfer_span_ns": network_transfer_span_ns,
        "control_exposed_ns": control_exposed_ns,
        "prediction_exposed_ns": prediction_exposed_ns,
        "receiver_total_delay_ns": receiver_total_delay_ns,
        "network_active_union_ns": network_active_union_ns,
        "plan_count": plan_count,
        "completed_bytes": completed_bytes,
        "prediction_hidden_ns": prediction_hidden_ns,
        "control_hidden_ns": control_hidden_ns,
        "binding_hidden_ns": binding_hidden_ns,
        "binding_exposed_ns": binding_exposed_ns,
        "target_bind_wait_ns": target_bind_wait_ns,
        "receiver_posting_service_ns": receiver_posting_service_ns,
        "receiver_posting_queue_wait_ns": receiver_posting_queue_wait_ns,
        "receiver_buffer_stall_ns": receiver_buffer_stall_ns,
        "receiver_drain_queue_wait_ns": receiver_drain_queue_wait_ns,
        "receiver_drain_service_ns": receiver_drain_service_ns,
    }
    for name, value in numeric.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    optional_prediction_numeric = {
        "prediction_absolute_error_bytes": prediction_absolute_error_bytes,
        "prediction_relative_absolute_error_ppm": prediction_relative_absolute_error_ppm,
        "prediction_matrix_overlap_ppm": prediction_matrix_overlap_ppm,
        "prediction_top_destination_accuracy_ppm": prediction_top_destination_accuracy_ppm,
    }
    for name, value in optional_prediction_numeric.items():
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{name} must be None or a non-negative integer")
    rank_releases = tuple(sorted((int(rank), int(at_ns)) for rank, at_ns in rank_release_times_ns))
    memory_peaks = tuple(sorted((int(rank), int(value)) for rank, value in memory_peak_bytes_by_rank))
    staging_peaks = tuple(sorted((int(rank), int(value)) for rank, value in peak_staging_bytes_by_rank))
    final_assembly_peaks = tuple(sorted((int(rank), int(value)) for rank, value in peak_final_assembly_bytes_by_rank))
    anchor_task_ids = tuple(str(item) for item in window_task_ids)
    anchor_local = window_key is not None or anchor_layer_id is not None or horizon is not None
    if anchor_local:
        if window_key is None:
            raise ValueError("window_key is required for anchor-local runtime record")
        if anchor_layer_id is None or int(anchor_layer_id) < 0:
            raise ValueError("anchor_layer_id must be non-negative")
        if horizon != "P12":
            raise ValueError("horizon must be P12")
        if window_start_ns is None or window_end_ns is None:
            raise ValueError("window_start_ns/window_end_ns are required")
        if int(window_start_ns) < 0 or int(window_end_ns) < int(window_start_ns):
            raise ValueError("invalid anchor-local window times")
        if int(window_end_ns) - int(window_start_ns) != int(window_makespan_ns):
            raise ValueError("window_makespan_ns must equal window_end_ns-window_start_ns")
        if not anchor_task_ids or len(set(anchor_task_ids)) != len(anchor_task_ids):
            raise ValueError("window_task_ids must be unique and non-empty")
        if not window_task_catalogue_digest or not window_truth_digest:
            raise ValueError("anchor-local catalogue/truth digests are required")
    if any(
        rank < 0 or value < 0
        for rank, value in (*rank_releases, *memory_peaks, *staging_peaks, *final_assembly_peaks)
    ):
        raise ValueError("rank/runtime metric pairs must be non-negative")
    for name, value in (
        ("algorithm_id", algorithm_id),
        ("terminal_status", terminal_status),
        ("fairness_digest", fairness_digest),
        ("physical_completion_digest", physical_completion_digest),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty")
    payload = {
        "paired_key_digest": paired_key.paired_key_digest,
        "algorithm_id": str(algorithm_id),
        "status": normalized_status.value,
        "provenance_digest": provenance.provenance_digest,
        **numeric,
        "rank_release_times_ns": rank_releases,
        "memory_peak_bytes_by_rank": memory_peaks,
        "terminal_status": str(terminal_status),
        "fairness_digest": str(fairness_digest),
        "physical_completion_digest": str(physical_completion_digest),
        "objective_unit": "nanoseconds",
        "window_key": window_key,
        "anchor_layer_id": anchor_layer_id,
        "horizon": horizon,
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "window_network_transfer_span_ns": network_transfer_span_ns if anchor_local else None,
        "window_rank_release_times": rank_releases if anchor_local else (),
        "window_task_ids": anchor_task_ids,
        "window_task_catalogue_digest": window_task_catalogue_digest,
        "window_truth_digest": window_truth_digest,
        "is_truncated_tail": bool(is_truncated_tail),
        "template_ready_margin_ns": template_ready_margin_ns,
        "reconciliation_status": reconciliation_status,
        "prediction_quality_digest": prediction_quality_digest,
        "peak_staging_bytes_by_rank": staging_peaks,
        "peak_final_assembly_bytes_by_rank": final_assembly_peaks,
    }
    return FormalRuntimeRecord(
        paired_key=paired_key,
        algorithm_id=str(algorithm_id),
        status=normalized_status,
        provenance=provenance,
        rank_release_times_ns=rank_releases,
        memory_peak_bytes_by_rank=memory_peaks,
        terminal_status=str(terminal_status),
        fairness_digest=str(fairness_digest),
        physical_completion_digest=str(physical_completion_digest),
        objective_unit="nanoseconds",
        record_digest=stable_digest(payload),
        window_key=window_key,
        anchor_layer_id=None if anchor_layer_id is None else int(anchor_layer_id),
        horizon=horizon,
        window_start_ns=None if window_start_ns is None else int(window_start_ns),
        window_end_ns=None if window_end_ns is None else int(window_end_ns),
        window_network_transfer_span_ns=(int(network_transfer_span_ns) if anchor_local else None),
        window_rank_release_times=rank_releases if anchor_local else (),
        window_task_ids=anchor_task_ids,
        window_task_catalogue_digest=window_task_catalogue_digest,
        window_truth_digest=window_truth_digest,
        is_truncated_tail=bool(is_truncated_tail),
        template_ready_margin_ns=(None if template_ready_margin_ns is None else int(template_ready_margin_ns)),
        reconciliation_status=reconciliation_status,
        prediction_quality_digest=prediction_quality_digest,
        peak_staging_bytes_by_rank=staging_peaks,
        peak_final_assembly_bytes_by_rank=final_assembly_peaks,
        **numeric,
    )


def validate_anchor_local_formal_runtime_records(
    records: Iterable[FormalRuntimeRecord],
) -> tuple[FormalRuntimeRecord, ...]:
    """Reject duplicate anchor-local identities and relabelled full-forward rows."""

    normalized = tuple(records)
    seen_keys: set[tuple[str, str, int, str, str, str]] = set()
    full_forward_digests: dict[str, set[tuple[int, str]]] = {}
    for item in normalized:
        if not isinstance(item, FormalRuntimeRecord):
            raise TypeError("records must contain FormalRuntimeRecord")
        if item.window_key is None:
            continue
        if item.anchor_layer_id is None or item.horizon is None or item.window_truth_digest is None:
            raise ValueError("anchor-local record is incomplete")
        key = (
            item.paired_key.fixture_id,
            item.paired_key.sample_id,
            int(item.anchor_layer_id),
            str(item.horizon),
            str(item.window_truth_digest),
            item.paired_key.hardware_profile_digest,
        )
        if key in seen_keys:
            raise ValueError("duplicate anchor-local paired identity")
        seen_keys.add(key)
        labels = full_forward_digests.setdefault(item.physical_completion_digest, set())
        labels.add((int(item.anchor_layer_id), str(item.horizon)))
        if len(labels) > 1 and item.run_forward_makespan_ns == item.window_makespan_ns:
            raise ValueError("full-forward output was relabelled as multiple anchor-local samples")
    return normalized


def paired_bootstrap_formal_runtime_records(
    records: Iterable[FormalRuntimeRecord],
    *,
    baseline_algorithm_id: str,
    treatment_algorithm_id: str,
    bootstrap_replicates: int = 1000,
    confidence_bp: int = 9500,
) -> PairedBootstrapResult:
    """Bootstrap formal runtime window makespan only; oracle units are rejected."""

    converted: list[AlgorithmRunRecord] = []
    for item in records:
        if not isinstance(item, FormalRuntimeRecord) or item.objective_unit != "nanoseconds":
            raise TypeError("formal runtime bootstrap accepts only nanosecond FormalRuntimeRecord")
        converted.append(
            _record(
                key=item.paired_key,
                algorithm_id=item.algorithm_id,
                status=item.status,
                provenance=item.provenance,
                plan_digest=item.physical_completion_digest,
                objective_value=item.window_makespan_ns,
                objective_unit="nanoseconds",
                failure_code=None if item.status is RunStatus.COMPLETED else item.terminal_status,
                failure_message=None,
                payload={"formal_runtime_record_digest": item.record_digest},
            )
        )
    return paired_bootstrap_completed_records(
        converted,
        baseline_algorithm_id=baseline_algorithm_id,
        treatment_algorithm_id=treatment_algorithm_id,
        bootstrap_replicates=bootstrap_replicates,
        confidence_bp=confidence_bp,
    )


__all__ = [
    "FormalRuntimeRecord",
    "AlgorithmRunRecord",
    "PairedInstanceKey",
    "PairedBootstrapResult",
    "Provenance",
    "RunStatus",
    "make_formal_runtime_record",
    "make_paired_key",
    "paired_bootstrap_completed_records",
    "paired_bootstrap_formal_runtime_records",
    "validate_anchor_local_formal_runtime_records",
]
