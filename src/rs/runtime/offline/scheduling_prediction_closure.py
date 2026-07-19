"""Unified offline closure for scheduling cores and P2 prediction value.

This module is the single supported analysis surface for:

1. strict same-core Local/Joint P01 and P012 comparisons;
2. reactive/predicted/perfect P2 information timing under a heuristic family;
3. exact information-ladder controls on deterministic tiny reductions;
4. planner-overhead reporting separated from abstract logical makespan.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Any, Iterable

from rs.runtime.offline.oracle_information_ladder import (
    ExactInformationLadder,
    evaluate_exact_information_ladder,
    reduce_forecast_for_exact,
    reduce_traffic_instance_for_exact,
)
from rs.runtime.offline.p2_information_value import simulate_p2_information
from rs.runtime.offline.traffic_dataset import TrafficInstanceRecord
from rs.prediction.traffic_envelope import evaluate_traffic_forecast
from rs.scheduling.contracts import (
    FlowDemand,
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    ReleaseConstraint,
)
from rs.scheduling.families import (
    canonical_family_policy_id,
    get_family_kernel_spec,
    resolve_scoped_family_policy,
)
from rs.scheduling.families.core import FamilyScope
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.scheduling.traffic_matrix import matrix_digest_remote, matrix_remote_bytes


Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FamilyClosureRecord:
    traffic_instance_id: str
    model_id: str
    family_id: str
    world_size: int
    layer_id: int
    p2_available: bool
    local_p01_makespan: float
    joint_p01_makespan: float
    local_p012_makespan: float
    joint_p012_perfect_makespan: float
    joint_p012_reactive_makespan: float
    joint_p012_predicted_makespan: float | None
    local_p012_planning_ms: float
    joint_p012_planning_ms: float
    reactive_planning_ms: float
    predicted_planning_ms: float | None
    forecast_remote_relative_l1: float | None
    forecast_rank_pressure_relative_l1: float | None
    forecast_remote_total_bias: float | None
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        local_p2 = self.local_p012_makespan - self.local_p01_makespan
        p01_serial_total = self.joint_p01_makespan + local_p2
        p01_gain = self.local_p012_makespan - p01_serial_total
        p2_gain = p01_serial_total - self.joint_p012_perfect_makespan
        total_gain = self.local_p012_makespan - self.joint_p012_perfect_makespan
        predicted_gain = None
        if self.joint_p012_predicted_makespan is not None:
            predicted_gain = self.joint_p012_reactive_makespan - self.joint_p012_predicted_makespan
        perfect_information_gain = self.joint_p012_reactive_makespan - self.joint_p012_perfect_makespan
        return {
            **self.__dict__,
            "p01_joint_gain_pct": _gain_pct(self.local_p01_makespan, self.joint_p01_makespan),
            "p012_joint_gain_pct": _gain_pct(self.local_p012_makespan, self.joint_p012_perfect_makespan),
            "p01_coupling_gain_pct_of_local_p012": _fraction_pct(p01_gain, self.local_p012_makespan),
            "p2_cross_phase_gain_pct_of_local_p012": _fraction_pct(p2_gain, self.local_p012_makespan),
            "perfect_p2_value_vs_reactive_pct": _gain_pct(self.joint_p012_reactive_makespan, self.joint_p012_perfect_makespan),
            "predicted_p2_value_vs_reactive_pct": (
                None if self.joint_p012_predicted_makespan is None
                else _gain_pct(self.joint_p012_reactive_makespan, self.joint_p012_predicted_makespan)
            ),
            "prediction_capture_ratio": (
                None
                if predicted_gain is None or perfect_information_gain <= 1e-12
                else predicted_gain / perfect_information_gain
            ),
            "prediction_regret_to_perfect_pct": (
                None
                if self.joint_p012_predicted_makespan is None
                else _fraction_pct(
                    self.joint_p012_predicted_makespan - self.joint_p012_perfect_makespan,
                    self.joint_p012_perfect_makespan,
                )
            ),
            "p2_share_of_positive_joint_gain": None if total_gain <= 1e-12 else p2_gain / total_gain,
        }


def _gain_pct(base: float, value: float) -> float:
    return 0.0 if base <= 0.0 else 100.0 * (base - value) / base


def _fraction_pct(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else 100.0 * numerator / denominator


def _zero(size: int) -> Matrix:
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))


def _flows(matrix: Matrix, *, phase: str, release_state: str) -> tuple[FlowDemand, ...]:
    return tuple(
        FlowDemand(
            flow_id=f"{phase}:{source}->{destination}",
            phase=phase,
            src_rank=source,
            dst_rank=destination,
            byte_count=int(value),
            release_state=release_state,
            is_executable=True,
        )
        for source, row in enumerate(matrix)
        for destination, value in enumerate(row)
        if source != destination and int(value) > 0
    )


def build_execution_problem(
    p0: Matrix,
    p1: Matrix,
    p2: Matrix,
    *,
    information_mode: str,
    prediction_confidence: float,
    forecast_matrix: Matrix | None = None,
    max_waves: int = 4096,
) -> MultiPhaseSchedulingProblem:
    size = len(p0)
    forecast = p2 if forecast_matrix is None else forecast_matrix
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows(p0, phase="p0_dispatch", release_state="ready"),
            blocked_flows=_flows(p1, phase="p1_return", release_state="blocked"),
            forecast_pressure=_flows(p2, phase="p2_next_dispatch", release_state="blocked"),
        ),
        topology=LogicalTopology(num_gpus=size),
        release_model=ReleaseConstraint(
            phase="rank_phase_release",
            rank=0,
            release_after_phase="p0_dispatch",
            expert_compute_delay=0.0,
        ),
        forecast=ForecastPressure(
            source=information_mode,
            digest=matrix_digest_remote(forecast),
            oracle=information_mode == "p012_perfect",
            evaluation_eligible=information_mode != "p012_perfect",
            matrix_shape=(size, size),
            matrix_total_bytes=int(matrix_remote_bytes(forecast)),
            matrix=forecast,
            metadata={
                "information_mode": information_mode,
                "planning_hint_matrix": [list(row) for row in forecast],
            },
        ),
        options=GlobalReadySetOptions(
            scheduling_mode="execution_window",
            information_mode=information_mode,
            prediction_confidence=float(prediction_confidence),
            max_waves=int(max_waves),
        ),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=p2,
    )


def _plan_family(
    *,
    family_id: str,
    scope: FamilyScope,
    p0: Matrix,
    p1: Matrix,
    p2: Matrix,
    forecast: Matrix | None = None,
    confidence: float = 1.0,
) -> tuple[float, float, bool]:
    problem = build_execution_problem(
        p0,
        p1,
        p2,
        information_mode="p012_perfect" if forecast is None else "p012_predicted",
        prediction_confidence=float(confidence),
        forecast_matrix=p2 if forecast is None else forecast,
    )
    policy = resolve_scoped_family_policy(
        canonical_family_policy_id(family_id, scope)
    )
    started = time.perf_counter()
    plan = policy.build_logical_plan(problem)
    wrapper_ms = (time.perf_counter() - started) * 1000.0
    audit = replay_and_audit_logical_plan(problem, plan)
    makespan = float(audit.get("makespan", plan.diagnostics.get("makespan", 0.0)))
    planning_ms = float(
        plan.diagnostics.get(
            "kernel_runtime_ms",
            plan.diagnostics.get("planning_time_ms", wrapper_ms),
        )
    )
    return makespan, planning_ms, bool(audit.get("valid", False))


def evaluate_family_closure(
    instance: TrafficInstanceRecord,
    *,
    family_id: str,
    p2_forecast: Matrix | None = None,
    prediction_confidence: float = 1.0,
) -> FamilyClosureRecord:
    instance.validate()
    get_family_kernel_spec(family_id)
    p0, p1, p2 = instance.p0, instance.p1, instance.p2
    zero = _zero(instance.world_size)
    local_p01, _, valid_l01 = _plan_family(
        family_id=family_id, scope=FamilyScope.LOCAL, p0=p0, p1=p1, p2=zero
    )
    joint_p01, _, valid_j01 = _plan_family(
        family_id=family_id, scope=FamilyScope.JOINT, p0=p0, p1=p1, p2=zero
    )
    local_p012, local_ms, valid_l012 = _plan_family(
        family_id=family_id, scope=FamilyScope.LOCAL, p0=p0, p1=p1, p2=p2
    )
    joint_p012, joint_ms, valid_j012 = _plan_family(
        family_id=family_id, scope=FamilyScope.JOINT, p0=p0, p1=p1, p2=p2
    )
    reactive = simulate_p2_information(
        p0_dispatch_matrix=[list(row) for row in p0],
        p1_return_matrix=[list(row) for row in p1],
        p2_truth_matrix=[list(row) for row in p2],
        family_id=family_id,
        information_mode="reactive",
    )
    predicted = None
    if p2_forecast is not None:
        predicted = simulate_p2_information(
            p0_dispatch_matrix=[list(row) for row in p0],
            p1_return_matrix=[list(row) for row in p1],
            p2_truth_matrix=[list(row) for row in p2],
            family_id=family_id,
            information_mode="predicted",
            p2_forecast_matrix=[list(row) for row in p2_forecast],
            prediction_confidence=float(prediction_confidence),
        )
    valid = all((valid_l01, valid_j01, valid_l012, valid_j012, reactive.valid)) and (
        predicted is None or predicted.valid
    )
    forecast_quality = (
        None if p2_forecast is None
        else evaluate_traffic_forecast(p2_forecast, p2)
    )
    return FamilyClosureRecord(
        traffic_instance_id=instance.traffic_instance_id,
        model_id=instance.model_id,
        family_id=family_id,
        world_size=instance.world_size,
        layer_id=instance.layer_id,
        p2_available=instance.p2_available,
        local_p01_makespan=local_p01,
        joint_p01_makespan=joint_p01,
        local_p012_makespan=local_p012,
        joint_p012_perfect_makespan=joint_p012,
        joint_p012_reactive_makespan=reactive.makespan,
        joint_p012_predicted_makespan=None if predicted is None else predicted.makespan,
        local_p012_planning_ms=local_ms,
        joint_p012_planning_ms=joint_ms,
        reactive_planning_ms=reactive.planning_time_ms,
        predicted_planning_ms=None if predicted is None else predicted.planning_time_ms,
        forecast_remote_relative_l1=(
            None if forecast_quality is None else float(forecast_quality["remote_relative_l1"])
        ),
        forecast_rank_pressure_relative_l1=(
            None if forecast_quality is None else float(forecast_quality["rank_pressure_relative_l1"])
        ),
        forecast_remote_total_bias=(
            None if forecast_quality is None else float(forecast_quality["remote_total_bias"])
        ),
        valid=valid,
    )


def evaluate_exact_closure(
    instance: TrafficInstanceRecord,
    *,
    p2_forecast: Matrix | None = None,
    time_limit_ms: int = 5000,
) -> ExactInformationLadder:
    reduction = reduce_traffic_instance_for_exact(instance)
    reduced_forecast = None if p2_forecast is None else reduce_forecast_for_exact(p2_forecast, reduction)
    return evaluate_exact_information_ladder(
        reduction,
        p2_forecast=reduced_forecast,
        time_limit_ms=time_limit_ms,
    )


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    rows = [float(value) for value in values]
    return {
        "n": len(rows),
        "mean": statistics.mean(rows) if rows else None,
        "median": statistics.median(rows) if rows else None,
        "p10": _percentile(rows, 0.10),
        "p90": _percentile(rows, 0.90),
        "p95": _percentile(rows, 0.95),
        "min": min(rows) if rows else None,
        "max": max(rows) if rows else None,
    }


def summarize_family_records(records: Iterable[FamilyClosureRecord]) -> dict[str, Any]:
    rows = [record.to_dict() for record in records]
    output: dict[str, Any] = {}
    for family_id in sorted({str(row["family_id"]) for row in rows}):
        family_rows = [row for row in rows if row["family_id"] == family_id]
        p2_rows = [row for row in family_rows if row["p2_available"]]
        metrics = {}
        for key in (
            "p01_joint_gain_pct",
            "p012_joint_gain_pct",
            "p01_coupling_gain_pct_of_local_p012",
            "p2_cross_phase_gain_pct_of_local_p012",
            "perfect_p2_value_vs_reactive_pct",
            "predicted_p2_value_vs_reactive_pct",
            "prediction_capture_ratio",
            "prediction_regret_to_perfect_pct",
            "local_p012_planning_ms",
            "joint_p012_planning_ms",
            "forecast_remote_relative_l1",
            "forecast_rank_pressure_relative_l1",
            "forecast_remote_total_bias",
        ):
            source = p2_rows if "p2" in key or "prediction" in key else family_rows
            metrics[key] = stats(
                row[key] for row in source if row.get(key) is not None
            )
        metrics["p012_outcomes"] = {
            "win": sum(float(row["p012_joint_gain_pct"]) > 1e-9 for row in family_rows),
            "tie": sum(abs(float(row["p012_joint_gain_pct"])) <= 1e-9 for row in family_rows),
            "loss": sum(float(row["p012_joint_gain_pct"]) < -1e-9 for row in family_rows),
        }
        metrics["by_world_size"] = {
            str(world_size): {
                "p012_joint_gain_pct": stats(
                    row["p012_joint_gain_pct"] for row in family_rows if int(row["world_size"]) == world_size
                ),
                "perfect_p2_value_vs_reactive_pct": stats(
                    row["perfect_p2_value_vs_reactive_pct"]
                    for row in p2_rows
                    if int(row["world_size"]) == world_size
                ),
            }
            for world_size in sorted({int(row["world_size"]) for row in family_rows})
        }
        output[family_id] = metrics
    return output


__all__ = [
    "FamilyClosureRecord",
    "build_execution_problem",
    "evaluate_exact_closure",
    "evaluate_family_closure",
    "stats",
    "summarize_family_records",
]
