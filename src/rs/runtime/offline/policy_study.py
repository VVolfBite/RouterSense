#!/usr/bin/env python3
"""Offline replay fixture problem builders and study helpers."""

from __future__ import annotations

import json
from typing import Any

from rs.runtime.offline.runner import replay_and_audit_logical_plan, summarize_schedule_tail_metrics
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_remote_bytes
from rs.scheduling import (
    FlowDemand,
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    ReleaseConstraint,
    resolve_policy,
)
from rs.scheduling.validation import stable_hash, validate_logical_plan


def _matrix(value: Any) -> tuple[tuple[int, ...], ...]:
    return canonicalize_remote_matrix(value)


def _flows(
    matrix: tuple[tuple[int, ...], ...],
    *,
    phase: str,
    release_state: str,
    executable: bool,
) -> tuple[FlowDemand, ...]:
    flows: list[FlowDemand] = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            flows.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    byte_count=int(byte_count),
                    release_state=release_state,
                    is_executable=executable,
                )
            )
    return tuple(flows)


def build_replay_problem(
    fixture: dict[str, Any],
    *,
    mode: str,
    p2_source: str,
    expert_compute_delay: float,
    predicted_p2_matrix: tuple[tuple[int, ...], ...] | None = None,
    max_waves: int = 256,
) -> MultiPhaseSchedulingProblem:
    p0 = _matrix(fixture["p0_dispatch_matrix"])
    p1 = _matrix(fixture["p1_return_matrix"])
    p2_actual = _matrix(fixture.get("p2_next_dispatch_matrix", fixture.get("p2_next_dispatch_forecast_matrix", [])))
    if mode == "execution_window":
        p2 = p2_actual
        forecast_source = "actual_trace" if p2_source == "actual_trace" else "perfect_trace"
        forecast_oracle = True
        forecast_eligible = False
        information_mode = "p0_p1_p2"
    elif p2_source == "copy_current_dispatch":
        p2 = p0
        forecast_source = "copy_current_dispatch"
        forecast_oracle = False
        forecast_eligible = True
        information_mode = "p0_p1_p2"
    elif p2_source in {"perfect_trace", "actual_trace"}:
        p2 = p2_actual
        forecast_source = str(p2_source)
        forecast_oracle = True
        forecast_eligible = False
        information_mode = "p0_p1_p2"
    elif p2_source in {"fate_style_history", "fate_style_linear"}:
        if predicted_p2_matrix is None:
            raise ValueError(f"predicted_p2_matrix required for {p2_source}")
        p2 = canonicalize_remote_matrix(predicted_p2_matrix)
        forecast_source = str(p2_source)
        forecast_oracle = False
        forecast_eligible = True
        information_mode = "p0_p1_p2"
    else:
        p2 = tuple(tuple(0 for _ in row) for row in p0)
        forecast_source = "zero_hint"
        forecast_oracle = False
        forecast_eligible = True
        information_mode = "p0_p1"
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows(p0, phase="p0_dispatch", release_state="ready", executable=True),
            blocked_flows=_flows(p1, phase="p1_return", release_state="blocked", executable=False),
            forecast_pressure=_flows(p2, phase="p2_next_dispatch_forecast", release_state="advisory_only", executable=False),
        ),
        topology=LogicalTopology(num_gpus=int(fixture["num_gpus"])),
        release_model=ReleaseConstraint(
            phase="p1_return",
            rank=0,
            release_after_phase="p0_dispatch",
            expert_compute_delay=float(expert_compute_delay),
        ),
        forecast=ForecastPressure(
            source=forecast_source,
            digest=stable_hash({"source": forecast_source, "matrix": p2}),
            oracle=forecast_oracle,
            evaluation_eligible=forecast_eligible,
            matrix_shape=(len(p2), len(p2[0]) if p2 else 0),
            matrix_total_bytes=matrix_remote_bytes(p2),
            matrix=p2,
        ),
        options=GlobalReadySetOptions(
            scheduling_mode=mode,
            information_mode=information_mode,
            prediction_confidence=0.0 if forecast_source == "zero_hint" else 1.0,
            max_waves=int(max_waves),
        ),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=p2,
    )


def expected_replay_flows(problem: MultiPhaseSchedulingProblem) -> tuple[FlowDemand, ...]:
    flows = list(problem.flow_window.ready_flows + problem.flow_window.blocked_flows)
    if problem.options.scheduling_mode == "execution_window":
        flows.extend(
            _flows(
                problem.p2_next_dispatch_forecast_matrix,
                phase="p2_next_dispatch",
                release_state="ready",
                executable=True,
            )
        )
    return tuple(flows)
def run_replay_policy_study(
    *,
    fixture: dict[str, Any],
    policy_names: list[str],
    mode: str,
    p2_source: str,
    expert_compute_delay: float,
    max_waves: int = 256,
) -> dict[str, Any]:
    problem = build_replay_problem(
        fixture,
        mode=mode,
        p2_source=p2_source,
        expert_compute_delay=expert_compute_delay,
        max_waves=int(max_waves),
    )
    expected = expected_replay_flows(problem)
    rows: list[dict[str, Any]] = []
    for policy_name in policy_names:
        policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
        plan = policy.build_logical_plan(problem)
        validation = validate_logical_plan(
            plan,
            expected_flows=expected,
            mode=str(mode),
            expert_compute_delay=float(expert_compute_delay),
        )
        audit = replay_and_audit_logical_plan(problem, plan)
        tail = summarize_schedule_tail_metrics(problem=problem, plan=plan, audit=audit)
        rows.append(
            {
                "policy_name": policy_name,
                "policy_version": getattr(policy, "policy_version", "v1"),
                "logical_model": plan.diagnostics.get("logical_model", "unknown"),
                "information_mode": plan.diagnostics.get("information_mode", ""),
                "future_information_mode": plan.diagnostics.get("future_information_mode", ""),
                "evaluation_eligible": bool(plan.diagnostics.get("evaluation_eligible", True)),
                "valid": bool(validation["valid"]) and bool(audit.get("valid", False)),
                "makespan": float(plan.diagnostics.get("makespan", audit.get("makespan", 0.0))),
                "wave_count": int(len(plan.waves)),
                "plan_hash": stable_hash(plan.to_dict()),
                "tail_metrics": tail,
            }
        )
    return {
        "mode": str(mode),
        "p2_source": str(p2_source),
        "expert_compute_delay": float(expert_compute_delay),
        "rows": rows,
    }


__all__ = [
    "build_replay_problem",
    "expected_replay_flows",
    "run_replay_policy_study",
]
