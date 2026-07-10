"""Debug-only explain traces for RouterSense paired policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rs.scheduling.contracts import LogicalSchedulePlan, MultiPhaseSchedulingProblem
from rs.scheduling.multiphase.scheduler_state import run_global_matching_scheduler
from rs.scheduling.registry import resolve_policy


_SAFE_TO_RAW_AND_B = {
    "RS_safe_barrier_criticality": ("U_barrier_criticality_global_matching", "B_barrier_criticality_matching"),
    "RS_safe_gated_greedy": ("U_gated_greedy_maximal", "B_gated_greedy_maximal"),
}

_U_SPECS = {
    "U_gated_greedy_maximal": {
        "exact_matching": False,
        "residual_weight": 1.0,
        "barrier_weight": 1.0,
        "age_weight": 0.1,
        "prediction_weight": 0.25,
    },
    "U_barrier_criticality_global_matching": {
        "exact_matching": True,
        "residual_weight": 0.75,
        "barrier_weight": 1.75,
        "age_weight": 0.15,
        "prediction_weight": 0.35,
    },
}


@dataclass(frozen=True)
class PolicyDecisionExplanation:
    policy_name: str
    layer_id: int
    p2_source: str
    ready_edges: tuple[Any, ...]
    blocked_edges: tuple[Any, ...]
    eligible_edges: tuple[Any, ...]
    p0_score_by_edge: tuple[Any, ...]
    p1_score_by_edge: tuple[Any, ...]
    p2_score_by_edge: tuple[Any, ...]
    barrier_score_by_edge: tuple[Any, ...]
    gate_score_by_edge: tuple[Any, ...]
    total_score_by_edge: tuple[Any, ...]
    selected_matching: tuple[Any, ...]
    selected_order: tuple[Any, ...]
    modeled_makespan: float
    bottleneck_edges: tuple[Any, ...]
    critical_edges: tuple[Any, ...]
    paired_b_makespan: float | None
    raw_u_makespan: float | None
    safe_selected: str | None
    fallback_to_b: bool | None
    fallback_reason: str | None
    trace_steps: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _selected_order_from_plan(plan: LogicalSchedulePlan) -> tuple[str, ...]:
    order: list[str] = []
    for wave in plan.waves:
        for flow in wave.flows:
            order.append(str(flow.flow_id))
    return tuple(order)


def _bottleneck_edges_from_schedule(raw_schedule: list[dict[str, Any]]) -> tuple[str, ...]:
    if not raw_schedule:
        return ()
    makespan = max(float(entry.get("end", 0.0)) for entry in raw_schedule)
    return tuple(
        str(entry.get("flow_id", ""))
        for entry in raw_schedule
        if abs(float(entry.get("end", 0.0)) - makespan) <= 1e-9
    )


def _score_rows(ready: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    def _edge(candidate: dict[str, Any]) -> str:
        return str(candidate["flow_id"])

    p0 = []
    p1 = []
    p2 = []
    barrier = []
    gate = []
    total = []
    for candidate in sorted(ready, key=lambda item: (float(item["score"]), str(item["flow_id"])), reverse=True):
        edge = _edge(candidate)
        phase = int(candidate["phase"])
        residual_component = float(candidate.get("residual_component", 0.0))
        p0.append((edge, residual_component if phase == 0 else 0.0))
        p1.append((edge, residual_component if phase == 1 else 0.0))
        p2.append((edge, float(candidate.get("prediction_component", 0.0))))
        barrier.append((edge, float(candidate.get("barrier_component", 0.0))))
        gate.append((edge, float(candidate.get("age_component", 0.0) + candidate.get("base_priority_component", 0.0))))
        total.append((edge, float(candidate.get("score", 0.0))))
    return tuple(p0), tuple(p1), tuple(p2), tuple(barrier), tuple(gate), tuple(total)


def _trace_u_policy(
    problem: MultiPhaseSchedulingProblem,
    *,
    policy_name: str,
) -> tuple[LogicalSchedulePlan, dict[str, Any]]:
    spec = _U_SPECS[policy_name]
    result = run_global_matching_scheduler(
        [list(row) for row in problem.p0_dispatch_matrix],
        [list(row) for row in problem.p1_return_matrix],
        [list(row) for row in problem.p2_next_dispatch_forecast_matrix],
        int(problem.topology.num_gpus),
        strategy=policy_name,
        mode=problem.options.scheduling_mode,
        prediction_confidence=float(problem.options.prediction_confidence),
        expert_compute_delay=float(problem.release_model.expert_compute_delay),
        exact_matching=bool(spec["exact_matching"]),
        wave_quantum=None,
        max_waves=int(problem.options.max_waves),
        residual_weight=float(spec["residual_weight"]),
        barrier_weight=float(spec["barrier_weight"]),
        age_weight=float(spec["age_weight"]),
        prediction_weight=float(spec["prediction_weight"]),
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=False,
        collect_debug_trace=True,
    )
    plan = resolve_policy(policy_name=policy_name, bucket_rows=0).build_logical_plan(problem)
    return plan, result


def _explain_raw_u(problem: MultiPhaseSchedulingProblem, *, policy_name: str, p2_source: str) -> PolicyDecisionExplanation:
    plan, result = _trace_u_policy(problem, policy_name=policy_name)
    trace_steps = tuple(dict(step) for step in result.get("debug_trace", ()))
    primary_step = next(
        (
            step
            for step in trace_steps
            if step.get("chosen")
            and any(float(item.get("prediction_component", 0.0)) != 0.0 for item in step.get("ready", ()))
        ),
        next((step for step in trace_steps if step.get("chosen")), trace_steps[0] if trace_steps else {}),
    )
    ready = [dict(item) for item in primary_step.get("ready", ())]
    blocked = [dict(item) for item in primary_step.get("blocked", ())]
    p0_score, p1_score, p2_score, barrier_score, gate_score, total_score = _score_rows(ready)
    selected_matching = tuple(str(item["flow_id"]) for item in primary_step.get("chosen", ()))
    critical_edges = tuple(edge for edge, _score in total_score[: min(4, len(total_score))])
    layer_id = 0
    return PolicyDecisionExplanation(
        policy_name=policy_name,
        layer_id=layer_id,
        p2_source=str(p2_source),
        ready_edges=tuple(str(item["flow_id"]) for item in ready),
        blocked_edges=tuple(str(item["flow_id"]) for item in blocked),
        eligible_edges=tuple(str(item["flow_id"]) for item in ready),
        p0_score_by_edge=p0_score,
        p1_score_by_edge=p1_score,
        p2_score_by_edge=p2_score,
        barrier_score_by_edge=barrier_score,
        gate_score_by_edge=gate_score,
        total_score_by_edge=total_score,
        selected_matching=selected_matching,
        selected_order=_selected_order_from_plan(plan),
        modeled_makespan=float(result.get("makespan", 0.0)),
        bottleneck_edges=_bottleneck_edges_from_schedule(list(result.get("schedule", ()))),
        critical_edges=critical_edges,
        paired_b_makespan=None,
        raw_u_makespan=float(result.get("makespan", 0.0)),
        safe_selected=None,
        fallback_to_b=None,
        fallback_reason=None,
        trace_steps=trace_steps,
    )


def _explain_b_policy(problem: MultiPhaseSchedulingProblem, *, policy_name: str, p2_source: str) -> PolicyDecisionExplanation:
    policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
    plan = policy.build_logical_plan(problem)
    waves = plan.waves
    first_wave = waves[0] if waves else None
    phase = first_wave.flows[0].phase if first_wave and first_wave.flows else "p0_dispatch"
    matrix = {
        "p0_dispatch": problem.p0_dispatch_matrix,
        "p1_return": problem.p1_return_matrix,
        "p2_next_dispatch": problem.p2_next_dispatch_forecast_matrix,
    }.get(phase, problem.p0_dispatch_matrix)
    eligible = []
    for src, row in enumerate(matrix):
        for dst, byte_count in enumerate(row):
            if src == dst or int(byte_count) <= 0:
                continue
            eligible.append((src, dst, int(byte_count)))
    score_rows = []
    for src, dst, byte_count in eligible:
        score = policy._score(phase=phase, src_rank=src, dst_rank=dst, byte_count=byte_count, matrix=matrix)  # type: ignore[attr-defined]
        score_rows.append((f"{phase}:{src}->{dst}", float(score[0]), float(score[1])))
    score_rows.sort(key=lambda item: (-item[1], item[0]))
    selected_order = _selected_order_from_plan(plan)
    selected_matching = tuple(selected_order[: len(first_wave.flows)]) if first_wave is not None else ()
    return PolicyDecisionExplanation(
        policy_name=policy_name,
        layer_id=0,
        p2_source=str(p2_source),
        ready_edges=tuple(edge for edge, _score, _aux in score_rows),
        blocked_edges=(),
        eligible_edges=tuple(edge for edge, _score, _aux in score_rows),
        p0_score_by_edge=tuple((edge, score if edge.startswith("p0_dispatch") else 0.0) for edge, score, _aux in score_rows),
        p1_score_by_edge=tuple((edge, score if edge.startswith("p1_return") else 0.0) for edge, score, _aux in score_rows),
        p2_score_by_edge=tuple((edge, 0.0) for edge, _score, _aux in score_rows),
        barrier_score_by_edge=tuple((edge, aux) for edge, _score, aux in score_rows),
        gate_score_by_edge=tuple((edge, 0.0) for edge, _score, _aux in score_rows),
        total_score_by_edge=tuple((edge, score) for edge, score, _aux in score_rows),
        selected_matching=selected_matching,
        selected_order=selected_order,
        modeled_makespan=float(plan.diagnostics.get("makespan", 0.0)),
        bottleneck_edges=tuple(selected_matching[-1:]),
        critical_edges=tuple(edge for edge, _score, _aux in score_rows[: min(4, len(score_rows))]),
        paired_b_makespan=float(plan.diagnostics.get("makespan", 0.0)),
        raw_u_makespan=None,
        safe_selected=None,
        fallback_to_b=None,
        fallback_reason=None,
        trace_steps=(),
    )


def explain_policy_decision(
    policy: Any,
    window: MultiPhaseSchedulingProblem,
    *,
    p2_hint: tuple[tuple[int, ...], ...] | None,
    p2_source: str,
) -> PolicyDecisionExplanation:
    problem = MultiPhaseSchedulingProblem(
        flow_window=window.flow_window,
        topology=window.topology,
        release_model=window.release_model,
        forecast=window.forecast,
        options=window.options,
        p0_dispatch_matrix=window.p0_dispatch_matrix,
        p1_return_matrix=window.p1_return_matrix,
        p2_next_dispatch_forecast_matrix=window.p2_next_dispatch_forecast_matrix if p2_hint is None else p2_hint,
    )
    policy_name = str(getattr(policy, "policy_name", policy))
    if policy_name in _SAFE_TO_RAW_AND_B:
        raw_name, b_name = _SAFE_TO_RAW_AND_B[policy_name]
        raw_explain = _explain_raw_u(problem, policy_name=raw_name, p2_source=p2_source)
        b_plan = resolve_policy(policy_name=b_name, bucket_rows=0).build_logical_plan(problem)
        safe_plan = policy.build_logical_plan(problem)
        return PolicyDecisionExplanation(
            policy_name=policy_name,
            layer_id=raw_explain.layer_id,
            p2_source=str(p2_source),
            ready_edges=raw_explain.ready_edges,
            blocked_edges=raw_explain.blocked_edges,
            eligible_edges=raw_explain.eligible_edges,
            p0_score_by_edge=raw_explain.p0_score_by_edge,
            p1_score_by_edge=raw_explain.p1_score_by_edge,
            p2_score_by_edge=raw_explain.p2_score_by_edge,
            barrier_score_by_edge=raw_explain.barrier_score_by_edge,
            gate_score_by_edge=raw_explain.gate_score_by_edge,
            total_score_by_edge=raw_explain.total_score_by_edge,
            selected_matching=raw_explain.selected_matching,
            selected_order=raw_explain.selected_order,
            modeled_makespan=float(safe_plan.diagnostics.get("safe_makespan", safe_plan.diagnostics.get("makespan", 0.0))),
            bottleneck_edges=raw_explain.bottleneck_edges,
            critical_edges=raw_explain.critical_edges,
            paired_b_makespan=float(b_plan.diagnostics.get("makespan", 0.0)),
            raw_u_makespan=float(raw_explain.modeled_makespan),
            safe_selected=str(safe_plan.diagnostics.get("selected_policy")),
            fallback_to_b=bool(safe_plan.diagnostics.get("fallback_to_paired_b", False)),
            fallback_reason=str(safe_plan.diagnostics.get("selection_reason")),
            trace_steps=raw_explain.trace_steps,
        )
    if policy_name in _U_SPECS:
        return _explain_raw_u(problem, policy_name=policy_name, p2_source=p2_source)
    if policy_name in {"B_barrier_criticality_matching", "B_gated_greedy_maximal"}:
        return _explain_b_policy(problem, policy_name=policy_name, p2_source=p2_source)
    raise ValueError(f"policy_explain unsupported for {policy_name!r}")


__all__ = ["PolicyDecisionExplanation", "explain_policy_decision"]
