"""Debug-only explain traces for RouterSense policies.

This module must report the same raw-U / paired-B / safe decisions that the
real policies made. It therefore reuses the real policy build paths and reads
their diagnostics/debug traces, rather than maintaining a second copy of the
scoring configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem
from rs.scheduling.multiphase.safe_joint import SafeJointPolicy, SafePlannerWrapper
from rs.scheduling.registry import resolve_policy


_SAFE_TO_RAW_AND_B = {
    "RS_safe_barrier_criticality": ("U_barrier_criticality_global_matching", "B_barrier_criticality_matching"),
    "RS_safe_gated_greedy": ("U_gated_greedy_maximal", "B_gated_greedy_maximal"),
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
    selected_matching: tuple[tuple[str, ...], ...]
    selected_order: tuple[str, ...]
    canonical_first_service_order: tuple[str, ...]
    canonical_wave_assignment: tuple[tuple[str, int], ...]
    canonical_chunk_sequence: tuple[tuple[str, str], ...]
    matching_by_wave: tuple[tuple[str, ...], ...]
    first_matching: tuple[str, ...]
    critical_matching_waves: tuple[int, ...]
    modeled_makespan: float
    bottleneck_edges: tuple[str, ...]
    top_score_edges: tuple[str, ...]
    critical_path_edges: tuple[str, ...] | None
    critical_path_unavailable_reason: str | None
    raw_u_selected_matching: tuple[tuple[str, ...], ...]
    raw_u_selected_order: tuple[str, ...]
    raw_u_bottleneck_edges: tuple[str, ...]
    raw_u_top_score_edges: tuple[str, ...]
    paired_b_selected_matching: tuple[tuple[str, ...], ...]
    paired_b_selected_order: tuple[str, ...]
    paired_b_bottleneck_edges: tuple[str, ...]
    paired_b_top_score_edges: tuple[str, ...]
    safe_selected_matching: tuple[tuple[str, ...], ...]
    safe_selected_order: tuple[str, ...]
    safe_bottleneck_edges: tuple[str, ...]
    safe_top_score_edges: tuple[str, ...]
    paired_b_makespan: float | None
    raw_u_makespan: float | None
    safe_selected: str | None
    fallback_to_b: bool | None
    fallback_reason: str | None
    trace_steps: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _layer_id(problem: MultiPhaseSchedulingProblem) -> int:
    layer_id = (problem.forecast.metadata or {}).get("layer_id") if problem.forecast is not None else None
    try:
        return int(layer_id)
    except Exception:
        return 0


def _origin_flow_id(flow: FlowDemand) -> str:
    origin = flow.dependency_metadata.get("origin_flow_id", flow.flow_id)
    return str(origin)


def _canonical_edge_id(flow: FlowDemand) -> str:
    return f"{flow.phase}:{int(flow.src_rank)}->{int(flow.dst_rank)}"


def _plan_order(plan: LogicalSchedulePlan) -> tuple[str, ...]:
    return tuple(str(flow.flow_id) for wave in plan.waves for flow in wave.flows)


def _matching_by_wave(plan: LogicalSchedulePlan) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(_canonical_edge_id(flow) for flow in wave.flows) for wave in plan.waves)


def _first_service_order(plan: LogicalSchedulePlan) -> tuple[str, ...]:
    seen: set[str] = set()
    order: list[str] = []
    for wave in plan.waves:
        for flow in wave.flows:
            edge = _canonical_edge_id(flow)
            if edge in seen:
                continue
            seen.add(edge)
            order.append(edge)
    return tuple(order)


def _wave_assignment(plan: LogicalSchedulePlan) -> tuple[tuple[str, int], ...]:
    assignment: dict[str, int] = {}
    for wave in plan.waves:
        for flow in wave.flows:
            edge = _canonical_edge_id(flow)
            assignment.setdefault(edge, int(wave.wave_id))
    return tuple(sorted(assignment.items(), key=lambda item: (item[1], item[0])))


def _chunk_sequence(plan: LogicalSchedulePlan) -> tuple[tuple[str, str], ...]:
    sequence: list[tuple[str, str]] = []
    for wave in plan.waves:
        for flow in wave.flows:
            sequence.append((_canonical_edge_id(flow), str(flow.flow_id)))
    return tuple(sequence)


def _bottleneck_edges(plan: LogicalSchedulePlan) -> tuple[str, ...]:
    schedule = list(plan.diagnostics.get("raw_schedule", ()))
    if not schedule:
        return ()
    makespan = max(float(entry.get("end", 0.0)) for entry in schedule)
    result = []
    for entry in schedule:
        if abs(float(entry.get("end", 0.0)) - makespan) <= 1e-9:
            phase = int(entry.get("phase", -1))
            src = int(entry.get("src_gpu", -1))
            dst = int(entry.get("dst_gpu", -1))
            result.append(f"{_phase_name(phase)}:{src}->{dst}")
    return tuple(result)


def _phase_name(phase: int) -> str:
    return {0: "p0_dispatch", 1: "p1_return", 2: "p2_next_dispatch"}.get(int(phase), f"phase{phase}")


def _top_score_edges_from_ready(ready: list[dict[str, Any]]) -> tuple[str, ...]:
    ranked = sorted(ready, key=lambda item: (float(item.get("score", 0.0)), str(item.get("flow_id", ""))), reverse=True)
    return tuple(f"{_phase_name(int(item['phase']))}:{int(item['src_gpu'])}->{int(item['dst_gpu'])}" for item in ranked[: min(4, len(ranked))])


def _score_rows(ready: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    p0 = []
    p1 = []
    p2 = []
    barrier = []
    gate = []
    total = []
    ranked = sorted(ready, key=lambda item: (float(item.get("score", 0.0)), str(item.get("flow_id", ""))), reverse=True)
    for candidate in ranked:
        edge = f"{_phase_name(int(candidate['phase']))}:{int(candidate['src_gpu'])}->{int(candidate['dst_gpu'])}"
        phase = int(candidate["phase"])
        residual_component = float(candidate.get("residual_component", 0.0))
        p0.append((edge, residual_component if phase == 0 else 0.0))
        p1.append((edge, residual_component if phase == 1 else 0.0))
        p2.append((edge, float(candidate.get("prediction_component", 0.0))))
        barrier.append((edge, float(candidate.get("barrier_component", 0.0))))
        gate.append((edge, float(candidate.get("age_component", 0.0) + candidate.get("base_priority_component", 0.0))))
        total.append((edge, float(candidate.get("score", 0.0))))
    return tuple(p0), tuple(p1), tuple(p2), tuple(barrier), tuple(gate), tuple(total)


def _primary_trace_step(trace_steps: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    for step in trace_steps:
        if step.get("chosen") and any(abs(float(item.get("prediction_component", 0.0))) > 1e-12 for item in step.get("ready", ())):
            return step
    for step in trace_steps:
        if step.get("chosen"):
            return step
    return trace_steps[0] if trace_steps else {}


def _debug_trace_from_plan(plan: LogicalSchedulePlan) -> tuple[dict[str, Any], ...]:
    return tuple(dict(step) for step in plan.diagnostics.get("scheduler_debug_trace", ()) or ())


def _collect_debug_plan(policy: Any, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    previous = getattr(policy, "collect_debug_trace", False)
    try:
        if hasattr(policy, "collect_debug_trace"):
            setattr(policy, "collect_debug_trace", True)
        return policy.build_logical_plan(problem)
    finally:
        if hasattr(policy, "collect_debug_trace"):
            setattr(policy, "collect_debug_trace", previous)


def _explain_from_raw_u_plan(
    *,
    policy_name: str,
    problem: MultiPhaseSchedulingProblem,
    raw_plan: LogicalSchedulePlan,
    p2_source: str,
) -> PolicyDecisionExplanation:
    trace_steps = _debug_trace_from_plan(raw_plan)
    primary_step = _primary_trace_step(trace_steps)
    ready = [dict(item) for item in primary_step.get("ready", ())]
    blocked = [dict(item) for item in primary_step.get("blocked", ())]
    p0_score, p1_score, p2_score, barrier_score, gate_score, total_score = _score_rows(ready)
    top_score_edges = _top_score_edges_from_ready(ready)
    matching_by_wave = _matching_by_wave(raw_plan)
    first_matching = matching_by_wave[0] if matching_by_wave else ()
    return PolicyDecisionExplanation(
        policy_name=policy_name,
        layer_id=_layer_id(problem),
        p2_source=str(p2_source),
        ready_edges=tuple(f"{_phase_name(int(item['phase']))}:{int(item['src_gpu'])}->{int(item['dst_gpu'])}" for item in ready),
        blocked_edges=tuple(f"{_phase_name(int(item['phase']))}:{int(item['src_gpu'])}->{int(item['dst_gpu'])}" for item in blocked),
        eligible_edges=tuple(f"{_phase_name(int(item['phase']))}:{int(item['src_gpu'])}->{int(item['dst_gpu'])}" for item in ready),
        p0_score_by_edge=p0_score,
        p1_score_by_edge=p1_score,
        p2_score_by_edge=p2_score,
        barrier_score_by_edge=barrier_score,
        gate_score_by_edge=gate_score,
        total_score_by_edge=total_score,
        selected_matching=matching_by_wave,
        selected_order=_plan_order(raw_plan),
        canonical_first_service_order=_first_service_order(raw_plan),
        canonical_wave_assignment=_wave_assignment(raw_plan),
        canonical_chunk_sequence=_chunk_sequence(raw_plan),
        matching_by_wave=matching_by_wave,
        first_matching=first_matching,
        critical_matching_waves=tuple(range(len(matching_by_wave))),
        modeled_makespan=float(raw_plan.diagnostics.get("makespan", 0.0)),
        bottleneck_edges=_bottleneck_edges(raw_plan),
        top_score_edges=top_score_edges,
        critical_path_edges=None,
        critical_path_unavailable_reason="critical_path_not_implemented_for_replay_schedule",
        raw_u_selected_matching=matching_by_wave,
        raw_u_selected_order=_plan_order(raw_plan),
        raw_u_bottleneck_edges=_bottleneck_edges(raw_plan),
        raw_u_top_score_edges=top_score_edges,
        paired_b_selected_matching=(),
        paired_b_selected_order=(),
        paired_b_bottleneck_edges=(),
        paired_b_top_score_edges=(),
        safe_selected_matching=matching_by_wave,
        safe_selected_order=_plan_order(raw_plan),
        safe_bottleneck_edges=_bottleneck_edges(raw_plan),
        safe_top_score_edges=top_score_edges,
        paired_b_makespan=None,
        raw_u_makespan=float(raw_plan.diagnostics.get("makespan", 0.0)),
        safe_selected=None,
        fallback_to_b=None,
        fallback_reason=None,
        trace_steps=trace_steps,
    )


def _explain_from_b_plan(
    *,
    policy_name: str,
    problem: MultiPhaseSchedulingProblem,
    b_plan: LogicalSchedulePlan,
    p2_source: str,
) -> PolicyDecisionExplanation:
    matching_by_wave = _matching_by_wave(b_plan)
    first_matching = matching_by_wave[0] if matching_by_wave else ()
    order = _plan_order(b_plan)
    # For B policies we do not have a per-candidate debug trace; use the actual plan.
    top_score_edges = tuple(_first_service_order(b_plan)[: min(4, len(_first_service_order(b_plan)))])
    return PolicyDecisionExplanation(
        policy_name=policy_name,
        layer_id=_layer_id(problem),
        p2_source=str(p2_source),
        ready_edges=tuple(_first_service_order(b_plan)),
        blocked_edges=(),
        eligible_edges=tuple(_first_service_order(b_plan)),
        p0_score_by_edge=(),
        p1_score_by_edge=(),
        p2_score_by_edge=tuple((edge, 0.0) for edge in _first_service_order(b_plan)),
        barrier_score_by_edge=(),
        gate_score_by_edge=(),
        total_score_by_edge=(),
        selected_matching=matching_by_wave,
        selected_order=order,
        canonical_first_service_order=_first_service_order(b_plan),
        canonical_wave_assignment=_wave_assignment(b_plan),
        canonical_chunk_sequence=_chunk_sequence(b_plan),
        matching_by_wave=matching_by_wave,
        first_matching=first_matching,
        critical_matching_waves=tuple(range(len(matching_by_wave))),
        modeled_makespan=float(b_plan.diagnostics.get("makespan", 0.0)),
        bottleneck_edges=_bottleneck_edges(b_plan),
        top_score_edges=top_score_edges,
        critical_path_edges=None,
        critical_path_unavailable_reason="critical_path_not_implemented_for_phase_local_plan",
        raw_u_selected_matching=(),
        raw_u_selected_order=(),
        raw_u_bottleneck_edges=(),
        raw_u_top_score_edges=(),
        paired_b_selected_matching=matching_by_wave,
        paired_b_selected_order=order,
        paired_b_bottleneck_edges=_bottleneck_edges(b_plan),
        paired_b_top_score_edges=top_score_edges,
        safe_selected_matching=matching_by_wave,
        safe_selected_order=order,
        safe_bottleneck_edges=_bottleneck_edges(b_plan),
        safe_top_score_edges=top_score_edges,
        paired_b_makespan=float(b_plan.diagnostics.get("makespan", 0.0)),
        raw_u_makespan=None,
        safe_selected=None,
        fallback_to_b=None,
        fallback_reason=None,
        trace_steps=(),
    )


def _merge_safe_explanation(
    *,
    policy_name: str,
    problem: MultiPhaseSchedulingProblem,
    raw_plan: LogicalSchedulePlan,
    b_plan: LogicalSchedulePlan,
    safe_plan: LogicalSchedulePlan,
    p2_source: str,
) -> PolicyDecisionExplanation:
    raw_base = _explain_from_raw_u_plan(policy_name=policy_name, problem=problem, raw_plan=raw_plan, p2_source=p2_source)
    b_base = _explain_from_b_plan(policy_name=policy_name, problem=problem, b_plan=b_plan, p2_source=p2_source)
    fallback_to_b = bool(safe_plan.diagnostics.get("fallback_to_paired_b", False))
    safe_matching = b_base.paired_b_selected_matching if fallback_to_b else raw_base.raw_u_selected_matching
    safe_order = b_base.paired_b_selected_order if fallback_to_b else raw_base.raw_u_selected_order
    safe_bottleneck = b_base.paired_b_bottleneck_edges if fallback_to_b else raw_base.raw_u_bottleneck_edges
    safe_top_score = b_base.paired_b_top_score_edges if fallback_to_b else raw_base.raw_u_top_score_edges
    safe_wave_assignment = b_base.canonical_wave_assignment if fallback_to_b else raw_base.canonical_wave_assignment
    safe_chunk_sequence = b_base.canonical_chunk_sequence if fallback_to_b else raw_base.canonical_chunk_sequence
    safe_first_service = b_base.canonical_first_service_order if fallback_to_b else raw_base.canonical_first_service_order
    return PolicyDecisionExplanation(
        policy_name=policy_name,
        layer_id=_layer_id(problem),
        p2_source=str(p2_source),
        ready_edges=raw_base.ready_edges,
        blocked_edges=raw_base.blocked_edges,
        eligible_edges=raw_base.eligible_edges,
        p0_score_by_edge=raw_base.p0_score_by_edge,
        p1_score_by_edge=raw_base.p1_score_by_edge,
        p2_score_by_edge=raw_base.p2_score_by_edge,
        barrier_score_by_edge=raw_base.barrier_score_by_edge,
        gate_score_by_edge=raw_base.gate_score_by_edge,
        total_score_by_edge=raw_base.total_score_by_edge,
        selected_matching=safe_matching,
        selected_order=safe_order,
        canonical_first_service_order=safe_first_service,
        canonical_wave_assignment=safe_wave_assignment,
        canonical_chunk_sequence=safe_chunk_sequence,
        matching_by_wave=safe_matching,
        first_matching=safe_matching[0] if safe_matching else (),
        critical_matching_waves=tuple(range(len(safe_matching))),
        modeled_makespan=float(safe_plan.diagnostics.get("safe_makespan", safe_plan.diagnostics.get("makespan", 0.0))),
        bottleneck_edges=safe_bottleneck,
        top_score_edges=safe_top_score,
        critical_path_edges=None,
        critical_path_unavailable_reason="critical_path_not_implemented_for_safe_policy",
        raw_u_selected_matching=raw_base.raw_u_selected_matching,
        raw_u_selected_order=raw_base.raw_u_selected_order,
        raw_u_bottleneck_edges=raw_base.raw_u_bottleneck_edges,
        raw_u_top_score_edges=raw_base.raw_u_top_score_edges,
        paired_b_selected_matching=b_base.paired_b_selected_matching,
        paired_b_selected_order=b_base.paired_b_selected_order,
        paired_b_bottleneck_edges=b_base.paired_b_bottleneck_edges,
        paired_b_top_score_edges=b_base.paired_b_top_score_edges,
        safe_selected_matching=safe_matching,
        safe_selected_order=safe_order,
        safe_bottleneck_edges=safe_bottleneck,
        safe_top_score_edges=safe_top_score,
        paired_b_makespan=float(b_plan.diagnostics.get("makespan", 0.0)),
        raw_u_makespan=float(raw_plan.diagnostics.get("makespan", 0.0)),
        safe_selected=str(safe_plan.diagnostics.get("selected_policy", "")),
        fallback_to_b=fallback_to_b,
        fallback_reason=str(safe_plan.diagnostics.get("selection_reason", "")),
        trace_steps=raw_base.trace_steps,
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
    if isinstance(policy, SafePlannerWrapper) or policy_name in _SAFE_TO_RAW_AND_B:
        safe_policy = policy if isinstance(policy, SafePlannerWrapper) else resolve_policy(policy_name=policy_name, bucket_rows=0)
        assert isinstance(safe_policy, SafePlannerWrapper)
        raw_plan = _collect_debug_plan(safe_policy._joint_policy, problem)
        b_plan = safe_policy._local_policy.build_logical_plan(problem)
        joint_eval, local_eval, selected, reason = safe_policy.evaluate_components(
            problem, joint_plan=raw_plan, local_plan=b_plan,
        )
        diagnostics = safe_policy.selection_diagnostics(
            joint_eval=joint_eval, local_eval=local_eval, selected=selected, reason=reason,
        )
        safe_plan = LogicalSchedulePlan(
            policy_name=safe_policy.policy_name, waves=selected.plan.waves, diagnostics=diagnostics,
        )
        return _merge_safe_explanation(
            policy_name=policy_name,
            problem=problem,
            raw_plan=raw_plan,
            b_plan=b_plan,
            safe_plan=safe_plan,
            p2_source=p2_source,
        )
    if policy_name.startswith("U_"):
        raw_policy = policy if hasattr(policy, "build_logical_plan") else resolve_policy(policy_name=policy_name, bucket_rows=0)
        raw_plan = _collect_debug_plan(raw_policy, problem)
        return _explain_from_raw_u_plan(policy_name=policy_name, problem=problem, raw_plan=raw_plan, p2_source=p2_source)
    b_policy = policy if hasattr(policy, "build_logical_plan") else resolve_policy(policy_name=policy_name, bucket_rows=0)
    b_plan = b_policy.build_logical_plan(problem)
    return _explain_from_b_plan(policy_name=policy_name, problem=problem, b_plan=b_plan, p2_source=p2_source)


__all__ = ["PolicyDecisionExplanation", "explain_policy_decision"]
