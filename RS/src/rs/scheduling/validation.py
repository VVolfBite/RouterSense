"""Scheduling validation helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Iterable

from rs.scheduling.observation_contracts import (
    PeerFlow,
    PhaseDemand,
    PlanWave,
    PolicyContext,
    RouterSensePlan,
    RuntimeObservation,
)


_PHASE_ORDER = {"P1": 0, "P0": 1}
_IDENTITY_FIELDS = (
    "run_id_digest",
    "step_id_digest",
    "microbatch_id_digest",
    "layer_id",
    "expert_placement_hash",
    "model_revision_hash",
    "ep_group_hash",
    "request_table_hash",
)


def stable_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def flow_priority(flow: PeerFlow) -> tuple[int, int, int, int, int]:
    return (
        _PHASE_ORDER.get(flow.phase, 99),
        -int(flow.bytes),
        int(flow.src_rank),
        int(flow.dst_rank),
        0 if flow.phase == "P0" else 1,
    )


def _flow_semantics(phase: str) -> tuple[str, str, str, bool]:
    if phase == "P0":
        return ("router_ready", "ready", "none", True)
    if phase == "P1":
        return ("router_ready", "blocked", "remote_expert_compute_complete", False)
    raise ValueError(f"Unsupported phase {phase!r}")


def _identity_mismatch(field: str, expected: str, actual: str, *, rank: int) -> ValueError:
    return ValueError(f"identity mismatch field={field} rank={rank} expected={expected} actual={actual}")


def build_remote_flows(observations: Iterable[RuntimeObservation]) -> tuple[PeerFlow, ...]:
    flows: list[PeerFlow] = []
    for obs in observations:
        demand_known_at, release_state, release_dependency, payload_exists = _flow_semantics(obs.phase)
        for peer_idx, rows in enumerate(obs.per_peer_rows):
            rows = int(rows)
            if peer_idx >= len(obs.ep_group_ranks):
                continue
            peer_rank = int(obs.ep_group_ranks[peer_idx])
            if peer_rank == obs.global_rank or rows <= 0:
                continue
            flow_bytes = int(obs.per_peer_bytes[peer_idx]) if peer_idx < len(obs.per_peer_bytes) else 0
            flows.append(
                PeerFlow(
                    flow_id=f"{obs.layer_id}:{obs.phase}:{obs.global_rank}->{peer_rank}",
                    src_rank=int(obs.global_rank),
                    dst_rank=int(peer_rank),
                    phase=str(obs.phase),
                    rows=rows,
                    bytes=flow_bytes,
                    demand_known_at=demand_known_at,
                    release_state=release_state,
                    release_dependency=release_dependency,
                    payload_exists=payload_exists,
                    is_cross_rank=True,
                    is_cross_node=False,
                )
            )
    node_by_rank = {int(obs.global_rank): int(obs.topology.node_index) for obs in observations}
    finalized: list[PeerFlow] = []
    for flow in flows:
        finalized.append(
            PeerFlow(
                flow_id=flow.flow_id,
                src_rank=flow.src_rank,
                dst_rank=flow.dst_rank,
                phase=flow.phase,
                rows=flow.rows,
                bytes=flow.bytes,
                demand_known_at=flow.demand_known_at,
                release_state=flow.release_state,
                release_dependency=flow.release_dependency,
                payload_exists=flow.payload_exists,
                is_cross_rank=flow.src_rank != flow.dst_rank,
                is_cross_node=node_by_rank.get(flow.src_rank, -1) != node_by_rank.get(flow.dst_rank, -1),
            )
        )
    finalized.sort(key=flow_priority)
    return tuple(finalized)


def build_phase_demands(flows: Iterable[PeerFlow]) -> tuple[PhaseDemand, ...]:
    grouped: dict[str, list[PeerFlow]] = defaultdict(list)
    for flow in flows:
        grouped[flow.phase].append(flow)
    demands: list[PhaseDemand] = []
    for phase in ("P0", "P1"):
        phase_flows = tuple(grouped.get(phase, []))
        if not phase_flows:
            continue
        demand_known_at, release_state, release_dependency, payload_exists = _flow_semantics(phase)
        demands.append(
            PhaseDemand(
                phase=phase,
                demand_known_at=demand_known_at,
                release_state=release_state,
                release_dependency=release_dependency,
                payload_exists=payload_exists,
                flows=phase_flows,
                total_remote_rows=sum(int(flow.rows) for flow in phase_flows),
                total_remote_bytes=sum(int(flow.bytes) for flow in phase_flows),
            )
        )
    return tuple(demands)


def validate_global_observations(context: PolicyContext, observations: tuple[RuntimeObservation, ...]) -> None:
    if not observations:
        raise ValueError("global_observation is empty")
    base = observations[0]
    for obs in observations:
        if obs.phase not in {"P0", "P1"}:
            raise ValueError(f"Unsupported phase {obs.phase!r}")
        if tuple(obs.ep_group_ranks) != tuple(context.ep_group_ranks):
            raise ValueError("ep_group_ranks mismatch in RuntimeObservation")
        if int(obs.ep_group_size) != int(context.ep_group_size):
            raise ValueError("ep_group_size mismatch in RuntimeObservation")
        if obs.ep_group_hash != context.ep_group_hash:
            raise _identity_mismatch("ep_group_hash", context.ep_group_hash, obs.ep_group_hash, rank=obs.global_rank)
        if len(obs.per_peer_rows) != len(context.ep_group_ranks):
            raise ValueError(f"per_peer_rows length mismatch rank={obs.global_rank}")
        for field in _IDENTITY_FIELDS:
            expected = str(getattr(base, field))
            actual = str(getattr(obs, field))
            if expected != actual:
                raise _identity_mismatch(field, expected, actual, rank=obs.global_rank)
        if obs.run_id_digest != context.run_id_digest:
            raise _identity_mismatch("run_id_digest", context.run_id_digest, obs.run_id_digest, rank=obs.global_rank)
        if obs.step_id_digest != context.step_id_digest:
            raise _identity_mismatch("step_id_digest", context.step_id_digest, obs.step_id_digest, rank=obs.global_rank)
        if obs.microbatch_id_digest != context.microbatch_id_digest:
            raise _identity_mismatch("microbatch_id_digest", context.microbatch_id_digest, obs.microbatch_id_digest, rank=obs.global_rank)
        if obs.request_table_hash != context.request_table_hash:
            raise _identity_mismatch("request_table_hash", context.request_table_hash, obs.request_table_hash, rank=obs.global_rank)
        if obs.model_revision_hash != context.model_revision_hash:
            raise _identity_mismatch("model_revision_hash", context.model_revision_hash, obs.model_revision_hash, rank=obs.global_rank)
        if obs.expert_placement_hash != context.expert_placement_hash:
            raise _identity_mismatch("expert_placement_hash", context.expert_placement_hash, obs.expert_placement_hash, rank=obs.global_rank)


def validate_shadow_plan(context: PolicyContext, plan: RouterSensePlan) -> None:
    seen_flow_ids: set[str] = set()
    expected_remote = {flow.flow_id for demand in plan.phase_demands for flow in demand.flows if flow.is_cross_rank}
    covered: set[str] = set()
    for wave in plan.waves:
        outgoing_used: set[int] = set()
        incoming_used: set[int] = set()
        for flow in wave.flows:
            if flow.flow_id in seen_flow_ids:
                raise ValueError(f"duplicate flow {flow.flow_id}")
            seen_flow_ids.add(flow.flow_id)
            if flow.src_rank == flow.dst_rank:
                raise ValueError("local flow must not enter network plan")
            if flow.rows < 0 or flow.bytes < 0:
                raise ValueError("negative rows/bytes in flow")
            if flow.src_rank in outgoing_used:
                raise ValueError(f"rank {flow.src_rank} has multiple outgoing flows in wave {wave.wave_id}")
            if flow.dst_rank in incoming_used:
                raise ValueError(f"rank {flow.dst_rank} has multiple incoming flows in wave {wave.wave_id}")
            if wave.release_state == "ready" and flow.phase != "P0":
                raise ValueError(f"ready wave contains non-P0 flow {flow.flow_id}")
            if wave.release_state == "blocked" and flow.phase != "P1":
                raise ValueError(f"blocked wave contains non-P1 flow {flow.flow_id}")
            outgoing_used.add(flow.src_rank)
            incoming_used.add(flow.dst_rank)
            covered.add(flow.flow_id)
    if covered != expected_remote:
        missing = sorted(expected_remote - covered)
        extra = sorted(covered - expected_remote)
        raise ValueError(f"shadow plan coverage mismatch missing={missing} extra={extra}")
    if not context.full_duplex:
        raise ValueError("PolicyContext must be full_duplex for shadow plan")


def phase_coverage_from_demands(demands: tuple[PhaseDemand, ...]) -> dict[str, dict[str, object]]:
    by_phase = {d.phase: d for d in demands}
    return {
        "P0": {
            "present": "P0" in by_phase,
            "rows": int(by_phase["P0"].total_remote_rows) if "P0" in by_phase else 0,
            "bytes": int(by_phase["P0"].total_remote_bytes) if "P0" in by_phase else 0,
            "release_state": str(by_phase["P0"].release_state) if "P0" in by_phase else "ready",
        },
        "P1": {
            "present": "P1" in by_phase,
            "rows": int(by_phase["P1"].total_remote_rows) if "P1" in by_phase else 0,
            "bytes": int(by_phase["P1"].total_remote_bytes) if "P1" in by_phase else 0,
            "release_state": str(by_phase["P1"].release_state) if "P1" in by_phase else "blocked",
        },
    }


def summarize_plan_metrics(plan: RouterSensePlan) -> dict[str, object]:
    per_rank_outgoing: dict[int, int] = defaultdict(int)
    per_rank_incoming: dict[int, int] = defaultdict(int)
    duplex_pair_count = 0
    for wave in plan.waves:
        directed_pairs = {(flow.src_rank, flow.dst_rank) for flow in wave.flows}
        for flow in wave.flows:
            per_rank_outgoing[flow.src_rank] += 1
            per_rank_incoming[flow.dst_rank] += 1
        for src_rank, dst_rank in directed_pairs:
            if (dst_rank, src_rank) in directed_pairs and src_rank < dst_rank:
                duplex_pair_count += 1
    phase_coverage = phase_coverage_from_demands(plan.phase_demands)
    return {
        "total_remote_rows": sum(d.total_remote_rows for d in plan.phase_demands),
        "total_remote_bytes": sum(d.total_remote_bytes for d in plan.phase_demands),
        "wave_count": len(plan.waves),
        "ready_wave_count": len(plan.ready_waves),
        "blocked_future_wave_count": len(plan.blocked_future_waves),
        "ready_P0_rows": sum(flow.rows for wave in plan.ready_waves for flow in wave.flows if flow.phase == "P0"),
        "blocked_P1_rows": sum(flow.rows for wave in plan.blocked_future_waves for flow in wave.flows if flow.phase == "P1"),
        "per_wave_active_flows": [len(wave.flows) for wave in plan.waves],
        "per_rank_outgoing_utilization": dict(sorted(per_rank_outgoing.items())),
        "per_rank_incoming_utilization": dict(sorted(per_rank_incoming.items())),
        "duplex_pair_count": duplex_pair_count,
        "phase_coverage": phase_coverage,
    }


def validate_logical_plan(
    plan,
    *,
    expected_flows: tuple | None = None,
    mode: str | None = None,
) -> dict[str, object]:
    expected = {}
    for flow in expected_flows or ():
        key = (flow.phase, int(flow.src_rank), int(flow.dst_rank))
        expected[key] = expected.get(key, 0) + int(flow.byte_count)
    served: dict[tuple[str, int, int], int] = defaultdict(int)
    served_by_origin: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()
    errors: list[str] = []
    for wave in plan.waves:
        used_src: set[int] = set()
        used_dst: set[int] = set()
        if not wave.flows:
            errors.append(f"wave {wave.wave_id} has no flows")
        for flow in wave.flows:
            if flow.flow_id in seen_ids:
                errors.append(f"duplicate flow id {flow.flow_id}")
            seen_ids.add(flow.flow_id)
            origin_flow_id = str(getattr(flow, "dependency_metadata", {}).get("origin_flow_id", flow.flow_id))
            if int(flow.src_rank) in used_src:
                errors.append(f"wave {wave.wave_id} repeats source {flow.src_rank}")
            if int(flow.dst_rank) in used_dst:
                errors.append(f"wave {wave.wave_id} repeats destination {flow.dst_rank}")
            if mode == "runtime_lookahead" and str(flow.phase) in {"p2_next_dispatch", "p2_next_dispatch_forecast", "phase2"}:
                errors.append(f"runtime_lookahead contains real phase-2 flow {flow.flow_id}")
            used_src.add(int(flow.src_rank))
            used_dst.add(int(flow.dst_rank))
            served[(flow.phase, int(flow.src_rank), int(flow.dst_rank))] += int(flow.byte_count)
            served_by_origin[origin_flow_id] += int(flow.byte_count)
    if expected and served != expected:
        errors.append(f"coverage mismatch expected={expected} served={dict(served)}")
    return {
        "valid": not errors,
        "errors": errors,
        "coverage_verified": not errors,
        "matching_constraints_verified": not any("repeats" in error for error in errors),
        "served_amounts": {str(key): value for key, value in served.items()},
        "served_by_origin": dict(sorted(served_by_origin.items())),
    }


def validate_phase_execution_plan(plan) -> dict[str, object]:
    errors: list[str] = []
    seen: set[str] = set()
    for wave in plan.waves:
        used_src: set[int] = set()
        used_dst: set[int] = set()
        if not wave.bucket_tasks:
            errors.append(f"wave {wave.wave_id} has no bucket tasks")
        for task in wave.bucket_tasks:
            if task.task_id in seen:
                errors.append(f"duplicate task {task.task_id}")
            seen.add(task.task_id)
            if int(task.src_rank) in used_src:
                errors.append(f"wave {wave.wave_id} repeats source {task.src_rank}")
            if int(task.dst_rank) in used_dst:
                errors.append(f"wave {wave.wave_id} repeats destination {task.dst_rank}")
            used_src.add(int(task.src_rank))
            used_dst.add(int(task.dst_rank))
    return {"valid": not errors, "errors": errors}


def validate_bvn_fluid_certificate(certificate: dict[str, object]) -> dict[str, object]:
    phase_certificates = certificate.get("phase_certificates", certificate)
    errors: list[str] = []
    certs = phase_certificates.values() if isinstance(phase_certificates, dict) and "reference_model" not in phase_certificates else (phase_certificates,)
    for cert in certs:
        if not isinstance(cert, dict):
            errors.append("certificate entry is not a mapping")
            continue
        if cert.get("reference_model") != "birkhoff_von_neumann_fluid":
            errors.append("wrong reference_model")
        horizon = int(cert.get("fluid_optimal_horizon", 0))
        lower = max(int(cert.get("max_source_load", 0)), int(cert.get("max_destination_load", 0)))
        if horizon != lower:
            errors.append(f"fluid horizon {horizon} does not match port-load lower bound {lower}")
        if not cert.get("coverage_verified", False):
            errors.append("coverage not verified")
        if not cert.get("matching_constraints_verified", False):
            errors.append("matching constraints not verified")
        if not cert.get("certificate_verified", False):
            errors.append("certificate_verified is false")
        for wave in cert.get("waves", []):
            for edge in wave.get("dummy_edges", []) if isinstance(wave.get("dummy_edges", []), list) else ():
                if isinstance(edge, dict) and edge.get("flow_id"):
                    errors.append("dummy edge leaked flow_id")
    return {"valid": not errors, "errors": errors}


def compare_plan_to_exact_reference(plan, exact_result: dict[str, object]) -> dict[str, object]:
    if not exact_result.get("supported", False):
        return {"available": False, "certified_optimal": False, "optimality_gap": None}
    plan_objective = sum(float(wave.duration) for wave in plan.waves)
    optimal = float(exact_result.get("objective_logical_makespan", 0) or 0)
    gap = None if optimal == 0 and plan_objective != 0 else float(plan_objective - optimal)
    return {
        "available": True,
        "certified_optimal": bool(exact_result.get("certified_optimal", False)),
        "plan_objective_logical_makespan": plan_objective,
        "exact_objective_logical_makespan": optimal,
        "optimality_gap": gap,
        "policy_reaches_optimum": bool(exact_result.get("certified_optimal", False)) and abs(plan_objective - optimal) < 1e-9,
    }


__all__ = [
    "build_phase_demands",
    "build_remote_flows",
    "flow_priority",
    "phase_coverage_from_demands",
    "stable_hash",
    "summarize_plan_metrics",
    "compare_plan_to_exact_reference",
    "validate_global_observations",
    "validate_bvn_fluid_certificate",
    "validate_logical_plan",
    "validate_phase_execution_plan",
    "validate_shadow_plan",
]
