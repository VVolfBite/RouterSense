from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import PlanningRequest, WindowPlan


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    reasons: tuple[str, ...]


def validate_window_plan_for_request(plan: WindowPlan, request: PlanningRequest) -> None:
    request.validate()
    plan.validate()
    if str(plan.request_digest) != str(request.semantic_digest()):
        raise ValueError("request_digest_mismatch")
    world_size = int(request.topology.world_size)
    wave_ids = [int(wave.wave_id) for wave in plan.waves]
    if len(set(wave_ids)) != len(wave_ids):
        raise ValueError("duplicate_wave_id")
    if wave_ids != sorted(wave_ids):
        raise ValueError("wave_ids_not_sorted")
    allowed_phases = {
        "p0_only": {"p0_dispatch"},
        "p0_p1": {"p0_dispatch", "p1_return"},
        "p0_p1_p2": {"p0_dispatch", "p1_return", "p2_next_dispatch_forecast", "p2_next_dispatch"},
    }[str(request.information_mode)]
    for wave in plan.waves:
        used_src: set[int] = set()
        used_dst: set[int] = set()
        for flow in wave.flows:
            if not 0 <= int(flow.src_rank) < world_size:
                raise ValueError("src_rank_out_of_range")
            if not 0 <= int(flow.dst_rank) < world_size:
                raise ValueError("dst_rank_out_of_range")
            if str(flow.phase) not in allowed_phases:
                raise ValueError("phase_not_allowed_for_information_mode")
            if str(request.p2_semantics) == "absent" and str(flow.phase).startswith("p2_"):
                raise ValueError("p2_present_when_absent")
            if str(request.p2_semantics) == "advisory_hint" and str(flow.phase) == "p2_next_dispatch" and bool(flow.executable):
                raise ValueError("advisory_p2_cannot_be_executable")
            if str(request.p2_semantics) == "executable_actual" and str(flow.phase) == "p2_next_dispatch_forecast":
                raise ValueError("execution_window_cannot_use_forecast_p2")
            if bool(flow.executable):
                if int(flow.src_rank) in used_src:
                    raise ValueError("multiple_outgoing_in_wave")
                if int(flow.dst_rank) in used_dst:
                    raise ValueError("multiple_incoming_in_wave")
                used_src.add(int(flow.src_rank))
                used_dst.add(int(flow.dst_rank))
