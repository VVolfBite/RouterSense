from __future__ import annotations

from typing import Any

from ..contracts import ONLINE_PIPELINE, TraceOrigin, FutureInformationMode, build_result_envelope


def online_claim_scope_for_world_size(world_size: int) -> tuple[str, bool]:
    if int(world_size) < 4:
        return "correctness_and_calibration_only", False
    return "online_ep_runtime", False


def build_online_result_envelope(
    *,
    run_id: str,
    world_size: int,
    trace_origin: str,
    future_information_mode: str,
    transport_backend: str,
    source_ownership_mode: str,
    expert_residency_mode: str,
    correctness_status: str,
    execution_mode: str,
    is_real_ep_runtime: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    claim_scope, performance_claim_eligible = online_claim_scope_for_world_size(world_size)
    return build_result_envelope(
        run_id=run_id,
        pipeline=ONLINE_PIPELINE,
        claim_scope=claim_scope,
        trace_origin=trace_origin,
        future_information_mode=future_information_mode,
        is_real_ep_runtime=is_real_ep_runtime,
        source_ownership_mode=source_ownership_mode,
        expert_residency_mode=expert_residency_mode,
        transport_backend=transport_backend,
        correctness_status=correctness_status,
        performance_claim_eligible=performance_claim_eligible,
        execution_mode=execution_mode,
        extra={
            "world_size": int(world_size),
            **(extra or {}),
        },
    )


def build_online_unimplemented_result(
    *,
    run_id: str,
    world_size: int,
    transport_backend: str,
    future_information_mode: str = FutureInformationMode.NONE,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_result_envelope(
        run_id=run_id,
        pipeline=ONLINE_PIPELINE,
        claim_scope="unsupported",
        trace_origin="not_collected",
        future_information_mode=future_information_mode,
        is_real_ep_runtime=False,
        source_ownership_mode="unimplemented",
        expert_residency_mode="unimplemented",
        transport_backend="unimplemented",
        correctness_status="unsupported",
        performance_claim_eligible=False,
        execution_mode="unsupported",
        extra={
            "world_size": int(world_size),
            "implemented": False,
            **(extra or {}),
        },
    )
