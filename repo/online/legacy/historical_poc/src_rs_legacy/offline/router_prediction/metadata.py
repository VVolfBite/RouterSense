from __future__ import annotations

from typing import Any

from ...contracts import OFFLINE_PIPELINE, TraceOrigin, FutureInformationMode, build_result_envelope


def build_router_prediction_result(*, run_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_result_envelope(
        run_id=run_id,
        pipeline=OFFLINE_PIPELINE,
        claim_scope="router_prediction_only",
        trace_origin=TraceOrigin.SINGLE_GPU_PROXY_ROUTER,
        future_information_mode=FutureInformationMode.ORACLE_FULL_TRACE,
        is_real_ep_runtime=False,
        source_ownership_mode="unavailable",
        expert_residency_mode="full_model_single_gpu_proxy",
        transport_backend="none",
        correctness_status="unsupported",
        performance_claim_eligible=False,
        extra={
            "analysis_scope": "router_prediction_only",
            "is_real_ep_trace": False,
            "source_ownership_available": False,
            "communication_cost_calibrated": False,
            **(extra or {}),
        },
    )
