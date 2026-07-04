from __future__ import annotations

from typing import Any

from ...contracts import OFFLINE_PIPELINE, TraceOrigin, FutureInformationMode, build_result_envelope


def build_calibrated_counterfactual_result(
    *,
    run_id: str,
    future_information_mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_result_envelope(
        run_id=run_id,
        pipeline=OFFLINE_PIPELINE,
        claim_scope="calibrated_offline_counterfactual",
        trace_origin=TraceOrigin.OBSERVED_ONLINE_NATIVE_EP,
        future_information_mode=future_information_mode,
        is_real_ep_runtime=False,
        source_ownership_mode="observed_online_native_ep",
        expert_residency_mode="observed_online_native_ep",
        transport_backend="calibrated_simulator",
        correctness_status="unsupported",
        performance_claim_eligible=False,
        extra={
            "deployable_scheduler_candidate": future_information_mode != FutureInformationMode.ORACLE_FULL_TRACE,
            **(extra or {}),
        },
    )
