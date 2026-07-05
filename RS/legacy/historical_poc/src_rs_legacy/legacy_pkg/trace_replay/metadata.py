from __future__ import annotations

from typing import Any

from ...contracts import LEGACY_TRACE_REPLAY_PIPELINE, TraceOrigin, FutureInformationMode, build_result_envelope


LEGACY_TRACE_REPLAY_MODE = "legacy_trace_replay"


def build_legacy_trace_replay_result(
    *,
    run_id: str,
    transport_backend: str,
    correctness_status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_result_envelope(
        run_id=run_id,
        pipeline=LEGACY_TRACE_REPLAY_PIPELINE,
        claim_scope="transport_replay_only",
        trace_origin=TraceOrigin.LEGACY_TRACE_REPLAY,
        future_information_mode=FutureInformationMode.ORACLE_FULL_TRACE,
        is_real_ep_runtime=False,
        source_ownership_mode="synthetic_token_position_modulo_partition",
        expert_residency_mode="rank_local_expert_weight_cache_from_full_model",
        transport_backend=transport_backend,
        correctness_status=correctness_status,
        performance_claim_eligible=False,
        execution_mode=LEGACY_TRACE_REPLAY_MODE,
        extra=extra or {},
    )
