from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts import ONLINE_PIPELINE, TraceOrigin, FutureInformationMode, build_result_envelope
from ...contracts import EpExecutionTrace
from ..observer_io import write_online_trace_artifacts


def build_native_ep_observer_metadata(*, run_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_result_envelope(
        run_id=run_id,
        pipeline=ONLINE_PIPELINE,
        claim_scope="correctness_and_calibration_only",
        trace_origin=TraceOrigin.OBSERVED_ONLINE_NATIVE_EP,
        future_information_mode=FutureInformationMode.NONE,
        is_real_ep_runtime=True,
        source_ownership_mode="per_rank_local_input_partition",
        expert_residency_mode="full_checkpoint_then_prune",
        transport_backend="online_native_a2a_ep",
        correctness_status="not_checked",
        performance_claim_eligible=False,
        extra=extra or {},
    )


def build_single_rank_local_moe_observer_metadata(*, run_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_result_envelope(
        run_id=run_id,
        pipeline=ONLINE_PIPELINE,
        claim_scope="correctness_and_calibration_only",
        trace_origin=TraceOrigin.OBSERVED_SINGLE_RANK_LOCAL_MOE,
        future_information_mode=FutureInformationMode.NONE,
        is_real_ep_runtime=False,
        source_ownership_mode="single_rank_local_prompt",
        expert_residency_mode="full_model_local_weight_extract_for_parity",
        transport_backend="single_rank_local_moe_reconstruction",
        correctness_status="not_checked",
        performance_claim_eligible=False,
        extra=extra or {},
    )


def build_ws2_route_partition_observer_metadata(*, run_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_result_envelope(
        run_id=run_id,
        pipeline=ONLINE_PIPELINE,
        claim_scope="distributed_route_partition_and_count_agreement_only",
        trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_ROUTE_PARTITION,
        future_information_mode=FutureInformationMode.NONE,
        is_real_ep_runtime=False,
        source_ownership_mode="dist_rank_local_prompt",
        expert_residency_mode="full_checkpoint_then_local_extract",
        transport_backend="torch_distributed_metadata_agreement",
        correctness_status="not_checked",
        performance_claim_eligible=False,
        extra=extra or {},
    )


def build_ws2_hidden_dispatch_observer_metadata(*, run_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_result_envelope(
        run_id=run_id,
        pipeline=ONLINE_PIPELINE,
        claim_scope="distributed_hidden_dispatch_only",
        trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_HIDDEN_DISPATCH,
        future_information_mode=FutureInformationMode.NONE,
        is_real_ep_runtime=False,
        source_ownership_mode="dist_rank_local_prompt",
        expert_residency_mode="full_checkpoint_then_local_extract",
        transport_backend="torch_distributed_hidden_dispatch_only",
        correctness_status="not_checked",
        performance_claim_eligible=False,
        extra=extra or {},
    )


def export_native_ep_trace_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    trace: EpExecutionTrace,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    metadata = build_native_ep_observer_metadata(run_id=run_id, extra=extra_metadata or {})
    return write_online_trace_artifacts(
        output_dir=output_dir,
        run_id=run_id,
        trace=trace,
        metadata=metadata,
    )


def export_single_rank_local_moe_trace_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    trace: EpExecutionTrace,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    metadata = build_single_rank_local_moe_observer_metadata(run_id=run_id, extra=extra_metadata or {})
    return write_online_trace_artifacts(
        output_dir=output_dir,
        run_id=run_id,
        trace=trace,
        metadata=metadata,
    )


def export_ws2_route_partition_trace_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    trace: EpExecutionTrace,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    metadata = build_ws2_route_partition_observer_metadata(run_id=run_id, extra=extra_metadata or {})
    return write_online_trace_artifacts(
        output_dir=output_dir,
        run_id=run_id,
        trace=trace,
        metadata=metadata,
    )


def export_ws2_hidden_dispatch_trace_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    trace: EpExecutionTrace,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    metadata = build_ws2_hidden_dispatch_observer_metadata(run_id=run_id, extra=extra_metadata or {})
    return write_online_trace_artifacts(
        output_dir=output_dir,
        run_id=run_id,
        trace=trace,
        metadata=metadata,
    )
