from __future__ import annotations

import json

import pytest

from rs.legacy.trace_replay import LEGACY_TRACE_REPLAY_MODE, build_legacy_trace_replay_result
from rs.offline.calibration import assert_online_native_ep_observation
from rs.offline.reporting import build_calibrated_counterfactual_result
from rs.offline.router_prediction import build_router_prediction_result
from rs.online import build_online_unimplemented_result
from rs.online.observer_io import write_online_trace_artifacts
from rs.online.scheduler_bridge import normalize_online_future_hint_mode
from rs.online.transport import transport_backend_realizes_matching
from rs.online.transport.native_alltoall import ONLINE_NATIVE_A2A_EP
from rs.contracts import (
    EpExecutionTrace,
    ExpertBucketRecord,
    FutureInformationMode,
    LayerRouteTrace,
    RankStageTiming,
    RouteIdentity,
    RouteRecord,
    TraceOrigin,
)


def test_legacy_trace_replay_is_marked_legacy() -> None:
    payload = build_legacy_trace_replay_result(
        run_id="legacy-run",
        transport_backend="scheduled_collective_partition_replay",
        correctness_status="not_checked",
    )
    assert payload["pipeline"] == "legacy"
    assert payload["execution_mode"] == LEGACY_TRACE_REPLAY_MODE
    assert payload["trace_origin"] == "legacy_trace_replay"
    assert payload["is_real_ep_runtime"] is False
    assert payload["performance_claim_eligible"] is False


def test_offline_proxy_trace_cannot_drive_calibrated_ep_analysis() -> None:
    payload = build_router_prediction_result(run_id="offline-run")
    with pytest.raises(RuntimeError, match="observed_online_native_ep"):
        assert_online_native_ep_observation(payload)


def test_online_scheduler_rejects_oracle_future_trace() -> None:
    with pytest.raises(RuntimeError, match="oracle_full_trace"):
        normalize_online_future_hint_mode("oracle_full_trace")


def test_all_to_all_backend_is_not_matching_realized() -> None:
    assert transport_backend_realizes_matching(ONLINE_NATIVE_A2A_EP) is False
    assert transport_backend_realizes_matching("scheduled_p2p") is True


def test_online_unimplemented_result_fails_closed() -> None:
    payload = build_online_unimplemented_result(
        run_id="online-run",
        world_size=2,
        transport_backend=ONLINE_NATIVE_A2A_EP,
    )
    assert payload["pipeline"] == "online"
    assert payload["claim_scope"] == "unsupported"
    assert payload["trace_origin"] == "not_collected"
    assert payload["performance_claim_eligible"] is False
    assert payload["implemented"] is False


def test_world_size_one_native_parity_result_uses_truthful_scope() -> None:
    payload = build_online_unimplemented_result(
        run_id="online-run",
        world_size=1,
        transport_backend=ONLINE_NATIVE_A2A_EP,
        extra={"implemented": True},
    )
    payload["execution_mode"] = "world_size_1_local_moe_reconstruction_parity"
    payload["trace_origin"] = TraceOrigin.OBSERVED_SINGLE_RANK_LOCAL_MOE
    payload["is_real_ep_runtime"] = False
    payload["expert_residency_mode"] = "full_model_local_weight_extract_for_parity"
    assert payload["claim_scope"] == "unsupported"
    assert payload["performance_claim_eligible"] is False
    assert payload["trace_origin"] == TraceOrigin.OBSERVED_SINGLE_RANK_LOCAL_MOE
    assert payload["is_real_ep_runtime"] is False
    assert payload["expert_residency_mode"] == "full_model_local_weight_extract_for_parity"


def test_offline_oracle_counterfactual_is_not_deployable() -> None:
    payload = build_calibrated_counterfactual_result(
        run_id="offline-calibrated",
        future_information_mode=FutureInformationMode.ORACLE_FULL_TRACE,
    )
    assert payload["claim_scope"] == "calibrated_offline_counterfactual"
    assert payload["performance_claim_eligible"] is False
    assert payload["deployable_scheduler_candidate"] is False


def test_online_observer_writes_metadata_and_jsonl(tmp_path) -> None:
    trace = EpExecutionTrace(
        trace_origin=TraceOrigin.OBSERVED_ONLINE_NATIVE_EP,
        future_information_mode=FutureInformationMode.NONE,
        route_traces=[
            LayerRouteTrace(
                trace_origin=TraceOrigin.OBSERVED_ONLINE_NATIVE_EP,
                future_information_mode=FutureInformationMode.NONE,
                layer_id=0,
                route_records=[
                    RouteRecord(
                        identity=RouteIdentity(
                            run_id="run-0",
                            request_id="req-0",
                            microbatch_id="mb-0",
                            layer_id=0,
                            source_rank=0,
                            destination_rank=1,
                            expert_id=3,
                            token_index_local=2,
                            topk_slot=0,
                        ),
                        routing_weight=0.75,
                        payload_rows=1,
                        payload_bytes=8192,
                        is_local_route=False,
                        is_remote_route=True,
                    )
                ],
            )
        ],
        stage_timings=[RankStageTiming(rank=0, stage="dispatch_post", wall_ms=1.25)],
        expert_buckets=[ExpertBucketRecord(rank=1, layer_id=0, expert_id=3, bucket_rows=1, bucket_bytes=8192)],
    )
    jsonl_path, metadata_path = write_online_trace_artifacts(
        output_dir=tmp_path,
        run_id="run-0",
        trace=trace,
        metadata={"pipeline": "online"},
    )
    assert jsonl_path.exists()
    assert metadata_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert row["record_type"] == "route_record"
    assert row["payload"]["identity"]["source_rank"] == 0
    assert json.loads(lines[1])["record_type"] == "rank_stage_timing"
    assert json.loads(lines[2])["record_type"] == "expert_bucket_record"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["trace_origin"] == TraceOrigin.OBSERVED_ONLINE_NATIVE_EP
    assert metadata["trace_artifact_schema_version"] == 2
    assert metadata["remote_route_rows"] == 1


def test_calibrated_analysis_rejects_single_rank_local_moe_trace(tmp_path) -> None:
    trace = EpExecutionTrace(
        trace_origin=TraceOrigin.OBSERVED_SINGLE_RANK_LOCAL_MOE,
        future_information_mode=FutureInformationMode.NONE,
        route_traces=[
            LayerRouteTrace(
                trace_origin=TraceOrigin.OBSERVED_SINGLE_RANK_LOCAL_MOE,
                future_information_mode=FutureInformationMode.NONE,
                layer_id=0,
                route_records=[],
            )
        ],
        stage_timings=[RankStageTiming(rank=0, stage="local_expert_compute", wall_ms=0.0)],
        expert_buckets=[ExpertBucketRecord(rank=0, layer_id=0, expert_id=0, bucket_rows=1, bucket_bytes=16)],
    )
    _jsonl_path, metadata_path = write_online_trace_artifacts(
        output_dir=tmp_path,
        run_id="single-rank-trace",
        trace=trace,
        metadata={
            "pipeline": "online",
            "is_real_ep_runtime": False,
            "world_size": 1,
            "transport_backend": "single_rank_local_moe_reconstruction",
        },
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="observed_online_native_ep"):
        assert_online_native_ep_observation(metadata, metadata_path)


def test_calibrated_analysis_requires_real_multirank_artifact(tmp_path) -> None:
    trace = EpExecutionTrace(
        trace_origin=TraceOrigin.OBSERVED_ONLINE_NATIVE_EP,
        future_information_mode=FutureInformationMode.NONE,
        route_traces=[
            LayerRouteTrace(
                trace_origin=TraceOrigin.OBSERVED_ONLINE_NATIVE_EP,
                future_information_mode=FutureInformationMode.NONE,
                layer_id=0,
                route_records=[
                    RouteRecord(
                        identity=RouteIdentity(
                            run_id="run-0",
                            request_id="req-0",
                            microbatch_id="mb-0",
                            layer_id=0,
                            source_rank=0,
                            destination_rank=1,
                            expert_id=3,
                            token_index_local=2,
                            topk_slot=0,
                        ),
                        routing_weight=0.75,
                        payload_rows=1,
                        payload_bytes=8192,
                        is_local_route=False,
                        is_remote_route=True,
                    )
                ],
            )
        ],
        stage_timings=[RankStageTiming(rank=0, stage="dispatch_post", wall_ms=1.25)],
        expert_buckets=[ExpertBucketRecord(rank=1, layer_id=0, expert_id=3, bucket_rows=1, bucket_bytes=8192)],
    )
    _jsonl_path, metadata_path = write_online_trace_artifacts(
        output_dir=tmp_path,
        run_id="real-trace",
        trace=trace,
        metadata={
            "pipeline": "online",
            "is_real_ep_runtime": True,
            "world_size": 2,
            "transport_backend": "online_native_a2a_ep",
            "correctness_status": "passed",
        },
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert_online_native_ep_observation(metadata, metadata_path)
