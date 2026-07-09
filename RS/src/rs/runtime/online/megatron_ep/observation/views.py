"""运行时观测视图构造。

这里专门负责把在线对象压成 artifact / replay trace 视图：
- phase context 轻量视图
- transport bundle 轻量视图
- scheduled plan 轻量视图
- control replay trace 轻量视图

这些函数不参与调度决策，只负责“如何表示出来”。
"""

from __future__ import annotations

from typing import Any

from .contracts import digest_text


def phase_context_artifact(*, context: Any, perf_profile: bool) -> dict[str, Any]:
    if not perf_profile:
        return context.to_dict()
    remote_segments = [
        segment
        for segment in context.outgoing_segments
        if not bool(segment.is_local) and int(segment.row_count) > 0
    ]
    return {
        "plan_key": dict(context.plan_key),
        "phase": str(context.phase),
        "layer_id": str(context.layer_id),
        "layer_name": str(context.layer_name),
        "global_rank": int(context.global_rank),
        "local_rank": int(context.local_rank),
        "control_mode": str(context.control_mode),
        "release_state": str(context.release_state),
        "demand_known_at": str(context.demand_known_at),
        "payload_exists": bool(context.payload_exists),
        "atomic_submit": bool(context.atomic_submit),
        "per_peer_rows": [int(v) for v in context.per_peer_rows],
        "per_peer_bytes": [int(v) for v in context.per_peer_bytes],
        "nonzero_edge_count": int(len(remote_segments)),
        "remote_row_count": int(sum(int(segment.row_count) for segment in remote_segments)),
        "remote_byte_count": int(sum(int(segment.byte_count) for segment in remote_segments)),
        "transport_bundle_count": int(len(context.transport_bundles)),
        "payload_roles": [str(spec.tensor_role) for spec in context.payload_specs],
        "p2_hint": {
            "hint_mode": str(context.p2_hint.hint_mode),
            "hint_digest": str(context.p2_hint.hint_digest),
            "hint_source": str(context.p2_hint.hint_source),
            "preferred_edge_count": int(context.p2_hint.metadata.get("preferred_edge_count", 0) or 0),
            "preferred_wave_count": int(context.p2_hint.metadata.get("preferred_wave_count", 0) or 0),
        },
    }


def transport_bundle_artifact(*, bundle: Any, perf_profile: bool) -> dict[str, Any]:
    if not perf_profile:
        return bundle.to_dict()
    segment = bundle.outgoing_segment
    return {
        "bundle_id": str(bundle.bundle_id),
        "phase": str(bundle.phase),
        "atomic_submit": bool(bundle.atomic_submit),
        "src_rank": int(segment.src_rank),
        "dst_rank": int(segment.dst_rank),
        "segment_ordinal": int(segment.segment_ordinal),
        "row_count": int(segment.row_count),
        "byte_count": int(segment.byte_count),
        "is_local": bool(segment.is_local),
        "payload_roles": [str(payload.tensor_role) for payload in bundle.payloads],
        "payload_count": int(len(bundle.payloads)),
    }


def scheduled_plan_artifact(*, plan: Any, perf_profile: bool) -> dict[str, Any]:
    if not perf_profile:
        return plan.to_dict()
    metrics = dict(plan.metrics)
    return {
        "plan_key": dict(plan.plan_key),
        "phase": str(plan.phase),
        "policy_name": str(plan.policy_name),
        "policy_version": str(plan.policy_version),
        "control_mode": str(plan.control_mode),
        "execution_mode": str(plan.execution_mode),
        "transport_mutation": bool(plan.transport_mutation),
        "future_hint_mode": str(plan.future_hint_mode),
        "root_rank": int(plan.root_rank),
        "observation_digest": str(plan.observation_digest),
        "plan_hash": str(plan.plan_hash),
        "wave_count": int(len(plan.waves)),
        "waves": [
            {
                "wave_id": int(wave.wave_id),
                "task_count": int(len(wave.bucket_tasks)),
                "task_ids": [str(task.task_id) for task in wave.bucket_tasks],
            }
            for wave in plan.waves
        ],
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key not in {"transfer_layouts", "policy_diagnostics", "bucket_order", "wave_edges"}
        },
    }


def control_replay_trace_row(
    *,
    run_id: str,
    ep_group_size: int,
    bucket_rows: int,
    phase_ctx: Any,
    plan: Any,
) -> dict[str, Any]:
    metrics = dict(plan.metrics)
    p2_summary = {
        "hint_mode": str(phase_ctx.p2_hint.hint_mode),
        "hint_digest": str(phase_ctx.p2_hint.hint_digest),
        "hint_source": str(phase_ctx.p2_hint.hint_source),
        "preferred_edge_count": int(phase_ctx.p2_hint.metadata.get("preferred_edge_count", 0) or 0),
        "preferred_wave_count": int(phase_ctx.p2_hint.metadata.get("preferred_wave_count", 0) or 0),
    }
    nonzero_edges = [
        {
            "src_rank": int(segment.src_rank),
            "dst_rank": int(segment.dst_rank),
            "row_count": int(segment.row_count),
            "byte_count": int(segment.byte_count),
        }
        for segment in phase_ctx.outgoing_segments
        if (not bool(segment.is_local)) and int(segment.row_count) > 0
    ]
    return {
        "run_id_digest": digest_text(str(run_id)),
        "layer_id": str(phase_ctx.layer_id),
        "layer_name": str(phase_ctx.layer_name),
        "phase": str(phase_ctx.phase),
        "global_rank": int(phase_ctx.global_rank),
        "local_rank": int(phase_ctx.local_rank),
        "ep_group_size": int(ep_group_size),
        "policy_name": str(plan.policy_name),
        "bucket_rows": int(bucket_rows),
        "per_rank_peer_bytes": [int(v) for v in phase_ctx.per_peer_bytes],
        "nonzero_edges": nonzero_edges,
        "nonzero_edge_count": int(len(nonzero_edges)),
        "p2_hint_summary": p2_summary,
        "abstract_plan_summary": {
            "plan_hash": str(plan.plan_hash),
            "wave_count": int(len(plan.waves)),
            "task_ref_count": int(metrics.get("abstract_plan_task_ref_count", 0) or 0),
            "abstract_plan_tensor_len": int(metrics.get("abstract_plan_tensor_len", 0) or 0),
        },
        "timing_summary": {
            "all_gather_time_us": float(metrics.get("all_gather_time_us", 0.0) or 0.0),
            "build_plan_time_us": float(metrics.get("build_plan_time_us", 0.0) or 0.0),
            "broadcast_time_us": float(metrics.get("broadcast_time_us", 0.0) or 0.0),
        },
        "transport_summary": {
            "planning_summary_tensor_len": int(metrics.get("planning_summary_tensor_len", 0) or 0),
            "abstract_plan_tensor_len": int(metrics.get("abstract_plan_tensor_len", 0) or 0),
            "abstract_plan_task_ref_count": int(metrics.get("abstract_plan_task_ref_count", 0) or 0),
            "bucket_count": int(metrics.get("bucket_count", 0) or sum(len(wave.bucket_tasks) for wave in plan.waves)),
            "total_wave_count": int(len(plan.waves)),
            "total_byte_count": int(metrics.get("total_byte_count", 0) or 0),
            "hint_match_rate": float(metrics.get("hint_match_rate", 0.0) or 0.0),
            "p2_matrix_source": str(metrics.get("p2_matrix_source", "")),
        },
    }
