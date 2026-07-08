"""PhaseExecutionPlan 的 root-authoritative 协商流程。

主要函数：
- run_phase_plan_agreement()
这是当前 sync_before_phase 路径里最核心的控制面入口之一。
"""

from __future__ import annotations

import time
from dataclasses import replace

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.phase import PhaseExecutionPlan, PhaseReadyContext
from rs.scheduling.phase_execution import (
    AbstractPhaseExecutionPlan,
    FutureDemandHint,
    PlanningSummaryContext,
    PhasePlanningSummary,
    PlanningEdgeSummary,
)
from rs.scheduling.phase_execution_utils import materialize_local_execution_plan

_WIRE_VERSION = 1
_PHASE_TO_CODE = {"P0": 0, "P1": 1}
_CODE_TO_PHASE = {value: key for key, value in _PHASE_TO_CODE.items()}
_HINT_MODE_TO_CODE = {"none": 0, "deterministic_stub": 1, "calibrated_artifact": 2}
_CODE_TO_HINT_MODE = {value: key for key, value in _HINT_MODE_TO_CODE.items()}
_POLICY_NAME_TO_CODE = {
    "phase_barrier_fifo": 0,
    "bucketed_fifo": 1,
    "greedy_ready_set": 2,
    "birkhoff_phase_local": 3,
    "islip_round_robin": 4,
    "power_of_two_choices": 5,
    "routersense_p0p1_reservation": 6,
    "routersense_p0p1p2_hint": 7,
    "fast_bvn_single_tier": 8,
    "aurora_order_fixed": 9,
    "trivial_reverse_bucket": 10,
}
_CODE_TO_POLICY_NAME = {value: key for key, value in _POLICY_NAME_TO_CODE.items()}


def _wire_device(group: dist.ProcessGroup | None) -> torch.device:
    try:
        backend = str(dist.get_backend(group if group is not None else dist.group.WORLD))
    except Exception:
        backend = ""
    if backend == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _cuda_synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hex_to_words(value: str, *, word_count: int) -> list[int]:
    normalized = str(value or "").lower().replace("0x", "")
    normalized = normalized[: 16 * word_count].ljust(16 * word_count, "0")
    words: list[int] = []
    for index in range(0, 16 * word_count, 16):
        raw = int(normalized[index : index + 16], 16)
        if raw >= (1 << 63):
            raw -= 1 << 64
        words.append(raw)
    return words


def _words_to_hex(words: list[int] | tuple[int, ...], *, word_count: int) -> str:
    return "".join(f"{int(word) & ((1 << 64) - 1):016x}" for word in list(words)[:word_count])


def _summary_tensor_length(world_size: int) -> int:
    # header(3) + per_peer_bytes(world_size) + edges(world_size * 4)
    return 3 + int(world_size) + (int(world_size) * 4)


def _encode_planning_summary_tensor(summary: PhasePlanningSummary, *, world_size: int, device: torch.device) -> torch.Tensor:
    payload = [-1 for _ in range(_summary_tensor_length(world_size))]
    payload[0] = _WIRE_VERSION
    payload[1] = int(summary.global_rank)
    payload[2] = len(summary.outgoing_edges)
    cursor = 3
    for value in tuple(int(v) for v in summary.per_peer_bytes)[:world_size]:
        payload[cursor] = int(value)
        cursor += 1
    edge_by_dst = {int(edge.dst_rank): edge for edge in summary.outgoing_edges}
    for dst_rank in range(world_size):
        edge = edge_by_dst.get(dst_rank)
        if edge is None:
            payload[cursor : cursor + 4] = [-1, -1, 0, 0]
        else:
            payload[cursor : cursor + 4] = [
                int(edge.dst_rank),
                int(edge.segment_ordinal),
                int(edge.row_count),
                int(edge.byte_count),
            ]
        cursor += 4
    return torch.tensor(payload, dtype=torch.long, device=device)


def _decode_planning_summary_tensor(
    tensor: torch.Tensor,
    *,
    phase: str,
    control_mode: str,
    layer_id: str,
    ep_group_ranks: tuple[int, ...],
    ep_group_root_rank: int,
    plan_key_factory,
) -> PhasePlanningSummary:
    values = [int(item) for item in tensor.detach().cpu().tolist()]
    global_rank = int(values[1])
    edge_count = int(values[2])
    world_size = len(tuple(ep_group_ranks))
    cursor = 3
    per_peer_bytes = tuple(int(values[cursor + index]) for index in range(world_size))
    cursor += world_size
    outgoing_edges: list[PlanningEdgeSummary] = []
    for _ in range(world_size):
        dst_rank, segment_ordinal, row_count, byte_count = values[cursor : cursor + 4]
        cursor += 4
        if dst_rank < 0 or row_count <= 0 or byte_count <= 0:
            continue
        outgoing_edges.append(
            PlanningEdgeSummary(
                phase=str(phase),
                src_rank=int(global_rank),
                dst_rank=int(dst_rank),
                segment_ordinal=int(segment_ordinal),
                row_count=int(row_count),
                byte_count=int(byte_count),
            )
        )
    outgoing_edges = outgoing_edges[:edge_count]
    return PhasePlanningSummary(
        plan_key=dict(plan_key_factory(int(global_rank))),
        phase=str(phase),
        control_mode=str(control_mode),
        layer_id=str(layer_id),
        global_rank=int(global_rank),
        ep_group_ranks=tuple(int(v) for v in ep_group_ranks),
        ep_group_root_rank=int(ep_group_root_rank),
        per_peer_bytes=per_peer_bytes,
        outgoing_edges=tuple(outgoing_edges),
    )


def _decode_planning_summary_rows(
    rows: list[list[int]],
    *,
    phase: str,
    control_mode: str,
    layer_id: str,
    ep_group_ranks: tuple[int, ...],
    ep_group_root_rank: int,
    plan_key_factory,
) -> tuple[PhasePlanningSummary, ...]:
    world_size = len(tuple(ep_group_ranks))
    result: list[PhasePlanningSummary] = []
    for values in rows:
        global_rank = int(values[1])
        edge_count = int(values[2])
        cursor = 3
        per_peer_bytes = tuple(int(values[cursor + index]) for index in range(world_size))
        cursor += world_size
        outgoing_edges: list[PlanningEdgeSummary] = []
        for _ in range(world_size):
            dst_rank, segment_ordinal, row_count, byte_count = values[cursor : cursor + 4]
            cursor += 4
            if dst_rank < 0 or row_count <= 0 or byte_count <= 0:
                continue
            outgoing_edges.append(
                PlanningEdgeSummary(
                    phase=str(phase),
                    src_rank=int(global_rank),
                    dst_rank=int(dst_rank),
                    segment_ordinal=int(segment_ordinal),
                    row_count=int(row_count),
                    byte_count=int(byte_count),
                )
            )
        result.append(
            PhasePlanningSummary(
                plan_key=dict(plan_key_factory(int(global_rank))),
                phase=str(phase),
                control_mode=str(control_mode),
                layer_id=str(layer_id),
                global_rank=int(global_rank),
                ep_group_ranks=tuple(int(v) for v in ep_group_ranks),
                ep_group_root_rank=int(ep_group_root_rank),
                per_peer_bytes=per_peer_bytes,
                outgoing_edges=tuple(outgoing_edges[:edge_count]),
            )
        )
    return tuple(result)


def _decode_abstract_plan_values(
    values: list[int],
    *,
    local_context: PhaseReadyContext,
) -> AbstractPhaseExecutionPlan:
    phase = _CODE_TO_PHASE[int(values[1])]
    policy_name = _CODE_TO_POLICY_NAME.get(int(values[2]), "phase_barrier_fifo")
    transport_mutation = bool(values[3])
    is_shadow_only = bool(values[4])
    root_rank = int(values[5])
    bucket_rows = int(values[6])
    wave_count = int(values[7])
    observation_digest = _words_to_hex(values[8:12], word_count=4)
    plan_hash = _words_to_hex(values[12:16], word_count=4)
    cursor = 16
    waves = []
    for _ in range(wave_count):
        wave_id = int(values[cursor])
        task_count = int(values[cursor + 1])
        cursor += 2
        task_refs = []
        for _ in range(task_count):
            src_rank, dst_rank, segment_ordinal, bucket_ordinal = values[cursor : cursor + 4]
            cursor += 4
            task_refs.append(
                {
                    "phase": phase,
                    "src_rank": int(src_rank),
                    "dst_rank": int(dst_rank),
                    "segment_ordinal": int(segment_ordinal),
                    "bucket_ordinal": int(bucket_ordinal),
                }
            )
        waves.append({"wave_id": int(wave_id), "phase": phase, "task_refs": task_refs})
    return AbstractPhaseExecutionPlan.from_dict(
        {
            "plan_key": dict(local_context.plan_key),
            "phase": phase,
            "policy_name": policy_name,
            "policy_version": "v1",
            "control_mode": str(local_context.control_mode),
            "execution_mode": "phase_sync_wave",
            "transport_mutation": transport_mutation,
            "is_shadow_only": is_shadow_only,
            "future_hint_mode": str(local_context.p2_hint.hint_mode),
            "root_rank": int(root_rank),
            "observation_digest": observation_digest,
            "plan_hash": plan_hash,
            "waves": waves,
            "metrics": {
                "bucket_rows": int(bucket_rows),
                "wave_count": int(wave_count),
                "transport_mutation": transport_mutation,
            },
        }
    )


def _encode_abstract_plan_tensor(plan: AbstractPhaseExecutionPlan, *, device: torch.device) -> torch.Tensor:
    payload: list[int] = [
        _WIRE_VERSION,
        _PHASE_TO_CODE[str(plan.phase)],
        _POLICY_NAME_TO_CODE.get(str(plan.policy_name), -1),
        int(plan.transport_mutation),
        int(plan.is_shadow_only),
        int(plan.root_rank),
        int(plan.metrics.get("bucket_rows", 0) or 0),
        len(plan.waves),
    ]
    payload.extend(_hex_to_words(str(plan.observation_digest), word_count=4))
    payload.extend(_hex_to_words(str(plan.plan_hash), word_count=4))
    for wave in plan.waves:
        payload.append(int(wave.wave_id))
        payload.append(len(wave.task_refs))
        for task_ref in wave.task_refs:
            payload.extend(
                [
                    int(task_ref.src_rank),
                    int(task_ref.dst_rank),
                    int(task_ref.segment_ordinal),
                    int(task_ref.bucket_ordinal),
                ]
            )
    return torch.tensor(payload, dtype=torch.long, device=device)


def _decode_abstract_plan_tensor(
    tensor: torch.Tensor,
    *,
    local_context: PhaseReadyContext,
) -> AbstractPhaseExecutionPlan:
    values = [int(item) for item in tensor.detach().cpu().tolist()]
    return _decode_abstract_plan_values(values, local_context=local_context)


def _get_process_group_root_safe(group: dist.ProcessGroup | None) -> int:
    if group is None:
        return 0
    if hasattr(dist, "get_process_group_ranks"):
        ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
        return int(ranks[0]) if ranks else 0
    return 0


def run_phase_plan_agreement(
    *,
    local_context: PhaseReadyContext,
    policy,
    group: dist.ProcessGroup | None,
) -> PhaseExecutionPlan:
    world_group = group if group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group=world_group)
    root_rank = int(_get_process_group_root_safe(world_group))
    device = _wire_device(world_group)
    rank = dist.get_rank(group=world_group)
    summary_build_start_ns = time.monotonic_ns()
    local_summary = local_context.to_planning_summary()
    summary_build_end_ns = time.monotonic_ns()
    summary_encode_start_ns = time.monotonic_ns()
    local_summary_tensor = _encode_planning_summary_tensor(local_summary, world_size=world_size, device=device)
    summary_encode_end_ns = time.monotonic_ns()
    gathered = [torch.empty_like(local_summary_tensor) for _ in range(world_size)]
    all_gather_submit_start_ns = time.monotonic_ns()
    dist.all_gather(gathered, local_summary_tensor, group=world_group)
    all_gather_submit_end_ns = time.monotonic_ns()
    all_gather_sync_start_ns = time.monotonic_ns()
    _cuda_synchronize_if_needed(device)
    all_gather_sync_end_ns = time.monotonic_ns()
    full_plan = None
    summary_stack_time_us = 0.0
    summary_tensor_to_cpu_time_us = 0.0
    summary_object_decode_time_us = 0.0
    if rank == root_rank:
        rebuilt_contexts: list[PhaseReadyContext] = []
        summary_stack_start_ns = time.monotonic_ns()
        gathered_matrix = torch.stack(gathered, dim=0)
        summary_stack_end_ns = time.monotonic_ns()
        summary_cpu_start_ns = time.monotonic_ns()
        gathered_cpu = gathered_matrix.detach().cpu()
        summary_cpu_end_ns = time.monotonic_ns()
        summary_object_decode_start_ns = time.monotonic_ns()
        summaries = _decode_planning_summary_rows(
            gathered_cpu.tolist(),
            phase=str(local_context.phase),
            control_mode=str(local_context.control_mode),
            layer_id=str(local_context.layer_id),
            ep_group_ranks=tuple(int(v) for v in local_context.ep_group_ranks),
            ep_group_root_rank=int(local_context.ep_group_root_rank),
            plan_key_factory=lambda global_rank: {
                **dict(local_context.plan_key),
                "rank": int(global_rank),
            },
        )
        summary_object_decode_end_ns = time.monotonic_ns()
        summary_stack_time_us = (summary_stack_end_ns - summary_stack_start_ns) / 1000.0
        summary_tensor_to_cpu_time_us = (summary_cpu_end_ns - summary_cpu_start_ns) / 1000.0
        summary_object_decode_time_us = (summary_object_decode_end_ns - summary_object_decode_start_ns) / 1000.0
        for summary in summaries:
            if int(summary.global_rank) == int(local_context.global_rank):
                rebuilt_contexts.append(local_context)
            else:
                rebuilt_contexts.append(PlanningSummaryContext.from_summary(summary))
        global_contexts = tuple(rebuilt_contexts)
        root_context = next(ctx for ctx in global_contexts if int(ctx.global_rank) == root_rank)
        build_plan_start_ns = time.monotonic_ns()
        full_plan = policy.build_plan(local_context=root_context, global_contexts=global_contexts)
        build_plan_end_ns = time.monotonic_ns()
        abstract_encode_start_ns = time.monotonic_ns()
        abstract_plan = full_plan.to_abstract_plan()
        payload_tensor = _encode_abstract_plan_tensor(abstract_plan, device=device)
        abstract_encode_end_ns = time.monotonic_ns()
    else:
        build_plan_start_ns = time.monotonic_ns()
        build_plan_end_ns = build_plan_start_ns
        abstract_encode_start_ns = build_plan_end_ns
        abstract_encode_end_ns = abstract_encode_start_ns
        abstract_plan = None
        payload_tensor = None
    payload_length = torch.tensor(
        [int(payload_tensor.numel()) if payload_tensor is not None else 0],
        dtype=torch.long,
        device=device,
    )
    broadcast_length_submit_start_ns = time.monotonic_ns()
    dist.broadcast(payload_length, src=root_rank, group=world_group)
    broadcast_length_submit_end_ns = time.monotonic_ns()
    broadcast_length_sync_start_ns = time.monotonic_ns()
    _cuda_synchronize_if_needed(device)
    broadcast_length_sync_end_ns = time.monotonic_ns()
    buffer_tensor = (
        payload_tensor
        if payload_tensor is not None
        else torch.empty(int(payload_length.item()), dtype=torch.long, device=device)
    )
    broadcast_payload_submit_start_ns = time.monotonic_ns()
    dist.broadcast(buffer_tensor, src=root_rank, group=world_group)
    broadcast_payload_submit_end_ns = time.monotonic_ns()
    broadcast_payload_sync_start_ns = time.monotonic_ns()
    _cuda_synchronize_if_needed(device)
    broadcast_payload_sync_end_ns = time.monotonic_ns()
    abstract_tensor_to_cpu_time_us = 0.0
    abstract_object_decode_time_us = 0.0
    decoded = (
        abstract_plan
        if isinstance(abstract_plan, AbstractPhaseExecutionPlan)
        else None
    )
    if decoded is None:
        abstract_cpu_start_ns = time.monotonic_ns()
        buffer_cpu = buffer_tensor.detach().cpu()
        abstract_cpu_end_ns = time.monotonic_ns()
        abstract_object_decode_start_ns = time.monotonic_ns()
        decoded = _decode_abstract_plan_values([int(item) for item in buffer_cpu.tolist()], local_context=local_context)
        abstract_object_decode_end_ns = time.monotonic_ns()
        abstract_tensor_to_cpu_time_us = (abstract_cpu_end_ns - abstract_cpu_start_ns) / 1000.0
        abstract_object_decode_time_us = (abstract_object_decode_end_ns - abstract_object_decode_start_ns) / 1000.0
    materialize_start_ns = time.monotonic_ns()
    local_plan = materialize_local_execution_plan(local_context=local_context, abstract_plan=decoded)
    materialize_end_ns = time.monotonic_ns()
    verify_start_ns = time.monotonic_ns()
    expected_plan_hash = str(decoded.plan_hash)
    actual_plan_hash = str(local_plan.plan_hash)
    if actual_plan_hash != expected_plan_hash:
        raise RuntimeError(
            "phase plan hash mismatch after local materialization: "
            f"expected={expected_plan_hash} actual={actual_plan_hash}"
        )
    verify_end_ns = time.monotonic_ns()
    all_gather_submit_time_us = (all_gather_submit_end_ns - all_gather_submit_start_ns) / 1000.0
    all_gather_sync_time_us = (all_gather_sync_end_ns - all_gather_sync_start_ns) / 1000.0
    all_gather_total_time_us = all_gather_submit_time_us + all_gather_sync_time_us
    broadcast_length_submit_time_us = (broadcast_length_submit_end_ns - broadcast_length_submit_start_ns) / 1000.0
    broadcast_length_sync_time_us = (broadcast_length_sync_end_ns - broadcast_length_sync_start_ns) / 1000.0
    broadcast_payload_submit_time_us = (broadcast_payload_submit_end_ns - broadcast_payload_submit_start_ns) / 1000.0
    broadcast_payload_sync_time_us = (broadcast_payload_sync_end_ns - broadcast_payload_sync_start_ns) / 1000.0
    broadcast_total_time_us = (
        broadcast_length_submit_time_us
        + broadcast_length_sync_time_us
        + broadcast_payload_submit_time_us
        + broadcast_payload_sync_time_us
    )
    timing_metrics = {
        "summary_build_time_us": (summary_build_end_ns - summary_build_start_ns) / 1000.0,
        "summary_encode_time_us": (summary_encode_end_ns - summary_encode_start_ns) / 1000.0,
        "all_gather_submit_time_us": all_gather_submit_time_us,
        "all_gather_sync_time_us": all_gather_sync_time_us,
        "all_gather_total_time_us": all_gather_total_time_us,
        "all_gather_time_us": all_gather_total_time_us,
        "summary_stack_time_us": summary_stack_time_us,
        "summary_tensor_to_cpu_time_us": summary_tensor_to_cpu_time_us,
        "summary_object_decode_time_us": summary_object_decode_time_us,
        "summary_decode_time_us": summary_stack_time_us + summary_tensor_to_cpu_time_us + summary_object_decode_time_us,
        "build_plan_time_us": (build_plan_end_ns - build_plan_start_ns) / 1000.0,
        "abstract_encode_time_us": (abstract_encode_end_ns - abstract_encode_start_ns) / 1000.0,
        "broadcast_length_submit_time_us": broadcast_length_submit_time_us,
        "broadcast_length_sync_time_us": broadcast_length_sync_time_us,
        "broadcast_payload_submit_time_us": broadcast_payload_submit_time_us,
        "broadcast_payload_sync_time_us": broadcast_payload_sync_time_us,
        "broadcast_total_time_us": broadcast_total_time_us,
        "broadcast_time_us": broadcast_total_time_us,
        "abstract_tensor_to_cpu_time_us": abstract_tensor_to_cpu_time_us,
        "abstract_object_decode_time_us": abstract_object_decode_time_us,
        "abstract_decode_time_us": abstract_tensor_to_cpu_time_us + abstract_object_decode_time_us,
        "materialize_local_plan_time_us": (materialize_end_ns - materialize_start_ns) / 1000.0,
        "verify_time_us": (verify_end_ns - verify_start_ns) / 1000.0,
        "planning_summary_tensor_len": int(local_summary_tensor.numel()),
        "planning_summary_total_elements": int(local_summary_tensor.numel() * world_size),
        "abstract_plan_tensor_len": int(buffer_tensor.numel()),
        "abstract_plan_total_elements": int(buffer_tensor.numel()),
        "abstract_plan_task_ref_count": int(sum(len(wave.task_refs) for wave in decoded.waves)),
        "broadcast_payload_elements": int(buffer_tensor.numel()),
        "total_agreement_time_us": (verify_end_ns - all_gather_submit_start_ns) / 1000.0,
    }
    return replace(local_plan, metrics={**local_plan.metrics, **timing_metrics})


__all__ = ["run_phase_plan_agreement"]
