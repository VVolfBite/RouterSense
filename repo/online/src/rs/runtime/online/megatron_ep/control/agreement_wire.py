"""分布式计划协商的 wire 层辅助逻辑。

主要负责：
- 组织 local observation / local plan 的跨 rank 传输
- 生成 agreement 元数据与校验 hash
它是控制面热路径的一部分。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import replace

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.contracts import PlanAgreement
from rs.scheduling.observation_contracts import (
    PhaseDemand,
    PlanWave,
    PeerFlow,
    RankTopologyRecord,
    RouterSensePlan,
    RuntimeObservation,
)
from rs.scheduling.validation import summarize_plan_metrics
from rs.scheduling.validation import stable_hash

_PHASE_TO_CODE = {"P0": 0, "P1": 1}
_CODE_TO_PHASE = {value: key for key, value in _PHASE_TO_CODE.items()}
_RELEASE_TO_CODE = {"ready": 0, "blocked": 1, "advisory_only": 2}
_CODE_TO_RELEASE = {value: key for key, value in _RELEASE_TO_CODE.items()}
_POLICY_TO_CODE = {
    "native_order": 0,
    "joint_shadow_p0p1": 1,
    "native_passthrough_identity": 2,
    "bucketed_fifo": 3,
    "phase_barrier_fifo": 3,
    "trivial_reverse_bucket": 4,
    "aurora_order_fixed": 5,
    "fast_bvn_single_tier": 6,
    "routersense_p0p1_reservation": 7,
    "routersense_p0p1p2_hint": 8,
    "greedy_ready_set": 9,
    "birkhoff_phase_local": 10,
    "islip_round_robin": 11,
}
_CODE_TO_POLICY = {
    0: "native_order",
    1: "joint_shadow_p0p1",
    2: "native_passthrough_identity",
    3: "phase_barrier_fifo",
    4: "trivial_reverse_bucket",
    5: "aurora_order_fixed",
    6: "fast_bvn_single_tier",
    7: "routersense_p0p1_reservation",
    8: "routersense_p0p1p2_hint",
    9: "greedy_ready_set",
    10: "birkhoff_phase_local",
    11: "islip_round_robin",
}
_EXEC_TO_CODE = {"native_passthrough": 0, "shadow_only": 1}
_CODE_TO_EXEC = {value: key for key, value in _EXEC_TO_CODE.items()}


def _hash_to_i64(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    value = int.from_bytes(digest, byteorder="big", signed=False)
    if value >= 2**63:
        value -= 2**64
    return value


def _text_to_i64(text: str) -> int:
    if re.fullmatch(r"[0-9a-f]{16}", text):
        value = int(text, 16)
        if value >= 2**63:
            value -= 2**64
        return value
    return _hash_to_i64(text)


def _i64_to_hex(value: int) -> str:
    unsigned = value if value >= 0 else value + 2**64
    return unsigned.to_bytes(8, byteorder="big", signed=False).hex()


def compute_ep_group_hash(ranks: Iterable[int]) -> str:
    return hashlib.sha256(",".join(str(int(rank)) for rank in ranks).encode("utf-8")).hexdigest()[:16]


def compute_node_index_map(hostname_digests: dict[int, str]) -> dict[int, int]:
    ordered = sorted(set(hostname_digests.values()))
    index_by_digest = {digest: idx for idx, digest in enumerate(ordered)}
    return {rank: index_by_digest[digest] for rank, digest in hostname_digests.items()}


def encode_runtime_observation(observation: RuntimeObservation) -> list[int]:
    peer_count = int(observation.ep_group_size)
    per_peer_rows = list(observation.per_peer_rows[:peer_count]) + [0] * max(0, peer_count - len(observation.per_peer_rows))
    per_peer_bytes = list(observation.per_peer_bytes[:peer_count]) + [0] * max(0, peer_count - len(observation.per_peer_bytes))
    input_splits = list(observation.input_splits[:peer_count]) + [0] * max(0, peer_count - len(observation.input_splits))
    output_splits = list(observation.output_splits[:peer_count]) + [0] * max(0, peer_count - len(observation.output_splits))
    expert_values = list(observation.tokens_per_expert[:64]) + [0] * max(0, 64 - len(observation.tokens_per_expert[:64]))
    return [
        2,
        _text_to_i64(observation.run_id_digest),
        _text_to_i64(observation.step_id_digest),
        _text_to_i64(observation.microbatch_id_digest),
        int(observation.layer_id) if str(observation.layer_id).isdigit() else -1,
        int(observation.global_rank),
        int(observation.local_rank),
        int(observation.topology.device_index),
        int(observation.ep_group_size),
        _PHASE_TO_CODE[str(observation.phase)],
        _text_to_i64(observation.dispatcher_hash),
        _text_to_i64(observation.expert_placement_hash),
        _text_to_i64(observation.model_revision_hash),
        _text_to_i64(observation.ep_group_hash),
        _text_to_i64(observation.request_table_hash),
        _text_to_i64(observation.topology.hostname_digest),
        int(observation.local_rows),
        int(observation.remote_rows),
        *[int(v) for v in per_peer_rows],
        *[int(v) for v in per_peer_bytes],
        *[int(v) for v in input_splits],
        *[int(v) for v in output_splits],
        *[int(v) for v in expert_values],
    ]


def decode_runtime_observation(
    payload: list[int],
    *,
    ep_group_ranks: tuple[int, ...],
    node_index_map: dict[int, int],
) -> RuntimeObservation:
    peer_count = int(payload[8])
    offset = 18
    per_peer_rows = tuple(int(v) for v in payload[offset : offset + peer_count])
    offset += peer_count
    per_peer_bytes = tuple(int(v) for v in payload[offset : offset + peer_count])
    offset += peer_count
    input_splits = tuple(int(v) for v in payload[offset : offset + peer_count])
    offset += peer_count
    output_splits = tuple(int(v) for v in payload[offset : offset + peer_count])
    offset += peer_count
    tokens_per_expert = tuple(int(v) for v in payload[offset : offset + 64])
    rank = int(payload[5])
    local_rank = int(payload[6])
    hostname_digest = _i64_to_hex(int(payload[15]))
    layer_id = str(int(payload[4])) if int(payload[4]) >= 0 else "unknown"
    topology = RankTopologyRecord(
        global_rank=rank,
        local_rank=local_rank,
        node_index=int(node_index_map[rank]),
        hostname_digest=hostname_digest,
        device_index=int(payload[7]),
        ep_group_rank=tuple(ep_group_ranks).index(rank),
    )
    observation = RuntimeObservation(
        run_id=f"digest:{_i64_to_hex(int(payload[1]))}",
        step_id=f"digest:{_i64_to_hex(int(payload[2]))}",
        microbatch_id=f"digest:{_i64_to_hex(int(payload[3]))}",
        layer_id=layer_id,
        layer_name=f"layer_{layer_id}",
        global_rank=rank,
        local_rank=local_rank,
        node_id=f"node:{topology.node_index}",
        device=f"cuda:{int(payload[7])}",
        ep_group_ranks=tuple(ep_group_ranks),
        ep_group_size=int(payload[8]),
        dispatcher_class=f"dispatcher:{_i64_to_hex(int(payload[10]))}",
        expert_placement_hash=_i64_to_hex(int(payload[11])),
        model_revision_hash=_i64_to_hex(int(payload[12])),
        dispatcher_hash=_i64_to_hex(int(payload[10])),
        ep_group_hash=_i64_to_hex(int(payload[13])),
        request_table_hash=_i64_to_hex(int(payload[14])),
        run_id_digest=_i64_to_hex(int(payload[1])),
        step_id_digest=_i64_to_hex(int(payload[2])),
        microbatch_id_digest=_i64_to_hex(int(payload[3])),
        phase=_CODE_TO_PHASE[int(payload[9])],
        per_peer_rows=per_peer_rows,
        per_peer_bytes=per_peer_bytes,
        local_rows=int(payload[16]),
        remote_rows=int(payload[17]),
        topology=topology,
        tokens_per_expert=tokens_per_expert,
        input_splits=input_splits,
        output_splits=output_splits,
        observation_digest="",
        availability={},
    )
    return observation.__class__(
        **{
            **observation.__dict__,
            "observation_digest": stable_hash(
                {
                    "run_id_digest": observation.run_id_digest,
                    "step_id_digest": observation.step_id_digest,
                    "microbatch_id_digest": observation.microbatch_id_digest,
                    "layer_id": observation.layer_id,
                    "global_rank": observation.global_rank,
                    "phase": observation.phase,
                    "per_peer_rows": list(observation.per_peer_rows),
                    "per_peer_bytes": list(observation.per_peer_bytes),
                    "local_rows": observation.local_rows,
                    "remote_rows": observation.remote_rows,
                    "expert_placement_hash": observation.expert_placement_hash,
                    "model_revision_hash": observation.model_revision_hash,
                    "request_table_hash": observation.request_table_hash,
                    "hostname_digest": observation.topology.hostname_digest,
                }
            ),
        }
    )


def _all_gather_variable_int64(local_values: list[int], *, device: torch.device) -> list[list[int]]:
    local_len = torch.tensor([len(local_values)], dtype=torch.int64, device=device)
    world_size = dist.get_world_size()
    gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len)
    lens = [int(item.item()) for item in gathered_lens]
    max_len = max(lens)
    padded = torch.zeros(max_len, dtype=torch.int64, device=device)
    if local_values:
        padded[: len(local_values)] = torch.tensor(local_values, dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return [tensor[: lens[idx]].detach().cpu().tolist() for idx, tensor in enumerate(gathered)]


def encode_plan_tensor(plan: RouterSensePlan, ep_group_size: int) -> list[int]:
    data = [
        2,
        _POLICY_TO_CODE[plan.policy_name],
        _EXEC_TO_CODE[plan.execution_mode],
        1 if plan.transport_mutation else 0,
        _hash_to_i64(plan.run_id),
        _hash_to_i64(plan.step_id),
        _hash_to_i64(plan.microbatch_id),
        int(plan.layer_id) if str(plan.layer_id).isdigit() else -1,
        _text_to_i64(plan.ep_group_hash),
        _text_to_i64(plan.observation_digest),
        _text_to_i64(plan.request_table_hash),
        _text_to_i64(plan.model_revision_hash),
        _text_to_i64(plan.expert_placement_hash),
        0 if plan.control_mode == "default_continue" else 1,
        1 if plan.is_shadow_only else 0,
        1 if plan.can_preempt else 0,
        len(plan.phase_demands),
        len(plan.ready_waves),
        len(plan.blocked_future_waves),
        sum(len(wave.flows) for wave in plan.waves),
    ]
    for demand in plan.phase_demands:
        data.extend([_PHASE_TO_CODE[demand.phase], _RELEASE_TO_CODE[demand.release_state], 1 if demand.payload_exists else 0, len(demand.flows)])
        for flow in demand.flows:
            data.extend(
                [
                    int(flow.src_rank),
                    int(flow.dst_rank),
                    _PHASE_TO_CODE[flow.phase],
                    int(flow.rows),
                    int(flow.bytes),
                    _RELEASE_TO_CODE[flow.release_state],
                    1 if flow.payload_exists else 0,
                    1 if flow.is_cross_rank else 0,
                    1 if flow.is_cross_node else 0,
                    _hash_to_i64(flow.flow_id),
                ]
            )
    for wave_group in (plan.ready_waves, plan.blocked_future_waves):
        for wave in wave_group:
            data.extend([int(wave.wave_id), _RELEASE_TO_CODE[wave.release_state], len(wave.flows)])
            for flow in wave.flows:
                data.extend(
                    [
                        int(flow.src_rank),
                        int(flow.dst_rank),
                        _PHASE_TO_CODE[flow.phase],
                        int(flow.rows),
                        int(flow.bytes),
                        _RELEASE_TO_CODE[flow.release_state],
                        1 if flow.payload_exists else 0,
                        1 if flow.is_cross_rank else 0,
                        1 if flow.is_cross_node else 0,
                        _hash_to_i64(flow.flow_id),
                    ]
                )
    return data


def decode_plan_tensor(payload: list[int], ep_group_size: int | None = None) -> RouterSensePlan:
    version = int(payload[0])
    if version != 2:
        raise ValueError(f"unsupported plan encoding version: {version}")
    phase_demand_count = int(payload[16])
    ready_wave_count = int(payload[17])
    blocked_wave_count = int(payload[18])
    offset = 20
    phase_demands: list[PhaseDemand] = []
    for _ in range(phase_demand_count):
        phase = _CODE_TO_PHASE[int(payload[offset])]
        release_state = _CODE_TO_RELEASE[int(payload[offset + 1])]
        payload_exists = bool(int(payload[offset + 2]))
        flow_count = int(payload[offset + 3])
        offset += 4
        flows: list[PeerFlow] = []
        for _flow_index in range(flow_count):
            src_rank = int(payload[offset])
            dst_rank = int(payload[offset + 1])
            phase_name = _CODE_TO_PHASE[int(payload[offset + 2])]
            rows = int(payload[offset + 3])
            bytes_ = int(payload[offset + 4])
            release = _CODE_TO_RELEASE[int(payload[offset + 5])]
            flow_payload_exists = bool(int(payload[offset + 6]))
            is_cross_rank = bool(int(payload[offset + 7]))
            is_cross_node = bool(int(payload[offset + 8]))
            flow_id = _i64_to_hex(int(payload[offset + 9]))
            offset += 10
            flows.append(
                PeerFlow(
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    phase=phase_name,
                    rows=rows,
                    bytes=bytes_,
                    demand_known_at="router_ready",
                    release_state=release,
                    release_dependency="none"
                    if release != "blocked"
                    else "remote_expert_compute_complete",
                    payload_exists=flow_payload_exists,
                    flow_id=flow_id,
                    is_cross_rank=is_cross_rank,
                    is_cross_node=is_cross_node,
                )
            )
        phase_demands.append(
            PhaseDemand(
                phase=phase,
                flows=tuple(flows),
                demand_known_at="router_ready",
                release_state=release_state,
                release_dependency="none" if release_state != "blocked" else "remote_expert_compute_complete",
                payload_exists=payload_exists,
                total_remote_rows=sum(int(flow.rows) for flow in flows),
                total_remote_bytes=sum(int(flow.bytes) for flow in flows),
            )
        )
    ready_waves: list[PlanWave] = []
    blocked_future_waves: list[PlanWave] = []
    for wave_group, target in ((ready_wave_count, ready_waves), (blocked_wave_count, blocked_future_waves)):
        for _wave_index in range(wave_group):
            wave_id = int(payload[offset])
            release_state = _CODE_TO_RELEASE[int(payload[offset + 1])]
            flow_count = int(payload[offset + 2])
            offset += 3
            flows: list[PeerFlow] = []
            for _flow_index in range(flow_count):
                src_rank = int(payload[offset])
                dst_rank = int(payload[offset + 1])
                phase_name = _CODE_TO_PHASE[int(payload[offset + 2])]
                rows = int(payload[offset + 3])
                bytes_ = int(payload[offset + 4])
                release = _CODE_TO_RELEASE[int(payload[offset + 5])]
                flow_payload_exists = bool(int(payload[offset + 6]))
                is_cross_rank = bool(int(payload[offset + 7]))
                is_cross_node = bool(int(payload[offset + 8]))
                flow_id = _i64_to_hex(int(payload[offset + 9]))
                offset += 10
                flows.append(
                    PeerFlow(
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        phase=phase_name,
                        rows=rows,
                        bytes=bytes_,
                        demand_known_at="router_ready",
                        release_state=release,
                        release_dependency="none"
                        if release != "blocked"
                        else "remote_expert_compute_complete",
                        payload_exists=flow_payload_exists,
                        flow_id=flow_id,
                        is_cross_rank=is_cross_rank,
                        is_cross_node=is_cross_node,
                    )
                )
            target.append(PlanWave(wave_id=wave_id, flows=tuple(flows), release_state=release_state))
    plan = RouterSensePlan(
        policy_name=_CODE_TO_POLICY[int(payload[1])],
        policy_version="v1",
        execution_mode=_CODE_TO_EXEC[int(payload[2])],
        transport_mutation=bool(int(payload[3])),
        run_id=f"plan:{_i64_to_hex(int(payload[4]))}",
        step_id=f"plan:{_i64_to_hex(int(payload[5]))}",
        microbatch_id=f"plan:{_i64_to_hex(int(payload[6]))}",
        layer_id=str(int(payload[7])) if int(payload[7]) >= 0 else "unknown",
        ep_group_hash=_i64_to_hex(int(payload[8])),
        observation_digest=_i64_to_hex(int(payload[9])),
        request_table_hash=_i64_to_hex(int(payload[10])),
        model_revision_hash=_i64_to_hex(int(payload[11])),
        expert_placement_hash=_i64_to_hex(int(payload[12])),
        plan_hash=stable_hash({"wire_payload": list(payload)}),
        control_mode="default_continue" if int(payload[13]) == 0 else "sync_before_phase",
        future_hint_mode="none",
        phase_demands=tuple(phase_demands),
        ready_waves=tuple(ready_waves),
        blocked_future_waves=tuple(blocked_future_waves),
        is_shadow_only=bool(int(payload[14])),
        can_preempt=bool(int(payload[15])),
        metrics={},
    )
    return replace(plan, metrics=summarize_plan_metrics(plan))


def validate_rank_hashes(hashes: Iterable[str]) -> None:
    unique = {item for item in hashes if item}
    if len(unique) > 1:
        raise RuntimeError(f"plan hash mismatch across ranks: {sorted(unique)}")


def run_policy_agreement(
    *,
    local_observation: RuntimeObservation,
    policy,
    context,
    group: dist.ProcessGroup | None,
    device: torch.device,
) -> tuple[RouterSensePlan, PlanAgreement]:
    world_group = group if group is not None else dist.group.WORLD
    local_payload = encode_runtime_observation(local_observation)
    gather_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    gather_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if gather_start is not None:
        gather_start.record()
    gathered_payloads = _all_gather_variable_int64(local_payload, device=device)
    if gather_end is not None:
        gather_end.record()
        torch.cuda.synchronize(device)
        observation_all_gather_ms = float(gather_start.elapsed_time(gather_end))
    else:
        observation_all_gather_ms = 0.0
    rank_order = tuple(int(rank) for rank in dist.get_process_group_ranks(world_group))
    hostname_digests = {
        rank_order[index]: _i64_to_hex(int(payload[15]))
        for index, payload in enumerate(gathered_payloads)
    }
    node_index_map = compute_node_index_map(hostname_digests)
    decoded_observations = tuple(
        decode_runtime_observation(payload, ep_group_ranks=rank_order, node_index_map=node_index_map)
        for payload in gathered_payloads
    )
    root_rank = int(rank_order[0]) if rank_order else 0
    if dist.get_rank(group=world_group) == root_rank:
        root_context = next(item for item in decoded_observations if int(item.global_rank) == root_rank)
        plan = policy.build_plan(local_context=root_context, global_contexts=decoded_observations)
        semantic_hash = plan.plan_hash
        wire_payload = encode_plan_tensor(plan, len(rank_order))
        wire_hash = stable_hash(wire_payload)
    else:
        plan = None
        semantic_hash = ""
        wire_payload = []
        wire_hash = ""
    gathered_wire = _all_gather_variable_int64(wire_payload, device=device)
    root_payload = gathered_wire[0]
    decoded_plan = decode_plan_tensor(root_payload, len(rank_order))
    decoded_hash = decoded_plan.plan_hash
    local_hash_tensor = [_hash_to_i64(decoded_hash)]
    gathered_hashes = _all_gather_variable_int64(local_hash_tensor, device=device)
    rank_hashes = tuple(_i64_to_hex(int(item[0])) for item in gathered_hashes if item)
    validate_rank_hashes(rank_hashes)
    agreement = PlanAgreement(
        root_rank=root_rank,
        rank_count=len(rank_order),
        root_wire_hash=wire_hash or stable_hash(root_payload),
        root_semantic_hash=semantic_hash or decoded_hash,
        decoded_semantic_hash=decoded_hash,
        observation_digest=decoded_plan.observation_digest,
        agreement_status="accepted",
        policy_name=decoded_plan.policy_name,
        policy_version=decoded_plan.policy_version,
        control_mode=decoded_plan.control_mode,
        observation_encode_ms=0.0,
        observation_all_gather_ms=observation_all_gather_ms,
        planner_ms=0.0,
        plan_broadcast_ms=0.0,
        agreement_ms=observation_all_gather_ms,
        rank_hashes=rank_hashes,
        accepted=True,
        reason="ok",
    )
    return decoded_plan, agreement


__all__ = [
    "compute_ep_group_hash",
    "compute_node_index_map",
    "decode_plan_tensor",
    "decode_runtime_observation",
    "encode_plan_tensor",
    "encode_runtime_observation",
    "run_policy_agreement",
    "validate_rank_hashes",
]
