from __future__ import annotations

import hashlib
import re
import time
from dataclasses import replace
from typing import Iterable

import torch
import torch.distributed as dist

from rs.scheduling.observation_contracts import (
    PhaseDemand,
    PlanAgreement,
    PlanWave,
    PeerFlow,
    PolicyContext,
    RankTopologyRecord,
    RouterSensePlan,
    RuntimeObservation,
)
from rs.scheduling.policy.base import RouterSensePolicy
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
    "trivial_reverse_bucket": 4,
    "aurora_order_fixed": 5,
    "fast_bvn_single_tier": 6,
    "routersense_p0p1_reservation": 7,
    "routersense_p0p1p2_hint": 8,
}
_CODE_TO_POLICY = {value: key for key, value in _POLICY_TO_CODE.items()}
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
    return replace(
        observation,
        observation_digest=stable_hash(
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
        data.extend(
            [
                _PHASE_TO_CODE[demand.phase],
                _RELEASE_TO_CODE[demand.release_state],
                1 if demand.payload_exists else 0,
                len(demand.flows),
            ]
        )
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


def decode_plan_tensor(payload: list[int]) -> RouterSensePlan:
    control_mode = "default_continue" if int(payload[13]) == 0 else "sync_before_phase"
    is_shadow_only = bool(payload[14])
    can_preempt = bool(payload[15])
    phase_demand_count = int(payload[16])
    ready_wave_count = int(payload[17])
    blocked_wave_count = int(payload[18])
    offset = 20

    def _decode_flow_items(count: int) -> tuple[PeerFlow, ...]:
        nonlocal offset
        flows = []
        for _ in range(count):
            src_rank = int(payload[offset])
            dst_rank = int(payload[offset + 1])
            phase = _CODE_TO_PHASE[int(payload[offset + 2])]
            rows = int(payload[offset + 3])
            bytes_ = int(payload[offset + 4])
            release_state = _CODE_TO_RELEASE[int(payload[offset + 5])]
            payload_exists = bool(payload[offset + 6])
            is_cross_rank = bool(payload[offset + 7])
            is_cross_node = bool(payload[offset + 8])
            flow_id = _i64_to_hex(int(payload[offset + 9]))
            offset += 10
            demand_known_at = "router_ready"
            release_dependency = "none" if phase == "P0" else "remote_expert_compute_complete"
            flows.append(
                PeerFlow(
                    flow_id=flow_id,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    phase=phase,
                    rows=rows,
                    bytes=bytes_,
                    demand_known_at=demand_known_at,
                    release_state=release_state,
                    release_dependency=release_dependency,
                    payload_exists=payload_exists,
                    is_cross_rank=is_cross_rank,
                    is_cross_node=is_cross_node,
                )
            )
        return tuple(flows)

    def _decode_phase_demands(count: int) -> tuple[PhaseDemand, ...]:
        nonlocal offset
        demands: list[PhaseDemand] = []
        for _ in range(count):
            phase = _CODE_TO_PHASE[int(payload[offset])]
            release_state = _CODE_TO_RELEASE[int(payload[offset + 1])]
            payload_exists = bool(payload[offset + 2])
            flow_count = int(payload[offset + 3])
            offset += 4
            flows = _decode_flow_items(flow_count)
            demands.append(
                PhaseDemand(
                    phase=phase,
                    demand_known_at="router_ready",
                    release_state=release_state,
                    release_dependency="none" if phase == "P0" else "remote_expert_compute_complete",
                    payload_exists=payload_exists,
                    flows=flows,
                    total_remote_rows=sum(int(flow.rows) for flow in flows),
                    total_remote_bytes=sum(int(flow.bytes) for flow in flows),
                )
            )
        return tuple(demands)

    def _decode_waves(count: int) -> tuple[PlanWave, ...]:
        nonlocal offset

        waves: list[PlanWave] = []
        for _ in range(count):
            wave_id = int(payload[offset])
            release_state = _CODE_TO_RELEASE[int(payload[offset + 1])]
            flow_count = int(payload[offset + 2])
            offset += 3
            flows = _decode_flow_items(flow_count)
            waves.append(PlanWave(wave_id=wave_id, release_state=release_state, flows=flows))
        return tuple(waves)

    phase_demands = _decode_phase_demands(phase_demand_count)
    ready_waves = _decode_waves(ready_wave_count)
    blocked_future_waves = _decode_waves(blocked_wave_count)

    from rs.scheduling.validation import build_phase_demands, summarize_plan_metrics

    if not phase_demands:
        all_flows = [flow for wave in ready_waves + blocked_future_waves for flow in wave.flows]
        phase_demands = build_phase_demands(all_flows)
    plan = RouterSensePlan(
        run_id=f"digest:{_i64_to_hex(int(payload[4]))}",
        step_id=f"digest:{_i64_to_hex(int(payload[5]))}",
        microbatch_id=f"digest:{_i64_to_hex(int(payload[6]))}",
        layer_id=str(int(payload[7])) if int(payload[7]) >= 0 else "unknown",
        ep_group_hash=_i64_to_hex(int(payload[8])),
        request_table_hash=_i64_to_hex(int(payload[10])),
        model_revision_hash=_i64_to_hex(int(payload[11])),
        expert_placement_hash=_i64_to_hex(int(payload[12])),
        observation_digest=_i64_to_hex(int(payload[9])),
        plan_hash="",
        policy_name=_CODE_TO_POLICY[int(payload[1])],
        policy_version="v1",
        execution_mode=_CODE_TO_EXEC[int(payload[2])],
        transport_mutation=bool(payload[3]),
        future_hint_mode="none",
        control_mode=control_mode,
        is_shadow_only=is_shadow_only,
        can_preempt=can_preempt,
        phase_demands=tuple(phase_demands),
        ready_waves=ready_waves,
        blocked_future_waves=blocked_future_waves,
        metrics={},
    )
    metrics = summarize_plan_metrics(plan)
    return replace(plan, metrics=metrics, plan_hash=stable_hash({"plan": plan.to_dict(), "metrics": metrics}))


def validate_rank_hashes(rank_hashes: tuple[str, ...]) -> None:
    if len(set(rank_hashes)) != 1:
        raise RuntimeError(f"plan hash mismatch across ranks: {rank_hashes}")


def run_policy_agreement(
    *,
    local_observations: tuple[RuntimeObservation, ...],
    context: PolicyContext,
    policy: RouterSensePolicy,
    device: torch.device,
    group: dist.ProcessGroup | None = None,
) -> tuple[RouterSensePlan, PlanAgreement]:
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group must be initialized for policy agreement")

    encode_t0 = time.perf_counter()
    local_values: list[int] = [len(local_observations)]
    for observation in local_observations:
        encoded = encode_runtime_observation(observation)
        local_values.append(len(encoded))
        local_values.extend(encoded)
    encode_ms = (time.perf_counter() - encode_t0) * 1000.0

    gather_t0 = time.perf_counter()
    gathered_payloads = _all_gather_variable_int64_with_group(local_values, device=device, group=group)
    gather_ms = (time.perf_counter() - gather_t0) * 1000.0

    hostname_digests: dict[int, str] = {}
    raw_observations: list[list[int]] = []
    for payload in gathered_payloads:
        obs_count = int(payload[0])
        offset = 1
        for _ in range(obs_count):
            obs_len = int(payload[offset])
            offset += 1
            encoded = payload[offset : offset + obs_len]
            offset += obs_len
            raw_observations.append(encoded)
            hostname_digests[int(encoded[5])] = _i64_to_hex(int(encoded[15]))
    node_index_map = compute_node_index_map(hostname_digests)
    global_observations = tuple(
        decode_runtime_observation(raw, ep_group_ranks=context.ep_group_ranks, node_index_map=node_index_map)
        for raw in raw_observations
    )
    global_observation_digest = stable_hash([obs.observation_digest for obs in global_observations])

    planner_t0 = time.perf_counter()
    rank = dist.get_rank(group)
    if rank == 0:
        root_plan = policy.build_plan(context, global_observations)
        if root_plan.observation_digest != global_observation_digest:
            root_plan = replace(root_plan, observation_digest=global_observation_digest)
        root_plan = replace(root_plan, plan_hash=stable_hash({"plan": root_plan.to_dict(), "metrics": root_plan.metrics}))
        plan_values = encode_plan_tensor(root_plan, context.ep_group_size)
    else:
        plan_values = []
    planner_ms = (time.perf_counter() - planner_t0) * 1000.0 if rank == 0 else 0.0

    broadcast_t0 = time.perf_counter()
    plan_len = torch.tensor([len(plan_values)], dtype=torch.int64, device=device)
    root_global_rank = 0 if group is None else dist.get_global_rank(group, 0)
    dist.broadcast(plan_len, src=root_global_rank, group=group)
    plan_tensor = torch.zeros(int(plan_len.item()), dtype=torch.int64, device=device)
    if rank == 0 and plan_values:
        plan_tensor[:] = torch.tensor(plan_values, dtype=torch.int64, device=device)
    dist.broadcast(plan_tensor, src=root_global_rank, group=group)
    broadcast_ms = (time.perf_counter() - broadcast_t0) * 1000.0

    decoded_plan = decode_plan_tensor(plan_tensor.detach().cpu().tolist())
    root_wire_hash = hashlib.sha256(plan_tensor.detach().cpu().numpy().tobytes()).hexdigest()
    decoded_semantic_hash = stable_hash({"plan": decoded_plan.to_dict(), "metrics": decoded_plan.metrics})
    decoded_plan = replace(decoded_plan, plan_hash=root_wire_hash, observation_digest=global_observation_digest)

    hash_t0 = time.perf_counter()
    local_hash_bytes = bytes.fromhex(decoded_semantic_hash)
    local_hash_tensor = torch.tensor(list(local_hash_bytes), dtype=torch.uint8, device=device)
    gathered_hashes = [torch.zeros_like(local_hash_tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered_hashes, local_hash_tensor, group=group)
    rank_hashes = tuple(bytes(t.detach().cpu().tolist()).hex() for t in gathered_hashes)
    hash_ms = (time.perf_counter() - hash_t0) * 1000.0
    validate_rank_hashes(rank_hashes)
    root_semantic_hash = rank_hashes[0]

    agreement = PlanAgreement(
        root_rank=0,
        rank_count=dist.get_world_size(group),
        root_wire_hash=root_wire_hash,
        root_semantic_hash=root_semantic_hash,
        decoded_semantic_hash=decoded_semantic_hash,
        observation_digest=global_observation_digest,
        agreement_status="agreed",
        policy_name=decoded_plan.policy_name,
        policy_version=decoded_plan.policy_version,
        control_mode=decoded_plan.control_mode,
        observation_encode_ms=encode_ms,
        observation_all_gather_ms=gather_ms,
        planner_ms=planner_ms,
        plan_broadcast_ms=broadcast_ms,
        agreement_ms=encode_ms + gather_ms + planner_ms + broadcast_ms + hash_ms,
        rank_hashes=rank_hashes,
        accepted=True,
        reason="root_authoritative_decoded_plan",
    )
    return decoded_plan, agreement


def _all_gather_variable_int64_with_group(
    local_values: list[int], *, device: torch.device, group: dist.ProcessGroup | None
) -> list[list[int]]:
    local_len = torch.tensor([len(local_values)], dtype=torch.int64, device=device)
    world_size = dist.get_world_size(group)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len, group=group)
    lens = [int(item.item()) for item in gathered_lens]
    max_len = max(lens)
    padded = torch.zeros(max_len, dtype=torch.int64, device=device)
    if local_values:
        padded[: len(local_values)] = torch.tensor(local_values, dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)
    return [tensor[: lens[idx]].detach().cpu().tolist() for idx, tensor in enumerate(gathered)]
