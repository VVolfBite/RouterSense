from __future__ import annotations

import torch
from dataclasses import replace

from rs.runtime.online.megatron_ep.contracts import PolicyContext, RankTopologyRecord, RuntimeObservation
from rs.runtime.online.megatron_ep.phase import FutureDemandHint, PhaseReadyContext, build_phase_ready_context
from rs.scheduling.policy.validation import stable_hash


def make_observation(
    *,
    rank: int,
    phase: str,
    rows: tuple[int, int],
    ep_group_ranks: tuple[int, ...] = (0, 1),
    local_rank: int | None = None,
    hostname_digest: str = "host-a",
    node_index: int = 0,
    request_table_hash: str = "request",
    placement_hash: str = "placement",
    model_revision_hash: str = "model",
) -> RuntimeObservation:
    local_rank = rank if local_rank is None else local_rank
    run_id = "run"
    step_id = "step"
    microbatch_id = "mb"
    layer_id = "0"
    ep_group_hash = stable_hash({"ep_group_ranks": list(ep_group_ranks)})
    dispatcher_hash = stable_hash("Dispatcher")
    run_id_digest = stable_hash(run_id)
    step_id_digest = stable_hash(step_id)
    microbatch_id_digest = stable_hash(microbatch_id)
    topology = RankTopologyRecord(
        global_rank=rank,
        local_rank=local_rank,
        node_index=node_index,
        hostname_digest=hostname_digest,
        device_index=local_rank,
        ep_group_rank=ep_group_ranks.index(rank),
    )
    observation_digest = stable_hash(
        {
            "run_id_digest": run_id_digest,
            "step_id_digest": step_id_digest,
            "microbatch_id_digest": microbatch_id_digest,
            "layer_id": layer_id,
            "global_rank": rank,
            "phase": phase,
            "per_peer_rows": list(rows),
            "per_peer_bytes": [v * 16 for v in rows],
            "local_rows": rows[ep_group_ranks.index(rank)],
            "remote_rows": sum(v for i, v in enumerate(rows) if i != ep_group_ranks.index(rank)),
            "expert_placement_hash": placement_hash,
            "model_revision_hash": model_revision_hash,
            "request_table_hash": request_table_hash,
            "hostname_digest": hostname_digest,
        }
    )
    return RuntimeObservation(
        run_id=run_id,
        step_id=step_id,
        microbatch_id=microbatch_id,
        layer_id=layer_id,
        layer_name="decoder.layers.0.mlp",
        global_rank=rank,
        local_rank=local_rank,
        node_id=f"node:{node_index}",
        device=f"cuda:{local_rank}",
        ep_group_ranks=ep_group_ranks,
        ep_group_size=len(ep_group_ranks),
        dispatcher_class="Dispatcher",
        expert_placement_hash=placement_hash,
        model_revision_hash=model_revision_hash,
        dispatcher_hash=dispatcher_hash,
        ep_group_hash=ep_group_hash,
        request_table_hash=request_table_hash,
        run_id_digest=run_id_digest,
        step_id_digest=step_id_digest,
        microbatch_id_digest=microbatch_id_digest,
        phase=phase,
        per_peer_rows=rows,
        per_peer_bytes=tuple(v * 16 for v in rows),
        local_rows=rows[ep_group_ranks.index(rank)],
        remote_rows=sum(v for i, v in enumerate(rows) if i != ep_group_ranks.index(rank)),
        topology=topology,
        input_splits=rows if phase == "P0" else (0,) * len(ep_group_ranks),
        output_splits=rows if phase == "P1" else (0,) * len(ep_group_ranks),
        observation_digest=observation_digest,
        availability={},
    )


def make_context(
    *,
    ep_group_ranks: tuple[int, ...] = (0, 1),
    request_table_hash: str = "request",
    placement_hash: str = "placement",
    model_revision_hash: str = "model",
) -> PolicyContext:
    return PolicyContext(
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        layer_id="0",
        run_id_digest=stable_hash("run"),
        step_id_digest=stable_hash("step"),
        microbatch_id_digest=stable_hash("mb"),
        request_table_hash=request_table_hash,
        model_revision_hash=model_revision_hash,
        expert_placement_hash=placement_hash,
        ep_group_ranks=ep_group_ranks,
        ep_group_size=len(ep_group_ranks),
        ep_group_hash=stable_hash({"ep_group_ranks": list(ep_group_ranks)}),
        future_hint_mode="none",
        control_mode="default_continue",
    )


def make_phase_context(
    *,
    rank: int,
    phase: str,
    input_splits: tuple[int, int],
    output_splits: tuple[int, int],
    rows: int,
    hidden_dim: int = 4,
    p2_hint_mode: str = "none",
) -> PhaseReadyContext:
    hidden = torch.arange(rows * hidden_dim, dtype=torch.float16).reshape(rows, hidden_dim)
    packed_tensors = (hidden, hidden[:, :1].clone()) if phase == "P0" else (hidden,)
    return build_phase_ready_context(
        plan_key={"layer_id": "0", "phase": phase, "rank": rank},
        phase=phase,
        control_mode="sync_before_phase",
        forward_epoch=0,
        layer_id="0",
        layer_name="module.decoder.layers.0.mlp",
        global_rank=rank,
        local_rank=rank,
        ep_group_ranks=(0, 1),
        ep_group_root_rank=0,
        topology={"global_rank": rank, "local_rank": rank, "node_index": 0, "hostname_digest": "host-a", "device_index": rank, "ep_group_rank": rank},
        dispatcher_class="MoEAlltoAllTokenDispatcher",
        dispatcher_fingerprint={"dispatcher_class": "MoEAlltoAllTokenDispatcher"},
        expert_placement_hash="placement",
        input_splits=input_splits,
        output_splits=output_splits,
        packed_tensors=packed_tensors,
        release_state="ready",
        demand_known_at="router_ready",
        payload_exists=True,
        p2_hint=FutureDemandHint(hint_mode=p2_hint_mode, hint_digest=f"digest:{p2_hint_mode}", hint_source=p2_hint_mode),
    )


def make_phase_context_generic(
    *,
    rank: int,
    phase: str,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    ep_group_ranks: tuple[int, ...],
    hidden_dim: int = 4,
    p2_hint_mode: str = "none",
    p2_hint_digest: str | None = None,
    p2_hint_source: str | None = None,
) -> PhaseReadyContext:
    send_rows = sum(input_splits) if phase == "P0" else sum(output_splits)
    hidden = torch.arange(max(send_rows, 1) * hidden_dim, dtype=torch.float16).reshape(max(send_rows, 1), hidden_dim)[:send_rows]
    packed_tensors = (hidden, hidden[:, :1].clone()) if phase == "P0" else (hidden,)
    hint = FutureDemandHint(
        hint_mode=p2_hint_mode,
        hint_digest=p2_hint_digest or f"digest:{p2_hint_mode}:{rank}:{phase}",
        hint_source=p2_hint_source or p2_hint_mode,
    )
    return build_phase_ready_context(
        plan_key={"layer_id": "0", "phase": phase, "rank": rank},
        phase=phase,
        control_mode="sync_before_phase",
        forward_epoch=0,
        layer_id="0",
        layer_name="module.decoder.layers.0.mlp",
        global_rank=rank,
        local_rank=rank,
        ep_group_ranks=ep_group_ranks,
        ep_group_root_rank=ep_group_ranks[0],
        topology={"global_rank": rank, "local_rank": rank, "node_index": 0, "hostname_digest": "host-a", "device_index": rank, "ep_group_rank": ep_group_ranks.index(rank)},
        dispatcher_class="MoEAlltoAllTokenDispatcher",
        dispatcher_fingerprint={"dispatcher_class": "MoEAlltoAllTokenDispatcher"},
        expert_placement_hash="placement",
        input_splits=input_splits,
        output_splits=output_splits,
        packed_tensors=packed_tensors,
        release_state="ready",
        demand_known_at="router_ready",
        payload_exists=True,
        p2_hint=hint,
    )


def make_contexts_from_matrix(
    *,
    phase: str,
    matrix: tuple[tuple[int, ...], ...],
    p2_hint_mode: str = "none",
) -> tuple[PhaseReadyContext, ...]:
    ep_group_ranks = tuple(range(len(matrix)))
    contexts: list[PhaseReadyContext] = []
    for rank, row in enumerate(matrix):
        col = tuple(int(matrix[src][rank]) for src in range(len(matrix)))
        if phase == "P0":
            input_splits = tuple(int(v) for v in row)
            output_splits = col
        else:
            input_splits = col
            output_splits = tuple(int(v) for v in row)
        contexts.append(
            make_phase_context_generic(
                rank=rank,
                phase=phase,
                input_splits=input_splits,
                output_splits=output_splits,
                ep_group_ranks=ep_group_ranks,
                p2_hint_mode=p2_hint_mode,
            )
        )
    return tuple(contexts)


def with_p2_digest(context: PhaseReadyContext, *, digest: str, source: str = "synthetic") -> PhaseReadyContext:
    return replace(context, p2_hint=FutureDemandHint(hint_mode="deterministic_stub", hint_digest=digest, hint_source=source))
