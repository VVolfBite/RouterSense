from __future__ import annotations

import torch

from rs.core.contracts.execution import ActualPhaseContext
from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    FutureDemandHint,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    RuntimeIdentity,
    build_phase_ready_context,
)


def build_phase_contexts_from_matrix(
    *,
    phase: str,
    matrix: tuple[tuple[int, ...], ...],
    hidden_dim: int = 4,
) -> tuple[object, ...]:
    contexts: list[object] = []
    group_ranks = tuple(range(len(matrix)))
    for rank, row in enumerate(matrix):
        recv = tuple(int(matrix[src][rank]) for src in range(len(matrix)))
        send = tuple(int(v) for v in row)
        active_rows = max(sum(send if phase == "P0" else recv), 1)
        hidden = torch.arange(active_rows * hidden_dim, dtype=torch.float16).reshape(active_rows, hidden_dim)
        packed = (hidden, hidden[:, :1].clone()) if phase == "P0" else (hidden,)
        contexts.append(
            build_phase_ready_context(
                PhaseContextBuildRequest(
                    plan_key={"layer_id": "0", "phase": phase, "rank": rank},
                    runtime_identity=RuntimeIdentity(
                        run_id="paper-eval",
                        forward_epoch=0,
                        layer_id="0",
                        layer_name="decoder.layers.0.mlp",
                        global_rank=rank,
                        local_rank=rank,
                        ep_group_ranks=group_ranks,
                        ep_group_root_rank=0,
                    ),
                    topology={
                        "global_rank": rank,
                        "local_rank": rank,
                        "node_index": 0,
                        "hostname_digest": "paper-host",
                        "device_index": rank,
                        "ep_group_rank": rank,
                    },
                    dispatcher_snapshot=DispatcherSnapshot(
                        dispatcher_class="MoEAlltoAllTokenDispatcher",
                        dispatcher_fingerprint={"dispatcher_class": "MoEAlltoAllTokenDispatcher"},
                        expert_placement_hash="paper-placement",
                        input_splits=send if phase == "P0" else recv,
                        output_splits=recv if phase == "P0" else send,
                    ),
                    payload_contract=PhasePayloadContract(
                        phase=phase,
                        payload_roles=("hidden_states", "routing_probs") if phase == "P0" else ("hidden_states",),
                        atomic_submit=phase == "P0",
                    ),
                    packed_tensors=packed,
                    control_mode="sync_before_phase",
                    release_state="ready",
                    demand_known_at="router_ready",
                    payload_exists=True,
                    p2_hint=FutureDemandHint(
                        hint_mode="deterministic_stub",
                        hint_digest="paper-digest",
                        hint_source="paper",
                    ),
                )
            )
        )
    return tuple(contexts)


def actual_phase_context_from_ready_context(
    ready_context,
    *,
    phase: str,
    layer_id: str = "0",
) -> ActualPhaseContext:
    return ActualPhaseContext(
        layer_id=str(layer_id),
        phase=str(phase),
        world_size=len(tuple(ready_context.ep_group_ranks)),
        rank_space="global",
        layout_digest=str(ready_context.canonical_receive_layout_id),
        metadata={"phase_ready_context": ready_context.to_dict()},
    )
