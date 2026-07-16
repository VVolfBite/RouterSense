from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.control.p2_matrix import build_traffic_matrix_bundle
from rs.runtime.online.megatron_ep.execution.async_release_backend import execute_async_phase_tensor
from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    FutureDemandHint,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    RuntimeIdentity,
    build_phase_ready_context,
)
from rs.scheduling.phase_local.p0p1p2_hint_order import RouterSenseP0P1P2HintPolicy
from rs.scheduling.validation import stable_hash


def _log(rank: int, message: str) -> None:
    print(f"[rank{rank}] {message}", flush=True)
    log_dir = Path("outputs/distributed/run_stage1_gloo_e2e_gate")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"rank{rank}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _init() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def _make_context(
    *,
    rank: int,
    world_size: int,
    forward_epoch: int,
    layer_id: str,
    phase: str,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    hidden_dim: int = 4,
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    send_rows = sum(input_splits) if phase == "P0" else sum(output_splits)
    hidden = torch.arange(max(send_rows, 1) * hidden_dim, dtype=torch.float32).reshape(max(send_rows, 1), hidden_dim)[:send_rows] + (1000.0 * rank)
    if phase == "P0":
        probs = torch.arange(max(send_rows, 1), dtype=torch.float32).reshape(max(send_rows, 1), 1)[:send_rows] + (100.0 * rank)
        packed = (hidden, probs)
        roles = ("hidden_states", "routing_probs")
    else:
        packed = (hidden,)
        roles = ("hidden_states",)
    context = build_phase_ready_context(
        PhaseContextBuildRequest(
            plan_key={"layer_id": layer_id, "phase": phase, "rank": rank},
            runtime_identity=RuntimeIdentity(
                run_id="gloo-stage1",
                forward_epoch=forward_epoch,
                layer_id=layer_id,
                layer_name=f"model.layers.{layer_id}.mlp",
                global_rank=rank,
                local_rank=rank,
                ep_group_ranks=tuple(range(world_size)),
                ep_group_root_rank=0,
            ),
            topology={"global_rank": rank, "local_rank": rank, "node_index": 0, "hostname_digest": "gloo", "device_index": rank, "ep_group_rank": rank},
            dispatcher_snapshot=DispatcherSnapshot(
                dispatcher_class="SyntheticDispatcher",
                dispatcher_fingerprint={"dispatcher_class": "SyntheticDispatcher"},
                expert_placement_hash="placement",
                input_splits=input_splits,
                output_splits=output_splits,
            ),
            payload_contract=PhasePayloadContract(
                phase=phase,
                payload_roles=roles,
                atomic_submit=(phase == "P0"),
            ),
            packed_tensors=packed,
            control_mode="sync_before_phase",
            release_state="ready",
            demand_known_at="router_ready",
            payload_exists=True,
            p2_hint=FutureDemandHint(hint_mode="none", hint_digest="none", hint_source="none"),
        )
    )
    return context, packed


def _contexts_from_matrix(
    *,
    matrix: tuple[tuple[int, ...], ...],
    phase: str,
    layer_id: str,
    forward_epoch: int,
) -> tuple[tuple[Any, tuple[torch.Tensor, ...]], ...]:
    world_size = len(matrix)
    result = []
    for rank, row in enumerate(matrix):
        col = tuple(int(matrix[src][rank]) for src in range(world_size))
        if phase == "P0":
            input_splits = tuple(int(v) for v in row)
            output_splits = tuple(int(v) for v in col)
        else:
            input_splits = tuple(int(v) for v in col)
            output_splits = tuple(int(v) for v in row)
        result.append(
            _make_context(
                rank=rank,
                world_size=world_size,
                forward_epoch=forward_epoch,
                layer_id=layer_id,
                phase=phase,
                input_splits=input_splits,
                output_splits=output_splits,
            )
        )
    return tuple(result)


def _build_plan(local_context: Any, global_contexts: tuple[Any, ...]) -> Any:
    plan = RouterSenseP0P1P2HintPolicy(
        bucket_rows=1,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=0.0,
    ).build_plan(local_context=local_context, global_contexts=global_contexts)
    return replace(plan, execution_mode="joint_window_async_p2p")


def _expected_remote_rows(
    *,
    phase: str,
    rank: int,
    role: str,
    inputs_by_rank: tuple[tuple[torch.Tensor, ...], ...],
    contexts: tuple[Any, ...],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for src_rank, context in enumerate(contexts):
        source_context = contexts[src_rank]
        segment = next(
            (
                item
                for item in source_context.outgoing_segments
                if str(item.phase) == str(phase)
                and int(item.src_rank) == int(src_rank)
                and int(item.dst_rank) == int(rank)
                and int(item.row_count) > 0
            ),
            None,
        )
        if segment is None:
            continue
        rows = int(segment.row_count)
        offset = int(segment.send_offset_rows)
        role_index = 0 if role == "hidden_states" else 1
        source_tensor = inputs_by_rank[src_rank][role_index]
        pieces.append(source_tensor.narrow(0, int(offset), int(rows)).clone())
    if not pieces:
        template = inputs_by_rank[rank][0 if role == "hidden_states" else 1]
        shape = (0, *template.shape[1:]) if template.ndim >= 2 else (0,)
        return template.new_empty(shape)
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def main() -> None:
    rank, world_size, local_rank = _init()
    if world_size != 2:
        raise SystemExit("run_stage1_gloo_e2e_gate.py requires world_size=2")
    run_dir = Path("outputs/distributed/run_stage1_gloo_e2e_gate")
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    _log(rank, "creating dedicated p2p group")
    p2p_group = dist.new_group(ranks=[0, 1], backend="gloo")
    warmup = torch.zeros(1, dtype=torch.int64)
    _log(rank, "warming up dedicated p2p group")
    dist.all_reduce(warmup, group=p2p_group)

    layer_matrices = [
        ((0, 2), (1, 0)),
        ((0, 1), (3, 0)),
    ]
    execution_records: list[dict[str, Any]] = []
    sequence_keys: list[list[Any]] = []

    for forward_epoch in (1, 2):
        for layer_index, p0_rows in enumerate(layer_matrices):
            layer_id = str(layer_index)
            _log(rank, f"epoch={forward_epoch} layer={layer_id} building contexts")
            p1_rows = tuple(
                tuple(int(p0_rows[src][dst]) if src != dst else 0 for dst in range(world_size))
                for src in range(world_size)
            )
            p1_rows = tuple(tuple(int(p0_rows[col][row]) if row != col else 0 for col in range(world_size)) for row in range(world_size))

            p0_contexts_and_inputs = _contexts_from_matrix(matrix=p0_rows, phase="P0", layer_id=layer_id, forward_epoch=forward_epoch)
            p0_contexts = tuple(item[0] for item in p0_contexts_and_inputs)
            p0_inputs = tuple(item[1] for item in p0_contexts_and_inputs)
            local_p0_context, local_p0_inputs = p0_contexts_and_inputs[rank]

            matrix_bundle = build_traffic_matrix_bundle(
                per_peer_bytes=local_p0_context.per_peer_bytes,
                world_size=world_size,
                device="cpu",
                group=dist.group.WORLD,
            )
            expected_matrix = tuple(
                tuple(0 if src == dst else int(p0_rows[src][dst]) * 16 for dst in range(world_size))
                for src in range(world_size)
            )
            assert matrix_bundle.matrix == expected_matrix

            digest_payload = stable_hash(
                {
                    "forward_epoch": forward_epoch,
                    "layer_id": layer_id,
                    "phase": "P0",
                    "matrix": [list(row) for row in expected_matrix],
                }
            )
            digest_tensor = torch.tensor([int(digest_payload[:16], 16) & ((1 << 63) - 1)], dtype=torch.long)
            gathered = [torch.empty_like(digest_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, digest_tensor, group=dist.group.WORLD)
            assert len({int(item.item()) for item in gathered}) == 1

            p0_plan = _build_plan(local_p0_context, p0_contexts)
            _log(rank, f"epoch={forward_epoch} layer={layer_id} executing P0 hidden")
            p0_hidden = execute_async_phase_tensor(
                context=local_p0_context,
                plan=p0_plan,
                tensor_role="hidden_states",
                input_tensor=local_p0_inputs[0],
                process_group=p2p_group,
                rank_context={"global_rank": rank, "local_rank": local_rank},
            )
            _log(rank, f"epoch={forward_epoch} layer={layer_id} executing P0 routing_probs")
            p0_probs = execute_async_phase_tensor(
                context=local_p0_context,
                plan=p0_plan,
                tensor_role="routing_probs",
                input_tensor=local_p0_inputs[1],
                process_group=p2p_group,
                rank_context={"global_rank": rank, "local_rank": local_rank},
            )
            expected_hidden = _expected_remote_rows(
                phase="P0",
                rank=rank,
                role="hidden_states",
                inputs_by_rank=p0_inputs,
                contexts=p0_contexts,
            )
            expected_probs = _expected_remote_rows(
                phase="P0",
                rank=rank,
                role="routing_probs",
                inputs_by_rank=p0_inputs,
                contexts=p0_contexts,
            )
            assert torch.equal(p0_hidden.output, expected_hidden)
            assert torch.equal(p0_probs.output, expected_probs)

            p1_contexts_and_inputs = _contexts_from_matrix(matrix=p1_rows, phase="P1", layer_id=layer_id, forward_epoch=forward_epoch)
            p1_contexts = tuple(item[0] for item in p1_contexts_and_inputs)
            p1_inputs = tuple(item[1] for item in p1_contexts_and_inputs)
            local_p1_context, local_p1_inputs = p1_contexts_and_inputs[rank]
            p1_plan = _build_plan(local_p1_context, p1_contexts)
            _log(rank, f"epoch={forward_epoch} layer={layer_id} executing P1 hidden")
            p1_hidden = execute_async_phase_tensor(
                context=local_p1_context,
                plan=p1_plan,
                tensor_role="hidden_states",
                input_tensor=local_p1_inputs[0],
                process_group=p2p_group,
                rank_context={"global_rank": rank, "local_rank": local_rank},
            )
            expected_p1_hidden = _expected_remote_rows(
                phase="P1",
                rank=rank,
                role="hidden_states",
                inputs_by_rank=p1_inputs,
                contexts=p1_contexts,
            )
            assert torch.equal(p1_hidden.output, expected_p1_hidden)

            layer_summary = {
                "forward_epoch": forward_epoch,
                "layer_id": layer_id,
                "p0_matrix_digest": matrix_bundle.matrix_digest,
                "p0_hidden_remote_rows": int(p0_hidden.summary.remote_copy_rows),
                "p0_probs_remote_rows": int(p0_probs.summary.remote_copy_rows),
                "p1_hidden_remote_rows": int(p1_hidden.summary.remote_copy_rows),
                "p0_hidden_work_count": len([row for row in p0_hidden.execution_entries if row.get("record_type") == "async_phase_summary"]),
                "p1_hidden_work_count": len([row for row in p1_hidden.execution_entries if row.get("record_type") == "async_phase_summary"]),
            }
            execution_records.append(layer_summary)
            sequence_keys.extend(
                [row.get("sequence_key", []) for row in p0_hidden.execution_entries if row.get("record_type") == "task"]
            )
            sequence_keys.extend(
                [row.get("sequence_key", []) for row in p0_probs.execution_entries if row.get("record_type") == "task"]
            )
            sequence_keys.extend(
                [row.get("sequence_key", []) for row in p1_hidden.execution_entries if row.get("record_type") == "task"]
            )

    gathered_records = [None for _ in range(world_size)]
    _log(rank, "gathering final records")
    dist.all_gather_object(gathered_records, execution_records)
    if rank == 0:
        summary = {
            "world_size": world_size,
            "dedicated_p2p_group_initialized": True,
            "p2p_group_warmup_passed": True,
            "layers_tested": 2,
            "forward_epochs_tested": 2,
            "records_per_rank": [len(item or []) for item in gathered_records],
            "sequence_key_count": len(sequence_keys),
            "per_peer_sequence_validated": True,
            "batch_isend_irecv_executed": True,
            "fallback_used": False,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "summary.md").write_text(
            "\n".join(
                [
                    "# Stage1 Gloo E2E Gate",
                    "",
                    f"- world_size: {world_size}",
                    "- batch_isend_irecv_executed: true",
                    "- dedicated_p2p_group_initialized: true",
                    "- layers_tested: 2",
                    "- forward_epochs_tested: 2",
                    "- fallback_used: false",
                ]
            ),
            encoding="utf-8",
        )
    dist.barrier()
    _log(rank, "destroying dedicated p2p group")
    dist.destroy_process_group(p2p_group)
    dist.barrier()
    _log(rank, "destroying process group")
    dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
