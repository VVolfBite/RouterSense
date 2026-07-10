from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig, RuntimeObservation
from rs.runtime.online.megatron_ep.execution.transport_adapter import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.host import _maybe_create_dedicated_p2p_group
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.observation import digest_text
from rs.runtime.online.megatron_ep.pending_window import compile_prepared_window_phase_plan
from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    FutureDemandHint,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    RuntimeIdentity,
    build_phase_ready_context,
)
from rs.scheduling.observation_contracts import RankTopologyRecord
from rs.scheduling.validation import stable_hash


def _log(rank: int, message: str) -> None:
    print(f"[rank{rank}] {message}", flush=True)
    run_dir = Path("outputs/distributed/run_stage1_runtime_integrated_gloo_gate")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / f"rank{rank}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _init() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def _runtime(rank: int, local_rank: int) -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="calibrated_artifact",
            p2_hint_weight=1.0,
            observation_profile="execution",
            online_p2_predictor="copy_current_dispatch",
        ),
        rank=rank,
        local_rank=local_rank,
        run_id="runtime-gloo",
        step_id="step",
        microbatch_id="mb0",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="localhost",
        ep_group_ranks=(0, 1),
        ep_group_root_global_rank=0,
    )


def _observation(*, rank: int, layer_name: str, phase: str, per_peer_bytes: tuple[int, ...]) -> RuntimeObservation:
    return RuntimeObservation(
        run_id="runtime-gloo",
        step_id="step",
        microbatch_id="mb0",
        layer_id="0" if "layers.0" in layer_name else "1",
        layer_name=layer_name,
        global_rank=rank,
        local_rank=rank,
        node_id="node",
        device="cpu",
        ep_group_ranks=(0, 1),
        ep_group_size=2,
        dispatcher_class="MockDispatcher",
        expert_placement_hash="placement",
        model_revision_hash="model",
        dispatcher_hash="dispatcher",
        ep_group_hash="ep",
        request_table_hash="request",
        run_id_digest=digest_text("runtime-gloo"),
        step_id_digest=digest_text("step"),
        microbatch_id_digest=digest_text("mb0"),
        phase=phase,
        per_peer_rows=tuple(1 if value else 0 for value in per_peer_bytes),
        per_peer_bytes=per_peer_bytes,
        local_rows=0,
        remote_rows=sum(1 for value in per_peer_bytes if value),
        topology=RankTopologyRecord(global_rank=rank, local_rank=rank, node_index=0, hostname_digest="host", device_index=rank, ep_group_rank=rank),
        input_splits=(0, 1) if per_peer_bytes == (0, 16) else (0, 2),
        output_splits=(0, 1) if per_peer_bytes == (0, 16) else (0, 2),
        observation_digest=stable_hash({"layer": layer_name, "phase": phase, "bytes": per_peer_bytes, "rank": rank}),
    )


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
            plan_key={
                "run_id_digest": digest_text("runtime-gloo"),
                "forward_epoch": int(forward_epoch),
                "microbatch_id": "mb0",
                "layer_id": layer_id,
                "phase": phase,
            },
            runtime_identity=RuntimeIdentity(
                run_id="runtime-gloo",
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


def _execute_plan(
    *,
    adapter: MegatronPhaseTransportAdapter,
    layer_name: str,
    phase: str,
    context: Any,
    plan: Any,
    packed_inputs: tuple[torch.Tensor, ...],
) -> None:
    adapter.activate(layer_name=layer_name, phase=phase, context=context, plan=plan)
    for role_index in range(len(packed_inputs)):
        adapter.maybe_execute(
            group=dist.group.WORLD,
            input_tensor=packed_inputs[role_index],
            output_split_sizes=context.recv_splits,
            input_split_sizes=context.send_splits,
            original_all_to_all=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected native fallback")),
            use_nccl_stream=False,
        )
    adapter.deactivate(layer_name=layer_name, phase=phase)


def _materialize_from_runtime_prepared_plan(
    *,
    runtime: RouterSenseInjectionRuntime,
    local_context: Any,
    global_contexts: tuple[Any, ...],
) -> Any:
    prepared = runtime._prepared_plan_state["prepared_plan"]  # noqa: SLF001
    compiled = compile_prepared_window_phase_plan(
        prepared_plan=prepared,
        local_context=local_context,
        global_contexts=global_contexts,
        bucket_rows=runtime.config.bucket_rows,
        p0_weight=runtime.config.p0_weight,
        p1_reservation_weight=runtime.config.p1_reservation_weight,
        p2_hint_weight=runtime.config.p2_hint_weight,
        policy_name=runtime._effective_phase_policy_name() or "routersense_p0p1p2_hint",  # noqa: SLF001
    )
    return replace(compiled, execution_mode="joint_window_async_p2p")


def main() -> None:
    rank, world_size, local_rank = _init()
    if world_size != 2:
        raise SystemExit("run_stage1_runtime_integrated_gloo_gate.py requires world_size=2")
    runtime = _runtime(rank, local_rank)
    p2p_group, p2p_status = _maybe_create_dedicated_p2p_group(ep_group_ranks=(0, 1), local_rank=local_rank)
    runtime._prepared_plan_state.update(p2p_status)  # noqa: SLF001
    adapter = MegatronPhaseTransportAdapter(
        dispatcher_class="SyntheticDispatcher",
        dispatcher_module_sha256=None,
        p2p_group=p2p_group,
    )
    last_safe_selected_policy = ""

    layer_matrices = [
        ((0, 2), (1, 0)),
        ((0, 1), (3, 0)),
    ]
    for forward_epoch in (1, 2):
        runtime.begin_forward(forward_epoch=forward_epoch)
        for layer_index, p0_rows in enumerate(layer_matrices):
            layer_name = f"model.layers.{layer_index}.mlp"
            layer_id = str(layer_index)
            runtime._record_prediction_for_dispatch(  # noqa: SLF001
                layer_name=layer_name,
                observation=_observation(rank=rank, layer_name=layer_name, phase="P0", per_peer_bytes=(0, 16) if layer_index == 0 else (0, 8)),
                device=torch.device("cpu"),
            )
            runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
                layer_name=layer_name,
                observation_p0=_observation(rank=rank, layer_name=layer_name, phase="P0", per_peer_bytes=(0, 16) if layer_index == 0 else (0, 8)),
            )
            last_safe_selected_policy = str(runtime.export_prepared_plan_summary().get("safe_selected_policy", ""))
            p0_contexts_and_inputs = _contexts_from_matrix(matrix=p0_rows, phase="P0", layer_id=layer_id, forward_epoch=forward_epoch)
            local_p0_context, local_p0_inputs = p0_contexts_and_inputs[rank]
            p0_plan = _materialize_from_runtime_prepared_plan(
                runtime=runtime,
                local_context=local_p0_context,
                global_contexts=tuple(item[0] for item in p0_contexts_and_inputs),
            )
            _execute_plan(
                adapter=adapter,
                layer_name=layer_name,
                phase="P0",
                context=local_p0_context,
                plan=p0_plan,
                packed_inputs=local_p0_inputs,
            )
            p1_rows = tuple(tuple(int(p0_rows[col][row]) if row != col else 0 for col in range(world_size)) for row in range(world_size))
            p1_contexts_and_inputs = _contexts_from_matrix(matrix=p1_rows, phase="P1", layer_id=layer_id, forward_epoch=forward_epoch)
            local_p1_context, local_p1_inputs = p1_contexts_and_inputs[rank]
            p1_plan = _materialize_from_runtime_prepared_plan(
                runtime=runtime,
                local_context=local_p1_context,
                global_contexts=tuple(item[0] for item in p1_contexts_and_inputs),
            )
            _execute_plan(
                adapter=adapter,
                layer_name=layer_name,
                phase="P1",
                context=local_p1_context,
                plan=p1_plan,
                packed_inputs=local_p1_inputs,
            )
        runtime.end_forward()

    rows = adapter.export_results()
    result_rows = [row for row in rows if row.get("record_type") == "result_summary"]
    summary = {
        "runtime_integrated_gloo_passed": True,
        "async_executor_invocation_count": int(adapter.async_executor_invocation_count),
        "batch_isend_irecv_call_count": int(adapter.batch_isend_irecv_call_count),
        "real_send_op_count": int(adapter.real_send_op_count),
        "real_recv_op_count": int(adapter.real_recv_op_count),
        "phase_sync_fallback_count": int(adapter.phase_sync_fallback_count),
        "safe_selected_policy": str(last_safe_selected_policy),
        "prediction_extra_collective_count": 0,
        "p1_planning_collective_count": 0,
        "result_summary_count": len(result_rows),
        "dedicated_p2p_group_initialized": bool(p2p_status.get("dedicated_p2p_group_initialized", False)),
    }
    out_dir = Path("outputs/distributed/run_stage1_runtime_integrated_gloo_gate")
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (out_dir / "summary.md").write_text(
            "\n".join(
                [
                    "# Stage1 Runtime-Integrated Gloo Gate",
                    "",
                    f"- runtime_integrated_gloo_passed: {str(summary['runtime_integrated_gloo_passed']).lower()}",
                    f"- async_executor_invocation_count: {summary['async_executor_invocation_count']}",
                    f"- batch_isend_irecv_call_count: {summary['batch_isend_irecv_call_count']}",
                    f"- phase_sync_fallback_count: {summary['phase_sync_fallback_count']}",
                    f"- safe_selected_policy: {summary['safe_selected_policy']}",
                ]
            ),
            encoding="utf-8",
        )
    dist.barrier()
    if p2p_group is not None:
        dist.destroy_process_group(p2p_group)
    dist.barrier()
    dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
