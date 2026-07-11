from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.execution.transport_adapter import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.host import _maybe_create_dedicated_p2p_group
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.observation import digest_text


RUN_DIR = Path("outputs/distributed/run_stage1_runtime_integrated_gloo_gate")


@dataclass
class SyntheticDispatcher:
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    router_topk: int = 1
    tokens_per_expert: torch.Tensor | None = None

    def _maybe_dtoh_and_synchronize(self, stage: str, tensor: Any) -> Any:
        return tensor



def _log(rank: int, message: str) -> None:
    print(f"[rank{rank}] {message}", flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / f"rank{rank}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")



def _init() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank



def _runtime(rank: int, local_rank: int, *, policy_name: str, p2_hint_mode: str, online_p2_predictor: str) -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy=policy_name,
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode=p2_hint_mode,
            p2_hint_weight=1.0,
            observation_profile="execution",
            online_p2_predictor=online_p2_predictor,
            executor_heartbeat_path=str(RUN_DIR),
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



def _make_p0_payloads(*, rank: int, send_rows: int, hidden_dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.arange(max(send_rows, 1) * hidden_dim, dtype=torch.float32).reshape(max(send_rows, 1), hidden_dim)[:send_rows] + (1000.0 * rank)
    probs = torch.arange(max(send_rows, 1), dtype=torch.float32).reshape(max(send_rows, 1), 1)[:send_rows] + (100.0 * rank)
    return hidden, probs



def _make_p1_payload(*, rank: int, send_rows: int, hidden_dim: int = 4) -> torch.Tensor:
    return torch.arange(max(send_rows, 1) * hidden_dim, dtype=torch.float32).reshape(max(send_rows, 1), hidden_dim)[:send_rows] + (2000.0 * rank)



def _dispatch_matrix_for_layer(layer_index: int) -> tuple[tuple[int, ...], ...]:
    matrices = [
        ((1, 2), (3, 1)),
        ((2, 1), (1, 2)),
    ]
    return matrices[layer_index]



def _execute_phase_calls(
    *,
    adapter: MegatronPhaseTransportAdapter,
    dispatcher: SyntheticDispatcher,
    phase_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    outputs: list[torch.Tensor] = []
    for tensor in phase_inputs:
        outputs.append(
            adapter.maybe_execute(
                group=dist.group.WORLD,
                input_tensor=tensor,
                output_split_sizes=dispatcher.output_splits,
                input_split_sizes=dispatcher.input_splits,
                original_all_to_all=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected native fallback")),
                use_nccl_stream=False,
            )
        )
    return tuple(outputs)



def main() -> None:
    rank, world_size, local_rank = _init()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / f"startup-rank{rank}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"init rank={rank} world_size={world_size} local_rank={local_rank}\\n")
    if world_size != 2:
        raise SystemExit("run_stage1_runtime_integrated_gloo_gate.py requires world_size=2")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    p2p_group, p2p_status = _maybe_create_dedicated_p2p_group(ep_group_ranks=(0, 1), local_rank=local_rank)
    with (RUN_DIR / f"startup-rank{rank}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"p2p_group_created={p2p_group is not None} status={p2p_status}\\n")
    strategy_specs = [
        ("fifo_async_p2p", "bucketed_fifo", "none", "none", (1,), 2),
        ("greedy_async_p2p", "greedy_ready_set", "none", "none", (1,), 2),
        ("birkhoff_phase_local_async_p2p", "birkhoff_phase_local", "none", "none", (1,), 2),
        ("routersense_joint_zero_hint_async_p2p", "routersense_p0p1p2_hint", "none", "none", (1,), 2),
        ("routersense_joint_predicted_async_p2p", "routersense_p0p1p2_hint", "calibrated_artifact", "copy_current_dispatch", (1, 2), 2),
    ]
    strategy_summaries: list[dict[str, Any]] = []
    for strategy_name, policy_name, p2_hint_mode, online_p2_predictor, forward_epochs, layer_count in strategy_specs:
        _log(rank, f"start strategy={strategy_name} epochs={list(forward_epochs)} layers={layer_count}")
        adapter = MegatronPhaseTransportAdapter(
            dispatcher_class="SyntheticDispatcher",
            dispatcher_module_sha256=None,
            p2p_group=p2p_group,
        )
        runtime = _runtime(
            rank,
            local_rank,
            policy_name=policy_name,
            p2_hint_mode=p2_hint_mode,
            online_p2_predictor=online_p2_predictor,
        )
        runtime._prepared_plan_state.update(p2p_status)  # noqa: SLF001
        runtime.transport_adapter = adapter
        adapter.timeline_hook = lambda event, **detail: runtime._timeline(
            event,
            layer_name=str(runtime.current_transport().get("layer_name") if runtime.current_transport() else "unknown"),
            **detail,
        )
        last_prepared_summary: dict[str, Any] = {}
        for forward_epoch in forward_epochs:
            _log(rank, f"strategy={strategy_name} begin_forward epoch={forward_epoch}")
            runtime.begin_forward(forward_epoch=forward_epoch)
            for layer_index in range(layer_count):
                _log(rank, f"strategy={strategy_name} layer={layer_index} phase=P0")
                layer_name = f"model.layers.{layer_index}.mlp"
                p0_rows = _dispatch_matrix_for_layer(layer_index)
                p0_row = tuple(int(v) for v in p0_rows[rank])
                p0_col = tuple(int(p0_rows[src][rank]) for src in range(world_size))
                p0_hidden, p0_probs = _make_p0_payloads(rank=rank, send_rows=sum(p0_row))
                p0_dispatcher = SyntheticDispatcher(input_splits=p0_row, output_splits=p0_col)
                runtime.before_token_dispatch(
                    layer_name=layer_name,
                    dispatcher=p0_dispatcher,
                    packed_hidden_states=p0_hidden,
                    packed_probs=p0_probs,
                )
                runtime.mark_token_dispatch_committed(layer_name=layer_name)
                _execute_phase_calls(
                    adapter=adapter,
                    dispatcher=p0_dispatcher,
                    phase_inputs=(p0_hidden, p0_probs),
                )
                runtime.after_token_dispatch(layer_name=layer_name)

                _log(rank, f"strategy={strategy_name} layer={layer_index} phase=P1")
                p1_dispatcher = SyntheticDispatcher(input_splits=p0_row, output_splits=p0_col)
                p1_hidden = _make_p1_payload(rank=rank, send_rows=sum(p0_col))
                runtime.before_token_combine(
                    layer_name=layer_name,
                    dispatcher=p1_dispatcher,
                    packed_hidden_states=p1_hidden,
                )
                _execute_phase_calls(
                    adapter=adapter,
                    dispatcher=p1_dispatcher,
                    phase_inputs=(p1_hidden,),
                )
                runtime.after_token_combine(layer_name=layer_name)
                last_prepared_summary = runtime.export_prepared_plan_summary()
            runtime.end_forward()
            _log(rank, f"strategy={strategy_name} end_forward epoch={forward_epoch}")
        rows = adapter.export_results()
        result_rows = [row for row in rows if row.get("record_type") == "result_summary"]
        async_phase_rows = [row for row in rows if row.get("record_type") == "async_phase_summary"]
        plan_summary = runtime.export_prepared_plan_summary() if not last_prepared_summary else last_prepared_summary
        strategy_summaries.append(
            {
                "strategy": strategy_name,
                "policy_name": policy_name,
                "p2_hint_mode": p2_hint_mode,
                "online_p2_predictor": online_p2_predictor,
                "forward_epochs_tested": list(forward_epochs),
                "layer_count_tested": int(layer_count),
                "planning_traffic_source": plan_summary.get("planning_traffic_source", ""),
                "pre_transport_observation_valid": bool(plan_summary.get("pre_transport_observation_valid", False)),
                "captured_before_transport": bool(plan_summary.get("captured_before_transport", False)),
                "actual_p0_full_row_matrix": plan_summary.get("actual_p0_full_row_matrix", []),
                "actual_p0_total_rows": int(plan_summary.get("actual_p0_total_rows", 0) or 0),
                "inferred_p1_row_matrix": plan_summary.get("inferred_p1_row_matrix", []),
                "p1_is_exact_transpose": bool(plan_summary.get("p1_is_exact_transpose", False)),
                "stored_p1_plan_digest": str(plan_summary.get("stored_p1_plan_digest", "")),
                "consumed_p1_plan_digest": str(plan_summary.get("consumed_p1_plan_digest", "")),
                "p0_traffic_matrix_gather_count": int(plan_summary.get("p0_traffic_matrix_gather_count", 0) or 0),
                "prediction_extra_collective_count": int(plan_summary.get("prediction_extra_collective_count", 0) or 0),
                "p1_planning_collective_count": int(plan_summary.get("p1_planning_collective_count", 0) or 0),
                "async_executor_invocation_count": int(adapter.async_executor_invocation_count),
                "batch_isend_irecv_call_count": int(adapter.batch_isend_irecv_call_count),
                "real_send_op_count": int(adapter.real_send_op_count),
                "real_recv_op_count": int(adapter.real_recv_op_count),
                "local_copy_task_count": int(adapter.local_copy_task_count),
                "phase_sync_fallback_count": int(adapter.phase_sync_fallback_count),
                "before_async_p2p_phase_count": int(plan_summary.get("before_async_p2p_phase_count", 0) or 0),
                "after_async_p2p_phase_count": int(plan_summary.get("after_async_p2p_phase_count", 0) or 0),
                "result_summary_count": len(result_rows),
                "async_phase_summary_count": len(async_phase_rows),
            }
        )
        _log(rank, f"strategy={strategy_name} summary={strategy_summaries[-1]}")
    summary = {
        "runtime_integrated_gloo_passed": all(
            bool(item["pre_transport_observation_valid"])
            and str(item["planning_traffic_source"]) == "pre_transport_phase_ready_context"
            and int(item["actual_p0_total_rows"]) > 0
            and bool(item["p1_is_exact_transpose"])
            and str(item["stored_p1_plan_digest"]) == str(item["consumed_p1_plan_digest"])
            and bool(item["stored_p1_plan_digest"])
            and int(item["p0_traffic_matrix_gather_count"]) == 1
            and int(item["prediction_extra_collective_count"]) == 0
            and int(item["p1_planning_collective_count"]) == 0
            and int(item["batch_isend_irecv_call_count"]) > 0
            and int(item["real_send_op_count"]) > 0
            and int(item["real_recv_op_count"]) > 0
            and int(item["phase_sync_fallback_count"]) == 0
            for item in strategy_summaries
        ),
        "dedicated_p2p_group_initialized": bool(p2p_status.get("dedicated_p2p_group_initialized", False)),
        "strategies": strategy_summaries,
    }
    if rank == 0:
        (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (RUN_DIR / "summary.md").write_text(
            "\n".join(
                [
                    "# Stage1 Runtime-Integrated Gloo Gate",
                    "",
                    f"- runtime_integrated_gloo_passed: {str(summary['runtime_integrated_gloo_passed']).lower()}",
                    f"- strategy_count: {len(strategy_summaries)}",
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
