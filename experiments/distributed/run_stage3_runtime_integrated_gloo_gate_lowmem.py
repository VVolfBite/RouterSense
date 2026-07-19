from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "outputs/distributed/run_stage3_runtime_integrated_gloo_gate_lowmem"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.execution.async_release_backend import validate_async_phase_preflight
from rs.runtime.online.megatron_ep.execution.transport_adapter import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.host import _maybe_create_dedicated_p2p_group
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime


@dataclass
class SyntheticDispatcher:
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    router_topk: int = 1
    tokens_per_expert: torch.Tensor | None = None

    def _maybe_dtoh_and_synchronize(self, stage: str, tensor: Any) -> Any:
        return tensor


def _log(rank: int, message: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / f"rank{rank}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")
    print(f"[rank{rank}] {message}", flush=True)


def _runtime(rank: int, *, policy_name: str, p2_hint_mode: str, online_p2_predictor: str) -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy=policy_name,
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode=p2_hint_mode,
            p2_hint_weight=1.0,
            observation_profile="execution",
            online_p2_predictor=online_p2_predictor,
            safe_projection_mode="disabled",
            executor_heartbeat_path="",
        ),
        rank=rank,
        local_rank=rank,
        run_id="runtime-gloo-lowmem",
        step_id="step",
        microbatch_id="mb0",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="localhost",
        ep_group_ranks=(0, 1),
        ep_group_root_global_rank=0,
    )


def _dispatch_matrix_for_layer(layer_index: int) -> tuple[tuple[int, ...], ...]:
    matrices = [
        ((1, 2), (3, 1)),
        ((2, 1), (1, 2)),
    ]
    return matrices[layer_index]


def _make_p0_payloads(*, rank: int, send_rows: int, hidden_dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.arange(max(send_rows, 1) * hidden_dim, dtype=torch.float32).reshape(max(send_rows, 1), hidden_dim)[:send_rows] + (1000.0 * rank)
    probs = torch.arange(max(send_rows, 1), dtype=torch.float32).reshape(max(send_rows, 1), 1)[:send_rows] + (100.0 * rank)
    return hidden, probs


def _make_p1_payload(*, rank: int, send_rows: int, hidden_dim: int = 4) -> torch.Tensor:
    return torch.arange(max(send_rows, 1) * hidden_dim, dtype=torch.float32).reshape(max(send_rows, 1), hidden_dim)[:send_rows] + (2000.0 * rank)


def _execute_phase_calls(
    *,
    adapter: MegatronPhaseTransportAdapter,
    dispatcher: SyntheticDispatcher,
    phase_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    outputs: list[torch.Tensor] = []
    active = adapter.current()
    phase = str(active.phase) if active is not None else "P0"
    if phase == "P1":
        output_split_sizes = dispatcher.input_splits
        input_split_sizes = dispatcher.output_splits
    else:
        output_split_sizes = dispatcher.output_splits
        input_split_sizes = dispatcher.input_splits
    for tensor in phase_inputs:
        outputs.append(
            adapter.maybe_execute(
                group=dist.group.WORLD,
                input_tensor=tensor,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                original_all_to_all=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected native fallback")),
                use_nccl_stream=False,
            )
        )
    return tuple(outputs)


def _assert_async_preflight_ready(*, adapter: MegatronPhaseTransportAdapter, tensor_role: str) -> None:
    active = adapter.current()
    if active is None:
        raise RuntimeError("missing active transport before preflight")
    result = validate_async_phase_preflight(
        context=active.context,
        plan=active.plan,
        tensor_role=tensor_role,
        process_group=adapter.p2p_group,
        rank_context={
            "global_rank": int(active.context.global_rank),
            "local_rank": int(active.context.local_rank),
        },
        mode=str((active.plan.metrics or {}).get("preflight_mode", "full")),
    )
    if not bool(result.ok) or not bool(result.all_ranks_ok):
        raise RuntimeError(
            f"gate_preflight_failed phase={active.phase} role={tensor_role} "
            f"reason={result.reason} local_send={result.local_send_count} "
            f"local_recv={result.local_recv_count} mode={result.preflight_mode}"
        )


def _worker(rank: int, init_file: str) -> None:
    stage = "init"
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
        )
        stage = "group"
        p2p_group, p2p_status = _maybe_create_dedicated_p2p_group(ep_group_ranks=(0, 1), local_rank=rank)
        adapter = MegatronPhaseTransportAdapter(
            dispatcher_class="SyntheticDispatcher",
            dispatcher_module_sha256=None,
            p2p_group=p2p_group,
        )
        runtime = _runtime(
            rank,
            policy_name="routersense_p0p1p2_hint",
            p2_hint_mode="none",
            online_p2_predictor="none",
        )
        runtime._runtime_state.merge(p2p_status)  # noqa: SLF001
        runtime.transport_adapter = adapter
        adapter.timeline_hook = lambda event, **detail: runtime._timeline(
            event,
            layer_name=str(runtime.current_transport().get("layer_name") if runtime.current_transport() else "unknown"),
            **detail,
        )
        runtime.begin_forward(forward_epoch=1)
        for layer_index in range(2):
            layer_name = f"model.layers.{layer_index}.mlp"
            stage = f"{layer_name}:P0"
            matrix = _dispatch_matrix_for_layer(layer_index)
            p0_row = tuple(int(v) for v in matrix[rank])
            p0_col = tuple(int(matrix[src][rank]) for src in range(2))
            p0_hidden, p0_probs = _make_p0_payloads(rank=rank, send_rows=sum(p0_row))
            p0_dispatcher = SyntheticDispatcher(input_splits=p0_row, output_splits=p0_col)
            runtime.before_token_dispatch(
                layer_name=layer_name,
                dispatcher=p0_dispatcher,
                packed_hidden_states=p0_hidden,
                packed_probs=p0_probs,
            )
            _assert_async_preflight_ready(adapter=adapter, tensor_role="hidden_states")
            _assert_async_preflight_ready(adapter=adapter, tensor_role="routing_probs")
            runtime.mark_token_dispatch_committed(layer_name=layer_name)
            _execute_phase_calls(adapter=adapter, dispatcher=p0_dispatcher, phase_inputs=(p0_hidden, p0_probs))
            runtime.after_token_dispatch(layer_name=layer_name)

            stage = f"{layer_name}:P1"
            p1_rows = tuple(tuple(int(matrix[src][dst]) for src in range(2)) for dst in range(2))
            p1_row = tuple(int(v) for v in p1_rows[rank])
            p1_col = tuple(int(p1_rows[src][rank]) for src in range(2))
            p1_hidden = _make_p1_payload(rank=rank, send_rows=sum(p1_row))
            p1_dispatcher = SyntheticDispatcher(input_splits=p1_col, output_splits=p1_row)
            runtime.before_token_combine(
                layer_name=layer_name,
                dispatcher=p1_dispatcher,
                packed_hidden_states=p1_hidden,
            )
            _assert_async_preflight_ready(adapter=adapter, tensor_role="hidden_states")
            _execute_phase_calls(adapter=adapter, dispatcher=p1_dispatcher, phase_inputs=(p1_hidden,))
            runtime.after_token_combine(layer_name=layer_name)

        summary = runtime.export_prepared_plan_summary()
        runtime.end_forward()
        payload = {
            "rank": rank,
            "planning_traffic_source": summary.get("planning_traffic_source", ""),
            "pre_transport_observation_valid": bool(summary.get("pre_transport_observation_valid", False)),
            "captured_before_transport": bool(summary.get("captured_before_transport", False)),
            "actual_p0_total_rows": int(summary.get("actual_p0_total_rows", 0) or 0),
            "actual_p0_full_row_matrix": summary.get("actual_p0_full_row_matrix", []),
            "actual_p0_matrix_unit": str(summary.get("actual_p0_matrix_unit", "")),
            "inferred_p1_row_matrix": summary.get("inferred_p1_row_matrix", []),
            "inferred_p1_matrix_unit": str(summary.get("inferred_p1_matrix_unit", "")),
            "p1_is_exact_transpose": bool(summary.get("p1_is_exact_transpose", False)),
            "stored_p1_plan_digest": str(summary.get("stored_p1_plan_digest", "")),
            "consumed_p1_plan_digest": str(summary.get("consumed_p1_plan_digest", "")),
            "stored_p1_logical_plan_digest": str(summary.get("stored_p1_logical_plan_digest", "")),
            "consumed_p1_logical_plan_digest": str(summary.get("consumed_p1_logical_plan_digest", "")),
            "stored_p1_compile_input_digest": str(summary.get("stored_p1_compile_input_digest", "")),
            "consumed_p1_compile_input_digest": str(summary.get("consumed_p1_compile_input_digest", "")),
            "compiler_id": str(summary.get("compiler_id", "")),
            "logical_plan_digest": str(summary.get("logical_plan_digest", "")),
            "compiled_plan_digest": str(summary.get("compiled_plan_digest", "")),
            "canonical_task_digest": str(summary.get("canonical_task_digest", "")),
            "canonical_task_count": int(summary.get("canonical_task_count", 0) or 0),
            "canonical_task_total_rows": int(summary.get("canonical_task_total_rows", 0) or 0),
            "compiler_shadow_status": str(summary.get("compiler_shadow_status", "")),
            "compiler_shadow_plan_hash_matches_legacy": bool(summary.get("compiler_shadow_plan_hash_matches_legacy", False)),
            "compiler_shadow_plan_hash": str(summary.get("compiler_shadow_plan_hash", "")),
            "compiler_shadow_missing_task_count": int(summary.get("compiler_shadow_missing_task_count", 0) or 0),
            "compiler_shadow_extra_task_count": int(summary.get("compiler_shadow_extra_task_count", 0) or 0),
            "compiler_shadow_execution_order_matches_legacy": bool(
                summary.get("compiler_shadow_execution_order_matches_legacy", False)
            ),
            "legacy_secondary_policy_invocation_count": int(summary.get("legacy_secondary_policy_invocation_count", 0) or 0),
            "p0_traffic_matrix_gather_count": int(summary.get("p0_traffic_matrix_gather_count", 0) or 0),
            "prediction_extra_collective_count": int(summary.get("prediction_extra_collective_count", 0) or 0),
            "p1_planning_collective_count": int(summary.get("p1_planning_collective_count", 0) or 0),
            "before_async_p2p_phase_count": int(summary.get("before_async_p2p_phase_count", 0) or 0),
            "after_async_p2p_phase_count": int(summary.get("after_async_p2p_phase_count", 0) or 0),
            "async_executor_invocation_count": int(adapter.async_executor_invocation_count),
            "batch_isend_irecv_call_count": int(adapter.batch_isend_irecv_call_count),
            "real_send_op_count": int(adapter.real_send_op_count),
            "real_recv_op_count": int(adapter.real_recv_op_count),
            "local_copy_task_count": int(adapter.local_copy_task_count),
            "local_copy_row_count": int(adapter.local_copy_row_count),
            "phase_sync_fallback_count": int(adapter.phase_sync_fallback_count),
        }
        (RUN_DIR / f"rank{rank}-summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        failure = {
            "rank": rank,
            "stage": stage,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (RUN_DIR / f"failure-rank{rank}.json").write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="rs_gloo_lowmem_") as tmp:
        init_file = str(Path(tmp) / "dist_init")
        ctx = mp.get_context("fork")
        processes: list[mp.Process] = []
        for rank in range(2):
            proc = ctx.Process(target=_worker, args=(rank, init_file))
            proc.start()
            processes.append(proc)
        exit_codes = []
        for proc in processes:
            proc.join()
            exit_codes.append(int(proc.exitcode or 0))
        if any(code != 0 for code in exit_codes):
            raise SystemExit(f"worker failure exit_codes={exit_codes}")

    rank_payloads = []
    for rank in range(2):
        path = RUN_DIR / f"rank{rank}-summary.json"
        rank_payloads.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {
        "passed": all(
            payload["actual_p0_total_rows"] > 0
            and payload["p1_is_exact_transpose"]
            and payload["batch_isend_irecv_call_count"] > 0
            and payload["phase_sync_fallback_count"] == 0
            and payload["stored_p1_plan_digest"] == payload["consumed_p1_plan_digest"]
            and payload["stored_p1_logical_plan_digest"] == payload["consumed_p1_logical_plan_digest"]
            and payload["stored_p1_compile_input_digest"] == payload["consumed_p1_compile_input_digest"]
            for payload in rank_payloads
        ),
        "ranks": rank_payloads,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not bool(summary["passed"]):
        raise SystemExit("lowmem runtime integrated gloo gate failed acceptance")


if __name__ == "__main__":
    main()
