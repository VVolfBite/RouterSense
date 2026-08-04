#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distributed._gpu_runner_common import available_cuda_count, copy_config, run_subprocess, write_json
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.execution.transport_adapter import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.target_planning import TargetLayerPlanningRequest
from rs.runtime.online.megatron_ep.target_planning.planner_service import TargetLayerPlannerMetrics
from rs.scheduling.contracts import LogicalSchedulePlan, LogicalWave
from rs.scheduling.unified_interface import PolicyOptions


class SyntheticDispatcher:
    def __init__(self, input_splits: tuple[int, ...], output_splits: tuple[int, ...]) -> None:
        self.input_splits = input_splits
        self.output_splits = output_splits
        self.router_topk = 1
        self.tokens_per_expert = None

    def _maybe_dtoh_and_synchronize(self, stage: str, tensor: Any) -> Any:
        return tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the target-plan lifecycle smoke on real 4GPU NCCL transport.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--selected-layers", default="0,1")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _runtime(rank: int, local_rank: int, world_size: int, *, safe_projection_mode: str = "disabled") -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="none",
            p2_hint_weight=1.0,
            online_p2_predictor="copy_current_dispatch",
            observation_profile="execution",
            invariant_mode="evaluation_strict",
            bucket_mode="dynamic_current",
            bucket_rows=0,
            safe_projection_mode=safe_projection_mode,
            residual_weight=0.75,
            barrier_weight=1.75,
            age_weight=0.15,
            prediction_weight=0.35,
            executor_heartbeat_path="",
        ),
        rank=rank,
        local_rank=local_rank,
        run_id="gpu-target-lifecycle",
        step_id="step",
        microbatch_id="mb0",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="localhost",
        ep_group_ranks=tuple(range(world_size)),
        ep_group_root_global_rank=0,
    )


def _payload(rank: int, rows: int, *, width: int = 4, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.arange(max(rows, 1) * width, dtype=torch.float32, device=device).reshape(max(rows, 1), width)[:rows] + (1000 * rank)
    probs = torch.arange(max(rows, 1), dtype=torch.float32, device=device).reshape(max(rows, 1), 1)[:rows] + (100 * rank)
    return hidden, probs


def _async_summary(results: list[dict[str, Any]], *, tensor_role: str) -> dict[str, Any]:
    for row in results:
        if str(row.get("record_type", "")) != "async_phase_summary":
            continue
        if str(row.get("tensor_role", "")) == str(tensor_role):
            return row
    return {}


def _build_runtime(rank: int, local_rank: int, world_size: int, *, safe_projection_mode: str) -> tuple[RouterSenseInjectionRuntime, MegatronPhaseTransportAdapter]:
    runtime = _runtime(rank, local_rank, world_size, safe_projection_mode=safe_projection_mode)
    adapter = MegatronPhaseTransportAdapter(
        dispatcher_class="SyntheticDispatcher",
        dispatcher_module_sha256=None,
        p2p_group=dist.group.WORLD,
    )
    runtime.transport_adapter = adapter
    adapter.timeline_hook = lambda event, **detail: runtime._timeline(  # noqa: SLF001
        event,
        layer_name=str(runtime.current_transport().get("layer_name") if runtime.current_transport() else "unknown"),
        **detail,
    )
    runtime.begin_forward(forward_epoch=1)
    return runtime, adapter


def _build_target_plan(
    runtime: RouterSenseInjectionRuntime,
    *,
    source_layer: str,
    target_layer: str,
    matrix: tuple[tuple[int, ...], ...],
    safe_projection_mode: str,
):
    service = runtime.target_planner_service
    if service is None:
        raise RuntimeError("missing target planner service")
    raw_policy = "U_barrier_criticality_global_matching"
    paired_policy = "B_barrier_criticality_core_independent" if safe_projection_mode == "host_select" else ""
    bundle, plan = service._build_target_plan(  # noqa: SLF001
        request=TargetLayerPlanningRequest(
            run_id=str(runtime.run_id),
            forward_epoch=int(runtime._forward_epoch),  # noqa: SLF001
            microbatch_id=str(runtime.microbatch_id),
            source_layer_id=str(source_layer),
            target_layer_id=str(target_layer),
            current_p0_rows=matrix,
            previous_p0_rows=matrix,
            predictor_name="copy_current_dispatch",
            policy_id=raw_policy,
            raw_u_policy_id=raw_policy,
            paired_b_policy_id=paired_policy,
            safe_projection_mode=safe_projection_mode,
            group_size=len(matrix),
            bucket_rows=0,
            policy_options=PolicyOptions(
                p0_weight=1.0,
                p1_weight=1.0,
                p2_hint_weight=1.0,
                residual_weight=0.75,
                barrier_weight=1.75,
                age_weight=0.15,
                prediction_weight=0.35,
            ),
            topology_digest=f"topo:{len(matrix)}",
            bucket_contract_digest="dynamic_current",
        ),
        metrics=TargetLayerPlannerMetrics(),
    )
    return bundle, plan


def _publish_plan(runtime: RouterSenseInjectionRuntime, *, key: Any, plan: Any) -> str | None:
    service = runtime.target_planner_service
    if service is None:
        raise RuntimeError("missing target planner service")
    try:
        service.publish_agreed_plan(key=key, plan=plan)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _execute_dispatch(
    *,
    runtime: RouterSenseInjectionRuntime,
    adapter: MegatronPhaseTransportAdapter,
    layer_name: str,
    matrix: tuple[tuple[int, ...], ...],
    device: torch.device,
    after_before_dispatch: Any | None = None,
) -> dict[str, Any]:
    rank = int(runtime.rank)
    row = tuple(int(v) for v in matrix[rank])
    col = tuple(int(matrix[src][rank]) for src in range(len(matrix)))
    hidden, probs = _payload(rank, sum(row), device=device)
    dispatcher = SyntheticDispatcher(input_splits=row, output_splits=col)
    runtime.before_token_dispatch(
        layer_name=layer_name,
        dispatcher=dispatcher,
        packed_hidden_states=hidden,
        packed_probs=probs,
    )
    if callable(after_before_dispatch):
        after_before_dispatch()
    runtime.mark_token_dispatch_committed(layer_name=layer_name)
    outputs = []
    for tensor in (hidden, probs):
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
    runtime.after_token_dispatch(layer_name=layer_name)
    return {
        "outputs": [tuple(int(v) for v in out.shape) for out in outputs],
        "summary": runtime.export_prepared_plan_summary(),
        "transport": adapter.export_results(),
    }


def _run_cases(rank: int, local_rank: int, world_size: int) -> dict[str, Any]:
    device = torch.device(f"cuda:{local_rank}")
    layer0 = "model.layers.0.mlp"
    layer1 = "model.layers.1.mlp"
    bootstrap_matrix = (
        (0, 4, 0, 2),
        (1, 0, 3, 0),
        (0, 2, 0, 5),
        (4, 0, 1, 0),
    )

    ready_runtime, ready_adapter = _build_runtime(rank, local_rank, world_size, safe_projection_mode="disabled")
    _execute_dispatch(runtime=ready_runtime, adapter=ready_adapter, layer_name=layer0, matrix=bootstrap_matrix, device=device)
    ready_key = ready_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001
    dist.barrier()
    ready_case = _execute_dispatch(runtime=ready_runtime, adapter=ready_adapter, layer_name=layer1, matrix=bootstrap_matrix, device=device)
    ready_terminal = ready_runtime.target_plan_store.get_terminal_record(ready_key)  # type: ignore[union-attr]

    late_runtime, late_adapter = _build_runtime(rank, local_rank, world_size, safe_projection_mode="disabled")
    _, late_plan = _build_target_plan(late_runtime, source_layer="0", target_layer="1", matrix=bootstrap_matrix, safe_projection_mode="disabled")
    late_plan = replace(
        late_plan,
        logical_plan=LogicalSchedulePlan(
            policy_name=str(late_plan.logical_plan.policy_name),
            waves=tuple(
                LogicalWave(
                    wave_id=int(w.wave_id),
                    flows=tuple(reversed(w.flows)),
                    duration=float(w.duration),
                )
                for w in late_plan.logical_plan.waves
            ),
            diagnostics=dict(late_plan.logical_plan.diagnostics or {}),
        ),
    )
    late_key = late_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001
    published_late = {"done": False}

    def _late_hook(release_epoch: int, _snapshot: dict[str, Any]) -> None:
        if release_epoch != 1 or published_late["done"]:
            return
        dist.barrier()
        publish_error = _publish_plan(late_runtime, key=late_key, plan=late_plan)
        if publish_error is not None:
            raise RuntimeError(publish_error)
        published_late["done"] = True

    def _late_setup() -> None:
        setattr(late_adapter, "on_release_batch_completed", _late_hook)

    late_case = _execute_dispatch(
        runtime=late_runtime,
        adapter=late_adapter,
        layer_name=layer1,
        matrix=bootstrap_matrix,
        device=device,
        after_before_dispatch=_late_setup,
    )
    late_terminal = late_runtime.target_plan_store.get_terminal_record(late_key)  # type: ignore[union-attr]

    too_late_runtime, too_late_adapter = _build_runtime(rank, local_rank, world_size, safe_projection_mode="disabled")
    _, too_late_plan = _build_target_plan(too_late_runtime, source_layer="0", target_layer="1", matrix=bootstrap_matrix, safe_projection_mode="disabled")
    too_late_key = too_late_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001
    too_late_case = _execute_dispatch(runtime=too_late_runtime, adapter=too_late_adapter, layer_name=layer1, matrix=bootstrap_matrix, device=device)
    dist.barrier()
    too_late_publish_error = _publish_plan(too_late_runtime, key=too_late_key, plan=too_late_plan)
    too_late_terminal = too_late_runtime.target_plan_store.get_terminal_record(too_late_key)  # type: ignore[union-attr]

    safe_ready_runtime, safe_ready_adapter = _build_runtime(rank, local_rank, world_size, safe_projection_mode="host_select")
    _execute_dispatch(runtime=safe_ready_runtime, adapter=safe_ready_adapter, layer_name=layer0, matrix=bootstrap_matrix, device=device)
    safe_ready_key = safe_ready_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001
    dist.barrier()
    safe_ready_plan = safe_ready_runtime.target_plan_store.peek(safe_ready_key)  # type: ignore[union-attr]
    safe_ready_case = _execute_dispatch(runtime=safe_ready_runtime, adapter=safe_ready_adapter, layer_name=layer1, matrix=bootstrap_matrix, device=device)
    safe_ready_terminal = safe_ready_runtime.target_plan_store.get_terminal_record(safe_ready_key)  # type: ignore[union-attr]

    safe_late_runtime, safe_late_adapter = _build_runtime(rank, local_rank, world_size, safe_projection_mode="host_select")
    _, safe_late_plan = _build_target_plan(safe_late_runtime, source_layer="0", target_layer="1", matrix=bootstrap_matrix, safe_projection_mode="host_select")
    safe_late_key = safe_late_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001
    safe_published_late = {"done": False}

    def _safe_late_hook(release_epoch: int, _snapshot: dict[str, Any]) -> None:
        if release_epoch != 1 or safe_published_late["done"]:
            return
        dist.barrier()
        publish_error = _publish_plan(safe_late_runtime, key=safe_late_key, plan=safe_late_plan)
        if publish_error is not None:
            raise RuntimeError(publish_error)
        safe_published_late["done"] = True

    def _safe_late_setup() -> None:
        setattr(safe_late_adapter, "on_release_batch_completed", _safe_late_hook)

    safe_late_case = _execute_dispatch(
        runtime=safe_late_runtime,
        adapter=safe_late_adapter,
        layer_name=layer1,
        matrix=bootstrap_matrix,
        device=device,
        after_before_dispatch=_safe_late_setup,
    )
    safe_late_terminal = safe_late_runtime.target_plan_store.get_terminal_record(safe_late_key)  # type: ignore[union-attr]
    return {
        "rank": rank,
        "ready": {
            "execution_origin": str(ready_case["summary"].get("execution_origin", "")),
            "terminal": ready_terminal.to_dict() if ready_terminal is not None else None,
        },
        "late": {
            "execution_origin": str(late_case["summary"].get("execution_origin", "")),
            "terminal": late_terminal.to_dict() if late_terminal is not None else None,
            "hidden_async_summary": _async_summary(late_case["transport"], tensor_role="hidden_states"),
        },
        "too_late": {
            "execution_origin": str(too_late_case["summary"].get("execution_origin", "")),
            "terminal": too_late_terminal.to_dict() if too_late_terminal is not None else None,
            "hidden_async_summary": _async_summary(too_late_case["transport"], tensor_role="hidden_states"),
            "publish_error": too_late_publish_error,
        },
        "safe_ready": {
            "execution_origin": str(safe_ready_case["summary"].get("execution_origin", "")),
            "terminal": safe_ready_terminal.to_dict() if safe_ready_terminal is not None else None,
            "selected_variant": str(getattr(safe_ready_plan, "selected_variant", "")) if safe_ready_plan is not None else "",
            "paired_b_logical_plan_digest": str(getattr(safe_ready_plan, "paired_b_logical_plan_digest", "")) if safe_ready_plan is not None else "",
        },
        "safe_late": {
            "execution_origin": str(safe_late_case["summary"].get("execution_origin", "")),
            "terminal": safe_late_terminal.to_dict() if safe_late_terminal is not None else None,
            "hidden_async_summary": _async_summary(safe_late_case["transport"], tensor_role="hidden_states"),
            "selected_variant": str(getattr(safe_late_plan, "selected_variant", "")),
            "paired_b_logical_plan_digest": str(getattr(safe_late_plan, "paired_b_logical_plan_digest", "")),
        },
    }


def _passed(gathered: list[dict[str, Any]]) -> bool:
    ready = [item["ready"] for item in gathered]
    late = [item["late"] for item in gathered]
    too_late = [item["too_late"] for item in gathered]
    safe_ready = [item["safe_ready"] for item in gathered]
    safe_late = [item["safe_late"] for item in gathered]
    if not all(((item.get("terminal") or {}).get("final_status") == "CONSUMED") for item in ready):
        return False
    if not all(str(item.get("execution_origin", "")).startswith("prepared_") for item in ready):
        return False
    if not all(((item.get("terminal") or {}).get("final_status") == "CONSUMED") for item in late):
        return False
    if not all(str(item.get("execution_origin", "")) == "provisional_then_late_suffix" for item in late):
        return False
    if not all(int((item.get("hidden_async_summary") or {}).get("suffix_splice_count", 0) or 0) == 1 for item in late):
        return False
    late_switch_epochs = {
        int((((item.get("hidden_async_summary") or {}).get("lineage") or [{}])[-1].get("switch_epoch", -1) or -1))
        for item in late
    }
    if len(late_switch_epochs) != 1:
        return False
    if not all(((item.get("terminal") or {}).get("final_status") == "EXPIRED") for item in too_late):
        return False
    if not all(bool(item.get("publish_error")) for item in too_late):
        return False
    if not all(((item.get("terminal") or {}).get("final_status") == "CONSUMED") for item in safe_ready):
        return False
    if not all(bool(item.get("paired_b_logical_plan_digest")) for item in safe_ready):
        return False
    if not all(((item.get("terminal") or {}).get("final_status") == "CONSUMED") for item in safe_late):
        return False
    if not all(int((item.get("hidden_async_summary") or {}).get("suffix_splice_count", 0) or 0) == 1 for item in safe_late):
        return False
    if not all(bool(item.get("paired_b_logical_plan_digest")) for item in safe_late):
        return False
    return True


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(Path(args.config), output_dir)
    if available_cuda_count() < int(args.world_size):
        payload = {
            "status": "gpu_environment_insufficient",
            "cuda_device_count": int(available_cuda_count()),
            "world_size": int(args.world_size),
        }
        write_json(output_dir / "gpu_target_lifecycle.json", payload)
        print(json.dumps(payload, indent=2))
        return 247
    if "LOCAL_RANK" not in os.environ or "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={int(args.world_size)}",
            str(Path(__file__).resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--selected-layers",
            str(args.selected_layers),
            "--world-size",
            str(int(args.world_size)),
            "--output-dir",
            str(output_dir),
        ]
        proc = run_subprocess(cmd)
        (output_dir / "worker_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (output_dir / "worker_stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0 and proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        return int(proc.returncode)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        gathered = [None for _ in range(world_size)]
        local_payload = _run_cases(rank, local_rank, world_size)
        dist.all_gather_object(gathered, local_payload)
        passed = _passed([item or {} for item in gathered])
        if rank == 0:
            payload = {
                "status": "passed" if passed else "failed",
                "world_size": int(world_size),
                "selected_layers": str(args.selected_layers),
                "all_ranks": gathered,
            }
            write_json(output_dir / "gpu_target_lifecycle.json", payload)
            print(json.dumps(payload, indent=2))
        dist.barrier()
        return 0 if passed else 1
    except Exception as exc:
        failure = {
            "status": "failed",
            "rank": rank,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / f"rank{rank}_gpu_target_lifecycle_failure.json", failure)
        if rank == 0:
            write_json(output_dir / "gpu_target_lifecycle.json", failure)
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
