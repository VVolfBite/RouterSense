from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs/closure/target_plan_hotpath_final"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.execution.transport_adapter import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.host import _maybe_create_dedicated_p2p_group
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


def _runtime(rank: int) -> RouterSenseInjectionRuntime:
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
            bucket_mode="fixed_rows",
            bucket_rows=1,
            executor_heartbeat_path="",
        ),
        rank=rank,
        local_rank=rank,
        run_id="target-plan-hotpath-gloo",
        step_id="step",
        microbatch_id="mb0",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="localhost",
        ep_group_ranks=(0, 1),
        ep_group_root_global_rank=0,
    )


def _payload(rank: int, rows: int, width: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.arange(max(rows, 1) * width, dtype=torch.float32).reshape(max(rows, 1), width)[:rows] + (1000 * rank)
    probs = torch.arange(max(rows, 1), dtype=torch.float32).reshape(max(rows, 1), 1)[:rows] + (100 * rank)
    return hidden, probs


def _phase_summary(results: list[dict[str, Any]], *, tensor_role: str) -> dict[str, Any]:
    for row in results:
        if str(row.get("tensor_role", "")) != str(tensor_role):
            continue
        result = row.get("result")
        if isinstance(result, dict):
            return result
    return {}


def _async_summary(results: list[dict[str, Any]], *, tensor_role: str) -> dict[str, Any]:
    for row in results:
        if str(row.get("record_type", "")) != "async_phase_summary":
            continue
        if str(row.get("tensor_role", "")) == str(tensor_role):
            return row
    return {}


def _execute_dispatch(
    *,
    runtime: RouterSenseInjectionRuntime,
    adapter: MegatronPhaseTransportAdapter,
    layer_name: str,
    matrix: tuple[tuple[int, ...], ...],
    after_before_dispatch: Any | None = None,
) -> dict[str, Any]:
    rank = int(runtime.rank)
    row = tuple(int(v) for v in matrix[rank])
    col = tuple(int(matrix[src][rank]) for src in range(len(matrix)))
    hidden, probs = _payload(rank, sum(row))
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


def _build_target_plan(runtime: RouterSenseInjectionRuntime, *, source_layer: str, target_layer: str, matrix: tuple[tuple[int, ...], ...]):
    service = runtime.target_planner_service
    if service is None:
        raise RuntimeError("missing target planner service")
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
            policy_id=str(runtime._effective_phase_policy_name()),  # noqa: SLF001
            group_size=2,
            bucket_rows=1,
            policy_options=PolicyOptions(
                p0_weight=1.0,
                p1_weight=1.0,
                p2_hint_weight=1.0,
                residual_weight=0.75,
                barrier_weight=1.75,
                age_weight=0.15,
                prediction_weight=0.35,
            ),
            topology_digest="topo",
            bucket_contract_digest="fixed_rows",
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
    except Exception as exc:  # pragma: no cover
        return f"{type(exc).__name__}: {exc}"


def _build_runtime(rank: int, p2p_group: Any) -> tuple[RouterSenseInjectionRuntime, MegatronPhaseTransportAdapter]:
    runtime = _runtime(rank)
    adapter = MegatronPhaseTransportAdapter(
        dispatcher_class="SyntheticDispatcher",
        dispatcher_module_sha256=None,
        p2p_group=p2p_group,
    )
    runtime.transport_adapter = adapter
    adapter.timeline_hook = lambda event, **detail: runtime._timeline(  # noqa: SLF001
        event,
        layer_name=str(runtime.current_transport().get("layer_name") if runtime.current_transport() else "unknown"),
        **detail,
    )
    runtime.begin_forward(forward_epoch=1)
    return runtime, adapter


def _run_attempt(rank: int, p2p_group: Any, attempt: int) -> dict[str, Any]:
    layer0 = "model.layers.0.mlp"
    layer1 = "model.layers.1.mlp"
    bootstrap_matrix = ((0, 4), (4, 0))

    ready_runtime, ready_adapter = _build_runtime(rank, p2p_group)
    _execute_dispatch(runtime=ready_runtime, adapter=ready_adapter, layer_name=layer0, matrix=bootstrap_matrix)
    ready_key = ready_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001
    deadline = time.time() + 5.0
    while ready_runtime.target_plan_store.peek(ready_key) is None and time.time() < deadline:  # type: ignore[union-attr]
        time.sleep(0.01)
    if ready_runtime.target_plan_store.peek(ready_key) is None:  # type: ignore[union-attr]
        raise RuntimeError(f"prepared target plan missing for attempt {attempt}")
    ready_case = _execute_dispatch(runtime=ready_runtime, adapter=ready_adapter, layer_name=layer1, matrix=bootstrap_matrix)
    ready_terminal = ready_runtime.target_plan_store.get_terminal_record(ready_key)  # type: ignore[union-attr]

    late_runtime, late_adapter = _build_runtime(rank, p2p_group)
    _, late_plan = _build_target_plan(late_runtime, source_layer="0", target_layer="1", matrix=bootstrap_matrix)
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
    provisional_hidden_ids: list[str] = []
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
        current = late_runtime.current_transport() or {}
        current_plan = current.get("plan")
        if current_plan is not None:
            provisional_hidden_ids[:] = [
                str(task.task_id)
                for task in late_runtime._build_release_batch_tasks_from_plan(  # noqa: SLF001
                    plan=current_plan,
                    tensor_role="hidden_states",
                )
            ]
        setattr(late_adapter, "on_release_batch_completed", _late_hook)

    late_case = _execute_dispatch(runtime=late_runtime, adapter=late_adapter, layer_name=layer1, matrix=bootstrap_matrix, after_before_dispatch=_late_setup)
    late_terminal = late_runtime.target_plan_store.get_terminal_record(late_key)  # type: ignore[union-attr]

    too_late_runtime, too_late_adapter = _build_runtime(rank, p2p_group)
    _, too_late_plan = _build_target_plan(too_late_runtime, source_layer="0", target_layer="1", matrix=bootstrap_matrix)
    too_late_key = too_late_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001
    too_late_case = _execute_dispatch(runtime=too_late_runtime, adapter=too_late_adapter, layer_name=layer1, matrix=((0, 1), (1, 0)))
    dist.barrier()
    too_late_publish_error = _publish_plan(too_late_runtime, key=too_late_key, plan=too_late_plan)
    too_late_terminal = too_late_runtime.target_plan_store.get_terminal_record(too_late_key)  # type: ignore[union-attr]

    reject_runtime, reject_adapter = _build_runtime(rank, p2p_group)
    _, reject_plan = _build_target_plan(reject_runtime, source_layer="0", target_layer="1", matrix=((0, 4), (0, 0)))
    reject_plan = replace(reject_plan, h1_rows=((0, 4), (0, 0)))
    reject_key = reject_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001
    published_reject = {"done": False}

    def _reject_hook(release_epoch: int, _snapshot: dict[str, Any]) -> None:
        if release_epoch != 1 or published_reject["done"]:
            return
        dist.barrier()
        publish_error = _publish_plan(reject_runtime, key=reject_key, plan=reject_plan)
        if publish_error is not None:
            raise RuntimeError(publish_error)
        published_reject["done"] = True

    def _reject_setup() -> None:
        setattr(reject_adapter, "on_release_batch_completed", _reject_hook)

    reject_case = _execute_dispatch(runtime=reject_runtime, adapter=reject_adapter, layer_name=layer1, matrix=((0, 0), (4, 0)), after_before_dispatch=_reject_setup)
    reject_terminal = reject_runtime.target_plan_store.get_terminal_record(reject_key)  # type: ignore[union-attr]

    return {
        "attempt": int(attempt),
        "rank": int(rank),
        "prepared_ready": {
            "execution_origin": str(ready_case["summary"].get("execution_origin", "")),
            "terminal": ready_terminal.to_dict() if ready_terminal is not None else None,
        },
        "late_suffix": {
            "execution_origin": str(late_case["summary"].get("execution_origin", "")),
            "terminal": late_terminal.to_dict() if late_terminal is not None else None,
            "hidden_async_summary": _async_summary(late_case["transport"], tensor_role="hidden_states"),
            "provisional_hidden_ids": list(provisional_hidden_ids),
            "hidden_phase_summary": _phase_summary(late_case["transport"], tensor_role="hidden_states"),
        },
        "too_late": {
            "execution_origin": str(too_late_case["summary"].get("execution_origin", "")),
            "terminal": too_late_terminal.to_dict() if too_late_terminal is not None else None,
            "publish_error": too_late_publish_error,
            "hidden_async_summary": _async_summary(too_late_case["transport"], tensor_role="hidden_states"),
        },
        "reject": {
            "execution_origin": str(reject_case["summary"].get("execution_origin", "")),
            "terminal": reject_terminal.to_dict() if reject_terminal is not None else None,
            "hidden_async_summary": _async_summary(reject_case["transport"], tensor_role="hidden_states"),
        },
    }


def _attempt_passed(payloads: list[dict[str, Any]]) -> bool:
    ready = [item.get("prepared_ready", {}) for item in payloads]
    late = [item.get("late_suffix", {}) for item in payloads]
    too_late = [item.get("too_late", {}) for item in payloads]
    reject = [item.get("reject", {}) for item in payloads]
    if not all(((item.get("terminal") or {}).get("final_status") == "CONSUMED") for item in ready):
        return False
    if not all(str(item.get("execution_origin", "")).startswith("prepared_") for item in ready):
        return False
    if not all(((item.get("terminal") or {}).get("final_status") == "CONSUMED") for item in late):
        return False
    if not all(str(item.get("execution_origin", "")) == "provisional_then_late_suffix" for item in late):
        return False
    late_digests = {str((item.get("hidden_async_summary") or {}).get("frontier_digest", "")) for item in late}
    late_switches = {int(((item.get("hidden_async_summary") or {}).get("lineage") or [{}])[-1].get("switch_epoch", -1)) for item in late}
    if len(late_digests) != 1 or len(late_switches) != 1:
        return False
    if not all(int((item.get("hidden_async_summary") or {}).get("suffix_splice_count", 0)) == 1 for item in late):
        return False
    if not all(list(item.get("provisional_hidden_ids", [])) != list((item.get("hidden_async_summary") or {}).get("final_task_ids", [])) for item in late):
        return False
    if not all(((item.get("terminal") or {}).get("final_status") == "EXPIRED") for item in too_late):
        return False
    if not all(bool(item.get("publish_error")) for item in too_late):
        return False
    if not all(int((item.get("hidden_async_summary") or {}).get("suffix_splice_count", 0)) == 0 for item in too_late):
        return False
    if not all(((item.get("terminal") or {}).get("final_status") == "REJECTED") for item in reject):
        return False
    if not all(int((item.get("hidden_async_summary") or {}).get("suffix_splice_count", 0)) == 0 for item in reject):
        return False
    return True


def _worker(rank: int, init_file: str) -> None:
    stage = "init"
    try:
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
        p2p_group, _ = _maybe_create_dedicated_p2p_group(ep_group_ranks=(0, 1), local_rank=rank)
        if rank == 0 and OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        dist.barrier()
        attempts: list[dict[str, Any]] = []
        for attempt in range(10):
            local_payload = _run_attempt(rank, p2p_group, attempt)
            gathered = [None, None]
            dist.all_gather_object(gathered, local_payload)
            attempts.append(
                {
                    "attempt": int(attempt),
                    "passed": bool(_attempt_passed([row or {} for row in gathered])),
                    "all_ranks": gathered,
                }
            )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"rank{rank}_target_hotpath.json").write_text(json.dumps({"attempts": attempts}, indent=2), encoding="utf-8")
        if rank == 0:
            summary = {
                "passed": bool(all(bool(item.get("passed", False)) for item in attempts)),
                "attempt_count": int(len(attempts)),
                "attempts": attempts,
            }
            (OUT_DIR / "gloo_target_plan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
        dist.barrier()
        dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "rank": rank,
            "stage": stage,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (OUT_DIR / f"rank{rank}_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


def main() -> None:
    with TemporaryDirectory(dir=str(REPO_ROOT / "outputs")) as tmpdir:
        init_file = Path(tmpdir) / "gloo-init"
        init_file.touch()
        ctx = mp.get_context("fork")
        workers = [ctx.Process(target=_worker, args=(rank, str(init_file))) for rank in range(2)]
        for proc in workers:
            proc.start()
        for proc in workers:
            proc.join()
        codes = [proc.exitcode for proc in workers]
        if any(code != 0 for code in codes):
            raise SystemExit(f"worker failure exit_codes={codes}")


if __name__ == "__main__":
    main()
