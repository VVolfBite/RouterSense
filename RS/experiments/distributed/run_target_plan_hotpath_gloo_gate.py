from __future__ import annotations

import json
import os
import sys
import threading
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


def _execute_dispatch(
    *,
    runtime: RouterSenseInjectionRuntime,
    adapter: MegatronPhaseTransportAdapter,
    layer_name: str,
    matrix: tuple[tuple[int, ...], ...],
    delay_before_after: float = 0.0,
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
    if delay_before_after > 0:
        time.sleep(delay_before_after)
    runtime.after_token_dispatch(layer_name=layer_name)
    return {
        "outputs": [tuple(int(v) for v in out.shape) for out in outputs],
        "summary": runtime.export_prepared_plan_summary(),
        "transport": adapter.export_results(),
    }


def _build_target_plan(runtime: RouterSenseInjectionRuntime, *, source_layer: str, target_layer: str, matrix: tuple[tuple[int, ...], ...]):
    service = runtime.target_planner_service
    assert service is not None
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


def _worker(rank: int, init_file: str) -> None:
    stage = "init"
    try:
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
        p2p_group, _ = _maybe_create_dedicated_p2p_group(ep_group_ranks=(0, 1), local_rank=rank)
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

        layer0 = "model.layers.0.mlp"
        layer1 = "model.layers.1.mlp"
        bootstrap_matrix = ((0, 4), (4, 0))
        _execute_dispatch(runtime=runtime, adapter=adapter, layer_name=layer0, matrix=bootstrap_matrix)

        key = runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001
        deadline = time.time() + 5.0
        while runtime.target_plan_store.peek(key) is None and time.time() < deadline:  # type: ignore[union-attr]
            time.sleep(0.01)
        prepared_plan = runtime.target_plan_store.peek(key)  # type: ignore[union-attr]
        if prepared_plan is None:
            raise RuntimeError("prepared plan not produced by formal planner")

        ready_case = _execute_dispatch(runtime=runtime, adapter=adapter, layer_name=layer1, matrix=bootstrap_matrix)
        ready_terminal = runtime.target_plan_store.get_terminal_record(key)  # type: ignore[union-attr]

        late_runtime = _runtime(rank)
        late_runtime.transport_adapter = MegatronPhaseTransportAdapter(
            dispatcher_class="SyntheticDispatcher",
            dispatcher_module_sha256=None,
            p2p_group=p2p_group,
        )
        late_runtime.transport_adapter.timeline_hook = lambda event, **detail: late_runtime._timeline(  # noqa: SLF001
            event,
            layer_name=str(late_runtime.current_transport().get("layer_name") if late_runtime.current_transport() else "unknown"),
            **detail,
        )
        late_runtime.begin_forward(forward_epoch=1)
        _, late_plan = _build_target_plan(late_runtime, source_layer="0", target_layer="1", matrix=bootstrap_matrix)
        reverse_plan = replace(
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

        def _late_after_before_dispatch() -> None:
            t = threading.Thread(
                target=lambda: (time.sleep(0.02), late_runtime.target_plan_store.put(late_key, reverse_plan)),  # type: ignore[union-attr]
                daemon=True,
            )
            t.start()

        late_case = _execute_dispatch(
            runtime=late_runtime,
            adapter=late_runtime.transport_adapter,
            layer_name=layer1,
            matrix=bootstrap_matrix,
            after_before_dispatch=_late_after_before_dispatch,
        )
        late_terminal = late_runtime.target_plan_store.get_terminal_record(late_key)  # type: ignore[union-attr]

        too_late_runtime = _runtime(rank)
        too_late_runtime.transport_adapter = MegatronPhaseTransportAdapter(
            dispatcher_class="SyntheticDispatcher",
            dispatcher_module_sha256=None,
            p2p_group=p2p_group,
        )
        too_late_runtime.transport_adapter.timeline_hook = lambda event, **detail: too_late_runtime._timeline(  # noqa: SLF001
            event,
            layer_name=str(too_late_runtime.current_transport().get("layer_name") if too_late_runtime.current_transport() else "unknown"),
            **detail,
        )
        too_late_runtime.begin_forward(forward_epoch=1)
        _, too_late_plan = _build_target_plan(too_late_runtime, source_layer="0", target_layer="1", matrix=bootstrap_matrix)
        too_late_key = too_late_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001

        def _too_late_after_before_dispatch() -> None:
            t2 = threading.Thread(
                target=lambda: (time.sleep(0.2), too_late_runtime.target_plan_store.put(too_late_key, too_late_plan)),  # type: ignore[union-attr]
                daemon=True,
            )
            t2.start()

        too_late_case = _execute_dispatch(
            runtime=too_late_runtime,
            adapter=too_late_runtime.transport_adapter,
            layer_name=layer1,
            matrix=((0, 1), (1, 0)),
            delay_before_after=0.25,
            after_before_dispatch=_too_late_after_before_dispatch,
        )
        too_late_terminal = too_late_runtime.target_plan_store.get_terminal_record(too_late_key)  # type: ignore[union-attr]

        reject_runtime = _runtime(rank)
        reject_runtime.transport_adapter = MegatronPhaseTransportAdapter(
            dispatcher_class="SyntheticDispatcher",
            dispatcher_module_sha256=None,
            p2p_group=p2p_group,
        )
        reject_runtime.transport_adapter.timeline_hook = lambda event, **detail: reject_runtime._timeline(  # noqa: SLF001
            event,
            layer_name=str(reject_runtime.current_transport().get("layer_name") if reject_runtime.current_transport() else "unknown"),
            **detail,
        )
        reject_runtime.begin_forward(forward_epoch=1)
        _, reject_plan = _build_target_plan(reject_runtime, source_layer="0", target_layer="1", matrix=((0, 4), (0, 0)))
        reject_plan = replace(reject_plan, h1_rows=((0, 4), (0, 0)))
        reject_key = reject_runtime._target_plan_key(layer_name=layer1)  # noqa: SLF001

        def _reject_after_before_dispatch() -> None:
            t3 = threading.Thread(
                target=lambda: (time.sleep(0.02), reject_runtime.target_plan_store.put(reject_key, reject_plan)),  # type: ignore[union-attr]
                daemon=True,
            )
            t3.start()

        reject_case = _execute_dispatch(
            runtime=reject_runtime,
            adapter=reject_runtime.transport_adapter,
            layer_name=layer1,
            matrix=((0, 0), (4, 0)),
            after_before_dispatch=_reject_after_before_dispatch,
        )
        reject_terminal = reject_runtime.target_plan_store.get_terminal_record(reject_key)  # type: ignore[union-attr]

        payload = {
            "rank": rank,
            "prepared_ready": {
                "execution_origin": ready_case["summary"].get("execution_origin", ""),
                "terminal": ready_terminal.to_dict() if ready_terminal is not None else None,
            },
            "late_suffix": {
                "execution_origin": late_case["summary"].get("execution_origin", ""),
                "terminal": late_terminal.to_dict() if late_terminal is not None else None,
                "transport": late_case["transport"],
            },
            "too_late": {
                "execution_origin": too_late_case["summary"].get("execution_origin", ""),
                "terminal": too_late_terminal.to_dict() if too_late_terminal is not None else None,
            },
            "reject": {
                "execution_origin": reject_case["summary"].get("execution_origin", ""),
                "terminal": reject_terminal.to_dict() if reject_terminal is not None else None,
            },
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"rank{rank}_target_hotpath.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        gathered = [None, None]
        dist.all_gather_object(gathered, payload)
        if rank == 0:
            summary = {
                "passed": True,
                "all_ranks": gathered,
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
