from __future__ import annotations

import json
import os
import socket
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rs.runtime.observation.instrumentation import BufferedEvidenceSink, build_runtime_instrumentation
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.execution.api import _source_input_offset, _target_output_offset
from rs.runtime.online.megatron_ep.execution.pipeline import RuntimeExecutionPipeline
from rs.runtime.online.megatron_ep.execution.transport_adapter import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.public_types import CombineCompleteEvent, CombineReadyEvent, DispatchCompleteEvent, DispatchReadyEvent

from experiments.distributed.run_m1_formal_lifecycle_publication_gloo import (
    _begin_forward,
    _dispatcher_for_phase,
    _emit_source_events,
    _end_forward,
    _runtime,
    _wait_until,
)
from experiments.distributed.run_m2_formal_execution_gloo import _input_tensor


RUN_DIR = Path("outputs/closure/m123_integrated_publication_execution_gloo")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _execute_role(
    *,
    runtime,
    adapter: MegatronPhaseTransportAdapter,
    dispatcher,
    tensor_role: str,
    phase: str,
    rank: int,
    group_ranks: tuple[int, ...],
) -> torch.Tensor:
    active = runtime.current_transport()
    assert active is not None
    prepared = active["prepared_execution"]
    assert prepared is not None
    spec = next(item for item in prepared.materialized_plan.payload_specs if str(item.payload_role) == str(tensor_role))
    tensor = _input_tensor(spec, source_global_rank=int(rank))
    target_dtype = getattr(torch, str(spec.dtype).replace("torch.", ""), None)
    if target_dtype is not None:
        tensor = tensor.to(dtype=target_dtype)
    if str(phase) == "P1":
        output_split_sizes = dispatcher.input_splits
        input_split_sizes = dispatcher.output_splits
    else:
        output_split_sizes = dispatcher.output_splits
        input_split_sizes = dispatcher.input_splits
    output = adapter.maybe_execute(
        group=dist.group.WORLD,
        input_tensor=tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        original_all_to_all=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected native fallback")),
        use_nccl_stream=False,
    )
    expected = torch.zeros_like(output)
    width = int(tensor.shape[1]) if tensor.ndim > 1 else 1
    max_rows_by_src = {int(global_rank): 0 for global_rank in group_ranks}
    for batch in prepared.materialized_plan.batches:
        for item in batch.slices:
            if str(item.payload_role) != str(tensor_role):
                continue
            source_extent = _source_input_offset(
                prepared.materialized_plan,
                str(tensor_role),
                int(item.dst_group_rank),
                int(item.send_offset_rows),
            ) + int(item.row_count)
            max_rows_by_src[int(item.src_global_rank)] = max(
                int(max_rows_by_src[int(item.src_global_rank)]),
                int(source_extent),
            )
    source_tensors: dict[int, torch.Tensor] = {}
    for global_rank in group_ranks:
        rows = max(int(max_rows_by_src[int(global_rank)]), 1)
        values = torch.arange(int(global_rank) * 10000, int(global_rank) * 10000 + rows, dtype=torch.float32)
        if width <= 1:
            source_tensors[int(global_rank)] = values.reshape(rows, 1).to(dtype=tensor.dtype)
        else:
            source_tensors[int(global_rank)] = values.unsqueeze(1).repeat(1, width).to(dtype=tensor.dtype)
    for batch in prepared.materialized_plan.batches:
        for item in batch.slices:
            if str(item.payload_role) != str(tensor_role):
                continue
            if int(item.dst_global_rank) != int(rank):
                continue
            source_tensor = source_tensors[int(item.src_global_rank)]
            src_offset = _source_input_offset(
                prepared.materialized_plan,
                str(tensor_role),
                int(item.dst_group_rank),
                int(item.send_offset_rows),
            )
            dst_offset = _target_output_offset(
                prepared.materialized_plan,
                str(tensor_role),
                int(item.src_group_rank),
                int(item.recv_offset_rows),
            )
            expected.narrow(0, int(dst_offset), int(item.row_count)).copy_(
                source_tensor.narrow(0, int(src_offset), int(item.row_count))
            )
    if tuple(int(dim) for dim in output.shape) != tuple(int(dim) for dim in expected.shape):
        raise AssertionError(
            f"unexpected {phase} {tensor_role} shape rank={rank}: expected={tuple(expected.shape)} actual={tuple(output.shape)}"
        )
    expected_source_ranks = {
        int(item.src_global_rank)
        for batch in prepared.materialized_plan.batches
        for item in batch.slices
        if str(item.payload_role) == str(tensor_role) and int(item.dst_global_rank) == int(rank)
    }
    if int(output.shape[0]) > 0:
        actual_source_ranks = {int(float(value) // 10000) for value in output[:, 0].float().tolist()}
        if not actual_source_ranks.issubset(expected_source_ranks):
            raise AssertionError(
                f"unexpected {phase} {tensor_role} source ranks rank={rank}: expected={sorted(expected_source_ranks)} actual={sorted(actual_source_ranks)}"
            )
        if any(item != int(rank) for item in expected_source_ranks) and actual_source_ranks == {int(rank)}:
            raise AssertionError(
                f"{phase} {tensor_role} rank={rank} only observed local rows despite remote incoming traffic"
            )
    return output


def _worker(rank: int, world_size: int, port: int, instrumentation_mode: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"rank": int(rank), "status": "failed"}
    try:
        group_ranks = tuple(range(world_size))
        runtime, trace, spy = _runtime(
            rank=rank,
            world_size=world_size,
            group_ranks=group_ranks,
            root_rank=0,
            process_group=dist.group.WORLD,
        )
        runtime.plan_publisher = CanonicalPlanPublisher(
            rank_map=RankMap(group_ranks=group_ranks, root_rank=group_ranks[0])
        )
        runtime.execution_pipeline = RuntimeExecutionPipeline()
        runtime.runtime_instrumentation = build_runtime_instrumentation(
            instrumentation_mode=instrumentation_mode,
            evidence_sink=BufferedEvidenceSink(),
        )
        runtime._instrumentation_mode = instrumentation_mode  # noqa: SLF001
        runtime._commit_sha = str(os.environ.get("ROUTERSENSE_COMMIT_SHA", "test-sha"))  # noqa: SLF001
        runtime._git_clean = True  # noqa: SLF001
        adapter = MegatronPhaseTransportAdapter(
            dispatcher_class="SyntheticDispatcher",
            dispatcher_module_sha256=None,
            p2p_group=None,
        )
        runtime.transport_adapter = adapter
        adapter.timeline_hook = lambda event, **detail: runtime._timeline(  # noqa: SLF001
            event,
            layer_name=str(runtime.current_transport().get("layer_name") if runtime.current_transport() else "unknown"),
            **detail,
        )

        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=group_ranks)
        _wait_until(lambda: runtime.target_planner_service.publication_state_for_slot(slot) is not None, timeout_seconds=10.0)  # type: ignore[union-attr]

        p0_dispatcher, p0_hidden, p0_probs = _dispatcher_for_phase(rank=rank, group_ranks=group_ranks, phase="P0")
        runtime.handle(
            DispatchReadyEvent(
                layer_name="model.layers.1.mlp",
                dispatcher=p0_dispatcher,
                packed_hidden_states=p0_hidden,
                packed_probs=p0_probs,
                layer_role="selected",
            )
        )
        hidden_out = _execute_role(runtime=runtime, adapter=adapter, dispatcher=p0_dispatcher, tensor_role="hidden_states", phase="P0", rank=rank, group_ranks=group_ranks)
        release_after_hidden = tuple(sorted(runtime.release_state_ledger.satisfied_release_ids))
        probs_out = _execute_role(runtime=runtime, adapter=adapter, dispatcher=p0_dispatcher, tensor_role="routing_probs", phase="P0", rank=rank, group_ranks=group_ranks)
        release_after_probs = tuple(sorted(runtime.release_state_ledger.satisfied_release_ids))
        runtime.handle(
            DispatchCompleteEvent(
                layer_name="model.layers.1.mlp",
                dispatcher=p0_dispatcher,
                packed_hidden_states=p0_hidden,
                result=(hidden_out, probs_out),
                layer_role="selected",
            )
        )

        p1_dispatcher, p1_hidden, _ = _dispatcher_for_phase(rank=rank, group_ranks=group_ranks, phase="P1")
        runtime.handle(
            CombineReadyEvent(
                layer_name="model.layers.1.mlp",
                dispatcher=p1_dispatcher,
                packed_hidden_states=p1_hidden,
            )
        )
        release_before_p1 = tuple(sorted(runtime.satisfied_release_dependency_ids_for(layer_id="1", phase="P1")))
        p1_out = _execute_role(runtime=runtime, adapter=adapter, dispatcher=p1_dispatcher, tensor_role="hidden_states", phase="P1", rank=rank, group_ranks=group_ranks)
        runtime.handle(
            CombineCompleteEvent(
                layer_name="model.layers.1.mlp",
                dispatcher=p1_dispatcher,
                packed_hidden_states=p1_hidden,
                result=p1_out,
            )
        )
        release_after_p1 = tuple(sorted(runtime.release_state_ledger.satisfied_release_ids))
        _end_forward(runtime)

        result_bundle = runtime._latest_result_bundle  # noqa: SLF001
        assert result_bundle is not None
        assert result_bundle.status == "success"
        assert release_after_hidden == ()
        local_release = f"release:p0_inbound_complete:{group_ranks.index(rank)}"
        assert local_release in release_after_probs
        assert local_release in release_before_p1
        assert f"release:p1_inbound_complete:{group_ranks.index(rank)}" in release_after_p1
        summary = {
            "rank": int(rank),
            "status": "passed",
            "publication_trace_count": int(len(trace)),
            "late_suffix_call_count": int(spy["late_suffix_call_count"]),
            "release_after_hidden": list(release_after_hidden),
            "release_after_probs": list(release_after_probs),
            "release_before_p1": list(release_before_p1),
            "release_after_p1": list(release_after_p1),
            "result_status": str(result_bundle.status),
            "measurement_event_count": int(result_bundle.summary.get("measurement_event_count", 0) or 0),
            "instrumentation_mode": instrumentation_mode,
        }
    except Exception as exc:  # noqa: BLE001
        summary = {
            "rank": int(rank),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        (RUN_DIR / f"rank{rank}.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()


def run_gate(*, instrumentation_mode: str = "perf_light") -> dict[str, object]:
    world_size = 4
    port = _free_port()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    mp.spawn(_worker, args=(world_size, port, str(instrumentation_mode)), nprocs=world_size, join=True)
    payloads = [json.loads((RUN_DIR / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(world_size)]
    assert all(item["status"] == "passed" for item in payloads), payloads
    summary = {
        "status": "passed",
        "world_size": world_size,
        "instrumentation_mode": str(instrumentation_mode),
        "ranks": payloads,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_gate(instrumentation_mode=str(os.environ.get("RS_M123_GATE_INSTRUMENTATION_MODE", "perf_light") or "perf_light"))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
