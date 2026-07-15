from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import traceback
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rs.runtime.observation.instrumentation import BufferedEvidenceSink, build_runtime_instrumentation
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.execution.pipeline import build_runtime_execution_pipeline
from rs.runtime.online.megatron_ep.execution.transport_adapter import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.public_types import CombineCompleteEvent, CombineReadyEvent, DispatchCompleteEvent, DispatchReadyEvent

from experiments.distributed.run_m1_formal_lifecycle_publication_gloo import (
    _begin_forward,
    _dispatcher_for_phase,
    _emit_source_events,
    _end_forward,
    _matrix_for_world_size,
    _runtime,
    _wait_until,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _input_tensor(spec, *, source_global_rank: int) -> torch.Tensor:
    rows = int(spec.row_count)
    hidden = int(spec.shape_suffix[0]) if spec.shape_suffix else 1
    base = int(source_global_rank) * 10000
    values = torch.arange(base, base + max(rows, 1), dtype=torch.float32)
    target_dtype = getattr(torch, str(spec.dtype).replace("torch.", ""), torch.float32)
    if hidden <= 1:
        return values[:rows].to(dtype=target_dtype).reshape(rows, 1)
    return values[:rows].unsqueeze(1).repeat(1, hidden).to(dtype=target_dtype)


def _execute_role(
    *,
    runtime,
    published_plan,
    source_payload_specs_by_rank,
    source_slice_oracle_by_rank,
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
    gathered_inputs: list[dict[str, object] | None] = [None for _ in group_ranks]
    dist.all_gather_object(
        gathered_inputs,
        {
            "rank": int(rank),
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype),
            "rows": tensor.float().tolist(),
        },
    )
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
    source_tensors: dict[int, torch.Tensor] = {}
    for payload in gathered_inputs:
        item = dict(payload or {})
        global_rank = int(item.get("rank", 0))
        rows = torch.tensor(item.get("rows", []), dtype=torch.float32)
        shape = tuple(int(dim) for dim in item.get("shape", []))
        if shape:
            rows = rows.reshape(shape)
        source_tensors[global_rank] = rows.to(dtype=tensor.dtype)
    for batch in prepared.materialized_plan.batches:
        for item in batch.slices:
            if str(item.payload_role) != str(tensor_role):
                continue
            if int(item.dst_global_rank) != int(rank):
                continue
            source_tensor = source_tensors[int(item.src_global_rank)]
            source_slice_key = (
                str(item.flow_id),
                str(item.payload_role),
                int(item.src_global_rank),
                int(item.dst_global_rank),
            )
            src_offset = int(
                source_slice_oracle_by_rank[int(item.src_global_rank)].get(source_slice_key, item.physical_send_offset_rows)
            )
            dst_offset = int(item.physical_recv_offset_rows)
            expected.narrow(0, int(dst_offset), int(item.row_count)).copy_(
                source_tensor.narrow(0, int(src_offset), int(item.row_count))
            )
    if tuple(int(dim) for dim in output.shape) != tuple(int(dim) for dim in expected.shape):
        raise AssertionError(
            f"unexpected {phase} {tensor_role} shape rank={rank}: expected={tuple(expected.shape)} actual={tuple(output.shape)}"
        )
    if not torch.equal(output, expected):
        mismatches = (output != expected).nonzero(as_tuple=False)
        first = mismatches[0].tolist() if int(mismatches.shape[0]) > 0 else [0, 0]
        row_index = int(first[0])
        col_index = int(first[1]) if len(first) > 1 else 0
        expected_value = expected[row_index, col_index].item() if expected.ndim > 1 else expected[row_index].item()
        actual_value = output[row_index, col_index].item() if output.ndim > 1 else output[row_index].item()
        local_slices = [
            {
                "flow_id": str(item.flow_id),
                "src_group_rank": int(item.src_group_rank),
                "dst_group_rank": int(item.dst_group_rank),
                "src_global_rank": int(item.src_global_rank),
                "dst_global_rank": int(item.dst_global_rank),
                "row_count": int(item.row_count),
                "send_offset_rows": int(item.send_offset_rows),
                "recv_offset_rows": int(item.recv_offset_rows),
                "transfer_tag": int(getattr(item, "transfer_tag", 0)),
            }
            for batch in prepared.materialized_plan.batches
            for item in batch.slices
            if str(item.payload_role) == str(tensor_role)
        ]
        raise AssertionError(
            f"tensor_mismatch phase={phase} payload_role={tensor_role} rank={rank} row={row_index} col={col_index} expected={expected_value} actual={actual_value} expected_rows={expected.float().tolist()} actual_rows={output.float().tolist()} slices={local_slices}"
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


def _close_runtime_services(runtime, adapter) -> dict[str, object]:
    errors: list[dict[str, object]] = []
    planner_service = getattr(runtime, "target_planner_service", None)
    if planner_service is not None:
        try:
            planner_service.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append({"component": "target_planner_service", "error_type": type(exc).__name__, "error": str(exc)})
    target_store = getattr(runtime, "target_plan_store", None)
    if target_store is not None and hasattr(target_store, "shutdown"):
        try:
            target_store.shutdown()
        except BaseException as exc:  # noqa: BLE001
            errors.append({"component": "target_plan_store", "error_type": type(exc).__name__, "error": str(exc)})
    instrumentation = getattr(runtime, "runtime_instrumentation", None)
    if instrumentation is not None and hasattr(instrumentation, "close"):
        try:
            instrumentation.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append({"component": "runtime_instrumentation", "error_type": type(exc).__name__, "error": str(exc)})
    if adapter is not None and hasattr(adapter, "close"):
        try:
            adapter.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append({"component": "transport_adapter", "error_type": type(exc).__name__, "error": str(exc)})
    planner_alive = bool(planner_service is not None and hasattr(planner_service, "is_alive") and planner_service.is_alive())
    return {
        "cleanup_errors": errors,
        "planner_thread_alive": planner_alive,
    }


def _transfer_keys_from_completed_tasks(
    materialized_plan,
    completed_task_ids: tuple[str, ...],
    *,
    owner_global_rank: int,
) -> tuple[dict[str, object], ...]:
    completed = set(str(item) for item in completed_task_ids)
    keys: list[dict[str, object]] = []
    for batch in materialized_plan.batches:
        for item in batch.slices:
            if str(item.task_id) not in completed:
                continue
            if int(item.src_global_rank) != int(owner_global_rank):
                continue
            keys.append(
                {
                    "phase": str(materialized_plan.phase),
                    "payload_role": str(item.payload_role),
                    "src_group_rank": int(item.src_group_rank),
                    "dst_group_rank": int(item.dst_group_rank),
                    "row_count": int(item.row_count),
                }
            )
    return tuple(keys)


def _all_task_ids(materialized_plan, *, payload_roles: set[str]) -> tuple[str, ...]:
    return tuple(
        str(item.task_id)
        for batch in materialized_plan.batches
        for item in batch.slices
        if str(item.payload_role) in payload_roles
    )


def _worker(rank: int, world_size: int, port: int, instrumentation_mode: str, execution_backend: str, run_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    run_root = Path(str(run_dir))
    run_root.mkdir(parents=True, exist_ok=True)
    summary = {"rank": int(rank), "status": "failed"}
    runtime = None
    adapter = None
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
        if str(execution_backend) == "async_release":
            object.__setattr__(runtime.config, "execution_mode", "joint_window_async_p2p")
        else:
            object.__setattr__(runtime.config, "execution_mode", "phase_sync_wave")
        runtime.execution_pipeline = build_runtime_execution_pipeline(
            execution_mode="joint_window_async_p2p" if str(execution_backend) == "async_release" else "phase_sync_wave"
        )
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
            p2p_group=dist.group.WORLD if str(execution_backend) == "async_release" else None,
        )
        runtime.transport_adapter = adapter
        adapter.timeline_hook = lambda event, **detail: runtime._timeline(  # noqa: SLF001
            event,
            layer_name=str(runtime.current_transport().get("layer_name") if runtime.current_transport() else "unknown"),
            **detail,
        )
        _begin_forward(runtime)
        slot = _emit_source_events(runtime, rank=rank, group_ranks=group_ranks)
        _wait_until(
            lambda: (
                runtime.target_planner_service.publication_state_for_slot(slot) is not None  # type: ignore[union-attr]
                and str(runtime.target_planner_service.publication_state_for_slot(slot).status).upper() == "READY"  # type: ignore[union-attr]
            ),
            timeout_seconds=10.0,
        )
        publication_state = runtime.target_planner_service.publication_state_for_slot(slot)  # type: ignore[union-attr]
        assert publication_state is not None
        publication_metadata = dict(publication_state.metadata)

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
        target_key = runtime._target_plan_key(layer_name="model.layers.1.mlp")  # noqa: SLF001
        store_key = runtime.target_plan_store._key(target_key)  # type: ignore[union-attr]  # noqa: SLF001
        published_plan = runtime._execution_plan_cache().get(store_key)  # noqa: SLF001
        if published_plan is None:
            publication_state = runtime.target_planner_service.publication_state_for_slot(slot)  # type: ignore[union-attr]
            raise KeyError(
                {
                    "store_key": store_key,
                    "publish_tokens_present": bool(getattr(runtime.target_plan_store, "_publish_tokens", {})),  # noqa: SLF001
                    "plans_present": bool(getattr(runtime.target_plan_store, "_plans", {})),  # noqa: SLF001
                    "expected_slots": list(getattr(runtime, "_expected_publication_slots", {}).keys()),  # noqa: SLF001
                    "published_slots": list(getattr(runtime, "_published_publication_slots", set())),  # noqa: SLF001
                    "publication_state_status": None if publication_state is None else str(publication_state.status),
                    "publication_state_metadata": {} if publication_state is None else dict(publication_state.metadata),
                    "control_timeline_tail": list(getattr(runtime, "control_timeline", [])[-10:]),
                }
            )
        prepared_p0 = runtime._prepared_execution_cache()[store_key]  # noqa: SLF001
        local_payload_specs = {
            str(spec.payload_role): {
                "row_count": int(spec.row_count),
                "dtype": str(spec.dtype),
                "shape_suffix": tuple(int(dim) for dim in spec.shape_suffix),
            }
            for spec in prepared_p0.materialized_plan.payload_specs
        }
        gathered_payload_specs: list[dict[str, object] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_payload_specs, local_payload_specs)
        source_payload_specs_by_rank = {
            int(global_rank): dict(gathered_payload_specs[index] or {})
            for index, global_rank in enumerate(group_ranks)
        }
        local_p0_slice_oracle = {
            (
                str(slice_.flow_id),
                str(slice_.payload_role),
                int(slice_.src_global_rank),
                int(slice_.dst_global_rank),
            ): int(slice_.physical_send_offset_rows)
            for batch in prepared_p0.materialized_plan.batches
            for slice_ in batch.slices
            if int(slice_.src_global_rank) == int(rank)
        }
        gathered_p0_slice_oracles: list[dict[tuple[str, str, int, int], int] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_p0_slice_oracles, local_p0_slice_oracle)
        p0_slice_oracle_by_rank = {
            int(global_rank): dict(gathered_p0_slice_oracles[index] or {})
            for index, global_rank in enumerate(group_ranks)
        }
        (run_root / f"rank{rank}_p0_materialized.json").write_text(
            json.dumps(
                {
                    "rank": int(rank),
                    "local_payload_specs": local_payload_specs,
                    "source_payload_specs_by_rank": source_payload_specs_by_rank,
                    "expected_outgoing_rows": prepared_p0.materialized_plan.expected_outgoing_rows,
                    "expected_incoming_rows": prepared_p0.materialized_plan.expected_incoming_rows,
                    "batches": [
                        {
                            "batch_id": str(batch.batch_id),
                            "wave_id": int(batch.wave_id),
                            "slices": [slice_.to_dict() for slice_ in batch.slices],
                        }
                        for batch in prepared_p0.materialized_plan.batches
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        hidden_debug_spec = next(item for item in prepared_p0.materialized_plan.payload_specs if str(item.payload_role) == "hidden_states")
        hidden_debug_tensor = _input_tensor(hidden_debug_spec, source_global_rank=int(rank))
        (run_root / f"rank{rank}_p0_hidden_input.json").write_text(
            json.dumps(
                {
                    "rank": int(rank),
                    "shape": tuple(int(dim) for dim in hidden_debug_tensor.shape),
                    "dtype": str(hidden_debug_tensor.dtype),
                    "rows": hidden_debug_tensor.float().tolist(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        hidden_out = _execute_role(runtime=runtime, published_plan=published_plan, source_payload_specs_by_rank=source_payload_specs_by_rank, source_slice_oracle_by_rank=p0_slice_oracle_by_rank, adapter=adapter, dispatcher=p0_dispatcher, tensor_role="hidden_states", phase="P0", rank=rank, group_ranks=group_ranks)
        release_after_hidden = tuple(sorted(runtime.release_state_ledger.satisfied_release_ids))
        probs_out = _execute_role(runtime=runtime, published_plan=published_plan, source_payload_specs_by_rank=source_payload_specs_by_rank, source_slice_oracle_by_rank=p0_slice_oracle_by_rank, adapter=adapter, dispatcher=p0_dispatcher, tensor_role="routing_probs", phase="P0", rank=rank, group_ranks=group_ranks)
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
        prepared_p1 = runtime._prepared_execution_cache()[store_key]  # noqa: SLF001
        local_p1_payload_specs = {
            str(spec.payload_role): {
                "row_count": int(spec.row_count),
                "dtype": str(spec.dtype),
                "shape_suffix": tuple(int(dim) for dim in spec.shape_suffix),
            }
            for spec in prepared_p1.materialized_plan.payload_specs
        }
        gathered_p1_payload_specs: list[dict[str, object] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_p1_payload_specs, local_p1_payload_specs)
        p1_source_payload_specs_by_rank = {
            int(global_rank): dict(gathered_p1_payload_specs[index] or {})
            for index, global_rank in enumerate(group_ranks)
        }
        local_p1_slice_oracle = {
            (
                str(slice_.flow_id),
                str(slice_.payload_role),
                int(slice_.src_global_rank),
                int(slice_.dst_global_rank),
            ): int(slice_.physical_send_offset_rows)
            for batch in prepared_p1.materialized_plan.batches
            for slice_ in batch.slices
            if int(slice_.src_global_rank) == int(rank)
        }
        gathered_p1_slice_oracles: list[dict[tuple[str, str, int, int], int] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_p1_slice_oracles, local_p1_slice_oracle)
        p1_slice_oracle_by_rank = {
            int(global_rank): dict(gathered_p1_slice_oracles[index] or {})
            for index, global_rank in enumerate(group_ranks)
        }
        release_before_p1 = tuple(sorted(runtime.satisfied_release_dependency_ids_for(layer_id="1", phase="P1")))
        p1_out = _execute_role(runtime=runtime, published_plan=published_plan, source_payload_specs_by_rank=p1_source_payload_specs_by_rank, source_slice_oracle_by_rank=p1_slice_oracle_by_rank, adapter=adapter, dispatcher=p1_dispatcher, tensor_role="hidden_states", phase="P1", rank=rank, group_ranks=group_ranks)
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
        local_release = f"release:1:p0_inbound_complete:{group_ranks.index(rank)}"
        assert local_release in release_after_probs
        assert local_release in release_before_p1
        assert f"release:1:p1_inbound_complete:{group_ranks.index(rank)}" in release_after_p1
        p0_outcomes = [
            item
            for item in runtime._latest_execution_outcomes  # noqa: SLF001
            if str(item.get("phase")) == "P0"
        ]
        p1_outcomes = [
            item
            for item in runtime._latest_execution_outcomes  # noqa: SLF001
            if str(item.get("phase")) == "P1"
        ]
        all_outcomes = [
            dict(item.get("outcome", item))
            for item in runtime._latest_execution_outcomes  # noqa: SLF001
        ]
        p0_completed_task_ids = tuple(
            str(task_id)
            for item in p0_outcomes
            if str(item.get("payload_role")) in {"hidden_states", "routing_probs"}
            for task_id in tuple(str(task_id) for task_id in dict(item.get("outcome", item)).get("completed_task_ids", ()))
        )
        p1_completed_task_ids = tuple(
            str(task_id)
            for item in p1_outcomes
            if str(item.get("payload_role")) == "hidden_states"
            for task_id in tuple(str(task_id) for task_id in dict(item.get("outcome", item)).get("completed_task_ids", ()))
        )
        if not p0_completed_task_ids:
            raise AssertionError("missing actual P0 completed_task_ids")
        if not p1_completed_task_ids:
            raise AssertionError("missing actual P1 completed_task_ids")
        distributed_operation_count = int(
            sum(int(dict(item.get("details", {})).get("distributed_operation_count", 0) or 0) for item in all_outcomes)
        )
        peak_inflight_batches = int(
            max((int(dict(item.get("details", {})).get("peak_inflight_batches", 0) or 0) for item in all_outcomes), default=0)
        )
        submitted_task_count = int(
            sum(len(tuple(item.get("submitted_task_ids", ()))) for item in all_outcomes)
        )
        completed_task_count = int(
            sum(len(tuple(item.get("completed_task_ids", ()))) for item in all_outcomes)
        )
        unresolved_task_count = int(
            sum(len(tuple(item.get("unresolved_task_ids", ()))) for item in all_outcomes)
        )
        summary = {
            "rank": int(rank),
            "status": "passed",
            "execution_backend": str(execution_backend),
            "publication_trace_count": int(len(trace)),
            "late_suffix_call_count": int(spy["late_suffix_call_count"]),
            "late_suffix_provider_call_count": int(spy["late_suffix_provider_call_count"]),
            "late_suffix_consume_count": int(spy["late_suffix_consume_count"]),
            "release_after_hidden": list(release_after_hidden),
            "release_after_probs": list(release_after_probs),
            "release_before_p1": list(release_before_p1),
            "release_after_p1": list(release_after_p1),
            "result_status": str(result_bundle.status),
            "measurement_event_count": int(result_bundle.summary.get("measurement_event_count", 0) or 0),
            "instrumentation_mode": instrumentation_mode,
            "planning_request_digest": str(published_plan.window_plan.request_digest),
            "window_plan_digest": str(published_plan.window_plan.semantic_digest()),
            "published_plan_digest": str(published_plan.published_plan_digest),
            "publication_slot": dict(published_plan.publication_slot),
            "rank_map": published_plan.rank_map.to_dict(),
            "window_plan": published_plan.window_plan.to_dict(),
            "p0_actual_phase_context": prepared_p0.actual_phase_context.to_dict(),
            "p1_actual_phase_context": prepared_p1.actual_phase_context.to_dict(),
            "p0_materialized_plan_digest": str(prepared_p0.materialized_plan.materialized_plan_digest),
            "p1_materialized_plan_digest": str(prepared_p1.materialized_plan.materialized_plan_digest),
            "p0_completed_task_ids": list(p0_completed_task_ids),
            "p1_completed_task_ids": list(p1_completed_task_ids),
            "distributed_operation_count": distributed_operation_count,
            "peak_inflight_batches": peak_inflight_batches,
            "submitted_task_count": submitted_task_count,
            "completed_task_count": completed_task_count,
            "unresolved_task_count": unresolved_task_count,
            "p0_completed_transfer_keys": list(
                _transfer_keys_from_completed_tasks(
                    prepared_p0.materialized_plan,
                    p0_completed_task_ids,
                    owner_global_rank=rank,
                )
            ),
            "p1_completed_transfer_keys": list(
                _transfer_keys_from_completed_tasks(
                    prepared_p1.materialized_plan,
                    p1_completed_task_ids,
                    owner_global_rank=rank,
                )
            ),
            "publication_candidate_status": str(publication_state.status),
            "publication_candidate_logical_plan_digest": str(publication_state.logical_plan_digest),
            "publication_candidate_planning_request": dict(publication_metadata.get("planning_request", {})),
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
        tmp_path = run_root / f"rank{rank}.json.tmp"
        final_path = run_root / f"rank{rank}.json"
        cleanup = _close_runtime_services(runtime, adapter)
        summary["cleanup_errors"] = list(cleanup["cleanup_errors"])
        summary["planner_thread_alive"] = bool(cleanup["planner_thread_alive"])
        if dist.is_initialized():
            dist.destroy_process_group()
        summary["dist_destroyed"] = not dist.is_initialized()
        summary["dist_initialized_after_cleanup"] = bool(dist.is_initialized())
        if summary.get("status") == "passed":
            if summary["cleanup_errors"] or bool(summary["planner_thread_alive"]) or not bool(summary["dist_destroyed"]):
                summary["status"] = "failed"
                summary["error_type"] = "CleanupError"
                summary["error"] = "runtime cleanup incomplete"
        tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, final_path)


def _terminate_all_processes(context: mp.SpawnContext) -> None:
    for pid in context.pids():
        if int(pid) <= 0:
            continue
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            )
            continue
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except Exception:
                pass
    if os.name != "nt":
        time.sleep(2.0)
        for pid in context.pids():
            if int(pid) <= 0:
                continue
            try:
                os.killpg(int(pid), signal.SIGKILL)
            except Exception:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass


def run_gate(*, instrumentation_mode: str = "perf_light") -> dict[str, object]:
    return run_gate_with_backend(instrumentation_mode=instrumentation_mode, execution_backend="phase_sync")


def run_gate_with_backend(
    *,
    instrumentation_mode: str = "perf_light",
    execution_backend: str = "phase_sync",
    output_dir: str | Path = Path("outputs/closure/m123_integrated_publication_execution_gloo"),
    timeout_seconds: float = 180.0,
) -> dict[str, object]:
    world_size = 4
    port = _free_port()
    run_token = uuid.uuid4().hex
    run_root = Path(output_dir).resolve() / f"m123_{execution_backend}_{run_token}"
    if run_root.exists():
        raise FileExistsError(f"run dir already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    started_at = time.time()
    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_token,
                "run_dir": str(run_root),
                "execution_backend": str(execution_backend),
                "instrumentation_mode": str(instrumentation_mode),
                "world_size": int(world_size),
                "port": int(port),
                "started_at": started_at,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    context = mp.spawn(
        _worker,
        args=(world_size, port, str(instrumentation_mode), str(execution_backend), str(run_root)),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + float(timeout_seconds)
    timed_out = False
    while True:
        if context.join(timeout=1.0):
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate_all_processes(context)
            break
    if timed_out:
        raise TimeoutError(f"m123 gate timed out after {timeout_seconds:.1f}s")
    payloads = [json.loads((run_root / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(world_size)]
    assert all(item["status"] == "passed" for item in payloads), payloads
    summary = {
        "status": "passed",
        "run_id": run_token,
        "run_dir": str(run_root),
        "started_at": started_at,
        "finished_at": time.time(),
        "world_size": world_size,
        "instrumentation_mode": str(instrumentation_mode),
        "execution_backend": str(execution_backend),
        "p0_matrix": [list(row) for row in _matrix_for_world_size(world_size)],
        "p1_matrix": [list(row) for row in tuple(tuple(_matrix_for_world_size(world_size)[src][dst] for src in range(world_size)) for dst in range(world_size))],
        "ranks": payloads,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-backend", default=str(os.environ.get("RS_M123_GATE_EXECUTION_BACKEND", "phase_sync") or "phase_sync"))
    parser.add_argument("--instrumentation-mode", default=str(os.environ.get("RS_M123_GATE_INSTRUMENTATION_MODE", "perf_light") or "perf_light"))
    parser.add_argument("--output-dir", default="outputs/closure/m123_integrated_publication_execution_gloo")
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    summary = run_gate_with_backend(
        instrumentation_mode=str(args.instrumentation_mode),
        execution_backend=str(args.execution_backend),
        output_dir=str(args.output_dir),
    )
    summary_digest = hashlib.sha256(json.dumps(summary, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()
    if str(args.summary_path).strip():
        Path(str(args.summary_path)).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.quiet:
        print(
            json.dumps(
                {
                    "status": str(summary["status"]),
                    "execution_backend": str(summary["execution_backend"]),
                    "summary_path": str(Path(str(args.summary_path)).resolve()) if str(args.summary_path).strip() else str(Path(summary["run_dir"]) / "summary.json"),
                    "summary_digest": summary_digest,
                    "duration_seconds": round(float(summary["finished_at"]) - float(summary["started_at"]), 3),
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
