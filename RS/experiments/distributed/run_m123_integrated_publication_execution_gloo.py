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
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rs.runtime.observation.instrumentation import BufferedEvidenceSink, build_runtime_instrumentation
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.communication_lane import slot_from_request
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.execution.pipeline import build_runtime_execution_pipeline
from rs.runtime.online.megatron_ep.execution.transport_adapter import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.public_types import CombineCompleteEvent, CombineReadyEvent, DispatchCompleteEvent, DispatchReadyEvent

from experiments.distributed.run_m1_formal_lifecycle_publication_gloo import (
    _begin_forward,
    _end_forward,
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


def _json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    payload = {
        "dtype": str(tensor.dtype),
        "shape": [int(dim) for dim in tensor.shape],
        "bytes": tensor.detach().contiguous().cpu().view(torch.uint8).tolist(),
    }
    return _json_digest(payload)


def _tensor_parity(expected: torch.Tensor, observed: torch.Tensor, *, atol: float = 1e-2, rtol: float = 1e-2) -> dict[str, object]:
    expected_cpu = expected.detach().float().cpu()
    observed_cpu = observed.detach().float().cpu()
    abs_diff = (observed_cpu - expected_cpu).abs()
    rel_diff = abs_diff / expected_cpu.abs().clamp_min(1e-12)
    mismatch = abs_diff > (atol + rtol * expected_cpu.abs())
    mismatch_indices = mismatch.nonzero(as_tuple=False)
    first_index = None if int(mismatch_indices.shape[0]) == 0 else [int(value) for value in mismatch_indices[0].tolist()]
    return {
        "allclose": bool(torch.allclose(observed_cpu, expected_cpu, atol=atol, rtol=rtol)),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_error": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "max_rel_error": float(rel_diff.max().item()) if rel_diff.numel() else 0.0,
        "mismatch_count": int(mismatch.sum().item()) if mismatch.numel() else 0,
        "first_mismatch_index": first_index,
        "nan_count": int(torch.isnan(observed_cpu).sum().item()),
        "inf_count": int(torch.isinf(observed_cpu).sum().item()),
        "reference_digest": _tensor_digest(expected),
        "executed_digest": _tensor_digest(observed),
    }


def _matrix_for_world_size(world_size: int) -> tuple[tuple[int, ...], ...]:
    if int(world_size) == 2:
        return ((0, 4), (3, 0))
    if int(world_size) == 4:
        return ((0, 4, 0, 2), (1, 0, 3, 0), (0, 2, 0, 5), (4, 0, 1, 0))
    raise ValueError(f"unsupported world_size {world_size!r}")


def _load_matrix_bundle(*, matrix_bundle_path: str, world_size: int) -> dict[str, tuple[tuple[int, ...], ...]]:
    if not str(matrix_bundle_path).strip():
        p0 = _matrix_for_world_size(world_size)
        p1 = tuple(tuple(int(p0[src][dst]) for src in range(world_size)) for dst in range(world_size))
        return {"p0": p0, "p1": p1, "full_p0": p0, "full_p1": p1}
    payload = json.loads(Path(str(matrix_bundle_path)).read_text(encoding="utf-8"))
    p0 = tuple(tuple(int(value) for value in row) for row in payload["p0_matrix"])
    p1 = tuple(tuple(int(value) for value in row) for row in payload["p1_matrix"])
    full_p0 = tuple(tuple(int(value) for value in row) for row in payload.get("full_p0_matrix", payload["p0_matrix"]))
    full_p1 = tuple(tuple(int(value) for value in row) for row in payload.get("full_p1_matrix", payload["p1_matrix"]))
    if len(p0) != int(world_size) or len(p1) != int(world_size):
        raise ValueError("matrix bundle world_size mismatch")
    return {"p0": p0, "p1": p1, "full_p0": full_p0, "full_p1": full_p1}


def _spec_for_role(payload_specs_by_role: dict[str, dict[str, object]], role: str) -> object:
    spec = dict(payload_specs_by_role.get(role, {}))
    return type("Spec", (), {"row_count": int(spec.get("row_count", 0) or 0), "dtype": str(spec.get("dtype", "torch.float32")), "shape_suffix": tuple(int(dim) for dim in spec.get("shape_suffix", ()))})()


def _build_full_reference_tensor(
    *,
    matrix: tuple[tuple[int, ...], ...],
    payload_specs_by_rank: dict[int, dict[str, object]],
    dst_rank: int,
    tensor_role: str,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for src_rank in range(len(matrix)):
        row_count = int(matrix[src_rank][dst_rank])
        if row_count <= 0:
            continue
        spec = _spec_for_role(dict(payload_specs_by_rank.get(int(src_rank), {})), tensor_role)
        source_tensor = _input_tensor(spec, source_global_rank=int(src_rank))
        rows.append(source_tensor.narrow(0, 0, row_count).clone())
    if not rows:
        spec = _spec_for_role(dict(payload_specs_by_rank.get(int(dst_rank), {})), tensor_role)
        empty = _input_tensor(spec, source_global_rank=int(dst_rank))
        return empty.narrow(0, 0, 0).clone()
    return torch.cat(rows, dim=0)


def _reconstruct_full_tensor_from_remote_output(
    *,
    remote_output: torch.Tensor,
    full_matrix: tuple[tuple[int, ...], ...],
    remote_matrix: tuple[tuple[int, ...], ...],
    payload_specs_by_rank: dict[int, dict[str, object]],
    dst_rank: int,
    tensor_role: str,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    remote_offset = 0
    for src_rank in range(len(full_matrix)):
        full_rows = int(full_matrix[src_rank][dst_rank])
        if full_rows <= 0:
            continue
        if src_rank == dst_rank:
            spec = _spec_for_role(dict(payload_specs_by_rank.get(int(src_rank), {})), tensor_role)
            source_tensor = _input_tensor(spec, source_global_rank=int(src_rank))
            chunks.append(source_tensor.narrow(0, 0, full_rows).clone())
            continue
        remote_rows = int(remote_matrix[src_rank][dst_rank])
        if remote_rows <= 0:
            continue
        chunks.append(remote_output.narrow(0, remote_offset, remote_rows).clone())
        remote_offset += remote_rows
    if not chunks:
        spec = _spec_for_role(dict(payload_specs_by_rank.get(int(dst_rank), {})), tensor_role)
        empty = _input_tensor(spec, source_global_rank=int(dst_rank))
        return empty.narrow(0, 0, 0).clone()
    return torch.cat(chunks, dim=0)


class _FakeDispatcher:
    def __init__(self, *, input_splits: tuple[int, ...], output_splits: tuple[int, ...]) -> None:
        self.input_splits = tuple(int(v) for v in input_splits)
        self.output_splits = tuple(int(v) for v in output_splits)
        self.tokens_per_expert = self.input_splits

    def _maybe_dtoh_and_synchronize(self, _stage: str, tokens_per_expert):
        return tokens_per_expert


def _dispatcher_for_phase_from_matrices(
    *,
    rank: int,
    group_ranks: tuple[int, ...],
    phase: str,
    p0_matrix: tuple[tuple[int, ...], ...],
    p1_matrix: tuple[tuple[int, ...], ...],
) -> tuple[_FakeDispatcher, torch.Tensor, torch.Tensor | None]:
    matrix = p0_matrix
    local_group_rank = tuple(int(v) for v in group_ranks).index(int(rank))
    row = tuple(int(value) for value in matrix[local_group_rank])
    col = tuple(int(matrix[src][local_group_rank]) for src in range(len(matrix)))
    if phase == "P0":
        hidden_rows = int(sum(row))
        probs_rows = int(sum(row))
        return (
            _FakeDispatcher(input_splits=row, output_splits=col),
            torch.zeros((hidden_rows, 4), dtype=torch.float32),
            torch.zeros((probs_rows, 1), dtype=torch.float32),
        )
    if phase == "P1":
        hidden_rows = int(sum(row))
        return (
            _FakeDispatcher(input_splits=row, output_splits=col),
            torch.zeros((hidden_rows, 4), dtype=torch.float32),
            None,
        )
    raise ValueError(f"unsupported phase {phase!r}")


def _emit_source_events_from_matrices(
    runtime,
    *,
    rank: int,
    group_ranks: tuple[int, ...],
    p0_matrix: tuple[tuple[int, ...], ...],
    p1_matrix: tuple[tuple[int, ...], ...],
) -> object:
    source_dispatcher, source_hidden, source_probs = _dispatcher_for_phase_from_matrices(
        rank=rank,
        group_ranks=group_ranks,
        phase="P0",
        p0_matrix=p0_matrix,
        p1_matrix=p1_matrix,
    )
    source_combine_dispatcher, source_combine_hidden, _ = _dispatcher_for_phase_from_matrices(
        rank=rank,
        group_ranks=group_ranks,
        phase="P1",
        p0_matrix=p0_matrix,
        p1_matrix=p1_matrix,
    )
    runtime.handle(
        DispatchReadyEvent(
            layer_name="model.layers.0.mlp",
            dispatcher=source_dispatcher,
            packed_hidden_states=source_hidden,
            packed_probs=source_probs,
            layer_role="prediction_source",
        )
    )
    runtime.handle(
        DispatchCompleteEvent(
            layer_name="model.layers.0.mlp",
            dispatcher=source_dispatcher,
            packed_hidden_states=source_hidden,
            result=(source_hidden.clone(), source_probs.clone() if source_probs is not None else None),
            layer_role="prediction_source",
        )
    )
    runtime.handle(
        CombineReadyEvent(
            layer_name="model.layers.0.mlp",
            dispatcher=source_combine_dispatcher,
            packed_hidden_states=source_combine_hidden,
        )
    )
    runtime.handle(
        CombineCompleteEvent(
            layer_name="model.layers.0.mlp",
            dispatcher=source_combine_dispatcher,
            packed_hidden_states=source_combine_hidden,
            result=source_combine_hidden.clone(),
        )
    )
    return slot_from_request(
        run_id=str(runtime.run_id),
        forward_generation=int(runtime._forward_epoch),  # noqa: SLF001
        microbatch_id=str(runtime.microbatch_id),
        source_layer_id="0",
        target_layer_id="1",
    )


def _materialized_task_descriptor(
    slice_dict: dict[str, object],
    *,
    phase: str,
    wave_id: int,
    payload_specs_by_role: dict[str, dict[str, object]],
) -> dict[str, object]:
    role = str(slice_dict["payload_role"])
    spec = dict(payload_specs_by_role.get(role, {}))
    return {
        "task_id": str(slice_dict["task_id"]),
        "phase": str(phase),
        "wave_id": int(wave_id),
        "flow_id": str(slice_dict["flow_id"]),
        "payload_role": role,
        "src_rank": int(slice_dict["src_global_rank"]),
        "dst_rank": int(slice_dict["dst_global_rank"]),
        "src_group_rank": int(slice_dict["src_group_rank"]),
        "dst_group_rank": int(slice_dict["dst_group_rank"]),
        "row_count": int(slice_dict["row_count"]),
        "dtype": str(spec.get("dtype", "")),
        "shape_suffix": list(spec.get("shape_suffix", [])),
        "send_offset_rows": int(slice_dict["send_offset_rows"]),
        "recv_offset_rows": int(slice_dict["recv_offset_rows"]),
        "physical_send_offset_rows": int(slice_dict["physical_send_offset_rows"]),
        "physical_recv_offset_rows": int(slice_dict["physical_recv_offset_rows"]),
        "peer_send_offset_rows": int(slice_dict.get("peer_send_offset_rows", 0) or 0),
        "peer_recv_offset_rows": int(slice_dict.get("peer_recv_offset_rows", 0) or 0),
        "dependency_ids": list(slice_dict.get("dependency_ids", [])),
        "transfer_tag": int(slice_dict.get("transfer_tag", 0) or 0),
    }


def _manifest_digest(rows: list[dict[str, object]]) -> str:
    canonical = sorted(
        rows,
        key=lambda item: (
            str(item["task_id"]),
            str(item["payload_role"]),
            int(item["src_rank"]),
            int(item["dst_rank"]),
            int(item["wave_id"]),
        ),
    )
    return _json_digest(canonical)


def _id_set_digest(values: list[str] | tuple[str, ...] | set[str]) -> str:
    return _json_digest(sorted(str(value) for value in values))


def _sequence_digest(values: list[str] | tuple[str, ...]) -> str:
    return _json_digest([str(value) for value in values])


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
) -> dict[str, object]:
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
    return {
        "output": output,
        "expected": expected,
        "parity": _tensor_parity(expected, output),
    }


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


def _worker(
    rank: int,
    world_size: int,
    port: int,
    instrumentation_mode: str,
    execution_backend: str,
    run_dir: str,
    policy_name: str,
    matrix_bundle_path: str,
) -> None:
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
        matrices = _load_matrix_bundle(matrix_bundle_path=matrix_bundle_path, world_size=world_size)
        object.__setattr__(runtime.config, "policy", str(policy_name))
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
        slot = _emit_source_events_from_matrices(
            runtime,
            rank=rank,
            group_ranks=group_ranks,
            p0_matrix=matrices["p0"],
            p1_matrix=matrices["p1"],
        )
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

        p0_dispatcher, p0_hidden, p0_probs = _dispatcher_for_phase_from_matrices(
            rank=rank,
            group_ranks=group_ranks,
            phase="P0",
            p0_matrix=matrices["p0"],
            p1_matrix=matrices["p1"],
        )
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
        hidden_result = _execute_role(runtime=runtime, published_plan=published_plan, source_payload_specs_by_rank=source_payload_specs_by_rank, source_slice_oracle_by_rank=p0_slice_oracle_by_rank, adapter=adapter, dispatcher=p0_dispatcher, tensor_role="hidden_states", phase="P0", rank=rank, group_ranks=group_ranks)
        release_after_hidden = tuple(sorted(runtime.release_state_ledger.satisfied_release_ids))
        probs_result = _execute_role(runtime=runtime, published_plan=published_plan, source_payload_specs_by_rank=source_payload_specs_by_rank, source_slice_oracle_by_rank=p0_slice_oracle_by_rank, adapter=adapter, dispatcher=p0_dispatcher, tensor_role="routing_probs", phase="P0", rank=rank, group_ranks=group_ranks)
        release_after_probs = tuple(sorted(runtime.release_state_ledger.satisfied_release_ids))
        runtime.handle(
            DispatchCompleteEvent(
                layer_name="model.layers.1.mlp",
                dispatcher=p0_dispatcher,
                packed_hidden_states=p0_hidden,
                result=(hidden_result["output"], probs_result["output"]),
                layer_role="selected",
            )
        )

        p1_dispatcher, p1_hidden, _ = _dispatcher_for_phase_from_matrices(
            rank=rank,
            group_ranks=group_ranks,
            phase="P1",
            p0_matrix=matrices["p0"],
            p1_matrix=matrices["p1"],
        )
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
        p1_result = _execute_role(runtime=runtime, published_plan=published_plan, source_payload_specs_by_rank=p1_source_payload_specs_by_rank, source_slice_oracle_by_rank=p1_slice_oracle_by_rank, adapter=adapter, dispatcher=p1_dispatcher, tensor_role="hidden_states", phase="P1", rank=rank, group_ranks=group_ranks)
        runtime.handle(
            CombineCompleteEvent(
                layer_name="model.layers.1.mlp",
                dispatcher=p1_dispatcher,
                packed_hidden_states=p1_hidden,
                result=p1_result["output"],
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
        submitted_task_ids = tuple(
            str(task_id)
            for item in all_outcomes
            for task_id in tuple(str(task_id) for task_id in item.get("submitted_task_ids", ()))
        )
        completed_task_ids = tuple(
            str(task_id)
            for item in all_outcomes
            for task_id in tuple(str(task_id) for task_id in item.get("completed_task_ids", ()))
        )
        unresolved_task_ids = tuple(
            str(task_id)
            for item in all_outcomes
            for task_id in tuple(str(task_id) for task_id in item.get("unresolved_task_ids", ()))
        )
        submitted_task_count = int(len(submitted_task_ids))
        completed_task_count = int(len(completed_task_ids))
        unresolved_task_count = int(len(unresolved_task_ids))
        fallback_count = int(getattr(adapter, "phase_sync_fallback_count", 0) or 0)
        fallback_reasons = [] if fallback_count == 0 else ["phase_sync_fallback"]
        native_fallback_invoked = bool(fallback_count > 0)
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
            "submitted_task_ids": list(submitted_task_ids),
            "completed_task_ids": list(completed_task_ids),
            "unresolved_task_ids": list(unresolved_task_ids),
            "submitted_task_id_set_digest": _id_set_digest(submitted_task_ids),
            "completed_task_id_set_digest": _id_set_digest(completed_task_ids),
            "unresolved_task_id_set_digest": _id_set_digest(unresolved_task_ids),
            "submit_sequence_digest": _sequence_digest(submitted_task_ids),
            "completion_sequence_digest": _sequence_digest(completed_task_ids),
            "fallback_count": fallback_count,
            "fallback_reasons": list(fallback_reasons),
            "native_fallback_invoked": native_fallback_invoked,
            "descriptor_source": "materialized_plan",
            "execution_identity_source": "formal_executor_outcome",
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
        rank_materialized_manifest = [
            _materialized_task_descriptor(
                slice_.to_dict(),
                phase=str(prepared_p0.materialized_plan.phase),
                wave_id=int(batch.wave_id),
                payload_specs_by_role=local_payload_specs,
            )
            for batch in prepared_p0.materialized_plan.batches
            for slice_ in batch.slices
        ] + [
            _materialized_task_descriptor(
                slice_.to_dict(),
                phase=str(prepared_p1.materialized_plan.phase),
                wave_id=int(batch.wave_id),
                payload_specs_by_role=local_p1_payload_specs,
            )
            for batch in prepared_p1.materialized_plan.batches
            for slice_ in batch.slices
        ]
        completed_ids = set(str(item) for item in (p0_completed_task_ids + p1_completed_task_ids))
        rank_executed_manifest = [row for row in rank_materialized_manifest if str(row["task_id"]) in completed_ids]
        full_p0_hidden_reference = _build_full_reference_tensor(
            matrix=matrices["full_p0"],
            payload_specs_by_rank=source_payload_specs_by_rank,
            dst_rank=int(rank),
            tensor_role="hidden_states",
        )
        full_p0_hidden_executed = _reconstruct_full_tensor_from_remote_output(
            remote_output=hidden_result["output"],
            full_matrix=matrices["full_p0"],
            remote_matrix=matrices["p0"],
            payload_specs_by_rank=source_payload_specs_by_rank,
            dst_rank=int(rank),
            tensor_role="hidden_states",
        )
        full_p0_probs_reference = _build_full_reference_tensor(
            matrix=matrices["full_p0"],
            payload_specs_by_rank=source_payload_specs_by_rank,
            dst_rank=int(rank),
            tensor_role="routing_probs",
        )
        full_p0_probs_executed = _reconstruct_full_tensor_from_remote_output(
            remote_output=probs_result["output"],
            full_matrix=matrices["full_p0"],
            remote_matrix=matrices["p0"],
            payload_specs_by_rank=source_payload_specs_by_rank,
            dst_rank=int(rank),
            tensor_role="routing_probs",
        )
        full_p1_hidden_reference = _build_full_reference_tensor(
            matrix=matrices["full_p1"],
            payload_specs_by_rank=p1_source_payload_specs_by_rank,
            dst_rank=int(rank),
            tensor_role="hidden_states",
        )
        full_p1_hidden_executed = _reconstruct_full_tensor_from_remote_output(
            remote_output=p1_result["output"],
            full_matrix=matrices["full_p1"],
            remote_matrix=matrices["p1"],
            payload_specs_by_rank=p1_source_payload_specs_by_rank,
            dst_rank=int(rank),
            tensor_role="hidden_states",
        )
        parity_payload = {
            "p0_hidden": dict(hidden_result["parity"]),
            "p0_routing_probs": dict(probs_result["parity"]),
            "p1_hidden": dict(p1_result["parity"]),
            "traffic_to_execution_p0_hidden": _tensor_parity(full_p0_hidden_reference, full_p0_hidden_executed),
            "traffic_to_execution_p0_routing_probs": _tensor_parity(full_p0_probs_reference, full_p0_probs_executed),
            "traffic_to_execution_p1_hidden": _tensor_parity(full_p1_hidden_reference, full_p1_hidden_executed),
        }
        parity_payload["reference_p0_digest"] = _json_digest(
            {
                "hidden_states": parity_payload["p0_hidden"]["reference_digest"],
                "routing_probs": parity_payload["p0_routing_probs"]["reference_digest"],
            }
        )
        parity_payload["executed_p0_digest"] = _json_digest(
            {
                "hidden_states": parity_payload["p0_hidden"]["executed_digest"],
                "routing_probs": parity_payload["p0_routing_probs"]["executed_digest"],
            }
        )
        parity_payload["reference_p1_digest"] = str(parity_payload["p1_hidden"]["reference_digest"])
        parity_payload["executed_p1_digest"] = str(parity_payload["p1_hidden"]["executed_digest"])
        parity_payload["traffic_reference_final_digest"] = _json_digest(
            {
                "p0_hidden": parity_payload["traffic_to_execution_p0_hidden"]["reference_digest"],
                "p0_routing_probs": parity_payload["traffic_to_execution_p0_routing_probs"]["reference_digest"],
                "p1_hidden": parity_payload["traffic_to_execution_p1_hidden"]["reference_digest"],
            }
        )
        parity_payload["traffic_executed_final_digest"] = _json_digest(
            {
                "p0_hidden": parity_payload["traffic_to_execution_p0_hidden"]["executed_digest"],
                "p0_routing_probs": parity_payload["traffic_to_execution_p0_routing_probs"]["executed_digest"],
                "p1_hidden": parity_payload["traffic_to_execution_p1_hidden"]["executed_digest"],
            }
        )
        parity_payload["reference_final_digest"] = str(parity_payload["traffic_reference_final_digest"])
        parity_payload["executed_final_digest"] = str(parity_payload["traffic_executed_final_digest"])
        parity_payload["transport_materialized_slice_parity"] = {
            "allclose": bool(
                parity_payload["p0_hidden"]["allclose"]
                and parity_payload["p0_routing_probs"]["allclose"]
                and parity_payload["p1_hidden"]["allclose"]
            ),
            "reference_digest": _json_digest(
                {
                    "p0_hidden": parity_payload["p0_hidden"]["reference_digest"],
                    "p0_routing_probs": parity_payload["p0_routing_probs"]["reference_digest"],
                    "p1_hidden": parity_payload["p1_hidden"]["reference_digest"],
                }
            ),
            "executed_digest": _json_digest(
                {
                    "p0_hidden": parity_payload["p0_hidden"]["executed_digest"],
                    "p0_routing_probs": parity_payload["p0_routing_probs"]["executed_digest"],
                    "p1_hidden": parity_payload["p1_hidden"]["executed_digest"],
                }
            ),
        }
        parity_payload["traffic_to_execution_parity"] = {
            "allclose": bool(
                parity_payload["traffic_to_execution_p0_hidden"]["allclose"]
                and parity_payload["traffic_to_execution_p0_routing_probs"]["allclose"]
                and parity_payload["traffic_to_execution_p1_hidden"]["allclose"]
            ),
            "reference_digest": str(parity_payload["traffic_reference_final_digest"]),
            "executed_digest": str(parity_payload["traffic_executed_final_digest"]),
        }
        parity_payload["full_reconstruction_parity"] = dict(parity_payload["traffic_to_execution_parity"])
        parity_payload["allclose"] = bool(
            parity_payload["transport_materialized_slice_parity"]["allclose"]
            and parity_payload["traffic_to_execution_parity"]["allclose"]
        )
        parity_payload["status"] = "PASS" if parity_payload["allclose"] else "FAILED"
        (run_root / f"rank{rank}_materialized_task_manifest.json").write_text(
            json.dumps(
                {
                    "rank": int(rank),
                    "policy_name": str(policy_name),
                    "execution_backend": str(execution_backend),
                    "manifest": rank_materialized_manifest,
                    "manifest_digest": _manifest_digest(rank_materialized_manifest),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (run_root / f"rank{rank}_executed_task_manifest.json").write_text(
            json.dumps(
                {
                    "rank": int(rank),
                    "policy_name": str(policy_name),
                    "execution_backend": str(execution_backend),
                    "manifest": rank_executed_manifest,
                    "manifest_digest": _manifest_digest(rank_executed_manifest),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (run_root / f"rank{rank}_parity.json").write_text(json.dumps(parity_payload, indent=2, sort_keys=True), encoding="utf-8")
        summary["policy_name"] = str(policy_name)
        summary["matrix_bundle_path"] = str(matrix_bundle_path)
        summary["materialized_task_manifest_digest"] = _manifest_digest(rank_materialized_manifest)
        summary["executed_task_manifest_digest"] = _manifest_digest(rank_executed_manifest)
        summary["reference_final_digest"] = str(parity_payload["reference_final_digest"])
        summary["executed_final_digest"] = str(parity_payload["executed_final_digest"])
        summary["parity_status"] = str(parity_payload["status"])
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
    policy_name: str = "U_barrier_criticality_global_matching",
    matrix_bundle_path: str = "",
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
                "policy_name": str(policy_name),
                "matrix_bundle_path": str(matrix_bundle_path),
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
        args=(
            world_size,
            port,
            str(instrumentation_mode),
            str(execution_backend),
            str(run_root),
            str(policy_name),
            str(matrix_bundle_path),
        ),
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
    materialized_rank_rows = [
        json.loads((run_root / f"rank{rank}_materialized_task_manifest.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    executed_rank_rows = [
        json.loads((run_root / f"rank{rank}_executed_task_manifest.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    parity_rank_rows = [
        json.loads((run_root / f"rank{rank}_parity.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    materialized_manifest = [item for row in materialized_rank_rows for item in list(row.get("manifest", []))]
    executed_manifest = [item for row in executed_rank_rows for item in list(row.get("manifest", []))]
    materialized_manifest_digest = _manifest_digest(materialized_manifest)
    executed_manifest_digest = _manifest_digest(executed_manifest)
    reference_final_digest = _json_digest([row["reference_final_digest"] for row in parity_rank_rows])
    executed_final_digest = _json_digest([row["executed_final_digest"] for row in parity_rank_rows])
    parity_pass = all(bool(row.get("allclose", False)) for row in parity_rank_rows)
    all_submitted_task_ids = [str(task_id) for row in payloads for task_id in list(row.get("submitted_task_ids", []))]
    all_completed_task_ids = [str(task_id) for row in payloads for task_id in list(row.get("completed_task_ids", []))]
    all_unresolved_task_ids = [str(task_id) for row in payloads for task_id in list(row.get("unresolved_task_ids", []))]
    all_fallback_reasons = [str(reason) for row in payloads for reason in list(row.get("fallback_reasons", []))]
    native_fallback_invoked = any(bool(row.get("native_fallback_invoked", False)) for row in payloads)
    (run_root / "materialized_task_manifest.json").write_text(
        json.dumps(
            {
                "policy_name": str(policy_name),
                "execution_backend": str(execution_backend),
                "manifest_digest": materialized_manifest_digest,
                "manifest": materialized_manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_root / "executed_task_manifest.json").write_text(
        json.dumps(
            {
                "policy_name": str(policy_name),
                "execution_backend": str(execution_backend),
                "manifest_digest": executed_manifest_digest,
                "manifest": executed_manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_root / "parity.json").write_text(
        json.dumps(
            {
                "status": "PASS" if parity_pass else "FAILED",
                "rank_parity": parity_rank_rows,
                "reference_final_digest": reference_final_digest,
                "executed_final_digest": executed_final_digest,
                "allclose": parity_pass,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "passed",
        "run_id": run_token,
        "run_dir": str(run_root),
        "started_at": started_at,
        "finished_at": time.time(),
        "world_size": world_size,
        "instrumentation_mode": str(instrumentation_mode),
        "execution_backend": str(execution_backend),
        "policy_name": str(policy_name),
        "matrix_bundle_path": str(matrix_bundle_path),
        "p0_matrix": [list(row) for row in _load_matrix_bundle(matrix_bundle_path=matrix_bundle_path, world_size=world_size)["p0"]],
        "p1_matrix": [list(row) for row in _load_matrix_bundle(matrix_bundle_path=matrix_bundle_path, world_size=world_size)["p1"]],
        "materialized_task_manifest_path": str(run_root / "materialized_task_manifest.json"),
        "executed_task_manifest_path": str(run_root / "executed_task_manifest.json"),
        "parity_path": str(run_root / "parity.json"),
        "materialized_task_manifest_digest": materialized_manifest_digest,
        "executed_task_manifest_digest": executed_manifest_digest,
        "plan_identity_match": materialized_manifest_digest == executed_manifest_digest,
        "reference_final_digest": reference_final_digest,
        "executed_final_digest": executed_final_digest,
        "tensor_parity_pass": parity_pass,
        "submitted_task_id_set_digest": _id_set_digest(all_submitted_task_ids),
        "completed_task_id_set_digest": _id_set_digest(all_completed_task_ids),
        "unresolved_task_id_set_digest": _id_set_digest(all_unresolved_task_ids),
        "submit_sequence_digest": _sequence_digest(all_submitted_task_ids),
        "completion_sequence_digest": _sequence_digest(all_completed_task_ids),
        "fallback_count": int(sum(int(row.get("fallback_count", 0) or 0) for row in payloads)),
        "fallback_reasons": all_fallback_reasons,
        "native_fallback_invoked": native_fallback_invoked,
        "descriptor_source": "materialized_plan",
        "execution_identity_source": "formal_executor_outcome",
        "ranks": payloads,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-backend", default=str(os.environ.get("RS_M123_GATE_EXECUTION_BACKEND", "phase_sync") or "phase_sync"))
    parser.add_argument("--policy-name", default=str(os.environ.get("RS_M123_GATE_POLICY_NAME", "U_barrier_criticality_global_matching") or "U_barrier_criticality_global_matching"))
    parser.add_argument("--matrix-bundle", default=str(os.environ.get("RS_M123_GATE_MATRIX_BUNDLE", "") or ""))
    parser.add_argument("--instrumentation-mode", default=str(os.environ.get("RS_M123_GATE_INSTRUMENTATION_MODE", "perf_light") or "perf_light"))
    parser.add_argument("--output-dir", default="outputs/closure/m123_integrated_publication_execution_gloo")
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    summary = run_gate_with_backend(
        instrumentation_mode=str(args.instrumentation_mode),
        execution_backend=str(args.execution_backend),
        policy_name=str(args.policy_name),
        matrix_bundle_path=str(args.matrix_bundle),
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
