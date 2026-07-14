from __future__ import annotations

import json
import os
import socket
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rs.core.contracts.execution import ActualPhaseContext, ExecutionContext
from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.execution import PayloadInvocation, PhaseSyncExecutor, P2PReleaseExecutor, RuntimeExecutionPipeline
from tests.contract.megatron_ep.helpers import make_phase_context_generic


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _matrix_for_group(group_ranks: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if len(group_ranks) == 4:
        return ((1, 2, 0, 0), (0, 1, 3, 0), (0, 0, 1, 1), (4, 0, 0, 1))
    if len(group_ranks) == 2:
        return ((1, 3), (2, 1))
    raise ValueError(f"unsupported group size {group_ranks!r}")


def _window_plan_from_matrices(group_ranks: tuple[int, ...], phase: str, matrices: tuple[tuple[tuple[int, ...], ...], ...]) -> WindowPlan:
    flow_phase = {"P0": "p0_dispatch", "P1": "p1_return"}[phase]
    return WindowPlan(
        planner_id="m2-formal",
        planner_family="joint",
        request_digest=f"{group_ranks[0]}->{group_ranks[-1]}:{phase}",
        waves=tuple(
            PlanWave(
                wave_id=wave_index,
                flows=tuple(
                    PlannedFlow(
                        flow_id=f"{flow_phase}_{wave_index}_{src_index}_{dst_index}",
                        phase=flow_phase,
                        src_rank=int(src_index),
                        dst_rank=int(dst_index),
                        row_count=int(rows),
                        release_state="ready",
                        executable=True,
                    )
                    for src_index, row in enumerate(matrix)
                    for dst_index, rows in enumerate(row)
                    if int(rows) > 0
                ),
                estimated_duration=float(sum(int(rows) for row in matrix for rows in row)),
            )
            for wave_index, matrix in enumerate(matrices)
        ),
        metadata={"source_layer_id": "0", "target_layer_id": "1"},
    )


def _window_plan(group_ranks: tuple[int, ...], phase: str) -> WindowPlan:
    matrix = _matrix_for_group(group_ranks)
    remote = tuple(tuple(0 if src == dst else int(value) for dst, value in enumerate(row)) for src, row in enumerate(matrix))
    local = tuple(tuple(int(value) if src == dst else 0 for dst, value in enumerate(row)) for src, row in enumerate(matrix))
    return _window_plan_from_matrices(group_ranks, phase, (remote, local))


def _context_for_rank(rank: int, *, group_ranks: tuple[int, ...], phase: str, matrix: tuple[tuple[int, ...], ...] | None = None):
    matrix = matrix or _matrix_for_group(group_ranks)
    group_index = group_ranks.index(int(rank))
    row = tuple(int(value) for value in matrix[group_index])
    col = tuple(int(matrix[src][group_index]) for src in range(len(matrix)))
    if phase == "P0":
        input_splits = row
        output_splits = col
    else:
        input_splits = col
        output_splits = row
    return make_phase_context_generic(
        rank=int(rank),
        phase=str(phase),
        input_splits=input_splits,
        output_splits=output_splits,
        ep_group_ranks=group_ranks,
        p2_hint_mode="none",
    )


def _actual_phase_context(rank: int, *, phase_context) -> ActualPhaseContext:
    return ActualPhaseContext(
        layer_id=str(phase_context.layer_id),
        phase=str(phase_context.phase),
        world_size=len(phase_context.ep_group_ranks),
        rank_space="global",
        layout_digest=str(phase_context.canonical_receive_layout_id),
        metadata={"phase_ready_context": phase_context.to_dict()},
    )


def _input_tensor(spec, *, source_global_rank: int) -> torch.Tensor:
    rows = int(spec.row_count)
    hidden = int(spec.shape_suffix[0]) if spec.shape_suffix else 1
    base = int(source_global_rank) * 10000
    values = torch.arange(base, base + max(rows, 1), dtype=torch.float32)
    if hidden <= 1:
        return values[:rows].to(dtype=torch.float16).reshape(rows, 1)
    return values[:rows].unsqueeze(1).repeat(1, hidden).to(dtype=torch.float16)


def _expected_output_tensor(
    *,
    local_global_rank: int,
    group_ranks: tuple[int, ...],
    phase: str,
    payload_role: str,
    shape_suffix: tuple[int, ...],
    matrix: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    matrix = matrix or _matrix_for_group(group_ranks)
    local_group_rank = group_ranks.index(int(local_global_rank))
    if phase == "P0":
        incoming_rows_by_peer = [int(matrix[src][local_group_rank]) for src in range(len(group_ranks))]
    else:
        incoming_rows_by_peer = [int(matrix[local_group_rank][dst]) for dst in range(len(group_ranks))]
    width = int(shape_suffix[0]) if shape_suffix else 1
    rows: list[torch.Tensor] = []
    for src_group_rank, row_count in enumerate(incoming_rows_by_peer):
        if row_count <= 0:
            continue
        source_global_rank = int(group_ranks[src_group_rank])
        source_peer_base = int(sum(int(matrix[src_group_rank][peer]) for peer in range(local_group_rank)))
        values = torch.arange(
            source_global_rank * 10000 + source_peer_base,
            source_global_rank * 10000 + source_peer_base + row_count,
            dtype=torch.float32,
        )
        if width <= 1:
            rows.append(values.reshape(row_count, 1).to(dtype=torch.float16))
        else:
            rows.append(values.unsqueeze(1).repeat(1, width).to(dtype=torch.float16))
    if not rows:
        return torch.zeros((0, max(width, 1)), dtype=torch.float16)
    return torch.cat(rows, dim=0)


def _serializable_outcome(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    tensor = result.get("output_payload")
    if isinstance(tensor, torch.Tensor):
        result["output_payload"] = {
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": str(tensor.dtype),
            "sum": float(tensor.float().sum().item()),
            "rows": tensor.float().tolist(),
        }
    return result


def _jsonify(value):
    if isinstance(value, torch.Tensor):
        return value.float().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _execute_for_rank(rank: int, *, group_ranks: tuple[int, ...], phase: str, process_group):
    matrix = _matrix_for_group(group_ranks)
    phase_context = _context_for_rank(rank, group_ranks=group_ranks, phase=phase, matrix=matrix)
    published = CanonicalPlanPublisher(
        rank_map=RankMap(group_ranks=group_ranks, root_rank=group_ranks[0])
    ).build(
        publication_slot={
            "run_id": "m2-gloo",
            "forward_generation": 0,
            "microbatch_id": "mb",
            "source_layer_id": "0",
            "target_layer_id": "1",
            "planning_slot": f"{group_ranks[0]}->{group_ranks[-1]}",
        },
        window_plan=_window_plan(group_ranks, phase),
    )
    pipeline = RuntimeExecutionPipeline()
    prepared = pipeline.prepare(published, _actual_phase_context(rank, phase_context=phase_context))
    results = {}
    for spec in prepared.materialized_plan.payload_specs:
        tensor = _input_tensor(spec, source_global_rank=int(rank))
        expected_output = _expected_output_tensor(
            local_global_rank=int(rank),
            group_ranks=group_ranks,
            phase=phase,
            payload_role=str(spec.payload_role),
            shape_suffix=tuple(int(dim) for dim in spec.shape_suffix),
        )
        sync_outcome = PhaseSyncExecutor().execute(
            plan=prepared.materialized_plan,
            invocation=PayloadInvocation(
                run_id="m2-gloo",
                forward_generation=0,
                layer_id=str(phase_context.layer_id),
                phase=str(phase_context.phase),
                payload_role=str(spec.payload_role),
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=str(spec.dtype),
                layout_digest=str(prepared.materialized_plan.layout_digest),
                invocation_id=f"sync:{rank}:{spec.payload_role}",
                input_tensor=tensor,
                process_group=process_group,
            ),
            context=ExecutionContext(
                run_id="m2-gloo",
                forward_generation=0,
                layer_id=str(phase_context.layer_id),
                phase=str(phase_context.phase),
                rank_space="global",
            ),
        )
        p2p_outcome = P2PReleaseExecutor().execute(
            plan=prepared.materialized_plan,
            invocation=PayloadInvocation(
                run_id="m2-gloo",
                forward_generation=0,
                layer_id=str(phase_context.layer_id),
                phase=str(phase_context.phase),
                payload_role=str(spec.payload_role),
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=str(spec.dtype),
                layout_digest=str(prepared.materialized_plan.layout_digest),
                invocation_id=f"p2p:{rank}:{spec.payload_role}",
                input_tensor=tensor,
                process_group=process_group,
            ),
            context=ExecutionContext(
                run_id="m2-gloo",
                forward_generation=0,
                layer_id=str(phase_context.layer_id),
                phase=str(phase_context.phase),
                rank_space="global",
                metadata={"max_inflight_batches": 2},
            ),
        )
        sync_tensor = sync_outcome.output_payload
        p2p_tensor = p2p_outcome.output_payload
        if not isinstance(sync_tensor, torch.Tensor) or not isinstance(p2p_tensor, torch.Tensor):
            raise AssertionError("executors must return tensor outputs")
        if not torch.equal(sync_tensor.cpu(), expected_output.cpu()):
            raise AssertionError(
                f"sync output mismatch for rank={rank} role={spec.payload_role}: "
                f"expected={expected_output.tolist()} actual={sync_tensor.tolist()}"
            )
        if not torch.equal(p2p_tensor.cpu(), expected_output.cpu()):
            raise AssertionError(
                f"p2p output mismatch for rank={rank} role={spec.payload_role}: "
                f"expected={expected_output.tolist()} actual={p2p_tensor.tolist()}"
            )
        if not torch.equal(sync_tensor.cpu(), p2p_tensor.cpu()):
            raise AssertionError(f"sync/p2p divergence for rank={rank} role={spec.payload_role}")
        if sync_outcome.details.get("distributed_operation_count", 0) <= 0:
            raise AssertionError(f"sync distributed_operation_count must be > 0 for rank={rank} role={spec.payload_role}")
        if p2p_outcome.details.get("distributed_operation_count", 0) <= 0:
            raise AssertionError(f"p2p distributed_operation_count must be > 0 for rank={rank} role={spec.payload_role}")
        if len(prepared.materialized_plan.batches) >= 2 and int(p2p_outcome.details.get("peak_inflight_batches", 0)) < 2:
            raise AssertionError(
                f"expected peak_inflight_batches >= 2 for rank={rank} role={spec.payload_role}, "
                f"got {p2p_outcome.details.get('peak_inflight_batches', 0)}"
            )
        results[str(spec.payload_role)] = {
            "sync": _serializable_outcome(sync_outcome.to_dict()),
            "p2p": _serializable_outcome(p2p_outcome.to_dict()),
            "expected_output": expected_output.float().tolist(),
        }
    return {
        "rank": int(rank),
        "group_ranks": list(group_ranks),
        "phase": str(phase),
        "published_plan_digest": str(published.published_plan_digest),
        "logical_plan_digest": str(published.logical_plan_digest),
        "materialized_plan_digest": str(prepared.materialized_plan.materialized_plan_digest),
        "validation": prepared.validation.to_dict(),
        "results": results,
    }


def _execute_sparse_for_rank(
    rank: int,
    *,
    group_ranks: tuple[int, ...],
    phase: str,
    matrices: tuple[tuple[tuple[int, ...], ...], ...],
    process_group,
):
    aggregate = tuple(
        tuple(sum(int(matrices[wave_index][src][dst]) for wave_index in range(len(matrices))) for dst in range(len(group_ranks)))
        for src in range(len(group_ranks))
    )
    phase_context = _context_for_rank(rank, group_ranks=group_ranks, phase=phase, matrix=aggregate)
    published = CanonicalPlanPublisher(
        rank_map=RankMap(group_ranks=group_ranks, root_rank=group_ranks[0])
    ).build(
        publication_slot={
            "run_id": "m2-gloo-sparse",
            "forward_generation": 0,
            "microbatch_id": "mb",
            "source_layer_id": "0",
            "target_layer_id": "1",
            "planning_slot": f"sparse:{group_ranks[0]}->{group_ranks[-1]}",
        },
        window_plan=_window_plan_from_matrices(group_ranks, phase, matrices),
    )
    prepared = RuntimeExecutionPipeline().prepare(published, _actual_phase_context(rank, phase_context=phase_context))
    spec = prepared.materialized_plan.payload_specs[0]
    tensor = _input_tensor(spec, source_global_rank=int(rank))
    expected_output = _expected_output_tensor(
        local_global_rank=int(rank),
        group_ranks=group_ranks,
        phase=phase,
        payload_role=str(spec.payload_role),
        shape_suffix=tuple(int(dim) for dim in spec.shape_suffix),
        matrix=aggregate,
    )
    sync_outcome = PhaseSyncExecutor().execute(
        plan=prepared.materialized_plan,
        invocation=PayloadInvocation(
            run_id="m2-gloo-sparse",
            forward_generation=0,
            layer_id=str(phase_context.layer_id),
            phase=str(phase_context.phase),
            payload_role=str(spec.payload_role),
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=str(spec.dtype),
            layout_digest=str(prepared.materialized_plan.layout_digest),
            invocation_id=f"sparse-sync:{rank}:{spec.payload_role}",
            input_tensor=tensor,
            process_group=process_group,
        ),
        context=ExecutionContext(
            run_id="m2-gloo-sparse",
            forward_generation=0,
            layer_id=str(phase_context.layer_id),
            phase=str(phase_context.phase),
            rank_space="global",
        ),
    )
    expected_collectives = int(len(matrices))
    if int(sync_outcome.details.get("collective_round_count", -1)) != expected_collectives:
        raise AssertionError(
            f"sparse collective round mismatch rank={rank}: expected={expected_collectives} "
            f"actual={sync_outcome.details.get('collective_round_count')}"
        )
    if sync_outcome.success is not True:
        raise AssertionError(f"sparse sync outcome failed rank={rank}: {sync_outcome.to_dict()}")
    if not isinstance(sync_outcome.output_payload, torch.Tensor) or not torch.equal(sync_outcome.output_payload.cpu(), expected_output.cpu()):
        raise AssertionError(
            f"sparse sync output mismatch rank={rank}: "
            f"expected={expected_output.tolist()} actual="
            f"{sync_outcome.output_payload.tolist() if isinstance(sync_outcome.output_payload, torch.Tensor) else sync_outcome.output_payload}"
        )
    return {
        "rank": int(rank),
        "group_ranks": list(group_ranks),
        "phase": str(phase),
        "expected_collective_round_count": int(expected_collectives),
        "materialized_plan_digest": str(prepared.materialized_plan.materialized_plan_digest),
        "sync": _serializable_outcome(sync_outcome.to_dict()),
    }


def _validate_global_results(summary: dict[str, object]) -> None:
    all_ranks = summary["all_ranks"]
    full_group_rounds = [
        int(rank_payload["full_group"]["results"]["hidden_states"]["sync"]["details"]["collective_round_count"])
        for rank_payload in all_ranks
    ]
    if any(value <= 0 for value in full_group_rounds):
        raise AssertionError(f"full_group collective rounds invalid: {full_group_rounds}")
    sparse_one = [int(rank_payload["sparse_one_wave"]["sync"]["details"]["collective_round_count"]) for rank_payload in all_ranks]
    sparse_two = [int(rank_payload["sparse_two_wave"]["sync"]["details"]["collective_round_count"]) for rank_payload in all_ranks]
    if sparse_one != [1, 1, 1, 1]:
        raise AssertionError(f"sparse_one_wave collective rounds mismatch: {sparse_one}")
    if sparse_two != [2, 2, 2, 2]:
        raise AssertionError(f"sparse_two_wave collective rounds mismatch: {sparse_two}")


def _worker(rank: int, world_size: int, master_port: int, out_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{int(master_port)}",
        rank=rank,
        world_size=world_size,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    subgroup = dist.new_group(ranks=[2, 3], backend="gloo")
    try:
        local = {
            "rank": int(rank),
            "full_group": _execute_for_rank(rank, group_ranks=(0, 1, 2, 3), phase="P0", process_group=dist.group.WORLD),
            "sparse_one_wave": _execute_sparse_for_rank(
                rank,
                group_ranks=(0, 1, 2, 3),
                phase="P0",
                matrices=(((0, 4, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),),
                process_group=dist.group.WORLD,
            ),
            "sparse_two_wave": _execute_sparse_for_rank(
                rank,
                group_ranks=(0, 1, 2, 3),
                phase="P0",
                matrices=(
                    ((0, 4, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 5), (0, 0, 0, 0)),
                ),
                process_group=dist.group.WORLD,
            ),
        }
        dist.barrier()
        if rank in {2, 3}:
            local["subgroup"] = _execute_for_rank(rank, group_ranks=(2, 3), phase="P0", process_group=subgroup)
        else:
            local["subgroup"] = {"rank": int(rank), "skipped": True}
        dist.barrier()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
        (out_path / f"rank{rank}_m2_formal_gloo.json").write_text(json.dumps(_jsonify(local), indent=2), encoding="utf-8")
        if rank == 0:
            summary = {
                "status": "passed",
                "world_size": int(world_size),
                "all_ranks": gathered,
            }
            _validate_global_results(summary)
            (out_path / "m2_formal_execution_gloo_summary.json").write_text(json.dumps(_jsonify(summary), indent=2), encoding="utf-8")
            print(json.dumps(_jsonify(summary), indent=2))
    except Exception as exc:
        failure = {
            "rank": int(rank),
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (out_path / f"rank{rank}_m2_formal_gloo_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    finally:
        dist.destroy_process_group(subgroup)
        dist.destroy_process_group()


def main() -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    port = _free_port()
    out_dir = "outputs/closure/m2_formal_execution_gloo"
    mp.spawn(_worker, args=(4, port, out_dir), nprocs=4, join=True)


if __name__ == "__main__":
    main()
