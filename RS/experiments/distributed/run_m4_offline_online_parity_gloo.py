from __future__ import annotations

import json
import os
import socket
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rs.core.contracts import ActualPhaseContext, EvaluationSpec, ExecutionContext, OfflineWindow, PredictionHint, PredictionIdentity, PredictionResult, TrafficProvenance
from rs.offline.parity import build_materialization_parity_case, build_planning_parity_case, expected_completed_task_ids
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.execution.api import GlooFunctionalExecutor, PayloadInvocation
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _window() -> OfflineWindow:
    matrix = (
        (0, 3, 0, 1),
        (2, 0, 1, 0),
        (0, 1, 0, 2),
        (1, 0, 2, 0),
    )
    return_matrix = (
        (0, 2, 0, 1),
        (3, 0, 1, 0),
        (0, 1, 0, 2),
        (1, 0, 2, 0),
    )
    p2_matrix = (
        (0, 2, 1, 0),
        (1, 0, 2, 1),
        (2, 0, 0, 1),
        (0, 1, 1, 0),
    )
    return OfflineWindow(
        window_identity="fixture:1->2",
        source_layer="1",
        target_layer="2",
        p0_actual=matrix,
        p1_actual=return_matrix,
        p2_actual=p2_matrix,
        placement_snapshot={"group_size": 4, "fixture_type": "offline_replay_smoke"},
        traffic_provenance=TrafficProvenance.ROUTE_RECONSTRUCTED,
        matrix_unit="rows",
        return_model="transpose_dispatch",
        raw_token_count=13,
        used_token_count=13,
        dropped_token_count=0,
        drop_reason=None,
        trace_digest="fixture-trace-layer1",
    )


def _prediction(window: OfflineWindow) -> PredictionResult:
    return PredictionResult(
        identity=PredictionIdentity(
            request_id=str(window.window_identity),
            source_layer_id=str(window.source_layer),
            target_layer_id=str(window.target_layer),
        ),
        hint=PredictionHint(
            predictor_id="copy_current",
            hint_type="traffic_matrix",
            target_dispatch_rows=window.p2_actual,
            confidence=1.0,
            oracle=False,
            source_layer_id=str(window.source_layer),
            target_layer_id=str(window.target_layer),
        ),
    )


def _spec() -> EvaluationSpec:
    return EvaluationSpec(
        track="runtime_lookahead",
        world_size=4,
        task_granularity="matrix_cell",
        matrix_unit="rows",
        time_unit="row_cost",
        cost_model_id="offline_common_v1",
        release_model="p1_return",
        return_model="transpose_dispatch",
        full_duplex=True,
        launch_cost=0.0,
        bytes_per_row=1,
        bandwidth=1.0,
        compute_delay=0.0,
        p2_semantics="lookahead",
        residual_policy="reject",
    )


def _input_tensor(rows: int, *, hidden: int, source_global_rank: int) -> torch.Tensor:
    base = int(source_global_rank) * 10000
    values = torch.arange(base, base + max(int(rows), 1), dtype=torch.float32)
    return values[:rows].unsqueeze(1).repeat(1, max(hidden, 1)).to(dtype=torch.float16)


def _peer_base_offset(rows_by_peer: list[int], peer_group_rank: int) -> int:
    return int(sum(int(rows_by_peer[index]) for index in range(int(peer_group_rank))))


def _expected_output_tensor(*, local_global_rank: int, matrix: tuple[tuple[int, ...], ...], shape_suffix: tuple[int, ...]) -> torch.Tensor:
    local_group_rank = int(local_global_rank)
    incoming_rows_by_peer = [int(matrix[src][local_group_rank]) for src in range(len(matrix))]
    width = int(shape_suffix[0]) if shape_suffix else 1
    rows: list[torch.Tensor] = []
    for src_group_rank, row_count in enumerate(incoming_rows_by_peer):
        if row_count <= 0:
            continue
        source_peer_base = _peer_base_offset([int(matrix[src_group_rank][peer]) for peer in range(len(matrix))], local_group_rank)
        values = torch.arange(
            src_group_rank * 10000 + source_peer_base,
            src_group_rank * 10000 + source_peer_base + row_count,
            dtype=torch.float32,
        )
        rows.append(values.unsqueeze(1).repeat(1, max(width, 1)).to(dtype=torch.float16))
    if not rows:
        return torch.zeros((0, max(width, 1)), dtype=torch.float16)
    return torch.cat(rows, dim=0)


def _jsonify(value):
    if isinstance(value, torch.Tensor):
        return value.float().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _worker(rank: int, world_size: int, master_port: int, out_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{int(master_port)}",
        rank=rank,
        world_size=world_size,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    try:
        window = _window()
        prediction = _prediction(window)
        spec = _spec()
        planning = build_planning_parity_case(
            window=window,
            prediction=prediction,
            spec=spec,
            planner_id="fifo_bucket",
            bucket_rows=4,
            max_waves=64,
        )
        if planning.offline_request.semantic_digest() != planning.online_request.semantic_digest():
            raise AssertionError("input parity failed")
        if planning.offline_plan.semantic_digest() != planning.online_plan.semantic_digest():
            raise AssertionError("plan parity failed")
        contexts = make_contexts_from_matrix(phase="P0", matrix=window.p0_actual, p2_hint_mode="deterministic_stub")
        actual_context = ActualPhaseContext(
            layer_id=str(contexts[rank].layer_id),
            phase="P0",
            world_size=4,
            rank_space="global",
            layout_digest=str(contexts[rank].canonical_receive_layout_id),
            metadata={"phase_ready_context": contexts[rank].to_dict()},
        )
        materialization = build_materialization_parity_case(
            window=window,
            prediction=prediction,
            spec=spec,
            planner_id="fifo_bucket",
            publication_slot={
                "run_id": str(window.trace_digest),
                "forward_generation": 0,
                "microbatch_id": "mb0",
                "source_layer_id": str(window.source_layer),
                "target_layer_id": str(window.target_layer),
                "planning_slot": f"{window.source_layer}->{window.target_layer}",
            },
            rank_map=RankMap(group_ranks=(0, 1, 2, 3), root_rank=0),
            actual_phase_context=actual_context,
            bucket_rows=4,
            max_waves=64,
        )
        if (
            materialization.offline_materialized_plan.materialized_plan_digest
            != materialization.online_prepared_execution.materialized_plan.materialized_plan_digest
        ):
            raise AssertionError("materialization parity failed")
        if materialization.online_prepared_execution.validation.valid is not True:
            raise AssertionError(f"materialization validation failed: {materialization.online_prepared_execution.validation.to_dict()}")
        role_results: dict[str, object] = {}
        for payload_spec in materialization.online_prepared_execution.materialized_plan.payload_specs:
            rows = int(payload_spec.row_count)
            hidden = int(payload_spec.shape_suffix[0]) if payload_spec.shape_suffix else 1
            input_tensor = _input_tensor(rows, hidden=hidden, source_global_rank=rank)
            expected_output = _expected_output_tensor(
                local_global_rank=rank,
                matrix=window.p0_actual,
                shape_suffix=tuple(int(dim) for dim in payload_spec.shape_suffix),
            )
            outcome = GlooFunctionalExecutor().execute(
                plan=materialization.online_prepared_execution.materialized_plan,
                invocation=PayloadInvocation(
                    run_id=str(window.trace_digest),
                    forward_generation=0,
                    layer_id=str(contexts[rank].layer_id),
                    phase="P0",
                    payload_role=str(payload_spec.payload_role),
                    shape=tuple(int(dim) for dim in input_tensor.shape),
                    dtype=str(payload_spec.dtype),
                    layout_digest=str(materialization.online_prepared_execution.materialized_plan.layout_digest),
                    invocation_id=f"m4-parity:{rank}:{payload_spec.payload_role}",
                    input_tensor=input_tensor,
                    process_group=dist.group.WORLD,
                ),
                context=ExecutionContext(
                    run_id=str(window.trace_digest),
                    forward_generation=0,
                    layer_id=str(contexts[rank].layer_id),
                    phase="P0",
                    rank_space="global",
                ),
            )
            expected_completed = expected_completed_task_ids(
                materialization.online_prepared_execution.materialized_plan,
                payload_role=str(payload_spec.payload_role),
            )
            if tuple(sorted(outcome.completed_task_ids)) != tuple(sorted(expected_completed)):
                raise AssertionError(
                    f"execution semantics parity failed for rank={rank} role={payload_spec.payload_role}: "
                    f"expected={expected_completed} actual={outcome.completed_task_ids}"
                )
            if not isinstance(outcome.output_payload, torch.Tensor) or not torch.equal(outcome.output_payload.cpu(), expected_output.cpu()):
                raise AssertionError(
                    f"output parity failed for rank={rank} role={payload_spec.payload_role}: "
                    f"expected={expected_output.tolist()} actual="
                    f"{outcome.output_payload.tolist() if isinstance(outcome.output_payload, torch.Tensor) else outcome.output_payload}"
                )
            role_results[str(payload_spec.payload_role)] = {
                "expected_completed_task_ids": list(expected_completed),
                "completed_task_ids": list(outcome.completed_task_ids),
                "distributed_operation_count": int(outcome.details.get("distributed_operation_count", 0)),
                "expected_output": expected_output.float().tolist(),
                "actual_output": outcome.output_payload.float().tolist(),
            }
        local = {
            "rank": int(rank),
            "request_digest": str(planning.offline_request.semantic_digest()),
            "plan_digest": str(planning.offline_plan.semantic_digest()),
            "materialized_plan_digest": str(materialization.offline_materialized_plan.materialized_plan_digest),
            "roles": role_results,
        }
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
        (out_path / f"rank{rank}_m4_parity.json").write_text(json.dumps(_jsonify(local), indent=2), encoding="utf-8")
        if rank == 0:
            request_digests = {str(item["request_digest"]) for item in gathered}
            plan_digests = {str(item["plan_digest"]) for item in gathered}
            materialized_digests = {str(item["materialized_plan_digest"]) for item in gathered}
            if len(request_digests) != 1:
                raise AssertionError(f"request digests diverged: {sorted(request_digests)}")
            if len(plan_digests) != 1:
                raise AssertionError(f"plan digests diverged: {sorted(plan_digests)}")
            if len(materialized_digests) != world_size:
                raise AssertionError("each rank should contribute its own materialized digest")
            summary = {
                "status": "passed",
                "world_size": int(world_size),
                "input_parity": {
                    "status": "PASS",
                    "offline_planning_request_digest": next(iter(request_digests)),
                    "online_planning_request_digest": next(iter(request_digests)),
                },
                "plan_parity": {
                    "status": "PASS",
                    "offline_window_plan_digest": next(iter(plan_digests)),
                    "online_window_plan_digest": next(iter(plan_digests)),
                },
                "materialization_parity": {
                    "status": "PASS",
                    "per_rank_materialized_plan_digests": [str(item["materialized_plan_digest"]) for item in gathered],
                },
                "execution_semantics_parity": {
                    "status": "PASS",
                    "per_rank_roles": {
                        str(item["rank"]): item["roles"]
                        for item in gathered
                    },
                },
            }
            (out_path / "m4_offline_online_parity_summary.json").write_text(json.dumps(_jsonify(summary), indent=2), encoding="utf-8")
            print(json.dumps(_jsonify(summary), indent=2))
    except Exception as exc:
        failure = {
            "rank": int(rank),
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (out_path / f"rank{rank}_m4_parity_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    finally:
        dist.destroy_process_group()


def main() -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    port = _free_port()
    out_dir = "outputs/closure/m4_offline_online_parity"
    mp.spawn(_worker, args=(4, port, out_dir), nprocs=4, join=True)


if __name__ == "__main__":
    main()
