from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..core.collective import CollectiveOps
from ..core.correctness import summarize_dispatch_plans
from ..core.manifest import DispatchPlan, DistributedManifest
from ..core.nccl_executor import NCCLExecutor
from ..core.wave_executor import CollectiveWaveExecutor, execute_native_baseline, verify_token_conservation
from ..core.wave_planner import scheduling_result_to_wave_schedule, verify_wave_conservation
from ..core.placement import PlacementStrategy
from ..core.worker_loop import WorkerLoop
from ....scheduler.strategy import SchedulingContext, get_strategy
from .expert_store import (
    count_local_expert_parameters,
    extract_local_expert_weights,
    plan_local_expert_ids,
    summarize_residency,
)
from .olmoe_adapter import build_dispatch_plan_from_trace, execute_local_experts, probe_olmoe_adapter_config


@dataclass
class DistributedRunnerConfig:
    world_size: int
    node_rank: int
    model_id: str
    origin_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DistributedRunnerPlan:
    adapter: dict[str, Any]
    placement: dict[int, int]
    residency: dict[str, Any]
    dispatch_summary: dict[str, Any]
    dispatch_plans: list[DispatchPlan]
    local_expert_weights: dict[str, Any]
    local_expert_weight_bundle: Any
    manifest: dict[str, Any]


def build_distributed_runner_plan(
    *,
    model: Any,
    trace: dict[str, Any],
    config: DistributedRunnerConfig,
    rank: int,
    host: str,
    gpu_name: str = "",
) -> DistributedRunnerPlan:
    adapter_config = probe_olmoe_adapter_config(model)
    owner_by_expert = PlacementStrategy.round_robin(adapter_config.num_experts, config.world_size)
    local_expert_ids = plan_local_expert_ids(owner_by_expert, rank)

    experts_module = getattr(model.model.layers[0].mlp, "experts", None)
    local_parameter_count = count_local_expert_parameters(experts_module, local_expert_ids)
    residency = summarize_residency(local_expert_ids, local_parameter_count=local_parameter_count)
    local_weights = extract_local_expert_weights(experts_module, local_expert_ids)

    dispatch_plans = _build_layer_dispatch_plans(
        trace=trace,
        owner_by_expert=owner_by_expert,
        origin_rank=rank,
        world_size=config.world_size,
    )
    manifest = DistributedManifest(
        ranks=list(range(config.world_size)),
        hosts=[host],
        gpu_names=[gpu_name] if gpu_name else [],
        placement=owner_by_expert,
        dispatch_plans=dispatch_plans,
        metadata={"model_id": config.model_id, "node_rank": config.node_rank, "rank": rank},
    )
    return DistributedRunnerPlan(
        adapter=adapter_config.to_dict(),
        placement=owner_by_expert,
        residency=residency.to_dict(),
        dispatch_summary=summarize_dispatch_plans(dispatch_plans),
        dispatch_plans=dispatch_plans,
        local_expert_weights=local_weights.to_dict(),
        local_expert_weight_bundle=local_weights,
        manifest=manifest.to_dict(),
    )


def simulate_rank_execution(plans: list[DispatchPlan], *, rank: int, bytes_per_row: int = 0) -> dict[str, Any]:
    collective = CollectiveOps(bytes_per_row=bytes_per_row)
    worker = WorkerLoop()
    for plan in plans:
        collective.dispatch(payload=None, plan=plan, rank=rank)
        worker.record_plan(plan, rank)
        collective.return_results(payload=None, plan=plan, rank=rank)
    return {
        "collectives": [asdict(record) for record in collective.records],
        "worker_state": asdict(worker.state),
    }


def simulate_local_expert_forward(
    *,
    hidden_states,
    route_items,
    local_weights,
):
    return execute_local_experts(hidden_states, route_items, local_weights)


def execute_scheduled_inference(
    *,
    dispatch_plans: list[DispatchPlan],
    rank: int,
    world_size: int,
    strategy_name: str = "greedy",
    hidden_size: int = 2048,
    expert_compute_delay: float = 0.0,
    executor: NCCLExecutor | None = None,
    payload=None,
    use_distributed: bool = False,
    execution_mode: str = "p2p_matching",
    local_expert_weights: Any | None = None,
    hidden_state_rows=None,
    plan_index: int = 0,
    max_waves: int | None = None,
) -> dict[str, Any]:
    """Bridge scheduling outputs to collective execution."""

    import torch  # type: ignore

    if not dispatch_plans:
        return {
            "strategy": strategy_name,
            "scheduling_result": {"makespan": 0.0, "solve_time_ms": 0.0, "chunk_count": 0},
            "nccl_execution": {},
            "collective_records": [],
        }

    collective = CollectiveOps(bytes_per_row=hidden_size * 2)
    executor = executor or NCCLExecutor(
        rank=rank,
        world_size=world_size,
        dtype=torch.float16,
        device=f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu",
    )

    active_plan = dispatch_plans[plan_index]
    import time

    matrix_start = time.perf_counter()
    local_dispatch_matrix = _build_matrix_from_plan(active_plan, rank, "send")
    local_combine_matrix = _build_matrix_from_plan(active_plan, rank, "recv")
    matrix_build_ms = (time.perf_counter() - matrix_start) * 1000.0
    aggregate_start = time.perf_counter()
    dispatch_matrix = _aggregate_matrix(
        local_dispatch_matrix,
        use_distributed=use_distributed,
        world_size=world_size,
        device=getattr(executor, "device", None),
    )
    combine_matrix = _aggregate_matrix(
        local_combine_matrix,
        use_distributed=use_distributed,
        world_size=world_size,
        device=getattr(executor, "device", None),
    )
    next_dispatch_matrix = (
        _aggregate_matrix(
            _build_matrix_from_plan(dispatch_plans[plan_index + 1], rank, "send"),
            use_distributed=use_distributed,
            world_size=world_size,
            device=getattr(executor, "device", None),
        )
        if len(dispatch_plans) > plan_index + 1
        else [[0] * world_size for _ in range(world_size)]
    )
    all_reduce_ms = (time.perf_counter() - aggregate_start) * 1000.0

    strategy = get_strategy(strategy_name)
    ctx = SchedulingContext(
        dispatch_matrix=dispatch_matrix,
        combine_matrix=combine_matrix,
        next_dispatch_matrix=next_dispatch_matrix,
        num_gpus=world_size,
        model="full_duplex",
        expert_compute_delay=expert_compute_delay,
    )
    planner_start = time.perf_counter()
    result = strategy.solve(ctx)
    planner_ms = (time.perf_counter() - planner_start) * 1000.0
    schedule = list(result.schedule) if result.schedule else _fallback_schedule(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        world_size,
    )

    if execution_mode in {"native_baseline", "wave_collective"}:
        if hidden_state_rows is None:
            raise RuntimeError(f"execution_mode={execution_mode} requires hidden_state_rows")
        if local_expert_weights is None:
            raise RuntimeError(f"execution_mode={execution_mode} requires local_expert_weights")
        wave_executor = CollectiveWaveExecutor(
            rank=rank,
            world_size=world_size,
            dtype=torch.float16,
            device=getattr(executor, "device", None) or (f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"),
        )
        if execution_mode == "native_baseline":
            verify_start = time.perf_counter()
            native = execute_native_baseline(
                rank=rank,
                world_size=world_size,
                dispatch_plan=active_plan,
                token_buffer=hidden_state_rows,
                hidden_size=hidden_size,
                device=wave_executor.device,
                dtype=wave_executor.dtype,
                local_weights=local_expert_weights,
            )
            correctness = verify_token_conservation(
                native.final_output,
                native.final_output,
                active_plan,
                native_route_items=native.combine_result.received_route_items,
                wave_route_items=native.combine_result.received_route_items,
            )
            conservation_check_ms = (time.perf_counter() - verify_start) * 1000.0
            return {
                "strategy": strategy_name,
                "execution_mode": execution_mode,
                "scheduling_result": {
                    "makespan": result.makespan,
                    "solve_time_ms": result.solve_time_ms,
                    "chunk_count": len(schedule),
                },
                "control_plane_ms": {
                    "matrix_build_ms": matrix_build_ms,
                    "all_reduce_ms": all_reduce_ms,
                    "planner_ms": planner_ms,
                    "wave_convert_ms": 0.0,
                    "conservation_check_ms": conservation_check_ms,
                },
                "native_baseline": {
                    "dispatch_comm_ms": native.dispatch_result.total_comm_ms,
                    "combine_comm_ms": native.combine_result.total_comm_ms,
                    "dispatch_pack_ms": native.dispatch_result.total_pack_ms,
                    "combine_pack_ms": native.combine_result.total_pack_ms,
                    "dispatch_wave_count": native.dispatch_result.wave_count,
                    "combine_wave_count": native.combine_result.wave_count,
                },
                "correctness": correctness,
            }

        bundle = scheduling_result_to_wave_schedule(
            result,
            dispatch_plan=active_plan,
            rank=rank,
            world_size=world_size,
            max_waves=max_waves,
        )
        wave_convert_end = time.perf_counter()
        wave_convert_ms = (wave_convert_end - planner_start) * 1000.0 - planner_ms
        verify_start = time.perf_counter()
        dispatch_conservation = verify_wave_conservation(bundle.dispatch_waves, rank=rank, dispatch_plan=active_plan, phase=0)
        combine_conservation = verify_wave_conservation(bundle.combine_waves, rank=rank, dispatch_plan=active_plan, phase=1)
        dispatch_exec = wave_executor.execute_waves(
            bundle.dispatch_waves,
            phase=0,
            direction="dispatch",
            token_buffer=hidden_state_rows,
            hidden_size=hidden_size,
        )
        local_output = execute_local_experts(
            dispatch_exec.received_tensor,
            dispatch_exec.received_route_items,
            local_expert_weights,
        )
        combine_exec = wave_executor.execute_waves(
            bundle.combine_waves,
            phase=1,
            direction="combine",
            token_buffer=local_output,
            hidden_size=hidden_size,
        )
        final_output = _aggregate_wave_outputs(combine_exec.received_tensor, combine_exec.received_route_items, hidden_size=hidden_size)
        native = execute_native_baseline(
            rank=rank,
            world_size=world_size,
            dispatch_plan=active_plan,
            token_buffer=hidden_state_rows,
            hidden_size=hidden_size,
            device=wave_executor.device,
            dtype=wave_executor.dtype,
            local_weights=local_expert_weights,
        )
        correctness = verify_token_conservation(native.final_output, final_output, active_plan)
        correctness = verify_token_conservation(
            native.final_output,
            final_output,
            active_plan,
            native_route_items=native.combine_result.received_route_items,
            wave_route_items=combine_exec.received_route_items,
        )
        conservation_check_ms = (time.perf_counter() - verify_start) * 1000.0
        return {
            "strategy": strategy_name,
            "execution_mode": execution_mode,
            "scheduling_result": {
                "makespan": result.makespan,
                "solve_time_ms": result.solve_time_ms,
                "chunk_count": len(schedule),
            },
            "control_plane_ms": {
                "matrix_build_ms": matrix_build_ms,
                "all_reduce_ms": all_reduce_ms,
                "planner_ms": planner_ms,
                "wave_convert_ms": wave_convert_ms,
                "conservation_check_ms": conservation_check_ms,
            },
            "wave_schedule": {
                "dispatch_wave_count": len(bundle.dispatch_waves),
                "combine_wave_count": len(bundle.combine_waves),
                "dispatch_conservation": dispatch_conservation,
                "combine_conservation": combine_conservation,
            },
            "wave_execution": {
                "dispatch_comm_ms": dispatch_exec.total_comm_ms,
                "dispatch_pack_ms": dispatch_exec.total_pack_ms,
                "dispatch_unpack_ms": dispatch_exec.total_unpack_ms,
                "combine_comm_ms": combine_exec.total_comm_ms,
                "combine_pack_ms": combine_exec.total_pack_ms,
                "combine_unpack_ms": combine_exec.total_unpack_ms,
                "dispatch_timings": [timing.__dict__ for timing in dispatch_exec.timings],
                "combine_timings": [timing.__dict__ for timing in combine_exec.timings],
            },
            "native_baseline": {
                "dispatch_comm_ms": native.dispatch_result.total_comm_ms,
                "combine_comm_ms": native.combine_result.total_comm_ms,
            },
            "correctness": correctness,
        }

    phase_specs = [
        (0, active_plan, "dispatch"),
        (1, active_plan, "combine"),
    ]
    if len(dispatch_plans) > plan_index + 1:
        phase_specs.append((2, dispatch_plans[plan_index + 1], "next_dispatch"))

    nccl_results: dict[str, Any] = {}
    for phase_idx, plan, direction in phase_specs:
        if direction == "combine" and expert_compute_delay > 0.0:
            _simulate_expert_compute_delay(
                expert_compute_delay=expert_compute_delay,
                hidden_size=hidden_size,
                device=getattr(executor, "device", None),
            )
        phase_schedule = [chunk for chunk in schedule if int(chunk["phase"]) == phase_idx]
        execution = collective.execute_scheduled_phase(
            plan=plan,
            rank=rank,
            schedule=phase_schedule,
            phase=phase_idx,
            direction=direction,
            executor=executor,
            hidden_size=hidden_size,
            payload=payload,
        )
        nccl_results[f"phase{phase_idx}_{direction}"] = {
            "wall_time_us": execution.total_wall_time_us,
            "ops_count": len(execution.ops),
            "ops": [
                {
                    "op": op.op,
                    "peer_rank": op.peer_rank,
                    "tensor_size": op.tensor_size,
                    "duration_us": op.duration_us,
                }
                for op in execution.ops
            ],
        }

    return {
        "strategy": strategy_name,
        "scheduling_result": {
            "makespan": result.makespan,
            "solve_time_ms": result.solve_time_ms,
            "chunk_count": len(schedule),
        },
        "nccl_execution": nccl_results,
        "collective_records": [
            {
                "op": record.op_name,
                "layer": record.layer_id,
                "send_bytes": record.send_bytes,
                "recv_bytes": record.recv_bytes,
            }
            for record in collective.records
        ],
    }


def _aggregate_wave_outputs(received_tensor, route_items, *, hidden_size: int):
    import torch  # type: ignore

    if not route_items:
        return torch.empty((0, hidden_size), dtype=received_tensor.dtype, device=received_tensor.device)
    token_count = max(int(item.token_flat_index) for item in route_items) + 1
    output = torch.zeros((token_count, hidden_size), dtype=received_tensor.dtype, device=received_tensor.device)
    for row_index, item in enumerate(route_items):
        output[int(item.token_flat_index)] += received_tensor[row_index]
    return output


def _build_layer_dispatch_plans(
    *,
    trace: dict[str, Any],
    owner_by_expert: dict[int, int],
    origin_rank: int,
    world_size: int,
) -> list[DispatchPlan]:
    records = trace.get("records", [])
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_layer.setdefault(int(record["layer_id"]), []).append(record)
    plans: list[DispatchPlan] = []
    for layer_id, layer_records in sorted(by_layer.items()):
        plans.append(
            build_dispatch_plan_from_trace(
                layer_records,
                owner_by_expert=owner_by_expert,
                origin_rank=origin_rank,
                layer_id=layer_id,
                world_size=world_size,
            )
        )
    return plans


def _build_matrix_from_plan(
    plan: DispatchPlan,
    rank: int,
    direction: str,
) -> list[list[int]]:
    del rank  # kept for interface stability
    matrix = [[0] * plan.world_size for _ in range(plan.world_size)]
    for shard in plan.shards:
        matrix[shard.source_rank][shard.destination_rank] += shard.rows
    if direction == "recv":
        return [[matrix[dst][src] for dst in range(plan.world_size)] for src in range(plan.world_size)]
    if direction != "send":
        raise ValueError(f"unsupported direction: {direction}")
    return matrix


def _aggregate_matrix(
    matrix: list[list[int]],
    *,
    use_distributed: bool,
    world_size: int,
    device=None,
    dist_module=None,
) -> list[list[int]]:
    if not use_distributed or world_size <= 1:
        return matrix

    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore

    dist_impl = dist if dist_module is None else dist_module
    target_device = device
    if target_device is None:
        if torch.cuda.is_available() and getattr(dist_impl, "get_backend", lambda: "")() == "nccl":
            target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            target_device = torch.device("cpu")

    tensor = torch.tensor(matrix, dtype=torch.float32, device=target_device)
    dist_impl.all_reduce(tensor, op=dist_impl.ReduceOp.SUM)
    return [[int(value) for value in row] for row in tensor.tolist()]


def _simulate_expert_compute_delay(
    *,
    expert_compute_delay: float,
    hidden_size: int,
    device=None,
) -> None:
    import time

    if expert_compute_delay <= 0.0:
        return

    try:
        import torch  # type: ignore
    except ImportError:
        time.sleep(min(expert_compute_delay, 1.0))
        return

    if device is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
        rows = max(1, int(expert_compute_delay))
        dummy = torch.zeros((rows, hidden_size), device=device, dtype=torch.float16)
        _ = dummy @ dummy.transpose(0, 1)
        torch.cuda.synchronize(device)
    time.sleep(min(expert_compute_delay, 1.0))


def _fallback_schedule(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> list[dict[str, int]]:
    schedule: list[dict[str, int]] = []
    for phase, matrix in enumerate((dispatch_matrix, combine_matrix, next_dispatch_matrix)):
        chunks: list[tuple[int, int, int]] = []
        for src in range(num_gpus):
            for dst in range(num_gpus):
                size = int(matrix[src][dst])
                if src == dst or size <= 0:
                    continue
                chunks.append((size, src, dst))
        chunks.sort(reverse=True)
        for size, src, dst in chunks:
            schedule.append(
                {
                    "phase": phase,
                    "src_gpu": src,
                    "dst_gpu": dst,
                    "size": size,
                }
            )
    return schedule
