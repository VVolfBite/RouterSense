from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..core.collective import CollectiveOps
from ..core.correctness import summarize_dispatch_plans
from ..core.manifest import DispatchPlan, DistributedManifest
from ..core.nccl_executor import NCCLExecutor
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
        origin_rank=config.origin_rank,
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

    dispatch_matrix = _build_matrix_from_plan(dispatch_plans[0], rank, "send")
    combine_matrix = _build_matrix_from_plan(dispatch_plans[0], rank, "recv")
    next_dispatch_matrix = (
        _build_matrix_from_plan(dispatch_plans[1], rank, "send")
        if len(dispatch_plans) > 1
        else [[0] * world_size for _ in range(world_size)]
    )

    strategy = get_strategy(strategy_name)
    ctx = SchedulingContext(
        dispatch_matrix=dispatch_matrix,
        combine_matrix=combine_matrix,
        next_dispatch_matrix=next_dispatch_matrix,
        num_gpus=world_size,
        model="full_duplex",
        expert_compute_delay=expert_compute_delay,
    )
    result = strategy.solve(ctx)
    schedule = list(result.schedule) if result.schedule else _fallback_schedule(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        world_size,
    )

    phase_specs = [
        (0, dispatch_plans[0], "dispatch"),
        (1, dispatch_plans[0], "combine"),
    ]
    if len(dispatch_plans) > 1:
        phase_specs.append((2, dispatch_plans[1], "next_dispatch"))

    nccl_results: dict[str, Any] = {}
    for phase_idx, plan, direction in phase_specs:
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
