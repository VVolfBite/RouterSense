from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..core.collective import CollectiveOps
from ..core.correctness import summarize_dispatch_plans
from ..core.manifest import DispatchPlan, DistributedManifest
from ..core.placement import PlacementStrategy
from ..core.worker_loop import WorkerLoop
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
