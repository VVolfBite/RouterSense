from __future__ import annotations

from rs.core.contracts.execution import ActualPhaseContext
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.materialization import CommonPlanMaterializer, CommonPlanValidator


def publish_plan(*, window_plan, world_size: int):
    publisher = CanonicalPlanPublisher(rank_map=RankMap(group_ranks=tuple(range(int(world_size))), root_rank=0))
    return publisher.build(
        publication_slot={
            "run_id": "paper-eval",
            "forward_generation": 0,
            "microbatch_id": "mb0",
            "source_layer_id": str(window_plan.metadata.get("source_layer_id", "0")),
            "target_layer_id": str(window_plan.metadata.get("target_layer_id", "1")),
            "planning_slot": f"{window_plan.metadata.get('source_layer_id', '0')}->{window_plan.metadata.get('target_layer_id', '1')}",
        },
        window_plan=window_plan,
    )


def materialize_and_validate(*, published_plan, actual_context: ActualPhaseContext):
    materialized = CommonPlanMaterializer().materialize(published_plan, actual_context)
    validation = CommonPlanValidator().validate(materialized, actual_context)
    return materialized, validation
