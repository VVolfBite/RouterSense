from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .manifest import DispatchPlan


@dataclass
class CorrectnessResult:
    generated_token_match: bool
    max_abs_logit_error: float | None = None
    mean_abs_logit_error: float | None = None
    cross_node_dispatch_route_count: int = 0
    cross_node_execute_count: int = 0
    cross_node_return_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_dispatch_plans(plans: list[DispatchPlan]) -> dict[str, Any]:
    cross_node = sum(plan.total_cross_node_routes() for plan in plans)
    total_routes = sum(len(shard.route_items) for plan in plans for shard in plan.shards)
    return {
        "plan_count": len(plans),
        "total_routes": total_routes,
        "cross_node_routes": cross_node,
    }
