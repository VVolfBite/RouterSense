from __future__ import annotations

from rs_sim.runtime import (
    make_default_synthetic_runtime_profile,
    make_runtime_profile_bundle,
)
from rs_sim.scheduler import PlanningCostModel


def synthetic_runtime_profile(
    *,
    max_batch_tasks: int = 64,
    local_assembly_latency_ns: int = 5,
    planning_cost_model: PlanningCostModel | None = None,
):
    base = make_default_synthetic_runtime_profile(
        max_batch_tasks=max_batch_tasks,
        local_assembly_latency_ns=local_assembly_latency_ns,
    )
    if planning_cost_model is None:
        return base
    return make_runtime_profile_bundle(
        profile_id=f"{base.profile_id}:test-planning",
        profile_kind=base.profile_kind,
        profile_provenance=base.profile_provenance,
        transport_profile=base.transport_profile,
        receiver_cost_model=base.receiver_cost_model,
        planning_cost_model=planning_cost_model,
        source_digests=base.source_digests,
        assumptions=base.assumptions + ("TEST_PLANNING_COST_OVERRIDE",),
        performance_eligible=False,
    )
