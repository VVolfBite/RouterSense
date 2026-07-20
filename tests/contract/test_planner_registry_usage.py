from __future__ import annotations

import pytest
from rs.planning import PlannerRegistry


def test_runtime_usage_rejects_reference_and_exact_planners() -> None:
    with pytest.raises(ValueError, match="not runtime-deployable"):
        PlannerRegistry.create("islip_reference", None, usage="runtime")
    with pytest.raises(ValueError, match="not runtime-deployable"):
        PlannerRegistry.create("oracle_local_exact", None, usage="runtime")


def test_retired_aliases_are_rejected() -> None:
    for planner_id in ("retired_joint_v0", "retired_local_v0", "retired_safe_v0"):
        with pytest.raises(ValueError, match="unknown formal algorithm"):
            PlannerRegistry.create(planner_id, None, usage="runtime")
