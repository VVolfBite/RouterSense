from __future__ import annotations

import pytest

from rs.planning import PlannerRegistry


def test_runtime_usage_rejects_reference_and_exact_planners() -> None:
    with pytest.raises(ValueError, match="not runtime-deployable"):
        PlannerRegistry.create("barrier_criticality_posthoc_best", None, usage="runtime")
    with pytest.raises(ValueError, match="not runtime-deployable"):
        PlannerRegistry.create("oracle_local_cp_sat", None, usage="runtime")
