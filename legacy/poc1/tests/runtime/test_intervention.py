from __future__ import annotations

import pytest

from routesense_poc1.runtime.intervention import RouteAblationContext


@pytest.mark.skip(reason="requires real OLMoE model and GPU runtime")
def test_ablate_zeros_target_weight() -> None:
    assert RouteAblationContext is not None
