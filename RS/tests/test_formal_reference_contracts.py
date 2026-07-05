from __future__ import annotations

import pytest

from rs.scheduling.baselines.birkhoff import (
    UnsupportedBaselineError,
    birkhoff_baseline_summary,
    birkhoff_schedule_single_layer,
)
from rs.scheduling.reference.exact_small_instance import (
    UnsupportedExactSolve,
    solve_exact_small_instance,
)
from rs.scheduling.reference.oracle_guided import (
    UnsupportedReferenceSolver,
    pairwise_oracle,
    unsupported_oracle_guided_result,
)


def test_formal_birkhoff_summary_is_not_placeholder() -> None:
    summary = birkhoff_baseline_summary([[0, 1], [2, 0]])
    assert summary["supported"] is False
    assert summary["solver_status"] == "unsupported"
    assert summary["makespan"] is None
    assert summary["optimality_gap"] is None


def test_formal_birkhoff_schedule_fails_closed() -> None:
    with pytest.raises(UnsupportedBaselineError):
        birkhoff_schedule_single_layer([[0, 1], [1, 0]])


def test_formal_oracle_reference_fails_closed() -> None:
    with pytest.raises(UnsupportedReferenceSolver):
        pairwise_oracle([[0, 1], [1, 0]], [[0, 1], [1, 0]], [[0, 1], [1, 0]], 2)
    result = unsupported_oracle_guided_result()
    assert result["solver_status"] == "unsupported"
    assert result["optimality_gap"] is None
    assert result["certified_optimal"] is False


def test_formal_exact_small_instance_fails_closed() -> None:
    with pytest.raises(UnsupportedExactSolve):
        solve_exact_small_instance()
