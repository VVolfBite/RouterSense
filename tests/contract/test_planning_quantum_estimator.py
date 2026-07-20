from __future__ import annotations

from rs.scheduling import estimate_planning_quantum_rows_from_contexts, estimate_planning_quantum_rows_from_values
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def test_planning_quantum_estimator_uses_nearest_power_of_two_cover() -> None:
    assert estimate_planning_quantum_rows_from_values([600, 60000, 1200, 1800]) == 512
    assert estimate_planning_quantum_rows_from_values([90, 60, 30, 90]) == 32
    assert estimate_planning_quantum_rows_from_values([1, 3, 7]) == 1
    assert estimate_planning_quantum_rows_from_values([600, 900, 1500, 2100]) == 512
    assert estimate_planning_quantum_rows_from_values([760, 1400, 2800]) == 1024


def test_planning_quantum_estimator_reads_remote_phase_contexts() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 600, 1200), (300, 0, 1800), (600, 900, 0)),
        p2_hint_mode="none",
    )
    assert estimate_planning_quantum_rows_from_contexts(global_contexts=contexts, phase="P0") == 256
