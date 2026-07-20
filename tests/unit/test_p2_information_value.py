from __future__ import annotations

from rs.runtime.offline.p2_information_value import simulate_p2_information


P0 = [
    [0, 8, 2, 0],
    [3, 0, 0, 7],
    [6, 0, 0, 1],
    [0, 5, 4, 0],
]
P1 = [list(row) for row in zip(*P0)]
P2 = [
    [0, 0, 9, 1],
    [0, 0, 2, 8],
    [7, 1, 0, 0],
    [4, 6, 0, 0],
]


def test_reactive_p2_reveals_truth_rows_without_executing_forecast_bytes() -> None:
    result = simulate_p2_information(
        p0_dispatch_matrix=P0,
        p1_return_matrix=P1,
        p2_truth_matrix=P2,
        family_id="rscf",
        information_mode="reactive",
    )
    assert result.valid is True
    assert result.revealed_rows == 4
    assert result.audit["served_volume_by_phase"][2] == sum(
        value for src, row in enumerate(P2) for dst, value in enumerate(row) if src != dst
    )


def test_predicted_mode_uses_advisory_matrix_but_audits_against_truth() -> None:
    forecast = [row[1:] + row[:1] for row in P2]
    result = simulate_p2_information(
        p0_dispatch_matrix=P0,
        p1_return_matrix=P1,
        p2_truth_matrix=P2,
        p2_forecast_matrix=forecast,
        family_id="rscf",
        information_mode="predicted",
        prediction_confidence=0.8,
    )
    assert result.valid is True
    assert result.audit["served_volume_by_phase"][2] == sum(
        value for src, row in enumerate(P2) for dst, value in enumerate(row) if src != dst
    )
