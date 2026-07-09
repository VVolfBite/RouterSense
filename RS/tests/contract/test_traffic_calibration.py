from __future__ import annotations

from rs.runtime.online.megatron_ep.prediction.traffic_calibration import calibrate_traffic_matrix


def test_total_traffic_calibration_can_reduce_relative_l1() -> None:
    predicted = ((0, 4), (2, 0))
    actual = ((0, 8), (4, 0))
    calibrated, audit = calibrate_traffic_matrix(predicted, actual_matrix=actual, mode="oracle_total")
    assert calibrated == ((0, 8), (4, 0))
    assert audit.after_relative_l1 <= audit.before_relative_l1


def test_row_col_calibration_preserves_zero_diagonal() -> None:
    calibrated, _audit = calibrate_traffic_matrix(
        ((9, 4), (2, 7)),
        current_dispatch_matrix=((0, 8), (4, 0)),
        mode="row_col_current",
    )
    assert calibrated[0][0] == 0
    assert calibrated[1][1] == 0
