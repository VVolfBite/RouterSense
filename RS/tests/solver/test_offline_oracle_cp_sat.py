from __future__ import annotations

from rs.core.contracts import EvaluationSpec, OfflineWindow, TrafficProvenance
from rs.offline import build_evaluation_task_set, solve_cp_sat


def test_cp_sat_oracle_reports_unsupported_without_ortools() -> None:
    window = OfflineWindow(
        window_identity="fixture:1->2",
        source_layer="1",
        target_layer="2",
        p0_actual=((0, 1), (1, 0)),
        p1_actual=((0, 1), (1, 0)),
        p2_actual=((0, 1), (1, 0)),
        placement_snapshot={},
        traffic_provenance=TrafficProvenance.REAL_EP_OBSERVED,
        matrix_unit="rows",
        return_model="transpose_dispatch",
        raw_token_count=2,
        used_token_count=2,
        dropped_token_count=0,
        drop_reason=None,
        trace_digest="trace",
    )
    spec = EvaluationSpec(
        track="execution_window",
        world_size=2,
        task_granularity="matrix_cell",
        matrix_unit="rows",
        time_unit="row_cost",
        cost_model_id="offline_common_v1",
        release_model="p1_return",
        return_model="transpose_dispatch",
        full_duplex=True,
        launch_cost=0.0,
        bytes_per_row=1,
        bandwidth=1.0,
        compute_delay=0.0,
        p2_semantics="actual",
        residual_policy="reject",
    )
    task_set = build_evaluation_task_set(window, spec)
    result = solve_cp_sat(task_set, mode="local")
    assert result.solver_status in {"UNSUPPORTED", "OPTIMAL", "FEASIBLE"}
    if result.solver_status == "UNSUPPORTED":
        assert result.certified_optimal is False
        assert result.objective is None
