from __future__ import annotations

import json
from pathlib import Path

from experiments.paper.adapters.scheduling_adapter import execute_policy, replay_window_from_matrices


def test_islip_execution_window_covers_p2() -> None:
    fixture = json.loads(Path("tests/fixtures/offline_replay_smoke/replay_layer_1.json").read_text(encoding="utf-8"))
    matrix = lambda key: tuple(tuple(int(value) for value in row) for row in fixture[key])
    window = replay_window_from_matrices(
        fixture_id="islip-p2",
        layer_id=1,
        p0_matrix=matrix("p0_dispatch_matrix"),
        p1_matrix=matrix("p1_return_matrix"),
        p2_matrix=matrix("p2_next_dispatch_matrix"),
    )
    result = execute_policy(
        replay_window=window,
        policy_name="islip_bucket",
        hint_type="perfect_trace_hint",
        p2_hint_rows=window.p2_truth_rows,
    )
    assert result["audit_valid"] is True, result["audit"]["validation_errors"]
    served = result["audit"]["served_volume_by_phase"]
    target = result["audit"]["target_volume_by_phase"]
    assert float(served[2]) == float(target[2]) > 0.0
