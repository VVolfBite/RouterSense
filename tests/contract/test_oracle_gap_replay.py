from __future__ import annotations

from pathlib import Path

from rs.runtime.offline.oracle_gap_replay import run_oracle_gap_replay
from rs.scheduling.reference.exact_small_instance import EXACT_REFERENCE_MODEL_ID


def test_oracle_gap_replay_cli(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    payload = run_oracle_gap_replay(
        fixture_dir=fixture_dir,
        small_only=True,
        policies=(
            "phase_barrier_fifo",
            "birkhoff_phase_local",
            "current:p012:local:global:rscf",
            "current:p012:joint:global:rscf",
        ),
    )
    assert payload["O_local_definition"] == "exact_runtime_bucket_wave_scope=local"
    assert payload["O_joint_definition"] == "exact_runtime_bucket_wave_scope=joint"
    assert payload["reference_model"] == EXACT_REFERENCE_MODEL_ID
    assert payload["scope_comparison_contract"]["same_tasks"] is True
    assert payload["scope_comparison_contract"]["same_cost_model"] is True
    assert payload["scope_comparison_contract"]["only_scope_changes"] is True
    assert payload["O_joint_small_fixture_available"] is True
    assert payload["oracle_gap_small_fixture_summary"]["dominance_violation"] is False
