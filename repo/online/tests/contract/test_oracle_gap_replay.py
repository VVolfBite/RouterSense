from __future__ import annotations

from pathlib import Path

from experiments.offline.run_oracle_gap_replay import run_oracle_gap_replay


def test_oracle_gap_replay_cli(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    payload = run_oracle_gap_replay(
        fixture_dir=fixture_dir,
        small_only=True,
        policies=(
            "birkhoff_von_neumann_fluid",
            "exact_small_instance_reference",
            "B_birkhoff",
            "U_barrier_criticality_global_matching",
            "RS_safe_barrier_criticality",
        ),
    )
    assert payload["O_local_definition"] == "birkhoff_von_neumann_fluid"
    assert payload["O_joint_small_fixture_available"] is True
    assert "oracle_gap_small_fixture_summary" in payload
