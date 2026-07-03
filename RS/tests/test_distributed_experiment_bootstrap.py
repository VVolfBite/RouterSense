from __future__ import annotations

from pathlib import Path


def test_distributed_bootstrap_helper_exists():
    helper = Path(__file__).resolve().parents[1] / "experiments" / "distributed" / "_bootstrap.py"
    assert helper.exists()


def test_render_torchrun_dry_run_uses_current_experiment_path():
    from rs.topology import load_inventory, render_torchrun_dry_run

    inventory = load_inventory(Path(__file__).resolve().parents[1] / "deploy" / "inventory" / "hosts.example.yaml")
    payload = render_torchrun_dry_run(inventory)
    assert "experiments/distributed/future_multinode_smoke.py --dry-run" in payload["node0_command"]
    assert "--nproc_per_node=2" in payload["node0_command"]
