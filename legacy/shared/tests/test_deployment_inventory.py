from __future__ import annotations

import json
from pathlib import Path

from routesense.topology import inventory_summary, load_inventory, render_torchrun_dry_run


def test_inventory_example_parses():
    inventory = load_inventory(Path("deploy/inventory/hosts.example.yaml"))
    summary = inventory_summary(inventory)
    assert summary["cluster_name"] == "rs-phase0b"
    assert summary["controller"]["role"] == "control"
    assert summary["executor"]["role"] == "gpu_executor"


def test_torchrun_dry_run_contains_future_contract():
    inventory = load_inventory(Path("deploy/inventory/hosts.example.yaml"))
    payload = render_torchrun_dry_run(inventory)
    assert payload["nnodes"] == 2
    assert payload["nproc_per_node"] == 2
    assert "torchrun" in payload["controller_command"]
    assert "--node_rank=0" in payload["controller_command"]
    assert "--node_rank=1" in payload["executor_command"]
