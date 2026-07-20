from __future__ import annotations

from pathlib import Path

import pytest

from rs.topology.inventory import Inventory, NodeSpec, RendezvousSpec, validate_inventory


def _inventory(*, current: int = 4, target: int = 4, root: str = "/workspace/RouterSense") -> Inventory:
    return Inventory(
        cluster_name="unit",
        nodes=[
            NodeSpec(
                name="node0",
                host="10.0.0.1",
                ssh_host="203.0.113.1",
                port=22,
                ssh_user="root",
                node_rank=0,
                current_gpu_count=current,
                target_gpu_count=target,
                paths={
                    "remote_rs_root": root,
                    "model_cache": "/models",
                    "artifact_root": "/artifacts",
                },
            )
        ],
        rendezvous=RendezvousSpec(master_node="node0", master_port=29500, backend="c10d"),
    )


def test_inventory_accepts_absolute_paths_and_sufficient_gpu_capacity() -> None:
    validate_inventory(_inventory())


def test_inventory_rejects_gpu_capacity_below_target() -> None:
    with pytest.raises(ValueError, match="below target_gpu_count"):
        validate_inventory(_inventory(current=2, target=4))


def test_inventory_rejects_relative_remote_paths() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        validate_inventory(_inventory(root="workspace/RouterSense"))
