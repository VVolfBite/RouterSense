from __future__ import annotations

from pathlib import Path

import pytest

from rs.topology.inventory import Inventory, NodeSpec, RendezvousSpec, load_inventory, validate_inventory


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


def test_compact_inventory_derives_stable_defaults(tmp_path: Path) -> None:
    path = tmp_path / "hosts.yaml"
    path.write_text(
        """
nodes:
  - host: 10.0.0.1
    ssh_host: 203.0.113.1
    gpu_count: 4
    model_path: /mnt/models/olmoe
""".strip() + "\n",
        encoding="utf-8",
    )
    inventory = load_inventory(path)
    node = inventory.nodes[0]
    assert inventory.cluster_name == "rs-1node"
    assert inventory.rendezvous.master_node == "node0"
    assert inventory.rendezvous.master_port == 29500
    assert node.name == "node0"
    assert node.node_rank == 0
    assert node.ssh_user == "root"
    assert node.port == 22
    assert node.current_gpu_count == 4
    assert node.target_gpu_count == 4
    assert node.paths == {
        "model_cache": "/mnt/models/olmoe",
        "remote_rs_root": "/workspace/RouterSense",
        "artifact_root": "/workspace/routersense-artifacts",
    }


def test_compact_inventory_accepts_ssh_port_and_user_overrides(tmp_path: Path) -> None:
    path = tmp_path / "hosts.yaml"
    path.write_text(
        """
nodes:
  - host: 10.0.0.1
    gpu_count: 2
    model_path: /models
    ssh_user: ubuntu
    ssh_port: 2222
""".strip() + "\n",
        encoding="utf-8",
    )
    node = load_inventory(path).nodes[0]
    assert node.ssh_host is None
    assert node.ssh_user == "ubuntu"
    assert node.port == 2222
