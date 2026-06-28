from __future__ import annotations

from pathlib import Path


def test_node_inventory_is_symmetric():
    from routesense.topology import load_inventory

    inventory = load_inventory(Path(__file__).resolve().parents[1] / "deploy" / "inventory" / "hosts.example.yaml")
    assert inventory.cluster_name == "rs-2node-ep"
    assert len(inventory.nodes) == 2
    assert {node.name for node in inventory.nodes} == {"node0", "node1"}
    assert inventory.rendezvous.master_node == "node0"
    assert inventory.rendezvous.backend == "c10d"
