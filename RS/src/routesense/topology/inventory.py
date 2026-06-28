from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import (
    resolve_inventory_path,
    resolve_model_name,
    resolve_node_artifact_root,
    resolve_node_model_cache,
    resolve_node_rs_root,
    resolve_rs_root,
)


@dataclass
class NodeSpec:
    name: str
    host: str
    port: int
    ssh_user: str
    node_rank: int
    current_gpu_count: int
    target_gpu_count: int
    paths: dict[str, Any] = field(default_factory=dict)


@dataclass
class RendezvousSpec:
    master_node: str
    master_port: int
    backend: str


@dataclass
class Inventory:
    cluster_name: str
    nodes: list[NodeSpec]
    rendezvous: RendezvousSpec


def load_inventory(path: str | Path) -> Inventory:
    payload = _load_mapping(Path(path).read_text(encoding="utf-8"))
    return Inventory(
        cluster_name=str(payload["cluster_name"]),
        nodes=[_load_node_spec(item) for item in payload.get("nodes", [])],
        rendezvous=_load_rendezvous_spec(payload["rendezvous"]),
    )


def inventory_summary(inventory: Inventory) -> dict[str, Any]:
    return {
        "cluster_name": inventory.cluster_name,
        "nodes": [asdict(node) for node in inventory.nodes],
        "rendezvous": asdict(inventory.rendezvous),
    }


def inventory_cli_summary(inventory: Inventory) -> dict[str, Any]:
    return {
        **inventory_summary(inventory),
        "resolved_paths": inventory_paths(inventory),
    }


def inventory_paths(inventory: Inventory) -> dict[str, Any]:
    data: dict[str, Any] = {
        "rs_root": str(resolve_rs_root()),
        "inventory_path": str(resolve_inventory_path()),
    }
    for node in inventory.nodes:
        data[f"{node.name}_remote_rs_root"] = _stringify_path(resolve_node_rs_root(inventory, node.name))
        data[f"{node.name}_model_cache"] = _stringify_path(resolve_node_model_cache(inventory, node.name))
        data[f"{node.name}_artifact_root"] = _stringify_path(resolve_node_artifact_root(inventory, node.name))
    return data


def render_torchrun_dry_run(inventory: Inventory, *, nnodes: int = 2, nproc_per_node: int = 2) -> dict[str, Any]:
    master_node = _get_node(inventory, inventory.rendezvous.master_node)
    master_addr = str(master_node.host)
    master_port = int(inventory.rendezvous.master_port)
    interface_hint = ""
    rdzv_id = f"{inventory.cluster_name}-phase0c"
    payload = {
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "master_addr": master_addr,
        "master_port": master_port,
        "rendezvous_backend": inventory.rendezvous.backend,
        "rendezvous_id": rdzv_id,
    }
    commands = {}
    for node in inventory.nodes:
        commands[f"{node.name}_command"] = _render_torchrun_command(
            node=node,
            node_rank=node.node_rank,
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            master_addr=master_addr,
            master_port=master_port,
            rendezvous_id=rdzv_id,
            interface_hint=interface_hint,
        )
    return {**payload, **commands}


def _render_torchrun_command(
    *,
    node: NodeSpec,
    node_rank: int,
    nnodes: int,
    nproc_per_node: int,
    master_addr: str,
    master_port: int,
    rendezvous_id: str,
    interface_hint: str,
) -> str:
    env = ["TORCHDISTRIBUTED_DEBUG=DETAIL"]
    if interface_hint:
        env.append(f"NCCL_SOCKET_IFNAME={interface_hint}")
    return (
        f"ssh -p {node.port} {node.ssh_user}@{node.host} "
        f"'{ ' '.join(env) } torchrun --nnodes={nnodes} --nproc_per_node={nproc_per_node} "
        f"--node_rank={node_rank} --rdzv-backend=c10d --rdzv-id={rendezvous_id} "
        f"--rdzv-endpoint={master_addr}:{master_port} "
        "RS/experiments/deployment/future_multinode_smoke.py --dry-run'"
    )


def _load_node_spec(payload: dict[str, Any]) -> NodeSpec:
    return NodeSpec(
        name=str(payload["name"]),
        host=str(payload["host"]),
        port=int(payload.get("port", 22)),
        ssh_user=str(payload["ssh_user"]),
        node_rank=int(payload["node_rank"]),
        current_gpu_count=int(payload.get("current_gpu_count", 0)),
        target_gpu_count=int(payload.get("target_gpu_count", 0)),
        paths=dict(payload.get("paths", {})),
    )


def _load_rendezvous_spec(payload: dict[str, Any]) -> RendezvousSpec:
    return RendezvousSpec(
        master_node=str(payload["master_node"]),
        master_port=int(payload["master_port"]),
        backend=str(payload.get("backend", "c10d")),
    )


def _get_node(inventory: Inventory, name: str) -> NodeSpec:
    for node in inventory.nodes:
        if node.name == name:
            return node
    raise KeyError(name)


def _load_mapping(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise RuntimeError("inventory file must parse to a mapping")
        return payload


def _stringify_path(value: Path | None) -> str | None:
    return None if value is None else str(value)
