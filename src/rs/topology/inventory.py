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
    ssh_host: str | None
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


DEFAULT_REMOTE_RS_ROOT = "/workspace/RouterSense"
DEFAULT_ARTIFACT_ROOT = "/workspace/routersense-artifacts"
DEFAULT_SSH_USER = "root"
DEFAULT_MASTER_PORT = 29500


def load_inventory(path: str | Path) -> Inventory:
    """Load either the full inventory schema or the compact deployment schema.

    The compact schema intentionally leaves only values that vary on PPIO:
    ``host``, optional ``ssh_host``/``ssh_port``/``ssh_user``, ``gpu_count``
    and ``model_path``.  Stable RouterSense paths, node names/ranks and
    rendezvous settings are derived deterministically.
    """

    payload = _load_mapping(Path(path).read_text(encoding="utf-8"))
    raw_nodes = payload.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raise RuntimeError("inventory nodes must parse to a list")
    nodes = [_load_node_spec(item, index=index) for index, item in enumerate(raw_nodes)]
    cluster_name = str(payload.get("cluster_name") or f"rs-{len(nodes)}node")
    inventory = Inventory(
        cluster_name=cluster_name,
        nodes=nodes,
        rendezvous=_load_rendezvous_spec(payload.get("rendezvous", {}), nodes=nodes),
    )
    validate_inventory(inventory)
    return inventory


def validate_inventory(inventory: Inventory) -> None:
    if not str(inventory.cluster_name).strip():
        raise ValueError("cluster_name must be non-empty")
    if not inventory.nodes:
        raise ValueError("inventory must contain at least one node")
    names = [str(node.name) for node in inventory.nodes]
    ranks = [int(node.node_rank) for node in inventory.nodes]
    if len(set(names)) != len(names):
        raise ValueError("inventory node names must be unique")
    if len(set(ranks)) != len(ranks) or sorted(ranks) != list(range(len(ranks))):
        raise ValueError("node_rank values must be unique and contiguous from zero")
    if str(inventory.rendezvous.master_node) not in set(names):
        raise ValueError("rendezvous.master_node must name an inventory node")
    if not 1 <= int(inventory.rendezvous.master_port) <= 65534:
        raise ValueError("rendezvous.master_port must leave room for the calibration port")
    target_counts = {int(node.target_gpu_count) for node in inventory.nodes}
    if min(target_counts, default=0) <= 0:
        raise ValueError("target_gpu_count must be positive on every node")
    if len(inventory.nodes) > 1 and len(target_counts) != 1:
        raise ValueError("multi-node deployment currently requires equal target_gpu_count per node")
    for node in inventory.nodes:
        if not str(node.host).strip() or not str(node.ssh_user).strip():
            raise ValueError(f"node {node.name} requires host and ssh_user")
        if not 1 <= int(node.port) <= 65535:
            raise ValueError(f"node {node.name} has invalid SSH port")
        if int(node.current_gpu_count) <= 0:
            raise ValueError(f"node {node.name} current_gpu_count must be positive")
        if int(node.current_gpu_count) < int(node.target_gpu_count):
            raise ValueError(
                f"node {node.name} current_gpu_count is below target_gpu_count "
                f"({node.current_gpu_count} < {node.target_gpu_count})"
            )
        required_paths = {"remote_rs_root", "model_cache", "artifact_root"}
        missing = sorted(key for key in required_paths if not str(node.paths.get(key, "")).strip())
        if missing:
            raise ValueError(f"node {node.name} is missing required paths: {', '.join(missing)}")
        non_absolute = sorted(
            key
            for key in required_paths
            if not Path(str(node.paths[key])).expanduser().is_absolute()
        )
        if non_absolute:
            raise ValueError(
                f"node {node.name} deployment paths must be absolute: {', '.join(non_absolute)}"
            )


def inventory_summary(inventory: Inventory) -> dict[str, Any]:
    return {
        "cluster_name": inventory.cluster_name,
        "nodes": [asdict(node) for node in inventory.nodes],
        "rendezvous": asdict(inventory.rendezvous),
    }


def inventory_cli_summary(inventory: Inventory, inventory_path: str | Path | None = None) -> dict[str, Any]:
    return {
        **inventory_summary(inventory),
        "resolved_paths": inventory_paths(inventory, inventory_path=inventory_path),
    }


def inventory_paths(inventory: Inventory, inventory_path: str | Path | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "rs_root": str(resolve_rs_root()),
        "inventory_path": str(resolve_inventory_path(inventory_path)),
    }
    for node in inventory.nodes:
        data[f"{node.name}_remote_rs_root"] = _stringify_path(resolve_node_rs_root(inventory, node.name))
        data[f"{node.name}_model_cache"] = _stringify_path(resolve_node_model_cache(inventory, node.name))
        data[f"{node.name}_artifact_root"] = _stringify_path(resolve_node_artifact_root(inventory, node.name))
    return data


def render_torchrun_dry_run(inventory: Inventory, *, nnodes: int = 2, nproc_per_node: int = 2) -> dict[str, Any]:
    if inventory.nodes:
        nnodes = len(inventory.nodes)
    master_node = _get_node(inventory, inventory.rendezvous.master_node)
    master_addr = str(master_node.host)
    master_port = int(inventory.rendezvous.master_port)
    interface_hint = ""
    rdzv_id = f"{inventory.cluster_name}-phase0c"
    target_gpu_counts = {node.name: int(node.target_gpu_count) for node in inventory.nodes}
    unique_proc_counts = sorted(set(target_gpu_counts.values()))
    payload = {
        "nnodes": nnodes,
        "nproc_per_node": unique_proc_counts[0] if len(unique_proc_counts) == 1 else None,
        "target_gpu_counts": target_gpu_counts,
        "world_size": sum(target_gpu_counts.values()),
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
            nproc_per_node=int(node.target_gpu_count),
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
        f"ssh -p {node.port} {node.ssh_user}@{node.ssh_host or node.host} "
        f"'{ ' '.join(env) } torchrun --nnodes={nnodes} --nproc_per_node={nproc_per_node} "
        f"--node_rank={node_rank} --rdzv-backend=c10d --rdzv-id={rendezvous_id} "
        f"--rdzv-endpoint={master_addr}:{master_port} "
        "experiments.online formal dry-run entrypoint'"
    )


def _load_node_spec(payload: dict[str, Any], *, index: int = 0) -> NodeSpec:
    if not isinstance(payload, dict):
        raise RuntimeError(f"inventory node {index} must parse to a mapping")
    host = str(payload.get("host", ""))
    gpu_count = int(
        payload.get(
            "gpu_count",
            payload.get("target_gpu_count", payload.get("current_gpu_count", 0)),
        )
    )
    target_gpu_count = int(payload.get("target_gpu_count", gpu_count))
    current_gpu_count = int(payload.get("current_gpu_count", gpu_count or target_gpu_count))
    paths = dict(payload.get("paths", {}))
    model_path = payload.get("model_path", payload.get("model_cache"))
    if model_path is not None and not paths.get("model_cache"):
        paths["model_cache"] = str(model_path)
    paths.setdefault(
        "remote_rs_root",
        str(payload.get("remote_rs_root", DEFAULT_REMOTE_RS_ROOT)),
    )
    paths.setdefault(
        "artifact_root",
        str(payload.get("artifact_root", DEFAULT_ARTIFACT_ROOT)),
    )
    return NodeSpec(
        name=str(payload.get("name") or f"node{index}"),
        host=host,
        ssh_host=str(payload["ssh_host"]) if payload.get("ssh_host") else None,
        port=int(payload.get("port", payload.get("ssh_port", 22))),
        ssh_user=str(payload.get("ssh_user") or DEFAULT_SSH_USER),
        node_rank=int(payload.get("node_rank", index)),
        current_gpu_count=current_gpu_count,
        target_gpu_count=target_gpu_count,
        paths=paths,
    )


def _load_rendezvous_spec(
    payload: dict[str, Any],
    *,
    nodes: list[NodeSpec] | None = None,
) -> RendezvousSpec:
    nodes = list(nodes or [])
    default_master = nodes[0].name if nodes else "node0"
    return RendezvousSpec(
        master_node=str(payload.get("master_node") or default_master),
        master_port=int(payload.get("master_port", DEFAULT_MASTER_PORT)),
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
