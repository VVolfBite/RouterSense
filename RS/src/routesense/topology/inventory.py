from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HostSpec:
    host: str
    ssh_user: str
    role: str
    node_rank_future: int
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_identity_file: str | None = None
    ssh_extra_args: list[str] = field(default_factory=list)


@dataclass
class Inventory:
    cluster_name: str
    controller: HostSpec
    executor: HostSpec
    paths: dict[str, Any]
    network: dict[str, Any]


def load_inventory(path: str | Path) -> Inventory:
    payload = _load_mapping(Path(path).read_text(encoding="utf-8"))
    return Inventory(
        cluster_name=str(payload["cluster_name"]),
        controller=_load_host_spec(payload["controller"]),
        executor=_load_host_spec(payload["executor"]),
        paths=dict(payload.get("paths", {})),
        network=dict(payload.get("network", {})),
    )


def inventory_summary(inventory: Inventory) -> dict[str, Any]:
    return {
        "cluster_name": inventory.cluster_name,
        "controller": asdict(inventory.controller),
        "executor": asdict(inventory.executor),
        "paths": dict(inventory.paths),
        "network": dict(inventory.network),
    }


def render_torchrun_dry_run(inventory: Inventory, *, nnodes: int = 2, nproc_per_node: int = 2) -> dict[str, Any]:
    master_port = int(inventory.network.get("master_port_future", 29500))
    master_addr = str(inventory.controller.ssh_host or inventory.controller.host)
    interface_hint = str(inventory.network.get("interface_hint", ""))
    return {
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "master_addr": master_addr,
        "master_port": master_port,
        "controller_command": _render_torchrun_command(
            host=inventory.controller,
            node_rank=0,
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            master_addr=master_addr,
            master_port=master_port,
            interface_hint=interface_hint,
        ),
        "executor_command": _render_torchrun_command(
            host=inventory.executor,
            node_rank=1,
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            master_addr=master_addr,
            master_port=master_port,
            interface_hint=interface_hint,
        ),
    }


def _render_torchrun_command(
    *,
    host: HostSpec,
    node_rank: int,
    nnodes: int,
    nproc_per_node: int,
    master_addr: str,
    master_port: int,
    interface_hint: str,
) -> str:
    env = ["TORCHDISTRIBUTED_DEBUG=DETAIL"]
    if interface_hint:
        env.append(f"NCCL_SOCKET_IFNAME={interface_hint}")
    return (
        f"ssh {host.ssh_user}@{host.ssh_host or host.host} "
        f"'{ ' '.join(env) } torchrun --standalone --nnodes={nnodes} --nproc_per_node={nproc_per_node} "
        f"--node_rank={node_rank} --master_addr={master_addr} --master_port={master_port} "
        "experiments/deployment/future_multinode_smoke.py --dry-run'"
    )


def _load_host_spec(payload: dict[str, Any]) -> HostSpec:
    return HostSpec(
        host=str(payload["host"]),
        ssh_user=str(payload["ssh_user"]),
        role=str(payload["role"]),
        node_rank_future=int(payload.get("node_rank_future", 0)),
        ssh_host=payload.get("ssh_host"),
        ssh_port=int(payload["ssh_port"]) if payload.get("ssh_port") is not None else None,
        ssh_identity_file=payload.get("ssh_identity_file"),
        ssh_extra_args=list(payload.get("ssh_extra_args", [])),
    )


def _load_mapping(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise RuntimeError("inventory file must parse to a mapping")
        return payload

