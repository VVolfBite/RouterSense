from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedPaths:
    rs_root: Path
    inventory_path: Path
    node_paths: dict[str, dict[str, Path | None]]


def resolve_rs_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_inventory_path(inventory_path: str | Path | None = None) -> Path:
    if inventory_path is not None:
        return Path(inventory_path)
    return resolve_rs_root() / "deploy" / "inventory" / "hosts.local.yaml"


def resolve_node_rs_root(inventory: Any, node_name: str) -> Path | None:
    return _resolve_node_path(inventory, node_name, "remote_rs_root")


def resolve_node_model_cache(inventory: Any, node_name: str) -> Path | None:
    return _resolve_node_path(inventory, node_name, "model_cache")


def resolve_node_artifact_root(inventory: Any, node_name: str) -> Path | None:
    return _resolve_node_path(inventory, node_name, "artifact_root")


def resolve_model_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _resolve_node_path(inventory: Any, node_name: str, key: str) -> Path | None:
    for node in getattr(inventory, "nodes", []):
        if getattr(node, "name", None) == node_name:
            value = getattr(node, "paths", {}).get(key)
            return None if value in (None, "") else Path(str(value))
    raise KeyError(f"unknown node: {node_name}")
