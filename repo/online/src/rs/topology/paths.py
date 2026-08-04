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


def resolve_model_path_for_node(
    inventory: Any,
    node_name: str,
    model_id: str,
) -> Path | None:
    cache_root = resolve_node_model_cache(inventory, node_name)
    if cache_root is None:
        return None
    return cache_root / resolve_model_name(model_id)


def resolve_preferred_model_path(
    inventory: Any,
    model_id: str,
    preferred_node_name: str | None = None,
) -> Path | None:
    node_names = [str(getattr(node, "name", "")) for node in getattr(inventory, "nodes", [])]
    ordered: list[str] = []
    if preferred_node_name:
        ordered.append(preferred_node_name)
    ordered.extend(name for name in node_names if name and name not in ordered)
    for node_name in ordered:
        candidate = resolve_model_path_for_node(inventory, node_name, model_id)
        if candidate is not None:
            return candidate
    return None


def _resolve_node_path(inventory: Any, node_name: str, key: str) -> Path | None:
    for node in getattr(inventory, "nodes", []):
        if getattr(node, "name", None) == node_name:
            value = getattr(node, "paths", {}).get(key)
            return None if value in (None, "") else Path(str(value))
    raise KeyError(f"unknown node: {node_name}")
