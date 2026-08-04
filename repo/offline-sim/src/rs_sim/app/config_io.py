from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {config_path}") from exc
    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            value = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ConfigError("YAML config requires PyYAML; use JSON or install PyYAML") from exc
            value = yaml.safe_load(text)
        else:
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                try:
                    import yaml  # type: ignore
                except ImportError as exc:
                    raise ConfigError("config must be JSON, or YAML with PyYAML installed") from exc
                value = yaml.safe_load(text)
    except Exception as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid config {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("config root must be an object")
    value = dict(value)
    value["__config_path"] = str(config_path)
    value["__config_dir"] = str(config_path.parent)
    return value


def resolve_path(value: str | Path, *, config: dict[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(config["__config_dir"]) / path).resolve()


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return dict(value)


def require_nonempty(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ConfigError(f"{name} must be non-empty")
    return text


def reject_unknown_fields(value: dict[str, Any], allowed: set[str] | frozenset[str], name: str) -> None:
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ConfigError(f"{name} contains unknown fields: {', '.join(unknown)}")


def require_bool(value: Any, name: str, *, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return bool(default)
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean, not {type(value).__name__}")
    return value
