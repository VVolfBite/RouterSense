from __future__ import annotations

"""Model-cache inspection used by deployment preflight tooling.

The deployment path must distinguish a configured path from a usable model
snapshot.  This module intentionally performs only filesystem checks: it does
not import Transformers, allocate a GPU, or contact the network.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


DEFAULT_DEPLOYMENT_MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"

_CONFIG_FILES = ("config.json",)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
)
_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_OPTIONAL_FILES = (
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def model_name(model_id: str) -> str:
    return str(model_id).rstrip("/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ModelCacheInspection:
    model_id: str
    configured_path: str
    model_path: str
    path_exists: bool
    is_directory: bool
    config_ready: bool
    tokenizer_ready: bool
    weights_ready: bool
    required_files_present: bool
    total_size_bytes: int
    manifest_hash: str
    discovered_files: tuple[str, ...]
    missing_requirements: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _index_weight_files(path: Path, index_name: str) -> tuple[Path, ...] | None:
    index_path = path / index_name
    if not index_path.is_file():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict):
        return ()
    names = sorted({str(value) for value in weight_map.values() if str(value).strip()})
    if not names:
        return ()
    return tuple(path / name for name in names)


def _has_weight_files(path: Path, names: set[str]) -> bool:
    for standalone in ("model.safetensors", "pytorch_model.bin"):
        if standalone in names and _nonempty_file(path / standalone):
            return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        referenced = _index_weight_files(path, index_name)
        if referenced is not None:
            return bool(referenced) and all(_nonempty_file(item) for item in referenced)
    sharded = tuple(path.glob("model-*.safetensors")) + tuple(path.glob("pytorch_model-*.bin"))
    return bool(sharded) and all(_nonempty_file(item) for item in sharded)


def _looks_like_model_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {item.name for item in path.iterdir() if item.is_file()}
    return bool(names.intersection(_CONFIG_FILES + _TOKENIZER_FILES + _WEIGHT_FILES + _OPTIONAL_FILES))


def resolve_model_directory(configured_path: str | Path | None, model_id: str = DEFAULT_DEPLOYMENT_MODEL_ID) -> Path:
    """Resolve either a direct snapshot directory or a cache-root child.

    Inventory files historically used both conventions.  A directory that
    already contains model files wins; otherwise ``<cache>/<model-name>`` is
    used.  The returned path may not exist so callers can report the exact
    missing target without guessing.
    """

    base = Path(str(configured_path or "")).expanduser()
    if _looks_like_model_directory(base):
        return base
    child = base / model_name(model_id)
    if child.exists() or not base.exists():
        return child
    # Compatibility for local snapshots whose directory omits '-Instruct'.
    compact_name = model_name(model_id).removesuffix("-Instruct")
    compact = base / compact_name
    if compact.exists():
        return compact
    return child


def _manifest(path: Path, files: Iterable[Path]) -> tuple[int, str]:
    import hashlib

    digest = hashlib.sha256()
    total = 0
    for file_path in sorted(files, key=lambda item: item.name):
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        total += int(size)
        digest.update(file_path.name.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        # Hash small metadata files fully.  For multi-GB weight shards, size and
        # filename provide a fast parity identity; byte-level transfer tools
        # still perform their own checks.
        if size <= 16 * 1024 * 1024:
            digest.update(file_path.read_bytes())
    return total, digest.hexdigest() if total or any(True for _ in files) else "missing"


def inspect_model_cache(
    configured_path: str | Path | None,
    model_id: str = DEFAULT_DEPLOYMENT_MODEL_ID,
) -> ModelCacheInspection:
    configured = Path(str(configured_path or "")).expanduser()
    model_path = resolve_model_directory(configured, model_id)
    exists = model_path.exists()
    is_dir = model_path.is_dir()
    files = tuple(item for item in model_path.iterdir() if item.is_file()) if is_dir else ()
    names = {item.name for item in files}
    config_ready = all(name in names for name in _CONFIG_FILES)
    tokenizer_ready = bool(names.intersection(_TOKENIZER_FILES))
    weights_ready = _has_weight_files(model_path, names) if is_dir else False
    required = bool(config_ready and tokenizer_ready and weights_ready)
    missing = []
    if not config_ready:
        missing.append("config")
    if not tokenizer_ready:
        missing.append("tokenizer")
    if not weights_ready:
        missing.append("weights")
    total, manifest_hash = _manifest(model_path, files)
    return ModelCacheInspection(
        model_id=str(model_id),
        configured_path=str(configured),
        model_path=str(model_path),
        path_exists=bool(exists),
        is_directory=bool(is_dir),
        config_ready=bool(config_ready),
        tokenizer_ready=bool(tokenizer_ready),
        weights_ready=bool(weights_ready),
        required_files_present=bool(required),
        total_size_bytes=int(total),
        manifest_hash=str(manifest_hash),
        discovered_files=tuple(sorted(names)),
        missing_requirements=tuple(missing),
    )


__all__ = [
    "DEFAULT_DEPLOYMENT_MODEL_ID",
    "ModelCacheInspection",
    "inspect_model_cache",
    "model_name",
    "resolve_model_directory",
]
