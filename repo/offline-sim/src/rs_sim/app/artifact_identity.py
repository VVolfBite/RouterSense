from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_digest(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _identity_files(root: Path) -> Iterable[Path]:
    preferred = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "latest_checkpointed_iteration.txt",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.name in preferred or path.suffix.lower() in {".yaml", ".yml"} and len(relative.parts) <= 2:
            yield path


def checkpoint_identity(model_path: Path, *, explicit_digest: str | None = None) -> dict[str, Any]:
    root = Path(model_path).expanduser().resolve()
    if explicit_digest:
        return {
            "schema_version": "RS_SIM_CHECKPOINT_IDENTITY",
            "status": "PASS",
            "mode": "EXPLICIT_SHA256",
            "model_path": str(root),
            "checkpoint_digest": str(explicit_digest),
        }
    if not root.exists():
        return {
            "schema_version": "RS_SIM_CHECKPOINT_IDENTITY",
            "status": "FAILED",
            "mode": "MISSING",
            "model_path": str(root),
        }
    rows: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            total_files += 1
            total_bytes += int(path.stat().st_size)
    for path in _identity_files(root):
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    inventory = {
        "root_name": root.name,
        "file_count": total_files,
        "total_bytes": total_bytes,
        "identity_files": rows,
    }
    return {
        "schema_version": "RS_SIM_CHECKPOINT_IDENTITY",
        "status": "PASS",
        "mode": "IDENTITY_FILES_AND_INVENTORY_SHA256",
        "model_path": str(root),
        "checkpoint_digest": stable_json_digest(inventory),
        "inventory": inventory,
        "note": "Digest covers identity/config/index files plus aggregate inventory; supply model.checkpoint_digest for a full external content digest.",
    }
