"""Small artifact helpers shared by offline and online code."""

from __future__ import annotations

import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def collect_environment_snapshot() -> dict[str, Any]:
    import torch  # type: ignore

    snapshot: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(torch.version, "cuda", None),
        "visible_gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
    }
    if torch.cuda.is_available():
        snapshot["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return snapshot
