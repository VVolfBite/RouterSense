from __future__ import annotations

import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def write_artifact_bundle(root: str | Path, *, summary: dict[str, Any], goal: str, config: dict[str, Any], environment: dict[str, Any]) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root_path / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (root_path / "goal.md").write_text(goal, encoding="utf-8")

