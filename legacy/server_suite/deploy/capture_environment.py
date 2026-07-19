from __future__ import annotations
import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(argv: list[str]) -> dict:
    try:
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        return {"argv": argv, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch_info = None
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "nccl_available": bool(torch.distributed.is_nccl_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "schema_version": "routersense.environment.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "run_id": os.environ.get("ROUTERSENSE_RUN_ID"),
        "torch": torch_info,
        "nvidia_smi_list": _run(["nvidia-smi", "-L"]),
        "nvidia_smi_query": _run([
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]),
        "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"]),
        "environment": {key: value for key, value in os.environ.items() if key.startswith(("NCCL_", "CUDA_", "TORCH_", "ROUTERSENSE_"))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
