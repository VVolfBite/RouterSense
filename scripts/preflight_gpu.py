#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc1.config import ensure_output_dir


def main() -> int:
    output_dir = ensure_output_dir(str(ROOT / "outputs"))
    payload = {
        "python_version": platform.python_version(),
        "torch_version": None,
        "transformers_version": None,
        "accelerate_version": None,
        "cuda_available": False,
        "gpu_names": [],
        "gpu_count": 0,
        "gpu_memory_gb": [],
        "cpu_count": os.cpu_count(),
        "system_memory_gb": None,
        "disk_checked_path": str(ROOT),
        "disk_available_gb": None,
        "disk_value_warning": None,
        "huggingface_cache_dir": os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or str(Path.home() / ".cache" / "huggingface"),
        "cwd_writable": os.access(os.getcwd(), os.W_OK),
        "float16_cuda_tensor": False,
        "network_reachable": False,
        "nvidia_smi_summary": None,
    }

    try:
        import torch  # type: ignore
        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            payload["gpu_count"] = torch.cuda.device_count()
            payload["gpu_memory_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2) for i in range(torch.cuda.device_count())]
            try:
                payload["float16_cuda_tensor"] = torch.tensor([1.0], dtype=torch.float16, device="cuda").is_cuda
            except Exception:
                payload["float16_cuda_tensor"] = False
    except Exception:
        payload["torch_version"] = None

    try:
        import transformers  # type: ignore
        payload["transformers_version"] = transformers.__version__
    except Exception:
        payload["transformers_version"] = None

    try:
        import accelerate  # type: ignore
        payload["accelerate_version"] = accelerate.__version__
    except Exception:
        payload["accelerate_version"] = None

    try:
        statvfs = shutil.disk_usage(str(ROOT))
        payload["disk_available_gb"] = round(statvfs.free / (1024**3), 2)
        if statvfs.free < 1_000_000 * 1024**3:
            payload["disk_value_warning"] = "available disk space is low"
        else:
            payload["disk_value_warning"] = "available disk space is sufficient"
    except Exception:
        payload["disk_available_gb"] = None
        payload["disk_value_warning"] = "disk_usage_failed"

    try:
        import urllib.request
        with urllib.request.urlopen("https://huggingface.co", timeout=10) as response:
            payload["network_reachable"] = response.status == 200
    except Exception:
        payload["network_reachable"] = False

    try:
        import subprocess
        completed = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=False)
        payload["nvidia_smi_summary"] = completed.stdout.strip() or None
    except Exception:
        payload["nvidia_smi_summary"] = None

    try:
        import psutil  # type: ignore
        payload["system_memory_gb"] = round(psutil.virtual_memory().available / (1024**3), 2)
    except Exception:
        payload["system_memory_gb"] = None

    with (output_dir / "preflight_gpu.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
