#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def _try_import(name: str) -> dict[str, object]:
    try:
        mod = __import__(name, fromlist=["*"])
        return {
            "available": True,
            "module": name,
            "file": getattr(mod, "__file__", None),
            "version": getattr(mod, "__version__", None),
        }
    except Exception as exc:
        return {
            "available": False,
            "module": name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/models/OLMoE-1B-7B-0125")
    args = parser.parse_args(argv)

    gpus = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                }
            )

    checks = {
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "nccl_available": bool(dist.is_nccl_available()),
        "visible_gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpus": gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
        "single_node_topology": True,
        "megatron_core": _try_import("megatron.core"),
        "megatron_bridge": _try_import("megatron.bridge"),
        "transformer_engine": _try_import("transformer_engine"),
        "transformers": _try_import("transformers"),
        "model_path": str(args.model),
        "model_exists": Path(args.model).exists(),
    }
    blocked = None
    if not checks["cuda_available"]:
        blocked = "cuda_unavailable"
    elif int(checks["visible_gpu_count"]) < 2:
        blocked = "only_one_visible_gpu"
    elif not checks["nccl_available"]:
        blocked = "nccl_unavailable"
    elif not checks["megatron_core"]["available"] or not checks["megatron_bridge"]["available"] or not checks["transformer_engine"]["available"]:
        blocked = "missing_dependency"
    elif not checks["model_exists"]:
        blocked = "model_not_found"

    payload = {
        "pipeline": "host_runtime_native_ep",
        "host_runtime": "megatron_core",
        "status": "blocked_environment" if blocked else "ready",
        "reason": blocked,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2))
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
