#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _emit(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, indent=2))
    return 2 if payload.get("status") == "blocked_environment" else 0


def _module_status(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return {
                "available": False,
                "module": name,
                "error": "ModuleNotFoundError",
            }
        package_name = name.split(".", 1)[0]
        try:
            version = importlib.metadata.version(package_name)
        except Exception:
            version = None
        return {
            "available": True,
            "module": name,
            "file": spec.origin,
            "version": version,
        }
    except Exception as exc:
        return {
            "available": False,
            "module": name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _blocked(reason: str, missing: list[str], checks: dict[str, Any]) -> int:
    return _emit(
        {
            "pipeline": "host_runtime_native_ep",
            "host_runtime": "megatron_core",
            "status": "blocked_environment",
            "reason": reason,
            "missing": missing,
            **checks,
        }
    )


def _safe_local_path(model: str) -> bool:
    try:
        return Path(model).expanduser().is_absolute() or model.startswith("/") or model.startswith(".")
    except Exception:
        return False


def _safe_path_exists(model: str) -> tuple[bool, str | None]:
    try:
        path = Path(model)
        if not path.exists():
            return False, "model_path_missing"
        try:
            list(path.iterdir()) if path.is_dir() else path.stat()
        except PermissionError:
            return False, "model_path_unreadable"
        return True, None
    except PermissionError:
        return False, "model_path_unreadable"
    except Exception:
        return False, "model_path_missing"


def main(argv: list[str] | None = None) -> int:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="/root/autodl-tmp/models/OLMoE-1B-7B-0924")
        args = parser.parse_args(argv)

        base = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

        torch_status = _module_status("torch")
        transformers_status = _module_status("transformers")
        megatron_core_status = _module_status("megatron.core")
        megatron_bridge_status = _module_status("megatron.bridge")
        te_status = _module_status("transformer_engine")

        checks: dict[str, Any] = {
            **base,
            "torch_version": torch_status.get("version"),
            "cuda_available": False,
            "cuda_version": None,
            "nccl_available": False,
            "visible_gpu_count": 0,
            "gpus": [],
            "single_node_topology": True,
            "megatron_core": megatron_core_status,
            "megatron_bridge": megatron_bridge_status,
            "transformer_engine": te_status,
            "transformers": transformers_status,
            "model_path": str(args.model),
            "model_is_local_path": False,
            "model_exists": False,
            "model_config": {"available": False},
        }

        missing: list[str] = []
        if not torch_status["available"]:
            missing.append("torch")
            return _blocked("missing_dependency", missing, checks)

        import torch
        import torch.distributed as dist

        checks["torch_version"] = torch.__version__
        checks["cuda_available"] = bool(torch.cuda.is_available())
        checks["cuda_version"] = torch.version.cuda
        checks["nccl_available"] = bool(dist.is_nccl_available())
        checks["visible_gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        checks["model_is_local_path"] = _safe_local_path(str(args.model))
        model_exists, model_reason = _safe_path_exists(str(args.model)) if checks["model_is_local_path"] else (False, None)
        checks["model_exists"] = model_exists

        if checks["cuda_available"]:
            gpus = []
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append(
                    {
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    }
                )
            checks["gpus"] = gpus

        if not transformers_status["available"]:
            missing.append("transformers")
        if not megatron_core_status["available"]:
            missing.append("megatron.core")
        if not megatron_bridge_status["available"]:
            missing.append("megatron.bridge")
        if not te_status["available"]:
            missing.append("transformer_engine")

        if not checks["cuda_available"]:
            return _blocked("cuda_unavailable", missing, checks)
        if int(checks["visible_gpu_count"]) < 2:
            return _blocked("insufficient_visible_gpus", missing, checks)
        if not checks["nccl_available"]:
            return _blocked("nccl_unavailable", missing, checks)
        if missing:
            return _blocked("missing_dependency", missing, checks)
        if model_reason == "model_path_unreadable":
            return _blocked("model_path_unreadable", missing, checks)

        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                str(args.model),
                trust_remote_code=True,
                local_files_only=bool(checks["model_is_local_path"]),
            )
            checks["model_config"] = {
                "available": True,
                "model_type": getattr(config, "model_type", None),
                "architectures": getattr(config, "architectures", None),
            }
        except Exception as exc:
            checks["model_config"] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if not checks["model_exists"] and not checks["model_config"]["available"]:
            return _blocked(model_reason or "model_path_missing", missing, checks)

        return _emit(
            {
                "pipeline": "host_runtime_native_ep",
                "host_runtime": "megatron_core",
                "status": "ready",
                "reason": None,
                "missing": missing,
                **checks,
            }
        )
    except Exception as exc:
        return _emit(
            {
                "pipeline": "host_runtime_native_ep",
                "host_runtime": "megatron_core",
                "status": "blocked_environment",
                "reason": "unexpected_environment_validation_failure",
                "missing": [],
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
