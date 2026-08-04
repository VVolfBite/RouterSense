#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig


def _fail(reason: str, **details: object) -> int:
    print(json.dumps({"ok": False, "reason": reason, **details}, indent=2))
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    model_path = argv[0] if argv else os.environ.get("RS_MODEL_PATH")
    if not torch.cuda.is_available():
        return _fail("no_cuda")
    if int(torch.cuda.device_count()) < 2:
        return _fail("only_one_visible_gpu", visible_gpu_count=int(torch.cuda.device_count()))
    if not dist.is_nccl_available():
        return _fail("nccl_unavailable")
    if not model_path:
        return _fail("model_not_found", detail="model path argument missing")
    model_root = Path(model_path)
    if not model_root.exists():
        return _fail("model_not_found", model_path=str(model_root))
    try:
        config = AutoConfig.from_pretrained(str(model_root), trust_remote_code=True)
    except Exception as exc:
        return _fail("model_not_found", model_path=str(model_root), error=f"{type(exc).__name__}: {exc}")
    print(
        json.dumps(
            {
                "ok": True,
                "torch_version": torch.__version__,
                "cuda_available": True,
                "visible_gpu_count": int(torch.cuda.device_count()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nccl_available": bool(dist.is_nccl_available()),
                "gpu0_name": torch.cuda.get_device_name(0),
                "gpu1_name": torch.cuda.get_device_name(1),
                "model_path": str(model_root),
                "model_type": getattr(config, "model_type", None),
                "transformers_version": __import__("transformers").__version__,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
