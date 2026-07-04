from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or torch.cuda.device_count() < 2 or (not dist.is_nccl_available()),
    reason="requires 2 visible CUDA devices and NCCL support",
)
def test_online_ws2_nccl_native_ep_layer(tmp_path) -> None:
    model_path = os.environ.get("RS_MODEL_PATH", r"D:\Project\Test\OLMoE")
    if not Path(model_path).exists():
        pytest.skip(f"model path not found: {model_path}")
    output_dir = tmp_path / "nccl-native-ep"
    command = [
        "torchrun",
        "--nproc_per_node=2",
        "experiments/online/bench_native_ep.py",
        "--world-size",
        "2",
        "--backend",
        "nccl",
        "--model-path",
        model_path,
        "--prompt-rank0",
        "Explain sparse expert routing in one paragraph.",
        "--prompt-rank1",
        "Summarize why expert parallel dispatch needs ownership tracking.",
        "--layer-index",
        "0",
        "--precision",
        "fp16",
        "--require-remote-route",
        "--validate",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"torchrun failed with code {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    merged_path = next(output_dir.glob("*-merged.json"))
    rank0_path = next(output_dir.glob("*-rank0.json"))
    rank1_path = next(output_dir.glob("*-rank1.json"))
    rank0 = json.loads(rank0_path.read_text(encoding="utf-8"))
    rank1 = json.loads(rank1_path.read_text(encoding="utf-8"))
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    assert rank0["backend"] == "nccl"
    assert rank1["backend"] == "nccl"
    assert rank0["device_info"]["device"] == "cuda:0"
    assert rank1["device_info"]["device"] == "cuda:1"
    assert rank0["correctness_status"] == "passed"
    assert rank1["correctness_status"] == "passed"
    assert rank0["numerical_correctness_pass"] is True
    assert rank1["numerical_correctness_pass"] is True
    assert rank0["remote_route_count"] > 0
    assert rank1["remote_route_count"] > 0
    assert rank0["dispatch_rows"] > 0
    assert rank1["combine_rows"] > 0
    assert merged["world_size"] == 2
