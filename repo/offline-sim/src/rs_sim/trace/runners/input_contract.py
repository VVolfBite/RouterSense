from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def global_sample_indices(*, source_rank: int, local_batch_size: int) -> tuple[int, ...]:
    start = int(source_rank) * int(local_batch_size)
    return tuple(range(start, start + int(local_batch_size)))


def sample_seed(*, base_seed: int, measured_sample: int, global_sample_index: int, warmup: bool = False) -> int:
    phase = 900_000_000 if warmup else 0
    return int(base_seed) + phase + int(measured_sample) * 10_000_019 + int(global_sample_index) * 100_003


def build_distributed_tokens(
    torch: Any,
    *,
    vocab_size: int,
    seq_length: int,
    local_batch_size: int,
    source_rank: int,
    base_seed: int,
    sample_index: int,
    device: str = "cuda",
    warmup: bool = False,
):
    rows = []
    indices = global_sample_indices(source_rank=source_rank, local_batch_size=local_batch_size)
    for global_index in indices:
        generator = torch.Generator(device=device)
        generator.manual_seed(sample_seed(
            base_seed=base_seed,
            measured_sample=sample_index,
            global_sample_index=global_index,
            warmup=warmup,
        ))
        rows.append(torch.randint(
            low=0,
            high=int(vocab_size),
            size=(int(seq_length),),
            generator=generator,
            device=device,
            dtype=torch.long,
        ))
    return torch.stack(rows, dim=0), indices


def persist_input_artifact(
    tokens: Any,
    *,
    output_dir: Path,
    rank: int,
    sample_index: int,
    global_indices: tuple[int, ...],
    seq_length: int,
    local_batch_size: int,
    global_batch_size: int,
    base_seed: int,
    save_token_ids: bool,
) -> dict[str, Any]:
    import numpy as np

    cpu = tokens.detach().to(device="cpu").contiguous()
    array = cpu.numpy().astype(np.dtype("<i4"), copy=False)
    raw = array.tobytes(order="C")
    digest = hashlib.sha256(raw).hexdigest()
    relative_path = None
    if save_token_ids:
        target_dir = Path(output_dir) / "inputs"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"rank{int(rank):04d}_sample{int(sample_index):04d}_input_ids.i32le"
        path.write_bytes(raw)
        relative_path = path.relative_to(output_dir).as_posix()
    return {
        "schema_version": "RS_SIM_INPUT_SAMPLE",
        "rank": int(rank),
        "sample_index": int(sample_index),
        "global_sample_indices": list(global_indices),
        "seq_length": int(seq_length),
        "local_micro_batch_size": int(local_batch_size),
        "global_source_batch_size": int(global_batch_size),
        "local_input_tokens": int(local_batch_size) * int(seq_length),
        "global_input_tokens": int(global_batch_size) * int(seq_length),
        "base_seed": int(base_seed),
        "generator_contract": "COUNTER_SEEDED_GLOBAL_SAMPLE",
        "dtype": "int32_le",
        "input_ids_sha256": digest,
        "input_ids_path": relative_path,
    }


def append_input_manifest(output_dir: Path, rank: int, payload: dict[str, Any]) -> Path:
    path = Path(output_dir) / "inputs" / f"rank{int(rank):04d}_input_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return path
