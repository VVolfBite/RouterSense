from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rs_sim.app.artifact_identity import checkpoint_identity
from rs_sim.app.collect import normalize_collect_config
from rs_sim.app.config_io import ConfigError, load_config
from rs_sim.trace.runners.input_contract import build_distributed_tokens, global_sample_indices, sample_seed


def _config(tmp_path: Path, *, qualification_samples: int = 1, sample_count: int = 2, external: bool = False):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": 64,
        "vocab_size": 128,
        "num_hidden_layers": 2,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }), encoding="utf-8")
    payload = {
        "version": 1,
        "name": "capture-merge",
        "model": {"path": str(model), "format": "hf", "dtype": "bfloat16", "hidden_size": 64},
        "launch": {"launcher": "torchrun", "nnodes": 1, "nproc_per_node": 4},
        "parallel": {"ep": 4, "tp": 1, "pp": 1, "dp": 1, "cp": 1, "etp": 1},
        "input": {
            "seq_length": 32,
            "micro_batch_size": 1,
            "global_batch_size": 4,
            "sample_count": sample_count,
            "qualification_samples": qualification_samples,
            "save_token_ids": True,
        },
        "capture": {"rank_to_node": [0, 0, 0, 0], "require_all_expected_layers": False},
        "payload": {"dispatch": {"alignment_bytes": 1}, "combine": {"alignment_bytes": 1}},
        "output": {"directory": str(tmp_path / "out")},
    }
    if external:
        payload["launch"]["command"] = ["python", "external_runner.py"]
    path = tmp_path / "collect.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_config(path)


def test_counter_seeded_input_contract_is_deterministic_and_rank_disjoint():
    torch = pytest.importorskip("torch")
    a, idx_a = build_distributed_tokens(
        torch, vocab_size=128, seq_length=8, local_batch_size=2,
        source_rank=0, base_seed=7, sample_index=3, device="cpu",
    )
    b, idx_b = build_distributed_tokens(
        torch, vocab_size=128, seq_length=8, local_batch_size=2,
        source_rank=0, base_seed=7, sample_index=3, device="cpu",
    )
    c, idx_c = build_distributed_tokens(
        torch, vocab_size=128, seq_length=8, local_batch_size=2,
        source_rank=1, base_seed=7, sample_index=3, device="cpu",
    )
    assert idx_a == idx_b == global_sample_indices(source_rank=0, local_batch_size=2)
    assert idx_c == global_sample_indices(source_rank=1, local_batch_size=2)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert sample_seed(base_seed=7, measured_sample=3, global_sample_index=0) != sample_seed(
        base_seed=7, measured_sample=4, global_sample_index=0
    )


def test_built_in_collect_records_formal_input_contract(tmp_path: Path):
    normalized = normalize_collect_config(_config(tmp_path))
    assert normalized["dataset"]["input_contract"] == "COUNTER_SEEDED_GLOBAL_SAMPLE"
    assert normalized["dataset"]["global_source_batch_size"] == 4
    assert "--global-batch-size" in normalized["launcher"]["command"]
    assert "--save-input-ids" in normalized["launcher"]["command"]


def test_external_runner_is_not_mislabeled_as_formal_input_contract(tmp_path: Path):
    normalized = normalize_collect_config(_config(tmp_path, external=True))
    assert normalized["dataset"]["input_contract"] == "EXTERNAL_RUNNER_UNVERIFIED"
    assert normalized["dataset"]["save_token_ids"] is False


def test_qualification_samples_must_fit_measured_samples(tmp_path: Path):
    with pytest.raises(ConfigError, match="must not exceed"):
        normalize_collect_config(_config(tmp_path, qualification_samples=3, sample_count=2))


def test_checkpoint_identity_supports_explicit_and_inventory_modes(tmp_path: Path):
    model = tmp_path / "checkpoint"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"mixtral"}', encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weights")
    inventory = checkpoint_identity(model)
    assert inventory["status"] == "PASS"
    assert inventory["mode"] == "IDENTITY_FILES_AND_INVENTORY_SHA256"
    assert inventory["inventory"]["file_count"] == 2
    explicit = checkpoint_identity(model, explicit_digest="abc123")
    assert explicit["mode"] == "EXPLICIT_SHA256"
    assert explicit["checkpoint_digest"] == "abc123"
