from __future__ import annotations

from pathlib import Path

from rs.topology import inspect_model_cache, resolve_model_directory


def test_model_cache_requires_config_tokenizer_and_weights(tmp_path: Path) -> None:
    model = tmp_path / "OLMoE-1B-7B-0924-Instruct"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    incomplete = inspect_model_cache(tmp_path)
    assert incomplete.model_path == str(model)
    assert incomplete.config_ready is True
    assert incomplete.tokenizer_ready is False
    assert incomplete.weights_ready is False
    assert incomplete.required_files_present is False

    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    ready = inspect_model_cache(tmp_path)
    assert ready.required_files_present is True
    assert ready.total_size_bytes > 0
    assert ready.manifest_hash != "missing"


def test_direct_model_directory_and_compact_compatibility(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    direct.mkdir()
    for name in ("config.json", "tokenizer.model", "model.safetensors"):
        (direct / name).write_bytes(b"x")
    assert resolve_model_directory(direct) == direct
    assert inspect_model_cache(direct).required_files_present is True

    compact = tmp_path / "OLMoE-1B-7B-0924"
    compact.mkdir()
    assert resolve_model_directory(tmp_path) == compact


def test_model_cache_rejects_empty_or_missing_weight_index_shards(tmp_path: Path) -> None:
    model = tmp_path / "cache" / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert inspect_model_cache(model).weights_ready is False

    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"missing.safetensors"}}',
        encoding="utf-8",
    )
    assert inspect_model_cache(model).weights_ready is False
