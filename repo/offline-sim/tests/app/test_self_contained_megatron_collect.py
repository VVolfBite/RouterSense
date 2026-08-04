from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rs_sim.app.collect import normalize_collect_config
from rs_sim.app.config_io import ConfigError, load_config
from rs_sim.trace.runners.megatron_bridge_moe import _forward_kwargs
from rs_sim.trace.runners.model_support import inspect_hf_model, validate_generic_text_moe


def _write_model(root: Path, *, architecture: str, model_type: str, experts: int = 8) -> Path:
    root.mkdir()
    payload = {
        "architectures": [architecture],
        "model_type": model_type,
        "hidden_size": 64,
        "vocab_size": 256,
        "num_hidden_layers": 4,
        "num_experts": experts,
        "num_experts_per_tok": 2,
    }
    (root / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _config(tmp_path: Path, model_path: Path, *, ep: int = 4, prediction: dict | None = None) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "name": "self-contained",
        "model": {
            "runner": "AUTO_MEGATRON_BRIDGE",
            "family": "auto",
            "path": str(model_path),
            "format": "hf",
            "dtype": "bfloat16",
        },
        "launch": {"launcher": "torchrun", "nnodes": 1, "nproc_per_node": ep},
        "parallel": {"ep": ep, "tp": 1, "pp": 1, "dp": 1, "cp": 1, "etp": 1},
        "input": {"seq_length": 32, "micro_batch_size": 1, "warmup_count": 1, "sample_count": 2, "seed": 7},
        "capture": {
            "backend": "MEGATRON_CORE_AUTO",
            "rank_to_node": [0] * ep,
            "minimum_consecutive_layers": 2,
            "require_all_expected_layers": False,
        },
        "prediction": prediction or {"mode": "FATE_P2", "confidence_ppm": 700000},
        "payload": {
            "dispatch": {"alignment_bytes": 1},
            "combine": {"alignment_bytes": 1},
        },
        "output": {"directory": str(tmp_path / "out")},
    }
    path = tmp_path / "collect.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_config(path)


@pytest.mark.parametrize(
    ("architecture", "model_type", "experts"),
    [
        ("OLMoEForCausalLM", "olmoe", 64),
        ("MixtralForCausalLM", "mixtral", 8),
        ("DeepseekV2ForCausalLM", "deepseek_v2", 64),
        ("Qwen2MoeForCausalLM", "qwen2_moe", 60),
        ("Qwen3MoeForCausalLM", "qwen3_moe", 128),
    ],
)
def test_supported_text_moe_families_use_built_in_module(
    tmp_path: Path, architecture: str, model_type: str, experts: int
):
    model = _write_model(tmp_path / "model", architecture=architecture, model_type=model_type, experts=experts)
    normalized = normalize_collect_config(_config(tmp_path, model, ep=4))
    command = normalized["launcher"]["command"]
    assert command[:3] == ["torchrun", "--nnodes=1", "--nproc_per_node=4"]
    assert "--module" in command
    assert "rs_sim.trace.runners.megatron_bridge_moe" in command
    assert command[command.index("--warmup-samples") + 1] == "1"
    assert command[command.index("--samples") + 1] == "2"
    assert not any("run_olmoe.py" in item for item in command)
    assert normalized["model_runner"]["self_contained"] is True
    assert normalized["payload"]["dispatch"]["token_payload_bytes_per_row"] == 128


def test_prediction_section_is_preserved_for_fate(tmp_path: Path):
    model = _write_model(tmp_path / "model", architecture="OLMoEForCausalLM", model_type="olmoe", experts=64)
    normalized = normalize_collect_config(
        _config(
            tmp_path,
            model,
            prediction={
                "mode": "FATE_P2",
                "provider": "MEGATRON_SAMPLED_FATE",
                "max_sample_tokens": 321,
                "confidence_ppm": 888000,
                "require_complete_fate_coverage": True,
            },
        )
    )
    assert normalized["prediction"] == {
        "mode": "FATE_P2",
        "provider": "MEGATRON_SAMPLED_FATE",
        "fate_artifact_path": None,
        "max_sample_tokens": 321,
        "confidence_ppm": 888000,
        "require_complete_fate_coverage": True,
    }


def test_expert_count_must_be_divisible_by_ep(tmp_path: Path):
    model = _write_model(tmp_path / "model", architecture="MixtralForCausalLM", model_type="mixtral", experts=8)
    with pytest.raises(ConfigError, match="not divisible"):
        normalize_collect_config(_config(tmp_path, model, ep=3))


def test_dense_and_multimodal_models_fail_closed(tmp_path: Path):
    dense = _write_model(tmp_path / "dense", architecture="LlamaForCausalLM", model_type="llama", experts=1)
    with pytest.raises(ConfigError, match="does not identify a MoE"):
        normalize_collect_config(_config(tmp_path / "dense_cfg", dense, ep=1))

    vlm = _write_model(tmp_path / "vlm", architecture="Qwen2VLForConditionalGeneration", model_type="qwen2_vl", experts=8)
    with pytest.raises(ConfigError, match="decoder-only text MoE"):
        normalize_collect_config(_config(tmp_path / "vlm_cfg", vlm, ep=4))


def test_inspection_does_not_treat_router_topk_as_total_experts(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({
            "architectures": ["UnknownForCausalLM"],
            "model_type": "unknown",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_experts_per_tok": 2,
        }),
        encoding="utf-8",
    )
    inspection = inspect_hf_model(model)
    assert inspection.num_experts is None
    with pytest.raises(ValueError, match="does not identify a MoE"):
        validate_generic_text_moe(inspection, ep=1)


def test_forward_signature_never_receives_duplicate_token_arguments():
    class InputIdsModel:
        def forward(self, input_ids, position_ids=None, **kwargs):
            raise NotImplementedError

    class TokensModel:
        def forward(self, tokens, position_ids=None):
            raise NotImplementedError

    batch = {"tokens": object(), "position_ids": object()}
    first = _forward_kwargs(InputIdsModel(), batch)
    second = _forward_kwargs(TokensModel(), batch)
    assert "input_ids" in first and "tokens" not in first
    assert "tokens" in second and "input_ids" not in second


def test_formal_configs_do_not_reference_placeholder_runner():
    root = Path(__file__).resolve().parents[2]
    checked = [
        root / "configs/collect/olmoe_ep4_smoke.yaml",
        root / "configs/collect/olmoe_ep4_formal.yaml",
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        assert "/workspace/run_olmoe.py" not in text
        assert "entrypoint:" not in text


def test_preflight_constructs_autobridge_provider_and_checks_hook_contract(tmp_path: Path, monkeypatch):
    import sys
    import types

    from rs_sim.trace.runners.preflight import run_megatron_model_preflight

    model = _write_model(tmp_path / "model", architecture="MixtralForCausalLM", model_type="mixtral", experts=8)

    class FakeProvider:
        pass

    class FakeBridge:
        def to_megatron_provider(self, load_weights=False):
            assert load_weights is False
            return FakeProvider()

    class FakeAutoBridge:
        @staticmethod
        def can_handle(path):
            return True

        @staticmethod
        def from_hf_pretrained(path, trust_remote_code=False):
            assert str(path) == str(model)
            return FakeBridge()

    bridge_module = types.ModuleType("megatron.bridge")
    bridge_module.AutoBridge = FakeAutoBridge
    megatron_module = types.ModuleType("megatron")
    megatron_module.__path__ = []
    core_module = types.ModuleType("megatron.core")
    core_module.__path__ = []
    transformer_module = types.ModuleType("megatron.core.transformer")
    transformer_module.__path__ = []
    moe_package = types.ModuleType("megatron.core.transformer.moe")
    moe_package.__path__ = []
    moe_layer_module = types.ModuleType("megatron.core.transformer.moe.moe_layer")
    dispatcher_module = types.ModuleType("megatron.core.transformer.moe.token_dispatcher")

    class MoELayer:
        def route(self):
            pass

        def preprocess(self):
            pass

        def routed_experts_compute(self):
            pass

        def forward(self):
            pass

    MoELayer.__module__ = moe_layer_module.__name__
    moe_layer_module.MoELayer = MoELayer

    class Dispatcher:
        def dispatch_postprocess(self):
            pass

    Dispatcher.__module__ = dispatcher_module.__name__
    dispatcher_module.Dispatcher = Dispatcher

    for name, module in {
        "megatron": megatron_module,
        "megatron.bridge": bridge_module,
        "megatron.core": core_module,
        "megatron.core.transformer": transformer_module,
        "megatron.core.transformer.moe": moe_package,
        moe_layer_module.__name__: moe_layer_module,
        dispatcher_module.__name__: dispatcher_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    report = run_megatron_model_preflight(
        model_path=model,
        hf_config_path=None,
        model_format="hf",
        ep=4,
        tp=1,
        pp=1,
        dp=1,
        cp=1,
        etp=1,
        nproc_per_node=4,
        trust_remote_code=False,
        require_cuda=False,
        require_fate_route=True,
    )
    assert report["status"] == "PASS"
    assert report["autobridge"]["provider_class"].endswith("FakeProvider")
    assert report["capture_contract"]["compatible_moe_layer_class_count"] == 1
    assert report["capture_contract"]["fate_route_compatible_class_count"] == 1


def test_preflight_accepts_megatron_core_015_split_lifecycle(tmp_path: Path, monkeypatch):
    import sys
    import types

    from rs_sim.trace.runners.preflight import run_megatron_model_preflight

    model = _write_model(tmp_path / "model015", architecture="OlmoeForCausalLM", model_type="olmoe", experts=64)

    class FakeProvider:
        pass

    class FakeBridge:
        def to_megatron_provider(self, load_weights=False):
            assert load_weights is False
            return FakeProvider()

    class FakeAutoBridge:
        @staticmethod
        def can_handle(path):
            return True

        @staticmethod
        def from_hf_pretrained(path, trust_remote_code=False):
            assert str(path) == str(model)
            return FakeBridge()

    bridge_module = types.ModuleType("megatron.bridge")
    bridge_module.AutoBridge = FakeAutoBridge
    megatron_module = types.ModuleType("megatron")
    megatron_module.__path__ = []
    core_module = types.ModuleType("megatron.core")
    core_module.__path__ = []
    transformer_module = types.ModuleType("megatron.core.transformer")
    transformer_module.__path__ = []
    moe_package = types.ModuleType("megatron.core.transformer.moe")
    moe_package.__path__ = []
    moe_layer_module = types.ModuleType("megatron.core.transformer.moe.moe_layer")
    dispatcher_module = types.ModuleType("megatron.core.transformer.moe.token_dispatcher")

    class MoELayer:
        def router_and_preprocess(self, hidden_states):
            pass

        def routed_experts_compute(self, hidden_states, probs, residual):
            pass

        def forward(self, hidden_states):
            pass

    MoELayer.__module__ = moe_layer_module.__name__
    moe_layer_module.MoELayer = MoELayer

    class Dispatcher:
        def dispatch_preprocess(self, tokens, routing_map, probs):
            pass

        def dispatch_postprocess(self, hidden_states, probs):
            pass

    Dispatcher.__module__ = dispatcher_module.__name__
    dispatcher_module.Dispatcher = Dispatcher

    for name, module in {
        "megatron": megatron_module,
        "megatron.bridge": bridge_module,
        "megatron.core": core_module,
        "megatron.core.transformer": transformer_module,
        "megatron.core.transformer.moe": moe_package,
        moe_layer_module.__name__: moe_layer_module,
        dispatcher_module.__name__: dispatcher_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    report = run_megatron_model_preflight(
        model_path=model,
        hf_config_path=None,
        model_format="hf",
        ep=4,
        tp=1,
        pp=1,
        dp=1,
        cp=1,
        etp=1,
        nproc_per_node=4,
        trust_remote_code=False,
        require_cuda=False,
        require_fate_route=True,
    )
    contract = report["capture_contract"]
    assert contract["compatible_moe_layer_class_count"] == 1
    assert contract["split_router_dispatch_compatible_class_count"] == 1
    assert contract["dispatch_preprocess_compatible_class_count"] == 1
    assert "SPLIT_ROUTER_DISPATCH" in contract["supported_lifecycle_profiles"]
