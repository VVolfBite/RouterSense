from __future__ import annotations

from types import SimpleNamespace

import torch

from routesense.trace import (
    collect_qwen_moe_architecture_probe,
    collect_qwen_moe_full_sequence_trace,
    collect_qwen_moe_router_trace,
)


class _FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt"):
        del text, return_tensors
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}


class _FakeGate:
    def __init__(self) -> None:
        self.weight = torch.ones((4, 8), dtype=torch.float32)


class _FakeMLP:
    def __init__(self) -> None:
        self.gate = _FakeGate()
        self.experts = object()


class _FakeLayer:
    def __init__(self) -> None:
        self.mlp = _FakeMLP()


class _FakeInnerModel:
    def __init__(self, layer_count: int) -> None:
        self.layers = [_FakeLayer() for _ in range(layer_count)]


class _FakeQwenMoeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="qwen2_moe",
            architectures=["Qwen2MoeForCausalLM"],
            num_experts_per_tok=2,
        )
        self.model = _FakeInnerModel(layer_count=2)
        self._parameter = torch.nn.Parameter(torch.ones(1))

    def eval(self):
        return self

    def parameters(self):
        yield self._parameter

    def __call__(self, **kwargs):
        del kwargs
        router_logits = [
            torch.tensor([[4.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, -2.0], [2.0, 1.0, 0.0, -1.0]]),
            torch.tensor([[1.0, 4.0, 0.0, -1.0], [3.0, 0.0, 1.0, -2.0], [1.0, 2.0, 0.0, -1.0]]),
        ]
        hidden_states = [
            torch.zeros((1, 3, 8)),
            torch.ones((1, 3, 8)),
            torch.full((1, 3, 8), 2.0),
        ]
        return SimpleNamespace(router_logits=router_logits, hidden_states=hidden_states)


def test_qwen_trace_collectors_smoke():
    model = _FakeQwenMoeModel()
    tokenizer = _FakeTokenizer()

    probe = collect_qwen_moe_architecture_probe(model)
    assert probe["trace_backend"] == "qwen_moe"
    assert probe["moe_layer_count"] == 2

    single = collect_qwen_moe_router_trace(model, tokenizer, "hello")
    assert single["summary"]["trace_backend"] == "qwen_moe"
    assert single["summary"]["route_record_count"] == 6

    full = collect_qwen_moe_full_sequence_trace(model, tokenizer, "hello")
    assert full["summary"]["trace_backend"] == "qwen_moe"
    assert full["summary"]["record_count"] == 12
    assert set(full["gate_weights"].keys()) == {0, 1}
    assert set(full["hidden_states"].keys()) == {0, 1}
