from routesense_poc1.model_inspector import inspect_model, select_middle_moe_layer
from routesense_poc1.schemas import MoELayerInfo


class LinearGate:
    def __init__(self):
        self.gate = object()


class RealMoELayer:
    def __init__(self):
        self.gate = object()
        self.experts = [object()]


class MockModel:
    def __init__(self):
        self.linear_gate = LinearGate()
        self.moe = RealMoELayer()

    def named_modules(self):
        yield "", self
        yield "linear_gate", self.linear_gate
        yield "moe", self.moe


def test_inspect_model_skips_plain_linear_gate_and_keeps_moe_candidate():
    layers = inspect_model(MockModel())
    assert any(layer.layer_path == "moe" for layer in layers)
    moe_layer = next(layer for layer in layers if layer.layer_path == "moe")
    assert moe_layer.candidate_score >= 4.0
    assert "experts" in moe_layer.candidate_reason
    assert select_middle_moe_layer(layers).layer_path == "moe"
