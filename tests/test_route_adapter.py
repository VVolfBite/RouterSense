import pytest

from routesense_poc1.route_patch import (
    MockRouteAblationContext,
    RouteAblationContext,
    RoutePatchError,
    UnsupportedOLMoEAdapter,
)


class DummyLayer:
    def __init__(self):
        self.gate_output = [0.5, 0.3, 0.2]
        self.topk_expert_ids = [10, 20, 30]
        self.topk_weights = [0.5, 0.3, 0.2]


class DummyModel:
    def __init__(self):
        self.layer = DummyLayer()

    def named_modules(self):
        yield "", self
        yield "layer", self.layer


def test_mock_route_ablation_context_preserves_semantics():
    model = DummyModel()
    with MockRouteAblationContext(model, "layer", token_pos=1, rank=1, expected_expert_id=20) as ctx:
        before, after = ctx.export_snapshots()
        assert before is not None and after is not None
        assert after.weights == [0.5, 0.0, 0.2]
        assert before.expert_ids == [10, 20, 30]
        assert model.layer.topk_weights == [0.5, 0.0, 0.2]
    assert model.layer.topk_weights == [0.5, 0.3, 0.2]


def test_real_route_ablation_context_requires_adapter():
    with pytest.raises(RoutePatchError):
        with RouteAblationContext(DummyModel(), "layer", token_pos=0, rank=0):
            pass


def test_unsupported_adapter_raises_clear_error():
    adapter = UnsupportedOLMoEAdapter()
    with pytest.raises(RoutePatchError, match="真实 OLMoE route patch 尚未适配"):
        adapter.extract_routing_state(None, None, None)
