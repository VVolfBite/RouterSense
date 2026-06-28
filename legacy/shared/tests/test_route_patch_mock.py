from __future__ import annotations

from routesense_poc1.route_patch import MockRouteAblationContext, RouteAblationContext, RoutePatchError


class MockMoELayer:
    def __init__(self):
        self.gate_output = [0.5, 0.3, 0.2]
        self.topk_expert_ids = [10, 20, 30]
        self.topk_weights = [0.5, 0.3, 0.2]


class MockModel:
    def __init__(self, layer):
        self.layer = layer

    def named_modules(self):
        yield "", self
        yield "layer", self.layer


def test_patch_context_zeroes_only_selected_weight_and_restores_on_exit():
    layer = MockMoELayer()
    model = MockModel(layer)

    with MockRouteAblationContext(model, "layer", token_pos=3, rank=1, expected_expert_id=20) as patch:
        result = patch.check_correctness()
        assert result.passed is True
        before, after = patch.export_snapshots()
        assert before is not None and after is not None
        assert before.weights == [0.5, 0.3, 0.2]
        assert after.weights == [0.5, 0.0, 0.2]
        assert after.expert_ids == [10, 20, 30]
        assert layer.topk_weights == [0.5, 0.0, 0.2]
        assert layer.gate_output == [0.5, 0.0, 0.2]

    assert layer.topk_weights == [0.5, 0.3, 0.2]
    assert layer.gate_output == [0.5, 0.3, 0.2]


def test_patch_context_raises_for_missing_interface():
    class BadLayer:
        pass

    class BadModel:
        def named_modules(self):
            yield "", self
            yield "layer", BadLayer()

    try:
        with RouteAblationContext(BadModel(), "layer", token_pos=0, rank=0):
            pass
    except RoutePatchError:
        return
    raise AssertionError("expected RoutePatchError")
