from pathlib import Path

from routesense_poc1.route_patch import MockRouteAblationContext, RouteAblationContext, RoutingState, UnsupportedOLMoEAdapter


class FakeLayer:
    def __init__(self):
        self.forward_called = False
        self.topk_expert_ids = [1, 2]
        self.topk_weights = [0.7, 0.3]
        self.gate_output = [0.7, 0.3]

    def forward(self, *args, **kwargs):
        self.forward_called = True
        return "ok"


class FakeModel:
    def __init__(self, layer):
        self.layer = layer
        self._modules = {"block": layer}

    def named_modules(self):
        yield "", self
        yield "block", self.layer


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def extract_routing_state(self, layer, args, kwargs):
        self.calls.append(("extract", layer))
        return RoutingState([1, 2], [0.7, 0.3], 1, 2, metadata={"source": "fake"})

    def apply_ablation(self, routing_state, token_pos, rank, expected_expert_id):
        self.calls.append(("apply", rank, expected_expert_id))
        routing_state.selected_expert_ids = [99]
        routing_state.effective_gate_weights = [0.0, 1.0]
        return routing_state

    def call_original_forward(self, original_forward, routing_state, args, kwargs):
        self.calls.append(("call", routing_state))
        return original_forward(*args, **kwargs)


def test_mock_context_keeps_existing_semantics():
    layer = FakeLayer()
    model = FakeModel(layer)
    with MockRouteAblationContext(model, "block", 0, 1) as context:
        assert context.export_snapshots()[0] is not None
        assert context.export_snapshots()[1] is not None


def test_real_context_delays_snapshots_until_wrapper_runs():
    layer = FakeLayer()
    model = FakeModel(layer)
    adapter = FakeAdapter()
    context = RouteAblationContext(model, "block", 0, 1, adapter=adapter)
    with context:
        assert context.export_snapshots() == (None, None)
        assert layer.forward_called is False
    assert layer.forward_called is False


def test_real_context_records_snapshots_after_wrapper_execution():
    layer = FakeLayer()
    model = FakeModel(layer)
    adapter = FakeAdapter()
    context = RouteAblationContext(model, "block", 0, 1, adapter=adapter)
    with context:
        layer.forward()
        before, after = context.export_snapshots()
        assert before is not None
        assert after is not None
        assert before.weights[1] == 0.3
        assert after.weights[1] == 1.0
    assert layer.forward_called is True
