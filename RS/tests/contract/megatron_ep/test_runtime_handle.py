from __future__ import annotations

import torch

from rs.runtime.online.megatron_ep.contracts import ExecutionSelection, OnlinePolicyParameters, OnlineRuntimeConfig
from rs.runtime.online.megatron_ep.host import attach_formal_online_runtime
from rs.runtime.online.megatron_ep.public_types import RuntimeDecision, RuntimeHandle


class _TokenDispatcherStub:
    def __init__(self) -> None:
        self.token_dispatch = lambda *args, **kwargs: ("dispatch", args, kwargs)
        self.token_combine = lambda *args, **kwargs: ("combine", args, kwargs)


class _MoELayerStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_dispatcher = _TokenDispatcherStub()


class _LayerContainerStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MoELayerStub()


class _ScopeModelStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_LayerContainerStub(), _LayerContainerStub(), _LayerContainerStub()])

    def forward(self):
        self.layers[0].mlp.token_dispatcher.token_dispatch("hidden", "probs")
        self.layers[1].mlp.token_dispatcher.token_dispatch("hidden", "probs")
        self.layers[1].mlp.token_dispatcher.token_combine("hidden")
        return "ok"


def _config() -> OnlineRuntimeConfig:
    return OnlineRuntimeConfig(
        policy_name="routersense_p0p1p2_hint",
        execution_mode="native_passthrough",
        control_mode="sync_before_phase",
        execution_selection=ExecutionSelection(layer_selector="selected", selected_layer_ids=("1",)),
        policy_parameters=OnlinePolicyParameters(
            online_p2_predictor="copy_current_dispatch",
            safe_projection_mode="disabled",
        ),
    )


def test_attach_formal_online_runtime_returns_runtime_handle_proxy() -> None:
    handle = attach_formal_online_runtime(
        model=torch.nn.Module(),
        runtime_config=_config(),
        rank=0,
        local_rank=0,
        run_id="run",
        model_revision="model",
        request_table_hash="request",
        hostname="host",
    )
    assert isinstance(handle, RuntimeHandle)
    assert handle.config.policy == "routersense_p0p1p2_hint"
    assert handle.closed is False


def test_runtime_handle_detach_restores_wrapped_dispatchers() -> None:
    model = _ScopeModelStub()
    layer0_dispatch = model.layers[0].mlp.token_dispatcher.token_dispatch
    layer0_combine = model.layers[0].mlp.token_dispatcher.token_combine
    layer1_dispatch = model.layers[1].mlp.token_dispatcher.token_dispatch
    layer1_combine = model.layers[1].mlp.token_dispatcher.token_combine
    handle = attach_formal_online_runtime(
        model=model,
        runtime_config=_config(),
        rank=0,
        local_rank=0,
        run_id="run",
        model_revision="model",
        request_table_hash="request",
        hostname="host",
    )
    assert model.layers[1].mlp.token_dispatcher.token_dispatch is not layer1_dispatch
    handle.detach()
    assert model.layers[0].mlp.token_dispatcher.token_dispatch is layer0_dispatch
    assert model.layers[0].mlp.token_dispatcher.token_combine is layer0_combine
    assert model.layers[1].mlp.token_dispatcher.token_dispatch is layer1_dispatch
    assert model.layers[1].mlp.token_dispatcher.token_combine is layer1_combine


def test_runtime_handle_close_is_idempotent() -> None:
    model = _ScopeModelStub()
    handle = attach_formal_online_runtime(
        model=model,
        runtime_config=_config(),
        rank=0,
        local_rank=0,
        run_id="run",
        model_revision="model",
        request_table_hash="request",
        hostname="host",
    )
    handle.close()
    handle.close()
    assert handle.closed is True


def test_attach_formal_online_runtime_emits_single_event_chain_per_hook() -> None:
    model = _ScopeModelStub()
    handle = attach_formal_online_runtime(
        model=model,
        runtime_config=_config(),
        rank=0,
        local_rank=0,
        run_id="run",
        model_revision="model",
        request_table_hash="request",
        hostname="host",
    )
    recorded: list[str] = []

    def _recording_handle(event) -> RuntimeDecision:
        recorded.append(type(event).__name__)
        return RuntimeDecision()

    handle.runtime.handle = _recording_handle
    model()

    assert recorded[0] == "ForwardBeginEvent"
    assert recorded[-1] == "ForwardEndEvent"
    assert recorded.count("DispatchReadyEvent") == recorded.count("DispatchCompleteEvent")
    assert recorded.count("CombineReadyEvent") == recorded.count("CombineCompleteEvent")
    assert "DispatchReadyEvent" in recorded
    assert "CombineCompleteEvent" in recorded


def test_runtime_handle_close_restores_model_forward_hooks() -> None:
    model = _ScopeModelStub()
    handle = attach_formal_online_runtime(
        model=model,
        runtime_config=_config(),
        rank=0,
        local_rank=0,
        run_id="run",
        model_revision="model",
        request_table_hash="request",
        hostname="host",
    )
    assert getattr(model, "_routersense_forward_wrapped", False) is True
    handle.close()
    assert getattr(model, "_routersense_forward_wrapped", False) is False
