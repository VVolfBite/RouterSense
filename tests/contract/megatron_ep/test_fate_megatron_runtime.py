from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rs.prediction.fate_future import FateFormalPredictor
from rs.runtime.online.megatron_ep.contracts import (
    ExecutionSelection,
    OnlinePolicyParameters,
    OnlineRuntimeConfig,
)
from rs.runtime.online.megatron_ep.host import attach_formal_online_runtime
from rs.runtime.online.megatron_ep.public_types import FormalRuntimeAttachPreflightError
from rs.runtime.online.megatron_ep.target_planning.fate_megatron import (
    build_megatron_fate_context_provider,
)
from rs.runtime.online.megatron_ep.target_planning.predictor import SharedTwoHorizonPredictor


class _Dispatcher:
    def __init__(self) -> None:
        self.token_dispatch = lambda *args, **kwargs: (args, kwargs)
        self.token_combine = lambda *args, **kwargs: (args, kwargs)


class _Router(torch.nn.Module):
    def __init__(self, hidden_size: int = 3, experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(experts * hidden_size, dtype=torch.float32).reshape(experts, hidden_size) / 10.0
        )
        self.config = SimpleNamespace(num_moe_experts=experts, moe_router_topk=top_k)

    def gating(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight)


class _MoE(torch.nn.Module):
    def __init__(self, *, with_router: bool = True) -> None:
        super().__init__()
        self.token_dispatcher = _Dispatcher()
        if with_router:
            self.router = _Router()


class _Layer(torch.nn.Module):
    def __init__(self, *, with_router: bool = True) -> None:
        super().__init__()
        self.mlp = _MoE(with_router=with_router)


class _Model(torch.nn.Module):
    def __init__(self, *, with_router: bool = True) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(with_router=with_router) for _ in range(3)])

    def forward(self):
        return None


def _future_fate_config() -> OnlineRuntimeConfig:
    return OnlineRuntimeConfig(
        policy_name="routersense_p0p1p2_hint",
        execution_mode="joint_window_async_p2p",
        control_mode="default_continue",
        execution_selection=ExecutionSelection(layer_selector="all"),
        policy_parameters=OnlinePolicyParameters(
            planner_id="future_prepared:global:rscf",
            planning_horizon="p012",
            planning_timing="previous_layer",
            online_p2_predictor="fate_cross_layer_gate",
            online_p2_predictor_config={"second_hop_predictor_id": "bridge_copy_current"},
            safe_projection_mode="disabled",
        ),
    )


def test_megatron_fate_provider_builds_canonical_expert_context() -> None:
    provider = build_megatron_fate_context_provider(
        model=_Model(),
        group_rank=0,
        world_size=2,
        run_id="run",
    )
    provider.capture(source_layer_id="0", hidden_features=torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]]))
    context = provider(source_layer_id="0", target_layer_id="1")
    result = FateFormalPredictor().predict(context)
    assert result.hint.predictor_id == "fate_cross_layer_gate"
    assert len(result.hint.target_dispatch_rows) == 2
    assert context.expert_owner_by_id == (0, 0, 1, 1)


def test_formal_attach_wires_fate_request_factory_and_provider() -> None:
    handle = attach_formal_online_runtime(
        model=_Model(),
        runtime_config=_future_fate_config(),
        rank=0,
        local_rank=0,
        run_id="run",
        model_revision="model",
        request_table_hash="request",
        hostname="host",
    )
    try:
        assert handle.expert_route_context_provider is not None
        assert handle.target_planner_service is not None
        assert handle.target_planner_service.two_horizon_predictor_request_factory is not None
        assert handle.config.planner_id == "future_prepared:global:rscf"
    finally:
        handle.close()


def test_formal_attach_fate_fails_closed_without_router_binding() -> None:
    with pytest.raises(FormalRuntimeAttachPreflightError, match="faithful FATE runtime adapter unavailable"):
        attach_formal_online_runtime(
            model=_Model(with_router=False),
            runtime_config=_future_fate_config(),
            rank=0,
            local_rank=0,
            run_id="run",
            model_revision="model",
            request_table_hash="request",
            hostname="host",
        )


def test_frozen_bridge_config_reaches_two_horizon_predictor() -> None:
    artifact = {
        "schema_version": "routersense.bridge.affine.v1",
        "artifact_id": "unit",
        "world_size": 2,
        "max_layer": 4,
        "coefficients": [[0.0] * 11 for _ in range(4)],
        "intercept": [0.0, 1.0, 1.0, 0.0],
        "confidence": 0.8,
    }
    predictor = SharedTwoHorizonPredictor(
        predictor_name="bridge_frozen_affine",
        predictor_config={"artifact": artifact},
    )
    bundle = predictor.predict_two_horizon(
        source_layer_id="0",
        current_dispatch_matrix=((0, 3), (2, 0)),
    )
    assert bundle.h1.predictor == "bridge_frozen_affine"
    assert bundle.h1.confidence == pytest.approx(0.8)
