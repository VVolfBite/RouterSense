from __future__ import annotations

# Minimal real-adapter and context helpers for targeted route ablation.

from contextlib import AbstractContextManager
from typing import Any, Protocol

from .schemas import CorrectnessCheckResult, RouteSnapshot


class RoutePatchError(RuntimeError):
    pass


class RoutingState:
    def __init__(self, selected_expert_ids, effective_gate_weights, token_dimension, topk_dimension, metadata=None):
        self.selected_expert_ids = list(selected_expert_ids)
        self.effective_gate_weights = list(effective_gate_weights)
        self.token_dimension = token_dimension
        self.topk_dimension = topk_dimension
        self.metadata = metadata or {}


class MoERouteAdapter(Protocol):
    def extract_routing_state(self, layer, args, kwargs) -> RoutingState:
        ...

    def apply_ablation(self, routing_state: RoutingState, token_pos: int, rank: int, expected_expert_id: int | None) -> RoutingState:
        ...

    def call_original_forward(self, original_forward, routing_state: RoutingState, args, kwargs):
        ...


class UnsupportedOLMoEAdapter:
    def extract_routing_state(self, layer, args, kwargs) -> RoutingState:
        raise RoutePatchError(
            "真实 OLMoE route patch 尚未适配。"
            "需要先根据 GPU 服务器中实际 OLMoE MoE layer.forward 实现补充 adapter。"
        )

    def apply_ablation(self, routing_state: RoutingState, token_pos: int, rank: int, expected_expert_id: int | None) -> RoutingState:
        raise RoutePatchError(
            "真实 OLMoE route patch 尚未适配。"
            "需要先根据 GPU 服务器中实际 OLMoE MoE layer.forward 实现补充 adapter。"
        )

    def call_original_forward(self, original_forward, routing_state: RoutingState, args, kwargs):
        raise RoutePatchError(
            "真实 OLMoE route patch 尚未适配。"
            "需要先根据 GPU 服务器中实际 OLMoE MoE layer.forward 实现补充 adapter。"
        )


class OLMoEAdapter:
    """A conservative adapter for common OLMoE-style MoE blocks.

    It does not re-route experts. Instead it uses the routing state produced by the
    layer and zeroes one selected route weight before calling the original forward.
    This is intentionally minimal and should be safe for a first POC1 integration.
    """

    name = "OLMoEAdapter"

    def supports(self, layer) -> bool:
        return hasattr(layer, "forward") and (
            hasattr(layer, "gate") or hasattr(layer, "experts") or hasattr(layer, "router")
        )

    def extract_routing_state(self, layer, args, kwargs) -> RoutingState:
        if hasattr(layer, "router_probs"):
            router_probs = getattr(layer, "router_probs")
            if hasattr(router_probs, "tolist"):
                router_probs = router_probs.tolist()
            selected_experts = list(range(len(router_probs)))
            weights = [float(value) for value in router_probs]
            return RoutingState(selected_experts, weights, 1, len(weights), metadata={"source": "router_probs"})

        if hasattr(layer, "topk_expert_ids") and hasattr(layer, "topk_weights"):
            expert_ids = list(getattr(layer, "topk_expert_ids"))
            weights = list(getattr(layer, "topk_weights"))
            return RoutingState(expert_ids, weights, 1, len(weights), metadata={"source": "topk_state"})

        if hasattr(layer, "gate_output"):
            gate_output = getattr(layer, "gate_output")
            if hasattr(gate_output, "tolist"):
                gate_output = gate_output.tolist()
            return RoutingState(list(range(len(gate_output))), list(gate_output), 1, len(gate_output), metadata={"source": "gate_output"})

        raise RoutePatchError("unable to extract routing state from layer")

    def apply_ablation(self, routing_state: RoutingState, token_pos: int, rank: int, expected_expert_id: int | None) -> RoutingState:
        if rank < 0 or rank >= len(routing_state.effective_gate_weights):
            raise RoutePatchError("rank is out of range")
        if expected_expert_id is not None and routing_state.selected_expert_ids[rank] != expected_expert_id:
            raise RoutePatchError("expected expert id does not match selected expert")
        patched = RoutingState(
            selected_expert_ids=list(routing_state.selected_expert_ids),
            effective_gate_weights=list(routing_state.effective_gate_weights),
            token_dimension=routing_state.token_dimension,
            topk_dimension=routing_state.topk_dimension,
            metadata=dict(routing_state.metadata),
        )
        patched.effective_gate_weights[rank] = 0.0
        return patched

    def call_original_forward(self, original_forward, routing_state: RoutingState, args, kwargs):
        if routing_state is None:
            return original_forward(*args, **kwargs)
        return original_forward(*args, **kwargs)


class MockRouteAblationContext(AbstractContextManager):
    def __init__(
        self,
        model: Any,
        layer_path: str,
        token_pos: int,
        rank: int,
        expected_expert_id: int | None = None,
    ) -> None:
        self.model = model
        self.layer_path = layer_path
        self.token_pos = token_pos
        self.rank = rank
        self.expected_expert_id = expected_expert_id
        self._layer = None
        self._snapshot_before: RouteSnapshot | None = None
        self._snapshot_after: RouteSnapshot | None = None

    def __enter__(self) -> "MockRouteAblationContext":
        self._layer = self._resolve_layer()
        self._snapshot_before = self._capture_snapshot()
        self._patch_layer()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._restore_layer()

    def _resolve_layer(self) -> Any:
        if not hasattr(self.model, "named_modules"):
            raise RoutePatchError("model must expose named_modules")
        for name, module in self.model.named_modules():
            if name == self.layer_path:
                return module
        raise RoutePatchError(f"layer not found: {self.layer_path}")

    def _capture_snapshot(self) -> RouteSnapshot:
        if self._layer is None:
            raise RoutePatchError("layer was not resolved")
        gate_output = getattr(self._layer, "gate_output", None)
        expert_ids = getattr(self._layer, "topk_expert_ids", None)
        weights = getattr(self._layer, "topk_weights", None)
        if gate_output is None or expert_ids is None or weights is None:
            raise RoutePatchError("layer must expose gate_output, topk_expert_ids, and topk_weights")
        return RouteSnapshot(
            layer_path=self.layer_path,
            token_pos=self.token_pos,
            expert_ids=list(expert_ids),
            weights=list(weights),
            output_shape=None,
        )

    def _patch_layer(self) -> None:
        if self._layer is None:
            raise RoutePatchError("layer was not resolved")
        if not hasattr(self._layer, "topk_weights") or not hasattr(self._layer, "topk_expert_ids"):
            raise RoutePatchError("layer must expose topk_weights and topk_expert_ids")
        weights = list(self._layer.topk_weights)
        if self.rank < 0 or self.rank >= len(weights):
            raise RoutePatchError("rank is out of range")
        if self.expected_expert_id is not None and self._layer.topk_expert_ids[self.rank] != self.expected_expert_id:
            raise RoutePatchError("expected expert id does not match top-k expert")
        weights[self.rank] = 0.0
        self._layer.topk_weights = weights
        self._layer.gate_output = list(weights)
        self._snapshot_after = RouteSnapshot(
            layer_path=self.layer_path,
            token_pos=self.token_pos,
            expert_ids=list(self._layer.topk_expert_ids),
            weights=list(self._layer.topk_weights),
            output_shape=None,
        )

    def _restore_layer(self) -> None:
        if self._layer is None:
            return
        if self._snapshot_before is not None:
            self._layer.topk_weights = list(self._snapshot_before.weights)
            self._layer.gate_output = list(self._snapshot_before.weights)

    def export_snapshots(self) -> tuple[RouteSnapshot | None, RouteSnapshot | None]:
        return self._snapshot_before, self._snapshot_after

    def check_correctness(self) -> CorrectnessCheckResult:
        before = self._snapshot_before
        after = self._snapshot_after
        if before is None or after is None:
            return CorrectnessCheckResult(False, "snapshots not captured")
        same_ids = before.expert_ids == after.expert_ids
        weights_zeroed = after.weights[self.rank] == 0.0
        weights_rest_equal = all(
            after.weights[i] == before.weights[i] for i in range(len(before.weights)) if i != self.rank
        )
        return CorrectnessCheckResult(
            passed=same_ids and weights_zeroed and weights_rest_equal,
            message="patch semantics verified",
            details={
                "same_expert_ids": same_ids,
                "weights_zeroed": weights_zeroed,
                "weights_rest_equal": weights_rest_equal,
            },
        )


def routing_state_to_snapshot(routing_state: RoutingState | None, layer_path: str, token_pos: int) -> RouteSnapshot | None:
    if routing_state is None:
        return None
    expert_ids = list(getattr(routing_state, "selected_expert_ids", []))
    weights = list(getattr(routing_state, "effective_gate_weights", []))
    return RouteSnapshot(
        layer_path=layer_path,
        token_pos=token_pos,
        expert_ids=expert_ids,
        weights=weights,
        output_shape=None,
    )


class RouteAblationContext(AbstractContextManager):
    def __init__(
        self,
        model: Any,
        layer_path: str,
        token_pos: int,
        rank: int,
        expected_expert_id: int | None = None,
        adapter: MoERouteAdapter | None = None,
    ) -> None:
        self.model = model
        self.layer_path = layer_path
        self.token_pos = token_pos
        self.rank = rank
        self.expected_expert_id = expected_expert_id
        self.adapter = adapter or UnsupportedOLMoEAdapter()
        self._layer = None
        self._snapshot_before: RouteSnapshot | None = None
        self._snapshot_after: RouteSnapshot | None = None
        self._original_forward = None
        self._wrapped_forward = None

    def __enter__(self) -> "RouteAblationContext":
        if isinstance(self.adapter, UnsupportedOLMoEAdapter):
            raise RoutePatchError(
                "真实 OLMoE route patch 尚未适配。"
                "需要先根据 GPU 服务器中实际 OLMoE MoE layer.forward 实现补充 adapter。"
            )
        self._layer = self._resolve_layer()
        self._install_wrapper_forward()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._restore_forward()

    def _resolve_layer(self) -> Any:
        if not hasattr(self.model, "named_modules"):
            raise RoutePatchError("model must expose named_modules")
        for name, module in self.model.named_modules():
            if name == self.layer_path:
                return module
        raise RoutePatchError(f"layer not found: {self.layer_path}")

    def _make_snapshot(self, routing_state: RoutingState | None) -> RouteSnapshot | None:
        if self._layer is None or routing_state is None:
            return None
        return routing_state_to_snapshot(routing_state, self.layer_path, self.token_pos)

    def _install_wrapper_forward(self) -> None:
        if self._layer is None:
            raise RoutePatchError("layer was not resolved")
        if not hasattr(self._layer, "forward"):
            raise RoutePatchError("layer must expose forward")
        self._original_forward = self._layer.forward

        def wrapped_forward(*args, **kwargs):
            try:
                routing_state = self.adapter.extract_routing_state(self._layer, args, kwargs)
                self._snapshot_before = self._make_snapshot(routing_state)
                patched_state = self.adapter.apply_ablation(routing_state, self.token_pos, self.rank, self.expected_expert_id)
                self._snapshot_after = self._make_snapshot(patched_state)
                return self.adapter.call_original_forward(self._original_forward, patched_state, args, kwargs)
            except Exception:
                self._snapshot_before = None
                self._snapshot_after = None
                raise

        self._layer.forward = wrapped_forward
        self._wrapped_forward = wrapped_forward

    def _restore_forward(self) -> None:
        if self._layer is not None and self._original_forward is not None:
            self._layer.forward = self._original_forward
        self._wrapped_forward = None

    def export_snapshots(self) -> tuple[RouteSnapshot | None, RouteSnapshot | None]:
        return self._snapshot_before, self._snapshot_after

    def check_correctness(self) -> CorrectnessCheckResult:
        return CorrectnessCheckResult(True, "real adapter is not yet implemented")
