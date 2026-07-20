from __future__ import annotations

"""Megatron adapter for faithful FATE cross-layer prediction.

The adapter is deliberately narrow: it snapshots the current MoE router input
at ``token_dispatch`` and evaluates that snapshot with the *next* MoE layer's
router gate.  The result is exposed as the canonical ``ExpertRouteContext``
consumed by :class:`rs.prediction.fate_future.FateFormalPredictor`.
"""

from dataclasses import dataclass
import re
import threading
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from rs.core.contracts import ExpertRouteContext, PredictionIdentity


def _layer_id(name: str) -> str:
    matches = re.findall(r"\d+", str(name))
    return str(int(matches[-1])) if matches else str(name)


def _read_int(candidates: Iterable[object]) -> int | None:
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _nested_attr(value: object, path: str) -> object | None:
    current = value
    for name in path.split("."):
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _router_weight(router: object) -> torch.Tensor | None:
    for path in ("weight", "gate.weight", "linear.weight", "router.weight"):
        value = _nested_attr(router, path)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value
    return None


def _num_experts(module: object, router: object) -> int | None:
    weight = _router_weight(router)
    return _read_int(
        (
            _nested_attr(router, "config.num_moe_experts"),
            _nested_attr(module, "config.num_moe_experts"),
            getattr(router, "num_experts", None),
            getattr(module, "num_experts", None),
            None if weight is None else int(weight.shape[0]),
        )
    )


def _top_k(module: object, router: object) -> int | None:
    return _read_int(
        (
            _nested_attr(router, "config.moe_router_topk"),
            _nested_attr(module, "config.moe_router_topk"),
            getattr(router, "top_k", None),
            getattr(router, "topk", None),
            getattr(module, "top_k", None),
        )
    )


def _contiguous_owner_map(num_experts: int, world_size: int) -> tuple[int, ...]:
    if int(num_experts) <= 0 or int(world_size) <= 0:
        raise ValueError("num_experts and world_size must be positive")
    base, remainder = divmod(int(num_experts), int(world_size))
    mapping: list[int] = []
    for rank in range(int(world_size)):
        mapping.extend([rank] * (base + (1 if rank < remainder else 0)))
    if len(mapping) != int(num_experts):
        raise AssertionError("expert owner mapping length mismatch")
    return tuple(mapping)


def _first_full_gate_tensor(output: object, *, token_count: int, num_experts: int) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        if output.ndim == 2 and tuple(output.shape) == (int(token_count), int(num_experts)):
            return output
        return None
    if isinstance(output, dict):
        for key in ("router_logits", "logits", "scores", "gate_logits"):
            found = _first_full_gate_tensor(output.get(key), token_count=token_count, num_experts=num_experts)
            if found is not None:
                return found
        for value in output.values():
            found = _first_full_gate_tensor(value, token_count=token_count, num_experts=num_experts)
            if found is not None:
                return found
        return None
    if isinstance(output, (tuple, list)):
        for value in output:
            found = _first_full_gate_tensor(value, token_count=token_count, num_experts=num_experts)
            if found is not None:
                return found
    return None


@dataclass(frozen=True)
class MegatronFateLayerBinding:
    layer_id: str
    layer_name: str
    router: object
    top_k: int
    num_experts: int
    expert_owner_by_id: tuple[int, ...]
    gate_output_domain: str = "logits"

    def gate(self, hidden: np.ndarray) -> np.ndarray:
        values = np.asarray(hidden)
        if values.ndim != 2:
            raise ValueError(f"FATE hidden snapshot must be 2-D, got {values.shape}")
        weight = _router_weight(self.router)
        parameter = weight
        if parameter is None and isinstance(self.router, torch.nn.Module):
            parameter = next(self.router.parameters(), None)
        device = torch.device("cpu") if parameter is None else parameter.device
        dtype = torch.float32 if parameter is None else parameter.dtype
        tensor = torch.as_tensor(values, device=device, dtype=dtype)
        with torch.inference_mode():
            gating = getattr(self.router, "gating", None)
            if callable(gating):
                output = gating(tensor)
            elif weight is not None:
                output = F.linear(tensor, weight)
            elif callable(self.router):
                output = self.router(tensor)
            else:
                raise ValueError(f"next-layer router for {self.layer_name!r} is not callable")
        logits = _first_full_gate_tensor(
            output,
            token_count=int(tensor.shape[0]),
            num_experts=int(self.num_experts),
        )
        if logits is None:
            raise ValueError(
                f"next-layer router for {self.layer_name!r} did not expose full "
                f"[tokens, experts] gate scores; provide a router.gating method or weight"
            )
        return logits.detach().float().cpu().numpy()


class MegatronFateContextProvider:
    """Thread-safe source-hidden snapshot and target-router binding store."""

    def __init__(
        self,
        *,
        bindings: dict[str, MegatronFateLayerBinding],
        group_rank: int,
        world_size: int,
        run_id: str,
    ) -> None:
        self.bindings = dict(bindings)
        self.group_rank = int(group_rank)
        self.world_size = int(world_size)
        self.run_id = str(run_id)
        self._hidden_by_source: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        if not self.bindings:
            raise ValueError("faithful FATE requires at least one router-equipped MoE layer")
        if self.group_rank < 0 or self.group_rank >= self.world_size:
            raise ValueError("group_rank outside FATE world")

    def capture(self, *, source_layer_id: str, hidden_features: object) -> None:
        value = hidden_features[0] if isinstance(hidden_features, tuple) and hidden_features else hidden_features
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.ndim == 3:
                tensor = tensor.reshape(-1, tensor.shape[-1])
            if tensor.ndim != 2:
                raise ValueError(f"FATE source hidden must be 2-D/3-D, got {tuple(tensor.shape)}")
            snapshot = tensor.float().cpu().numpy().copy()
        else:
            snapshot = np.asarray(value, dtype=np.float32)
            if snapshot.ndim == 3:
                snapshot = snapshot.reshape(-1, snapshot.shape[-1])
            if snapshot.ndim != 2:
                raise ValueError(f"FATE source hidden must be 2-D/3-D, got {snapshot.shape}")
            snapshot = np.ascontiguousarray(snapshot).copy()
        with self._lock:
            self._hidden_by_source[str(source_layer_id)] = snapshot

    def __call__(self, *, source_layer_id: str, target_layer_id: str) -> ExpertRouteContext:
        source = str(source_layer_id)
        target = str(target_layer_id)
        with self._lock:
            hidden = self._hidden_by_source.get(source)
        if hidden is None:
            raise RuntimeError(f"missing FATE gate-input snapshot for source layer {source!r}")
        binding = self.bindings.get(target)
        if binding is None:
            raise RuntimeError(f"missing FATE next-layer router binding for target layer {target!r}")
        source_ranks = np.full((int(hidden.shape[0]),), int(self.group_rank), dtype=np.int64)
        context = ExpertRouteContext(
            identity=PredictionIdentity(
                request_id=f"{self.run_id}:fate:{source}->{target}",
                run_id=self.run_id,
                source_layer_id=source,
                target_layer_id=target,
            ),
            hidden_features=hidden,
            gate_features={
                "next_layer_gate": binding.gate,
                "source_ranks": source_ranks,
                "gate_output_domain": binding.gate_output_domain,
            },
            top_k=int(binding.top_k),
            expert_owner_by_id=tuple(int(value) for value in binding.expert_owner_by_id),
            world_size=int(self.world_size),
        )
        context.validate()
        return context


def build_megatron_fate_context_provider(
    *,
    model: torch.nn.Module,
    group_rank: int,
    world_size: int,
    run_id: str,
) -> MegatronFateContextProvider:
    bindings: dict[str, MegatronFateLayerBinding] = {}
    failures: list[str] = []
    for name, module in model.named_modules():
        if getattr(module, "token_dispatcher", None) is None:
            continue
        router = getattr(module, "router", None)
        if router is None:
            failures.append(f"{name}:missing_router")
            continue
        experts = _num_experts(module, router)
        top_k = _top_k(module, router)
        if experts is None or top_k is None:
            failures.append(f"{name}:missing_num_experts_or_topk")
            continue
        if int(top_k) > int(experts):
            failures.append(f"{name}:topk_exceeds_experts")
            continue
        layer = _layer_id(name)
        bindings[layer] = MegatronFateLayerBinding(
            layer_id=layer,
            layer_name=str(name),
            router=router,
            top_k=int(top_k),
            num_experts=int(experts),
            expert_owner_by_id=_contiguous_owner_map(int(experts), int(world_size)),
            gate_output_domain="logits",
        )
    if not bindings:
        detail = ",".join(failures[:8]) or "no_token_dispatcher_layers"
        raise ValueError(f"unable to build faithful FATE Megatron bindings: {detail}")
    return MegatronFateContextProvider(
        bindings=bindings,
        group_rank=int(group_rank),
        world_size=int(world_size),
        run_id=str(run_id),
    )


__all__ = [
    "MegatronFateContextProvider",
    "MegatronFateLayerBinding",
    "build_megatron_fate_context_provider",
]
