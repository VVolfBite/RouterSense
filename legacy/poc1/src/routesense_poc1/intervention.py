from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any


class RoutePatchError(RuntimeError):
    pass


class RouteAblationContext(AbstractContextManager):
    def __init__(
        self,
        model: Any,
        layer_path: str,
        token_pos: int,
        expert_rank: int,
        expected_expert_id: int,
    ) -> None:
        self.model = model
        self.layer_path = layer_path
        self.token_pos = token_pos
        self.expert_rank = expert_rank
        self.expected_expert_id = expected_expert_id
        self._layer = None
        self._original_forward = None

    def __enter__(self) -> "RouteAblationContext":
        self._layer = self.model.get_submodule(self.layer_path)
        self._original_forward = self._layer.forward
        self._layer.forward = self._wrapped_forward
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._layer is not None and self._original_forward is not None:
            self._layer.forward = self._original_forward

    def _wrapped_forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if batch_size != 1:
            raise RoutePatchError("batch size must be 1 for precise route ablation")
        flat_hidden = hidden_states.view(-1, hidden_dim)
        router_logits, top_k_weights, top_k_indices = self._layer.gate(flat_hidden)
        top_k_weights = top_k_weights.clone()
        flat_target_pos = self.token_pos
        if int(top_k_indices[flat_target_pos, self.expert_rank].item()) != self.expected_expert_id:
            raise RoutePatchError("expected_expert_id does not match selected expert")
        top_k_weights[flat_target_pos, self.expert_rank] = 0.0
        final_hidden_states = self._layer.experts(flat_hidden, top_k_indices, top_k_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def verify_intervention_correctness() -> dict[str, str]:
    return {
        "status": "requires_gpu_runtime",
        "message": (
            "Real OLMoE intervention correctness must be verified on GPU with the target model; "
            "CPU-side build only provides the exact patching contract and hard failure behavior."
        ),
    }
