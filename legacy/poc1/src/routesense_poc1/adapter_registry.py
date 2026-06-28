from __future__ import annotations

from typing import Protocol


class MoERouteAdapter(Protocol):
    name: str

    def supports(self, layer) -> bool:
        ...

    def extract_routing_state(self, layer, args, kwargs):
        ...

    def apply_ablation(self, routing_state, token_pos, rank, expected_expert_id):
        ...

    def call_original_forward(self, original_forward, routing_state, args, kwargs):
        ...


class UnsupportedOLMoEAdapter:
    name = "UnsupportedOLMoEAdapter"

    def supports(self, layer) -> bool:
        return False

    def extract_routing_state(self, layer, args, kwargs):
        raise RuntimeError(
            "真实 OLMoE route patch 尚未适配。请先查看 outputs/model_inspection.json 和 outputs/selected_moe_layer_forward.txt。"
        )

    def apply_ablation(self, routing_state, token_pos, rank, expected_expert_id):
        raise RuntimeError(
            "真实 OLMoE route patch 尚未适配。请先查看 outputs/model_inspection.json 和 outputs/selected_moe_layer_forward.txt。"
        )

    def call_original_forward(self, original_forward, routing_state, args, kwargs):
        raise RuntimeError(
            "真实 OLMoE route patch 尚未适配。请先查看 outputs/model_inspection.json 和 outputs/selected_moe_layer_forward.txt。"
        )


def get_adapter_for_layer(layer) -> MoERouteAdapter | None:
    return UnsupportedOLMoEAdapter()
