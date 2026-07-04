from __future__ import annotations

import torch

from rs.runtime.distributed_ep.adapter.expert_store import extract_local_expert_weights


class _Linear:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight


class _Expert:
    def __init__(self, gate_weight: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor) -> None:
        self.gate_proj = _Linear(gate_weight)
        self.up_proj = _Linear(up_weight)
        self.down_proj = _Linear(down_weight)

    def parameters(self):
        return [self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight]


def test_extract_local_expert_weights_from_modulelist_style_experts() -> None:
    experts = [
        _Expert(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
            torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32),
        ),
        _Expert(
            torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
            torch.tensor([[6.0, 7.0], [8.0, 9.0]], dtype=torch.float32),
            torch.tensor([[10.0, 11.0], [12.0, 13.0]], dtype=torch.float32),
        ),
    ]
    local = extract_local_expert_weights(experts, [1])
    assert local.local_expert_ids == [1]
    assert tuple(local.gate_up_proj.shape) == (1, 4, 2)
    assert tuple(local.down_proj.shape) == (1, 2, 2)
    assert torch.equal(local.gate_up_proj[0, :2], experts[1].gate_proj.weight)
    assert torch.equal(local.gate_up_proj[0, 2:], experts[1].up_proj.weight)
    assert torch.equal(local.down_proj[0], experts[1].down_proj.weight)
