import pytest

from routesense_poc1.metrics import next_token_nll

pytest.importorskip("torch")


def test_next_token_nll_supports_torch_tensor():
    import torch

    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
    input_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    nll = next_token_nll(logits, input_ids)
    assert isinstance(nll, float)
