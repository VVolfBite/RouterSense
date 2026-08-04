from __future__ import annotations

from routesense.trace import decode_metadata_tensor_rows


def test_decode_metadata_tensor_rows_round_trip():
    torch = __import__("torch")
    tensor = torch.tensor([[11, 22, 0, 1, 3, 2]], dtype=torch.int64)
    rows = decode_metadata_tensor_rows(tensor)
    assert rows == [{
        "token_id": 11,
        "global_route_item_index": 22,
        "origin_rank": 0,
        "destination_rank": 1,
        "expert_id": 3,
        "route_rank": 2,
    }]
