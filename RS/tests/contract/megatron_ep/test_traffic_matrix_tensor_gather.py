from __future__ import annotations

import torch

from rs.runtime.online.megatron_ep.control.p2_matrix import (
    build_local_peer_bytes_tensor,
    build_traffic_matrix_bundle,
    gather_global_peer_bytes_matrix,
)


def test_build_local_peer_bytes_tensor_shapes_and_dtype() -> None:
    tensor = build_local_peer_bytes_tensor((4, 8), 4, "cpu")
    assert tensor.dtype == torch.int64
    assert tensor.device.type == "cpu"
    assert tensor.tolist() == [4, 8, 0, 0]


def test_gather_global_peer_bytes_matrix_single_rank_fallback() -> None:
    matrix, metadata = gather_global_peer_bytes_matrix(build_local_peer_bytes_tensor((7,), 1, "cpu"))
    assert matrix.tolist() == [[7]]
    assert metadata["matrix_source"] == "single_rank_fallback"
    assert metadata["is_global"] is False
    assert metadata["total_bytes"] == 7
    assert metadata["row_sums"] == (7,)
    assert metadata["col_sums"] == (7,)
    assert metadata["nonzero_edge_count"] == 0


def test_build_traffic_matrix_bundle_reports_replicated_local_row_fallback(monkeypatch) -> None:
    from rs.runtime.online.megatron_ep.control import p2_matrix as mod

    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(mod.dist, "all_gather_object", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("object collective forbidden")))
    bundle = build_traffic_matrix_bundle(per_peer_bytes=(0, 9), world_size=2, device="cpu", group=None)
    assert bundle.matrix_source == "replicated_local_row_fallback"
    assert bundle.is_global is False
    assert bundle.matrix == ((0, 9), (0, 0))
    assert bundle.total_bytes == 9
    assert bundle.row_sums == (9, 0)
    assert bundle.col_sums == (0, 9)

