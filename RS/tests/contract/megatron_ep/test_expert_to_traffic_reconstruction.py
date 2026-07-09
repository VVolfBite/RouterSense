from __future__ import annotations

from rs.runtime.online.megatron_ep.prediction.expert_to_traffic import compare_reconstructed_traffic, source_expert_counts_to_traffic_matrix
from rs.runtime.online.megatron_ep.prediction.expert_trace import SourceExpertCountMatrix


def test_expert_to_traffic_reconstruction_respects_remote_only() -> None:
    counts = SourceExpertCountMatrix(
        layer_id=0,
        world_size=2,
        num_experts=4,
        counts=((2, 1, 0, 0), (0, 0, 3, 1)),
    )
    expert_to_rank = {0: 0, 1: 1, 2: 0, 3: 1}
    matrix = source_expert_counts_to_traffic_matrix(counts, expert_to_rank, bytes_per_token=10)
    assert matrix == ((0, 10), (30, 0))
    audit = compare_reconstructed_traffic(matrix, ((999, 10), (30, 111)))
    assert audit.relative_l1_error == 0.0
    assert audit.self_bytes_ignored == 1110
