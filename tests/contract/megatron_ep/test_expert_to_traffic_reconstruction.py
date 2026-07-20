from __future__ import annotations

import json
from pathlib import Path

import pytest

from rs.runtime.offline.expert_to_traffic_reconstruction import (
    merge_source_expert_counts_by_layer_and_source_rank,
    run_expert_to_traffic_reconstruction,
)
from rs.runtime.online.megatron_ep.prediction.expert_to_traffic import (
    compare_reconstructed_traffic,
    source_expert_counts_to_traffic_matrix,
)
from rs.runtime.online.megatron_ep.prediction.expert_trace import SourceExpertCountMatrix


def test_expert_to_traffic_reconstruction_respects_remote_only() -> None:
    counts = SourceExpertCountMatrix(
        layer_id=0,
        world_size=2,
        num_experts=4,
        counts=((2, 1, 0, 0), (0, 0, 3, 1)),
        expert_to_rank_map=(0, 1, 0, 1),
    )
    expert_to_rank = {0: 0, 1: 1, 2: 0, 3: 1}
    matrix = source_expert_counts_to_traffic_matrix(counts, expert_to_rank, bytes_per_token=10)
    assert matrix == ((0, 10), (30, 0))
    audit = compare_reconstructed_traffic(matrix, ((999, 10), (30, 111)))
    assert audit.relative_l1_error == 0.0
    assert audit.self_bytes_ignored == 1110


def test_missing_expert_to_rank_raises() -> None:
    counts = SourceExpertCountMatrix(
        layer_id=0,
        world_size=2,
        num_experts=2,
        counts=((1, 0), (0, 1)),
        expert_to_rank_map=(0, 1),
    )
    with pytest.raises(ValueError, match="missing expert_to_rank"):
        source_expert_counts_to_traffic_matrix(counts, {0: 0}, bytes_per_token=4)


def test_global_rank_mapping_is_rejected_for_traffic_matrix_columns() -> None:
    counts = SourceExpertCountMatrix(
        layer_id=0,
        world_size=2,
        num_experts=2,
        counts=((1, 0), (0, 1)),
        expert_to_rank_map=(4, 5),
    )
    with pytest.raises(ValueError, match="EP-local rank indices"):
        source_expert_counts_to_traffic_matrix(counts, {0: 4, 1: 5}, bytes_per_token=4)


def test_merge_source_expert_counts_builds_complete_world_matrix() -> None:
    records = [
        SourceExpertCountMatrix(layer_id=7, world_size=4, num_experts=4, counts=((3, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), source_rank=0, expert_to_rank_map=(0, 1, 2, 3)),
        SourceExpertCountMatrix(layer_id=7, world_size=4, num_experts=4, counts=((0, 0, 0, 0), (0, 2, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0)), source_rank=1, expert_to_rank_map=(0, 1, 2, 3)),
        SourceExpertCountMatrix(layer_id=7, world_size=4, num_experts=4, counts=((0, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 2), (0, 0, 0, 0)), source_rank=2, expert_to_rank_map=(0, 1, 2, 3)),
        SourceExpertCountMatrix(layer_id=7, world_size=4, num_experts=4, counts=((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 1, 0, 1)), source_rank=3, expert_to_rank_map=(0, 1, 2, 3)),
    ]
    merged, diagnostics = merge_source_expert_counts_by_layer_and_source_rank(
        records,
        layer_id=7,
        world_size=4,
        num_experts=4,
    )
    assert diagnostics["complete_world_matrix"] is True
    assert diagnostics["missing_source_ranks"] == []
    matrix = source_expert_counts_to_traffic_matrix(
        merged,
        {0: 0, 1: 1, 2: 2, 3: 3},
        bytes_per_token=10,
    )
    assert matrix == (
        (0, 10, 0, 0),
        (0, 0, 10, 0),
        (10, 0, 0, 20),
        (0, 10, 0, 0),
    )


def test_merge_source_expert_counts_reports_missing_source_rank() -> None:
    records = [
        SourceExpertCountMatrix(layer_id=3, world_size=4, num_experts=2, counts=((1, 0), (0, 0), (0, 0), (0, 0)), source_rank=0, expert_to_rank_map=(0, 1)),
        SourceExpertCountMatrix(layer_id=3, world_size=4, num_experts=2, counts=((0, 0), (0, 1), (0, 0), (0, 0)), source_rank=1, expert_to_rank_map=(0, 1)),
        SourceExpertCountMatrix(layer_id=3, world_size=4, num_experts=2, counts=((0, 0), (0, 0), (1, 0), (0, 0)), source_rank=2, expert_to_rank_map=(0, 1)),
    ]
    merged, diagnostics = merge_source_expert_counts_by_layer_and_source_rank(
        records,
        layer_id=3,
        world_size=4,
        num_experts=2,
    )
    assert merged.counts[3] == (0, 0)
    assert diagnostics["complete_world_matrix"] is False
    assert diagnostics["missing_source_ranks"] == [3]


def test_merge_source_expert_counts_rejects_conflicting_rank_rows() -> None:
    records = [
        SourceExpertCountMatrix(layer_id=5, world_size=2, num_experts=2, counts=((1, 0), (0, 0)), source_rank=0, expert_to_rank_map=(0, 1)),
        SourceExpertCountMatrix(layer_id=5, world_size=2, num_experts=2, counts=((2, 0), (0, 0)), source_rank=0, expert_to_rank_map=(0, 1)),
    ]
    with pytest.raises(ValueError, match="conflicting source_expert_counts"):
        merge_source_expert_counts_by_layer_and_source_rank(
            records,
            layer_id=5,
            world_size=2,
            num_experts=2,
        )


def test_corrected_expert_to_traffic_o1_is_zero_on_gpu_trace_fixture(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "trace"
    fixture_dir.mkdir()
    rows = [
        SourceExpertCountMatrix(
            layer_id=0,
            world_size=2,
            num_experts=4,
            counts=((1, 1, 0, 0), (0, 0, 0, 0)),
            source_rank=0,
            expert_to_rank_map=(0, 1, 0, 1),
            bytes_per_token=4096,
        ),
        SourceExpertCountMatrix(
            layer_id=0,
            world_size=2,
            num_experts=4,
            counts=((0, 0, 0, 0), (0, 0, 2, 1)),
            source_rank=1,
            expert_to_rank_map=(0, 1, 0, 1),
            bytes_per_token=4096,
        ),
    ]
    from rs.runtime.online.megatron_ep.prediction.expert_trace import write_source_expert_counts_jsonl

    write_source_expert_counts_jsonl(fixture_dir / "rank0_source_expert_counts.jsonl", rows)
    phase_context_rows = [
        {"phase": "P0", "layer_id": 0, "global_rank": 0, "per_peer_bytes": [4096, 4096]},
        {"phase": "P0", "layer_id": 0, "global_rank": 1, "per_peer_bytes": [8192, 4096]},
    ]
    (fixture_dir / "rank0_phase_contexts.jsonl").write_text(
        "\n".join(json.dumps(row) for row in phase_context_rows) + "\n",
        encoding="utf-8",
    )
    payload = run_expert_to_traffic_reconstruction(fixture_dir=fixture_dir, bytes_per_token=4096)
    assert payload["summary"]["o1_corrected_relative_l1"] == 0.0
    assert payload["summary"]["bytes_model_used"] == "hidden_only"
    assert payload["summary"]["actual_matrix_source"] == "phase_context_aggregated_p0_dispatch"
    assert payload["summary"]["matrix_scope"] == "remote_only"
