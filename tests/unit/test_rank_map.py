from __future__ import annotations

import pytest

from rs.runtime.online.megatron_ep.control.rank_map import RankMap


def test_rank_map_roundtrip_for_noncontiguous_group() -> None:
    rank_map = RankMap(group_ranks=(2, 3), root_rank=2)
    assert rank_map.group_rank_to_global_rank(0) == 2
    assert rank_map.group_rank_to_global_rank(1) == 3
    assert rank_map.global_rank_to_group_rank(2) == 0
    assert rank_map.global_rank_to_group_rank(3) == 1
    assert rank_map.root_group_rank == 0


def test_rank_map_rejects_invalid_rank() -> None:
    rank_map = RankMap(group_ranks=(0, 2), root_rank=0)
    with pytest.raises(ValueError):
        rank_map.group_rank_to_global_rank(2)
    with pytest.raises(ValueError):
        rank_map.global_rank_to_group_rank(1)
