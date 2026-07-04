from __future__ import annotations

from integrations.megatron_ep.native_runtime import get_process_group_ranks_safe


def test_smoke_group_helper_returns_tuple() -> None:
    assert isinstance(get_process_group_ranks_safe(None), tuple)
