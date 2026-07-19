from __future__ import annotations

from rs.runtime.online.megatron_ep.host import get_process_group_ranks_safe


def test_smoke_group_helper_returns_world_tuple_only_when_explicitly_requested(monkeypatch) -> None:
    monkeypatch.setattr("rs.runtime.online.megatron_ep.host.dist.is_initialized", lambda: True)
    monkeypatch.setattr("rs.runtime.online.megatron_ep.host.dist.get_world_size", lambda: 2)
    assert get_process_group_ranks_safe(None, allow_world_group=True) == (0, 1)
