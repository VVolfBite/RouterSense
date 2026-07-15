from __future__ import annotations

from rs.runtime.online.megatron_ep import host as host_mod


def test_dedicated_group_registry_uses_global_creation_order(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []

    class FakeDist:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_world_size() -> int:
            return 4

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def new_group(*, ranks):
            calls.append(tuple(int(rank) for rank in ranks))
            return tuple(int(rank) for rank in ranks)

        @staticmethod
        def all_gather_object(gathered, payload):
            rows = [(0, 1), (0, 1), (2, 3), (2, 3)]
            for index, row in enumerate(rows):
                gathered[index] = row

        @staticmethod
        def all_reduce(tensor, group=None):
            return None

        @staticmethod
        def get_backend(group=None):
            return "gloo"

    monkeypatch.setattr(host_mod, "_DEDICATED_P2P_GROUP_REGISTRY", {})
    monkeypatch.setattr(host_mod, "dist", FakeDist)
    monkeypatch.setattr(host_mod.torch.cuda, "is_available", lambda: False)

    group, status = host_mod._maybe_create_dedicated_p2p_group(
        ep_group_ranks=(2, 3),
        local_rank=1,
    )
    assert group == (2, 3)
    assert calls == [(0, 1), (2, 3)]
    assert status["dedicated_p2p_groups_created"] == [[0, 1], [2, 3]]
    assert status["local_dedicated_group_ranks"] == [2, 3]
    assert status["p2p_group_warmup_passed"] is True
    assert status["new_group_call_order"] == [[0, 1], [2, 3]]
    assert status["hotpath_new_group_count"] == 0
