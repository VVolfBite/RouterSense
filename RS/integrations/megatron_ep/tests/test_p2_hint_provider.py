from __future__ import annotations

from integrations.megatron_ep.routersense.p2 import build_p2_hint_provider
from integrations.megatron_ep.routersense.p2.contracts import P2HintRequest


def test_deterministic_stub_p2_hint_is_stable_and_nonempty() -> None:
    provider = build_p2_hint_provider("deterministic_stub")
    request = P2HintRequest(
        plan_key={"layer_id": "0", "phase": "P0"},
        layer_id="0",
        phase="P0",
        global_rank=0,
        local_rank=0,
        ep_group_ranks=(0, 1),
    )
    first = provider.build_hint(request)
    second = provider.build_hint(request)
    assert first.hint_mode == "deterministic_stub"
    assert first.hint_digest
    assert first.hint_digest == second.hint_digest
    assert first.hint_source == "deterministic_stub_from_current_plan_key"


def test_no_p2_hint_provider_returns_none_mode() -> None:
    provider = build_p2_hint_provider("none")
    hint = provider.build_hint(
        P2HintRequest(
            plan_key={"layer_id": "0", "phase": "P1"},
            layer_id="0",
            phase="P1",
            global_rank=1,
            local_rank=1,
            ep_group_ranks=(0, 1),
        )
    )
    assert hint.hint_mode == "none"
    assert hint.hint_digest == "none"
