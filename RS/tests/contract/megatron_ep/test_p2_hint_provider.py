from __future__ import annotations

import pytest

from rs.runtime.online.megatron_ep.p2_provider import build_p2_hint_provider
from rs.runtime.online.megatron_ep.p2_contracts import P2HintRequest


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


def test_calibrated_artifact_p2_provider_fails_closed() -> None:
    provider = build_p2_hint_provider("calibrated_artifact")
    with pytest.raises(RuntimeError):
        provider.build_hint(
            P2HintRequest(
                plan_key={"layer_id": "0", "phase": "P0"},
                layer_id="0",
                phase="P0",
                global_rank=0,
                local_rank=0,
                ep_group_ranks=(0, 1),
            )
        )
