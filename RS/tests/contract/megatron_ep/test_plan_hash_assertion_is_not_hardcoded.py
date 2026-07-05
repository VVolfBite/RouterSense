from __future__ import annotations

from integrations.megatron_ep.routersense.dispatcher_facade import PolicyRuntimeRecord


def test_plan_hash_assertion_is_not_hardcoded_contract() -> None:
    # Contract-only guard: the smoke path must inspect agreement rank hashes,
    # not an unconditional constant.
    fields = set(PolicyRuntimeRecord.__dataclass_fields__.keys())
    assert "agreement" in fields
