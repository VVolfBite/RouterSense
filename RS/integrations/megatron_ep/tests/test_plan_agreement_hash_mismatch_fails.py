from __future__ import annotations

import pytest

from integrations.megatron_ep.routersense.policy.agreement import validate_rank_hashes


def test_plan_agreement_hash_mismatch_fails() -> None:
    with pytest.raises(RuntimeError):
        validate_rank_hashes(("a" * 64, "b" * 64))
