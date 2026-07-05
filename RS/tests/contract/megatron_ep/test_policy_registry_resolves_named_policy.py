from __future__ import annotations

from rs.scheduling.phase_local.fifo import BucketedFIFOPolicy
from rs.scheduling.phase_local.trivial_reverse_bucket import TrivialReverseBucketPolicy
from rs.scheduling.registry import resolve_phase_policy


def test_policy_registry_resolves_named_policy() -> None:
    fifo = resolve_phase_policy(policy_name="bucketed_fifo", bucket_rows=16)
    reverse = resolve_phase_policy(policy_name="trivial_reverse_bucket", bucket_rows=16)
    assert isinstance(fifo, BucketedFIFOPolicy)
    assert isinstance(reverse, TrivialReverseBucketPolicy)
