from __future__ import annotations

from integrations.megatron_ep.routersense.policy.bucketed_fifo import BucketedFIFOPolicy
from integrations.megatron_ep.routersense.policy.registry import resolve_phase_policy
from integrations.megatron_ep.routersense.policy.trivial_reverse_bucket import TrivialReverseBucketPolicy


def test_policy_registry_resolves_named_policy() -> None:
    fifo = resolve_phase_policy(policy_name="bucketed_fifo", bucket_rows=16)
    reverse = resolve_phase_policy(policy_name="trivial_reverse_bucket", bucket_rows=16)
    assert isinstance(fifo, BucketedFIFOPolicy)
    assert isinstance(reverse, TrivialReverseBucketPolicy)
