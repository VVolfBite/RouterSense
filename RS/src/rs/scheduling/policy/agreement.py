"""Deprecated scheduling-side agreement helpers.

The formal control-plane agreement implementation lives under
``rs.runtime.online.megatron_ep.control``.  The scheduling layer must remain
pure and must not depend on ``torch.distributed`` or runtime orchestration.

This module intentionally retains only small, pure helpers needed by older
callers during the migration window.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def compute_ep_group_hash(ranks: Iterable[int]) -> str:
    return hashlib.sha256(",".join(str(int(rank)) for rank in ranks).encode("utf-8")).hexdigest()[:16]


def run_policy_agreement(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError(
        "run_policy_agreement moved to rs.runtime.online.megatron_ep.control.plan_agreement; "
        "scheduling-side agreement is no longer supported"
    )


__all__ = ["compute_ep_group_hash", "run_policy_agreement"]
