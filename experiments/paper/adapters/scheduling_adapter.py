from __future__ import annotations

from typing import Any

from rs.runtime.offline.scheduling_adapter import (
    execute_policy as _execute_formal_policy,
    replay_window_from_matrices,
)
from rs.runtime.offline.replay_unified import ReplayWindow
from rs.scheduling.algorithm_catalog import get_algorithm_metadata
from rs.scheduling.catalog import resolve_algorithm_id


def execute_policy(
    *,
    replay_window: ReplayWindow,
    policy_name: str,
    hint_type: str,
    p2_hint_rows: tuple[tuple[int, ...], ...],
    confidence: float = 1.0,
    expert_compute_delay: float = 0.0,
    bucket_rows: int = 1,
    max_waves: int = 256,
) -> dict[str, Any]:
    result = _execute_formal_policy(
        replay_window=replay_window,
        policy_name=policy_name,
        hint_type=hint_type,
        p2_hint_rows=p2_hint_rows,
        confidence=confidence,
        expert_compute_delay=expert_compute_delay,
        bucket_rows=bucket_rows,
        max_waves=max_waves,
    )
    try:
        result["policy_metadata"] = get_algorithm_metadata(policy_name)
    except KeyError:
        resolved = resolve_algorithm_id(policy_name)
        try:
            result["policy_metadata"] = get_algorithm_metadata(str(resolved.canonical_name))
        except KeyError:
            result["policy_metadata"] = {
                "algorithm_id": str(resolved.canonical_name),
                "heuristic_family": "unclassified",
                "role": "unclassified",
            }
    return result


__all__ = ["execute_policy", "replay_window_from_matrices"]
