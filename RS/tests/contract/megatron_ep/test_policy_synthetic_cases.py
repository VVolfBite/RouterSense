from __future__ import annotations

import json
from pathlib import Path

from integrations.megatron_ep.routersense.policy.registry import resolve_phase_policy, supported_phase_policies
from .helpers import make_contexts_from_matrix


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "policy_cases"


def _load_case(name: str) -> tuple[str, int, tuple[tuple[int, ...], ...]]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    matrix = tuple(tuple(int(v) for v in row) for row in payload["matrix"])
    return str(payload["phase"]), int(payload["bucket_rows"]), matrix


def test_policy_synthetic_cases_cover_exact_once() -> None:
    for fixture_name in ["asymmetric_4rank.json", "receiver_incast_4rank.json", "duplex_pair_4rank.json", "skewed_8rank.json"]:
        phase, bucket_rows, matrix = _load_case(fixture_name)
        contexts = make_contexts_from_matrix(phase=phase, matrix=matrix)
        for policy_name in ["bucketed_fifo", "trivial_reverse_bucket", "aurora_order_fixed", "fast_bvn_single_tier"]:
            plan = resolve_phase_policy(policy_name=policy_name, bucket_rows=bucket_rows).build_plan(
                local_context=contexts[0],
                global_contexts=contexts,
            )
            expected_rows = sum(sum(int(v) for i, v in enumerate(row) if i != src) for src, row in enumerate(matrix))
            actual_rows = sum(int(task.row_count) for wave in plan.waves for task in wave.bucket_tasks)
            assert actual_rows == expected_rows


def test_policy_library_produces_distinct_valid_plans_on_at_least_one_case() -> None:
    phase, bucket_rows, matrix = _load_case("receiver_incast_4rank.json")
    contexts = make_contexts_from_matrix(phase=phase, matrix=matrix)
    signatures = {}
    for policy_name in ["bucketed_fifo", "trivial_reverse_bucket", "aurora_order_fixed", "fast_bvn_single_tier"]:
        plan = resolve_phase_policy(policy_name=policy_name, bucket_rows=bucket_rows).build_plan(
            local_context=contexts[0],
            global_contexts=contexts,
        )
        signatures[policy_name] = tuple(task.task_id for wave in plan.waves for task in wave.bucket_tasks)
    assert len(set(signatures.values())) >= 2


def test_supported_policy_registry_contains_all_expected_names() -> None:
    supported = set(supported_phase_policies())
    assert {
        "bucketed_fifo",
        "trivial_reverse_bucket",
        "aurora_order_fixed",
        "fast_bvn_single_tier",
        "routersense_p0p1_reservation",
        "routersense_p0p1p2_hint",
    }.issubset(supported)
