#!/usr/bin/env python3
"""Small-instance oracle gap replay and real-fixture oracle proxy summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline.replay_fixture_policy_study import _build_problem
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.scheduling import resolve_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--policies", nargs="*", default=None)
    return parser.parse_args()


def _small_fixture() -> dict[str, Any]:
    return {
        "num_gpus": 2,
        "p0_dispatch_matrix": [[0, 8], [3, 0]],
        "p1_return_matrix": [[0, 5], [7, 0]],
        "p2_next_dispatch_forecast_matrix": [[0, 4], [2, 0]],
        "p2_next_dispatch_matrix": [[0, 4], [2, 0]],
        "metadata": {"layer_id": "0", "next_layer_id": "1"},
    }


def _run_policy(problem, policy_name: str) -> dict[str, Any]:
    plan = resolve_policy(policy_name=policy_name, bucket_rows=0).build_logical_plan(problem)
    audit = replay_and_audit_logical_plan(problem, plan)
    return {
        "policy_name": policy_name,
        "makespan": float(audit.get("makespan", plan.diagnostics.get("makespan", 0.0))),
        "valid": bool(audit.get("valid", False)),
    }


def run_oracle_gap_replay(
    *,
    fixture_dir: Path,
    small_only: bool = False,
    policies: Iterable[str] | None = None,
) -> dict[str, Any]:
    small_problem = _build_problem(_small_fixture(), mode="execution_window", p2_source="actual_trace", expert_compute_delay=0.0)
    selected_policies = list(
        policies
        or [
        "birkhoff_von_neumann_fluid",
        "exact_small_instance_reference",
        "B_birkhoff",
        "B_barrier_criticality_matching",
        "U_barrier_criticality_global_matching",
        "RS_safe_barrier_criticality",
        ]
    )
    small_rows = [_run_policy(small_problem, policy_name) for policy_name in selected_policies]
    by_name = {row["policy_name"]: row for row in small_rows}
    o_local = by_name.get("birkhoff_von_neumann_fluid", {}).get("makespan", 0.0)
    o_joint = by_name.get("exact_small_instance_reference", {}).get("makespan", 0.0)
    payload = {
        "O_local_definition": "birkhoff_von_neumann_fluid",
        "O_joint_definition": "exact_small_instance_reference_small_fixture",
        "O_joint_small_fixture_available": True,
        "O_joint_real_fixture_available": False,
        "real_fixture_joint_proxy": "best_execution_window_U",
        "selected_policies": selected_policies,
        "small_fixture_rows": small_rows,
        "oracle_gap_small_fixture_summary": {
            "O_joint_vs_O_local_gap": None if o_local == 0 else float((o_joint - o_local) / o_local),
            "B_gap_to_O_local": None if o_local == 0 or "B_birkhoff" not in by_name else float((by_name["B_birkhoff"]["makespan"] - o_local) / o_local),
            "raw_U_gap_to_O_joint": None if o_joint == 0 or "U_barrier_criticality_global_matching" not in by_name else float((by_name["U_barrier_criticality_global_matching"]["makespan"] - o_joint) / o_joint),
            "safe_U_gap_to_O_joint": None if o_joint == 0 or "RS_safe_barrier_criticality" not in by_name else float((by_name["RS_safe_barrier_criticality"]["makespan"] - o_joint) / o_joint),
        },
    }
    if not small_only:
        fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"))
        payload["real_fixture_joint_proxy_status"] = "available" if fixture_paths else "missing_fixture_dir"
    return payload


def main() -> None:
    args = _parse_args()
    payload = run_oracle_gap_replay(
        fixture_dir=Path(args.fixture_dir),
        small_only=bool(args.small_only),
        policies=args.policies,
    )
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
