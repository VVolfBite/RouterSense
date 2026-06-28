#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.distributed_runtime import available_audit_strategies


CORE_POLICIES = ["fifo", "random-order", "strong-state", "full"]
ROUND_SIZES = [1, 2, 4, 8, 999999]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit policy leverage and round bottlenecks from existing POC2 NCCL artifacts.")
    parser.add_argument("--artifact-dir", action="append", required=True)
    parser.add_argument("--audit-output-tag", type=str, default="poc2_policy_leverage_audit")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-repetitions", type=int, default=3)
    parser.add_argument("--replay-warmup", type=int, default=1)
    args = parser.parse_args(argv)

    audit_root = ROOT / "artifacts" / args.audit_output_tag / "policy_leverage_audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_round_rows = []
    pair_rows = []
    overlap_rows = []
    position_rows = []
    replay_rows = []

    for artifact_dir in [Path(item) for item in args.artifact_dir]:
        summary, round_rows, pair_payload, overlap_payload, position_payload = audit_existing_artifact(artifact_dir)
        all_summaries.append(summary)
        all_round_rows.extend(round_rows)
        pair_rows.extend(pair_payload)
        overlap_rows.extend(overlap_payload)
        position_rows.extend(position_payload)
        if args.replay:
            replay_rows.extend(run_round_size_replay(artifact_dir, args.replay_repetitions, args.replay_warmup))

    diagnosis = diagnose_case(all_summaries, replay_rows)
    summary = {
        "artifacts": all_summaries,
        "diagnosis": diagnosis,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (audit_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(audit_root / "round_metrics.jsonl", all_round_rows + replay_rows)
    (audit_root / "policy_pair_comparison.json").write_text(json.dumps(pair_rows, indent=2), encoding="utf-8")
    (audit_root / "round_composition_overlap.json").write_text(json.dumps(overlap_rows, indent=2), encoding="utf-8")
    (audit_root / "position_effect_audit.json").write_text(json.dumps(position_rows, indent=2), encoding="utf-8")
    (audit_root / "goal.md").write_text(
        "# POC2 Policy Leverage Audit\n\n"
        "This audit checks whether scheduler policies materially change round-level bottlenecks, NCCL split sizes, and payload layout.\n"
        "It does not claim dependency value; it only diagnoses runtime leverage under the current harness.\n",
        encoding="utf-8",
    )
    (audit_root / "README.md").write_text(
        "# Policy Leverage Audit Outputs\n\n"
        "- `summary.json`: per-artifact diagnosis and final Case A/B/C/D label.\n"
        "- `round_metrics.jsonl`: round-level load, layout, and timing rows.\n"
        "- `policy_pair_comparison.json`: pairwise bottleneck/load/E2E comparisons.\n"
        "- `round_composition_overlap.json`: bucket/rank load overlap across policies.\n"
        "- `position_effect_audit.json`: position, outlier, and replay summaries.\n",
        encoding="utf-8",
    )
    print(audit_root)
    return 0


def audit_existing_artifact(artifact_dir: Path):
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact_dir / "workload_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((artifact_dir / "protocol.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (artifact_dir / "per_repetition.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [record for record in records if not record.get("warmup")]

    round_rows = []
    by_policy_plan = defaultdict(list)
    for record in records:
        strategy = str(record["strategy"])
        if strategy not in CORE_POLICIES:
            continue
        by_policy_plan[(record["plan_id"], record["repetition_index"], strategy)].append(record)
        for round_record in record["rounds"]:
            row = build_round_row(artifact_dir.name, record, round_record)
            round_rows.append(row)

    overlap_payload = []
    pair_payload = []
    plans = sorted({record["plan_id"] for record in records})
    repetitions = sorted({int(record["repetition_index"]) for record in records})
    for plan_id in plans:
        for repetition_index in repetitions:
            selected = {
                strategy: next(
                    (
                        record
                        for record in records
                        if record["plan_id"] == plan_id and int(record["repetition_index"]) == repetition_index and record["strategy"] == strategy
                    ),
                    None,
                )
                for strategy in CORE_POLICIES
            }
            if not all(selected.values()):
                continue
            overlap_payload.extend(compare_round_overlap(artifact_dir.name, plan_id, repetition_index, selected))
            pair_payload.extend(compare_policy_pairs(artifact_dir.name, plan_id, repetition_index, selected))

    position_payload = [build_position_audit(artifact_dir.name, summary, records, config, manifest, protocol)]
    artifact_summary = {
        "artifact_dir": str(artifact_dir),
        "model_key": manifest["model_key"],
        "placement_policy": manifest["placement_policy"],
        "scenario": manifest["scenario"],
        "regime": manifest["regime"],
        "release_round_size": config["release_round_size"],
        "policy_count": len({record["strategy"] for record in records}),
        "record_count": len(records),
        "round_count": len(round_rows),
    }
    return artifact_summary, round_rows, pair_payload, overlap_payload, position_payload


def build_round_row(artifact_name: str, record: dict[str, Any], round_record: dict[str, Any]) -> dict[str, Any]:
    rank_projection = round_record.get("rank_projection", [])
    recv_bytes = [int(item["projected_recv_bytes"]) for item in rank_projection]
    gini = gini_coefficient(recv_bytes)
    nonzero = [value for value in recv_bytes if value > 0]
    max_recv = max(recv_bytes) if recv_bytes else 0
    min_nonzero = min(nonzero) if nonzero else 0
    return {
        "artifact": artifact_name,
        "plan_id": record["plan_id"],
        "repetition_index": int(record["repetition_index"]),
        "microbatch_id": int(record["microbatch_id"]),
        "strategy": str(record["strategy"]),
        "strategy_order": list(record.get("strategy_order", [])),
        "release_round": int(round_record["release_round"]),
        "bucket_ids": list(round_record["bucket_ids"]),
        "bucket_count": int(round_record.get("bucket_count", len(round_record["bucket_ids"]))),
        "dispatch_bytes": int(round_record["dispatch_bytes"]),
        "return_bytes": int(round_record["return_bytes"]),
        "dispatch_time_ms": float(round_record["dispatch_time_ms"]),
        "compute_time_ms": float(round_record["local_compute_time_ms"]),
        "return_time_ms": float(round_record["return_time_ms"]),
        "round_completion_ms": float(round_record["total_service_time_ms"]),
        "rank_projection": rank_projection,
        "planned_send_splits_rows": list(round_record.get("planned_send_splits_rows", [])),
        "actual_send_splits_rows": list(round_record.get("actual_send_splits_rows", [])),
        "planned_return_splits_rows": list(round_record.get("planned_return_splits_rows", [])),
        "actual_return_splits_rows": list(round_record.get("actual_return_splits_rows", [])),
        "bottleneck_ranks": list(round_record.get("bottleneck_ranks", [])),
        "empty_rank_count": int(round_record.get("empty_rank_count", 0)),
        "max_rank_recv_bytes": max_recv,
        "min_nonzero_rank_recv_bytes": min_nonzero,
        "rank_byte_gini": gini,
        "max_min_nonzero_ratio": (max_recv / min_nonzero) if min_nonzero > 0 else None,
        "collective_participation_mask": list(round_record.get("collective_participation_mask", [])),
        "round_bucket_details": list(round_record.get("round_bucket_details", [])),
    }


def compare_round_overlap(artifact_name: str, plan_id: str, repetition_index: int, selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    pairs = [("full", "random-order"), ("full", "strong-state"), ("full", "fifo"), ("random-order", "fifo")]
    for left, right in pairs:
        left_record = selected[left]
        right_record = selected[right]
        for left_round, right_round in zip(left_record["rounds"], right_record["rounds"]):
            left_set = set(left_round["bucket_ids"])
            right_set = set(right_round["bucket_ids"])
            left_vec = rank_byte_vector(left_round)
            right_vec = rank_byte_vector(right_round)
            rows.append(
                {
                    "artifact": artifact_name,
                    "plan_id": plan_id,
                    "repetition_index": repetition_index,
                    "pair": f"{left}_vs_{right}",
                    "release_round": int(left_round["release_round"]),
                    "bucket_set_jaccard": jaccard(left_set, right_set),
                    "destination_rank_overlap": vector_overlap(left_vec, right_vec),
                    "rank_byte_cosine": cosine(left_vec, right_vec),
                    "rank_byte_correlation": corr(left_vec, right_vec),
                    "bottleneck_rank_exact_match": set(left_round.get("bottleneck_ranks", [])) == set(right_round.get("bottleneck_ranks", [])),
                    "send_split_exact_match": list(left_round.get("planned_send_splits_rows", [])) == list(right_round.get("planned_send_splits_rows", [])),
                    "return_split_exact_match": list(left_round.get("planned_return_splits_rows", [])) == list(right_round.get("planned_return_splits_rows", [])),
                }
            )
    return rows


def compare_policy_pairs(artifact_name: str, plan_id: str, repetition_index: int, selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    pairs = [("full", "random-order"), ("full", "strong-state"), ("full", "fifo"), ("random-order", "fifo")]
    for left, right in pairs:
        left_record = selected[left]
        right_record = selected[right]
        deltas = []
        bottleneck_match = []
        split_identity = []
        for left_round, right_round in zip(left_record["rounds"], right_record["rounds"]):
            deltas.append(int(left_round.get("max_rank_recv_bytes", 0) if "max_rank_recv_bytes" in left_round else max(rank_byte_vector(left_round), default=0)) - int(right_round.get("max_rank_recv_bytes", 0) if "max_rank_recv_bytes" in right_round else max(rank_byte_vector(right_round), default=0)))
            bottleneck_match.append(set(left_round.get("bottleneck_ranks", [])) == set(right_round.get("bottleneck_ranks", [])))
            split_identity.append(list(left_round.get("planned_send_splits_rows", [])) == list(right_round.get("planned_send_splits_rows", [])))
        rows.append(
            {
                "artifact": artifact_name,
                "plan_id": plan_id,
                "repetition_index": repetition_index,
                "pair": f"{left}_vs_{right}",
                "round_count": len(left_record["rounds"]),
                "e2e_delta_ms": float(left_record["end_to_end_batch_completion_ms"]) - float(right_record["end_to_end_batch_completion_ms"]),
                "barrier_delta_ms": float(left_record["barrier_wait_ms"]) - float(right_record["barrier_wait_ms"]),
                "mean_max_rank_byte_delta": statistics.fmean(deltas) if deltas else 0.0,
                "bottleneck_exact_match_ratio": sum(1 for item in bottleneck_match if item) / len(bottleneck_match) if bottleneck_match else 0.0,
                "split_exact_match_ratio": sum(1 for item in split_identity if item) / len(split_identity) if split_identity else 0.0,
                "release_order_equal": left_record["scheduler_decision"]["release_order"] == right_record["scheduler_decision"]["release_order"],
                "layout_differs_but_splits_same": left_record["scheduler_decision"]["release_order"] != right_record["scheduler_decision"]["release_order"] and all(split_identity),
            }
        )
    return rows


def build_position_audit(
    artifact_name: str,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    config: dict[str, Any],
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    by_policy_position = defaultdict(list)
    for record in records:
        order = list(record.get("strategy_order", []))
        if record["strategy"] in order:
            by_policy_position[(record["strategy"], order.index(record["strategy"]) + 1)].append(float(record["end_to_end_batch_completion_ms"]))
    outliers = []
    for pair_name, pair_payload in summary.get("paired_deltas", {}).items():
        for item in pair_payload.get("per_repetition_pairs", []):
            if abs(float(item["delta_ms"])) >= 100.0:
                outliers.append({"pair": pair_name, **item})
    return {
        "artifact": artifact_name,
        "config": config,
        "manifest": manifest,
        "protocol": protocol,
        "by_policy_position": {
            f"{policy}@{position}": {
                "sample_count": len(values),
                "mean": statistics.fmean(values) if values else 0.0,
                "median": statistics.median(values) if values else 0.0,
                "p95": sorted(values)[min(len(values) - 1, int(0.95 * (len(values) - 1)))] if values else 0.0,
            }
            for (policy, position), values in sorted(by_policy_position.items())
        },
        "position_effects": summary.get("position_effects", {}),
        "outliers": outliers,
    }


def run_round_size_replay(artifact_dir: Path, repetitions: int, warmup: int) -> list[dict[str, Any]]:
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact_dir / "workload_manifest.json").read_text(encoding="utf-8"))
    plan_snapshot = artifact_dir / "plan_snapshot.json"
    if not plan_snapshot.exists():
        return []
    plan_id = manifest["plan_ids"][0]
    rows = []
    for round_size in ROUND_SIZES:
        artifact_out = ROOT / "artifacts" / f"audit_replay_{artifact_dir.name}_rs{round_size}"
        output_out = ROOT / "outputs" / f"audit_replay_{artifact_dir.name}_rs{round_size}"
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "experiment/poc2/nccl_4gpu_benchmark.py",
            "--model",
            str(manifest["model_key"]),
            "--artifact-dir",
            str(artifact_out),
            "--output-dir",
            str(output_out),
            "--placement-policy",
            str(manifest["placement_policy"]),
            "--scenario",
            str(manifest["scenario"]),
            "--regime",
            str(manifest["regime"]),
            "--warmup-repetitions",
            str(warmup),
            "--repetitions",
            str(repetitions),
            "--microbatch-size",
            "1",
            "--microbatch-count",
            "1",
            "--max-length",
            str(config["max_length"]),
            "--strategy-order-mode",
            "counterbalanced",
            "--release-round-size",
            str(round_size),
            "--audit-plan-id",
            str(plan_id),
            "--plan-snapshot-path",
            str(plan_snapshot),
            "--strategy-subset",
            *available_audit_strategies(),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        replay_summary = json.loads((artifact_out / "summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "artifact": artifact_dir.name,
                "replay_artifact": str(artifact_out),
                "round_size": "all-at-once" if round_size >= 999999 else round_size,
                "summary": replay_summary,
            }
        )
    return rows


def diagnose_case(artifact_summaries: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not replay_rows:
        return {
            "case": "Case A",
            "label": "policy has insufficient leverage under the current round formation and placement",
        }
    leverage = []
    for item in replay_rows:
        paired = item["summary"]["paired_deltas"]
        full_random = paired.get("full_vs_random_order", {})
        full_strong = paired.get("full_vs_strong_state", {})
        leverage.append(
            {
                "round_size": item["round_size"],
                "full_vs_random_boot_ci": full_random.get("bootstrap_ci95", [0.0, 0.0]),
                "full_vs_strong_boot_ci": full_strong.get("bootstrap_ci95", [0.0, 0.0]),
            }
        )
    if all(pair["full_vs_random_boot_ci"][0] <= 0.0 <= pair["full_vs_random_boot_ci"][1] for pair in leverage):
        return {
            "case": "Case A",
            "label": "policy has insufficient leverage under the current round formation and placement",
            "replay": leverage,
        }
    return {
        "case": "Case B",
        "label": "synchronous collective and barrier structure masks scheduling gains",
        "replay": leverage,
    }


def rank_byte_vector(round_record: dict[str, Any]) -> list[float]:
    projection = round_record.get("rank_projection", [])
    if projection:
        return [float(item["projected_recv_bytes"]) for item in projection]
    splits = list(round_record.get("planned_send_splits_rows", []))
    return [float(item) for item in splits]


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return (len(left & right) / len(union)) if union else 1.0


def vector_overlap(left: list[float], right: list[float]) -> float:
    total = sum(max(a, b) for a, b in zip(left, right))
    matched = sum(min(a, b) for a, b in zip(left, right))
    return (matched / total) if total else 1.0


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denom = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return (numerator / denom) if denom else 0.0


def corr(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    return (numerator / (denom_left * denom_right)) if denom_left and denom_right else 0.0


def gini_coefficient(values: list[int]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0 or sum(ordered) == 0:
        return 0.0
    cumulative = 0
    for index, value in enumerate(ordered, start=1):
        cumulative += index * value
    return (2 * cumulative) / (n * sum(ordered)) - (n + 1) / n


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
