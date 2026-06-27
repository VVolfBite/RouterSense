from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from itertools import combinations

from .distributed_runtime import (
    PlacementEntry,
    WorkloadBucket,
    WorkloadPlan,
    dependency_score_for_bucket,
    evaluate_release_rounds,
    random_order,
    release_rounds_for_order,
    strong_state_score_for_bucket,
    _greedy_minimax_round_pack,
    _round_pressure_objective,
    _stable_bucket_order_key,
)


TARGET_SKEW_RANGES = {
    "balanced": (0.25, 0.30),
    "mild-skew": (0.35, 0.40),
    "moderate-skew": (0.45, 0.50),
    "high-skew": (0.55, 0.60),
}

EPISODE_PHASE_COUNTS = {
    "benign": 8,
    "ramp": 8,
    "burst": 16,
    "recovery": 8,
}


def hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_calibration_and_evaluation(
    plans: list[WorkloadPlan],
    *,
    calibration_fraction: float = 0.5,
) -> tuple[list[WorkloadPlan], list[WorkloadPlan]]:
    if not plans:
        return [], []
    cutoff = max(1, min(len(plans) - 1, int(math.ceil(len(plans) * calibration_fraction))))
    calibration = list(plans[:cutoff])
    evaluation = list(plans[cutoff:])
    if not evaluation:
        evaluation = [plans[-1]]
    return calibration, evaluation


def scenario_trace_bundle_hash(plans: list[WorkloadPlan]) -> str:
    rows = []
    for plan in plans:
        rows.append(
            {
                "plan_id": plan.plan_id,
                "microbatch_id": plan.microbatch_id,
                "layer_id": plan.layer_id,
                "routing_summary": plan.routing_summary,
            }
        )
    return hash_json(rows)


def scenario_token_origin_map_hash(plans: list[WorkloadPlan]) -> str:
    rows = []
    for plan in plans:
        rows.append(
            {
                "plan_id": plan.plan_id,
                "token_origin_rank_detail": plan.routing_summary.get("token_origin_rank_detail", {}),
            }
        )
    return hash_json(rows)


def scenario_route_item_set_hash(plans: list[WorkloadPlan]) -> str:
    rows = []
    for plan in plans:
        for bucket in plan.bucket_workloads:
            for route_item_index, token_id, route_rank in zip(bucket.route_item_indices, bucket.token_ids, bucket.route_ranks):
                rows.append(
                    {
                        "plan_id": plan.plan_id,
                        "route_item_index": int(route_item_index),
                        "token_id": int(token_id),
                        "origin_rank": int(bucket.origin_rank),
                        "destination_rank": int(bucket.destination_rank),
                        "expert_id": int(bucket.expert_id),
                        "route_rank": int(route_rank),
                    }
                )
    rows.sort(
        key=lambda item: (
            item["plan_id"],
            item["route_item_index"],
            item["token_id"],
            item["origin_rank"],
            item["destination_rank"],
            item["expert_id"],
            item["route_rank"],
        )
    )
    return hash_json(rows)


def scenario_payload_checksum(plans: list[WorkloadPlan]) -> str:
    rows = []
    for plan in plans:
        for bucket in plan.bucket_workloads:
            rows.append(
                {
                    "plan_id": plan.plan_id,
                    "bucket_id": bucket.bucket_id,
                    "payload_rows": bucket.payload_rows,
                    "hidden_dim": bucket.hidden_dim,
                    "payload_bytes": bucket.payload_bytes,
                }
            )
    return hash_json(rows)


def scenario_metadata_checksum(plans: list[WorkloadPlan]) -> str:
    rows = []
    for plan in plans:
        for bucket in plan.bucket_workloads:
            rows.append(
                {
                    "plan_id": plan.plan_id,
                    "bucket_id": bucket.bucket_id,
                    "route_item_indices": list(bucket.route_item_indices),
                    "token_ids": list(bucket.token_ids),
                    "route_ranks": list(bucket.route_ranks),
                }
            )
    return hash_json(rows)


def scenario_expert_weight_checksum(plans: list[WorkloadPlan]) -> str:
    if not plans:
        return hash_json([])
    return hash_json(
        {
            "hidden_dim": plans[0].hidden_dim,
            "intermediate_dim": plans[0].intermediate_dim,
            "experts": sorted({bucket.expert_id for plan in plans for bucket in plan.bucket_workloads}),
        }
    )


def placement_hash(placement: list[PlacementEntry]) -> str:
    return hash_json([asdict(item) for item in placement])


def episode_schedule_hash(rows: list[dict[str, Any]]) -> str:
    return hash_json(rows)


def plan_rank_metrics(plan: WorkloadPlan, target_rank: int) -> dict[str, float]:
    inbound_bytes = [0 for _ in range(max(1, len({item.destination_rank for item in plan.placement_mapping}) or 1))]
    outbound_bytes = [0 for _ in inbound_bytes]
    expert_rows: dict[int, int] = {}
    convergence_sources: dict[int, set[int]] = {}
    for bucket in plan.bucket_workloads:
        if bucket.destination_rank >= len(inbound_bytes):
            inbound_bytes.extend([0] * (bucket.destination_rank - len(inbound_bytes) + 1))
            outbound_bytes.extend([0] * (bucket.destination_rank - len(outbound_bytes) + 1))
        inbound_bytes[bucket.destination_rank] += int(bucket.payload_bytes)
        outbound_bytes[bucket.origin_rank] += int(bucket.payload_bytes)
        expert_rows[bucket.expert_id] = expert_rows.get(bucket.expert_id, 0) + int(bucket.payload_rows)
        convergence_sources.setdefault(bucket.destination_rank, set()).add(bucket.origin_rank)
    total_inbound = sum(inbound_bytes)
    total_outbound = sum(outbound_bytes)
    total_rows = sum(expert_rows.values())
    per_expert = sorted(expert_rows.values(), reverse=True)
    target_inbound = inbound_bytes[target_rank] if target_rank < len(inbound_bytes) else 0
    target_rows = sum(bucket.payload_rows for bucket in plan.bucket_workloads if bucket.destination_rank == target_rank)
    target_expert_rows = [
        rows
        for expert_id, rows in expert_rows.items()
        if any(bucket.expert_id == expert_id and bucket.destination_rank == target_rank for bucket in plan.bucket_workloads)
    ]
    convergence_score = float(sum(len(values) for values in convergence_sources.values())) / max(1, len(convergence_sources))
    peak_rank = max(inbound_bytes) if inbound_bytes else 0
    mean_rank = (statistics.fmean(inbound_bytes) if inbound_bytes else 0.0)
    peak_expert = max(per_expert) if per_expert else 0
    mean_expert = (statistics.fmean(per_expert) if per_expert else 0.0)
    return {
        "target_rank_inbound_bytes": float(target_inbound),
        "target_rank_inbound_share": float(target_inbound / total_inbound) if total_inbound else 0.0,
        "target_rank_compute_rows": float(target_rows),
        "target_rank_compute_row_share": float(target_rows / total_rows) if total_rows else 0.0,
        "target_rank_peak_expert_share": float(max(target_expert_rows) / total_rows) if total_rows and target_expert_rows else 0.0,
        "peak_mean_rank_load_ratio": float(peak_rank / mean_rank) if mean_rank else 0.0,
        "peak_mean_expert_load_ratio": float(peak_expert / mean_expert) if mean_expert else 0.0,
        "multi_source_convergence_score": convergence_score,
        "per_rank_inbound_byte_share": [
            (float(value) / total_inbound) if total_inbound else 0.0
            for value in inbound_bytes
        ],
        "per_rank_outbound_byte_share": [
            (float(value) / total_outbound) if total_outbound else 0.0
            for value in outbound_bytes
        ],
        "per_expert_row_share": {
            str(expert_id): (float(rows) / total_rows) if total_rows else 0.0
            for expert_id, rows in sorted(expert_rows.items())
        },
    }


def aggregate_scenario_metrics(plans: list[WorkloadPlan], target_rank: int) -> dict[str, float]:
    metrics = [plan_rank_metrics(plan, target_rank) for plan in plans]
    if not metrics:
        return {}
    keys = [
        "target_rank_inbound_bytes",
        "target_rank_inbound_share",
        "target_rank_compute_rows",
        "target_rank_compute_row_share",
        "target_rank_peak_expert_share",
        "peak_mean_rank_load_ratio",
        "peak_mean_expert_load_ratio",
        "multi_source_convergence_score",
    ]
    summary = {key: float(statistics.fmean(float(item[key]) for item in metrics)) for key in keys}
    summary["target_rank_bottleneck_frequency"] = sum(
        1 for item in metrics if item["target_rank_inbound_share"] == max(item["per_rank_inbound_byte_share"])
    ) / len(metrics)
    return summary


def expert_calibration_stats(plans: list[WorkloadPlan]) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for plan in plans:
        for bucket in plan.bucket_workloads:
            entry = rows.setdefault(
                bucket.expert_id,
                {
                    "expected_rows": 0.0,
                    "expected_remote_bytes": 0.0,
                    "popularity": 0.0,
                    "dependency_affinity": 0.0,
                },
            )
            entry["expected_rows"] += float(bucket.payload_rows)
            entry["expected_remote_bytes"] += float(bucket.payload_bytes if bucket.origin_rank != bucket.destination_rank else 0)
            entry["popularity"] += float(bucket.token_count)
            entry["dependency_affinity"] += float(
                dependency_score_for_bucket(bucket) + bucket.coactive_peer_degree + bucket.bridge_score
            )
    return rows


def _clone_bucket(bucket: WorkloadBucket) -> WorkloadBucket:
    return WorkloadBucket(**asdict(bucket))


def _clone_plan(plan: WorkloadPlan) -> WorkloadPlan:
    return WorkloadPlan.from_dict(plan.to_dict())


def remap_plan_to_placement(
    plan: WorkloadPlan,
    placement: list[PlacementEntry],
    *,
    plan_id: str | None = None,
    microbatch_id: int | None = None,
    route_index_offset: int = 0,
    phase: str | None = None,
) -> WorkloadPlan:
    placement_map = {item.expert_id: item.destination_rank for item in placement}
    updated = _clone_plan(plan)
    updated.plan_id = plan_id or plan.plan_id
    if microbatch_id is not None:
        updated.microbatch_id = microbatch_id
    updated.placement_policy = "stress-custom"
    updated.placement_mapping = [PlacementEntry(expert_id=item.expert_id, destination_rank=item.destination_rank) for item in placement]
    next_buckets: list[WorkloadBucket] = []
    for bucket_index, bucket in enumerate(updated.bucket_workloads):
        cloned = _clone_bucket(bucket)
        cloned.destination_rank = int(placement_map[cloned.expert_id])
        cloned.bucket_id = f"{updated.plan_id}_b{bucket_index}"
        cloned.microbatch_id = updated.microbatch_id
        cloned.route_item_indices = [int(value) + int(route_index_offset) for value in cloned.route_item_indices]
        next_buckets.append(cloned)
    updated.bucket_workloads = next_buckets
    updated.routing_summary = dict(updated.routing_summary)
    if phase is not None:
        updated.routing_summary["episode_phase"] = phase
    return updated


def choose_skewed_placement(
    base_placement: list[PlacementEntry],
    calibration_plans: list[WorkloadPlan],
    evaluation_plans: list[WorkloadPlan],
    *,
    target_rank: int,
    strength: str,
) -> tuple[list[PlacementEntry], dict[str, float]]:
    if strength == "balanced":
        return list(base_placement), aggregate_scenario_metrics(evaluation_plans, target_rank)
    expert_stats = expert_calibration_stats(calibration_plans)
    by_rank: dict[int, list[int]] = {}
    for item in base_placement:
        by_rank.setdefault(item.destination_rank, []).append(item.expert_id)
    ranked_target_candidates = sorted(
        (expert_id for expert_id, item in expert_stats.items() if expert_id not in set(by_rank.get(target_rank, []))),
        key=lambda expert_id: (
            -expert_stats[expert_id]["expected_rows"],
            -expert_stats[expert_id]["popularity"],
            -expert_stats[expert_id]["dependency_affinity"],
        ),
    )
    ranked_target_evictions = sorted(
        list(by_rank.get(target_rank, [])),
        key=lambda expert_id: (
            expert_stats.get(expert_id, {}).get("expected_rows", 0.0),
            expert_stats.get(expert_id, {}).get("popularity", 0.0),
            expert_stats.get(expert_id, {}).get("dependency_affinity", 0.0),
        ),
    )
    desired_low, desired_high = TARGET_SKEW_RANGES[strength]
    best_placement = list(base_placement)
    best_metrics = aggregate_scenario_metrics(evaluation_plans, target_rank)
    best_distance = abs(best_metrics.get("target_rank_inbound_share", 0.0) - desired_low)
    max_swaps = min(len(ranked_target_candidates), len(ranked_target_evictions))
    for swap_count in range(1, max_swaps + 1):
        current = [PlacementEntry(expert_id=item.expert_id, destination_rank=item.destination_rank) for item in base_placement]
        current_map = {item.expert_id: item.destination_rank for item in current}
        for index in range(swap_count):
            incoming = ranked_target_candidates[index]
            outgoing = ranked_target_evictions[index]
            donor_rank = current_map[incoming]
            current_map[incoming] = target_rank
            current_map[outgoing] = donor_rank
        candidate = [PlacementEntry(expert_id=expert_id, destination_rank=destination_rank) for expert_id, destination_rank in sorted(current_map.items())]
        remapped_eval = [
            remap_plan_to_placement(plan, candidate, plan_id=plan.plan_id, microbatch_id=plan.microbatch_id)
            for plan in evaluation_plans
        ]
        metrics = aggregate_scenario_metrics(remapped_eval, target_rank)
        share = metrics.get("target_rank_inbound_share", 0.0)
        distance = 0.0 if desired_low <= share <= desired_high else min(abs(share - desired_low), abs(share - desired_high))
        if distance < best_distance or (distance == best_distance and share > best_metrics.get("target_rank_inbound_share", 0.0)):
            best_distance = distance
            best_placement = candidate
            best_metrics = metrics
    return best_placement, best_metrics


def clone_episode_plans(
    source_plans: list[WorkloadPlan],
    placement: list[PlacementEntry],
    *,
    scenario_id: str,
    phase_lookup: dict[str, str] | None = None,
) -> list[WorkloadPlan]:
    cloned: list[WorkloadPlan] = []
    route_offset = 0
    for episode_index, plan in enumerate(source_plans):
        phase = phase_lookup.get(plan.plan_id) if phase_lookup else None
        current = remap_plan_to_placement(
            plan,
            placement,
            plan_id=f"{scenario_id}_ep{episode_index:03d}_{plan.plan_id}",
            microbatch_id=episode_index,
            route_index_offset=route_offset,
            phase=phase,
        )
        route_offset += sum(len(bucket.route_item_indices) for bucket in current.bucket_workloads)
        cloned.append(current)
    return cloned


def classify_episode_windows(plans: list[WorkloadPlan], target_rank: int) -> dict[str, list[WorkloadPlan]]:
    scored = []
    for plan in plans:
        metrics = plan_rank_metrics(plan, target_rank)
        severity = (
            metrics["target_rank_inbound_bytes"]
            + 4.0 * metrics["target_rank_compute_rows"]
            + 512.0 * metrics["multi_source_convergence_score"]
        )
        scored.append((severity, plan))
    ordered = [item[1] for item in sorted(scored, key=lambda pair: pair[0])]
    total = len(ordered)
    benign = ordered[: max(1, total // 4)]
    ramp = ordered[max(1, total // 4) : max(2, total // 2)]
    burst = ordered[max(2, total // 2) :]
    recovery = list(reversed(ramp or benign))
    return {
        "benign": benign or ordered[:1],
        "ramp": ramp or ordered[:1],
        "burst": burst or ordered[-1:],
        "recovery": recovery or ordered[:1],
    }


def build_burst_episode_schedule(
    evaluation_plans: list[WorkloadPlan],
    *,
    target_rank: int,
) -> tuple[list[WorkloadPlan], list[dict[str, Any]]]:
    phases = classify_episode_windows(evaluation_plans, target_rank)
    selected_plans: list[WorkloadPlan] = []
    manifest_rows: list[dict[str, Any]] = []
    for phase, count in EPISODE_PHASE_COUNTS.items():
        pool = phases[phase]
        for index in range(count):
            plan = pool[index % len(pool)]
            metrics = plan_rank_metrics(plan, target_rank)
            selected_plans.append(plan)
            manifest_rows.append(
                {
                    "episode_index": len(manifest_rows),
                    "phase": phase,
                    "source_plan_id": plan.plan_id,
                    "target_rank_inbound_bytes": metrics["target_rank_inbound_bytes"],
                    "target_rank_compute_rows": metrics["target_rank_compute_rows"],
                    "multi_source_convergence_score": metrics["multi_source_convergence_score"],
                }
            )
    return selected_plans, manifest_rows


def scenario_manifest(
    *,
    scenario_id: str,
    plans: list[WorkloadPlan],
    placement: list[PlacementEntry],
    episode_rows: list[dict[str, Any]],
    target_rank: int,
) -> dict[str, Any]:
    metrics = aggregate_scenario_metrics(plans, target_rank)
    return {
        "scenario_id": scenario_id,
        "trace_bundle_hash": scenario_trace_bundle_hash(plans),
        "placement_hash": placement_hash(placement),
        "episode_schedule_hash": episode_schedule_hash(episode_rows),
        "token_origin_map_hash": scenario_token_origin_map_hash(plans),
        "route_item_set_hash": scenario_route_item_set_hash(plans),
        "payload_checksum": scenario_payload_checksum(plans),
        "metadata_checksum": scenario_metadata_checksum(plans),
        "expert_weight_checksum": scenario_expert_weight_checksum(plans),
        "target_rank": target_rank,
        "metrics": metrics,
        "episode_length": len(plans),
    }


def _feature_row(plan: WorkloadPlan, target_rank: int) -> dict[str, float]:
    metrics = plan_rank_metrics(plan, target_rank)
    target_buckets = [bucket for bucket in plan.bucket_workloads if bucket.destination_rank == target_rank]
    dep_scores = [dependency_score_for_bucket(bucket) for bucket in target_buckets]
    top_dep = sorted(dep_scores, reverse=True)[:3]
    return {
        "inbound_bytes": metrics["target_rank_inbound_bytes"],
        "compute_rows": metrics["target_rank_compute_rows"],
        "peak_expert_share": metrics["target_rank_peak_expert_share"],
        "convergence": metrics["multi_source_convergence_score"],
        "dep_mean": float(statistics.fmean(dep_scores)) if dep_scores else 0.0,
        "dep_top3_mean": float(statistics.fmean(top_dep)) if top_dep else 0.0,
        "dep_max": max(dep_scores) if dep_scores else 0.0,
    }


def dependency_predictiveness_gate(
    calibration_plans: list[WorkloadPlan],
    evaluation_plans: list[WorkloadPlan],
    *,
    target_rank: int,
) -> dict[str, Any]:
    def _pairs(plans: list[WorkloadPlan]) -> list[tuple[dict[str, float], dict[str, float]]]:
        return [(_feature_row(plans[index], target_rank), _feature_row(plans[index + 1], target_rank)) for index in range(len(plans) - 1)]

    calibration_pairs = _pairs(calibration_plans)
    evaluation_pairs = _pairs(evaluation_plans)
    if not calibration_pairs or not evaluation_pairs:
        return {
            "gate": "fail",
            "reason": "insufficient temporal pairs",
            "state_only": {},
            "state_plus_dependency": {},
            "residual_improvement": 0.0,
            "spearman_dependency_vs_t1_hotspot": 0.0,
            "topk_risky_flow_precision": 0.0,
            "topk_risky_flow_recall": 0.0,
        }
    state_features = ["inbound_bytes", "compute_rows", "peak_expert_share", "convergence"]
    dependency_features = ["dep_mean", "dep_top3_mean", "dep_max"]
    target_name = "inbound_bytes"
    state_result = _fit_and_eval(calibration_pairs, evaluation_pairs, state_features, target_name)
    state_dep_result = _fit_and_eval(calibration_pairs, evaluation_pairs, state_features + dependency_features, target_name)
    residual_improvement = float(state_result["mae"] - state_dep_result["mae"])
    dep_values = [left["dep_top3_mean"] for left, _ in evaluation_pairs]
    hotspot_values = [right["inbound_bytes"] for _, right in evaluation_pairs]
    spearman = _spearman(dep_values, hotspot_values)
    topk = max(1, len(evaluation_pairs) // 4)
    predicted = sorted(
        range(len(evaluation_pairs)),
        key=lambda index: dep_values[index],
        reverse=True,
    )[:topk]
    actual = sorted(
        range(len(evaluation_pairs)),
        key=lambda index: hotspot_values[index],
        reverse=True,
    )[:topk]
    overlap = len(set(predicted) & set(actual))
    precision = overlap / len(predicted)
    recall = overlap / len(actual)
    gate = "pass" if residual_improvement > 0.01 * max(1.0, state_result["mae"]) and state_dep_result["r2"] >= state_result["r2"] else "fail"
    return {
        "gate": gate,
        "state_only": state_result,
        "state_plus_dependency": state_dep_result,
        "residual_improvement": residual_improvement,
        "spearman_dependency_vs_t1_hotspot": spearman,
        "topk_risky_flow_precision": precision,
        "topk_risky_flow_recall": recall,
    }


def _fit_and_eval(
    calibration_pairs: list[tuple[dict[str, float], dict[str, float]]],
    evaluation_pairs: list[tuple[dict[str, float], dict[str, float]]],
    feature_names: list[str],
    target_name: str,
) -> dict[str, float]:
    x_train = [[1.0] + [left[name] for name in feature_names] for left, _ in calibration_pairs]
    y_train = [right[target_name] for _, right in calibration_pairs]
    x_eval = [[1.0] + [left[name] for name in feature_names] for left, _ in evaluation_pairs]
    y_eval = [right[target_name] for _, right in evaluation_pairs]
    if np is not None:
        coef, *_ = np.linalg.lstsq(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float), rcond=None)
        predictions = (np.asarray(x_eval, dtype=float) @ coef).tolist()
    else:  # pragma: no cover
        baseline = statistics.fmean(y_train)
        predictions = [baseline for _ in x_eval]
    mae = float(statistics.fmean(abs(pred - actual) for pred, actual in zip(predictions, y_eval)))
    mean_target = statistics.fmean(y_eval)
    ss_res = sum((actual - pred) ** 2 for pred, actual in zip(predictions, y_eval))
    ss_tot = sum((actual - mean_target) ** 2 for actual in y_eval)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "r2": float(r2)}


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_order = {index: rank for rank, index in enumerate(sorted(range(len(left)), key=lambda idx: left[idx]))}
    right_order = {index: rank for rank, index in enumerate(sorted(range(len(right)), key=lambda idx: right[idx]))}
    left_ranks = [left_order[index] for index in range(len(left))]
    right_ranks = [right_order[index] for index in range(len(right))]
    mean_left = statistics.fmean(left_ranks)
    mean_right = statistics.fmean(right_ranks)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left_ranks, right_ranks))
    denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in left_ranks))
    denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in right_ranks))
    if denom_left == 0.0 or denom_right == 0.0:
        return 0.0
    return numerator / (denom_left * denom_right)


def lookahead_greedy_round_packer(plan: WorkloadPlan, round_size: int) -> dict[str, Any]:
    buckets = sorted(
        plan.bucket_workloads,
        key=lambda bucket: (
            -float(bucket.payload_bytes + bucket.payload_rows * 32),
            bucket.destination_rank,
            bucket.origin_rank,
            bucket.expert_id,
            bucket.bucket_id,
        ),
    )
    round_count = max(1, math.ceil(len(buckets) / max(1, round_size)))
    rounds: list[list[WorkloadBucket]] = [[] for _ in range(round_count)]
    round_bytes = [0.0 for _ in range(round_count)]
    round_rows = [0.0 for _ in range(round_count)]
    round_rank_penalty: list[dict[int, float]] = [dict() for _ in range(round_count)]
    round_expert_penalty: list[dict[int, float]] = [dict() for _ in range(round_count)]
    for bucket in buckets:
        candidates = []
        for round_index in range(round_count):
            if len(rounds[round_index]) >= max(1, round_size):
                continue
            bytes_penalty = round_bytes[round_index] + float(bucket.payload_bytes)
            rows_penalty = round_rows[round_index] + float(bucket.payload_rows)
            rank_penalty = round_rank_penalty[round_index].get(bucket.destination_rank, 0.0) + float(bucket.payload_bytes)
            expert_penalty = round_expert_penalty[round_index].get(bucket.expert_id, 0.0) + float(bucket.payload_rows)
            candidates.append((max(bytes_penalty, rows_penalty * 64.0, rank_penalty, expert_penalty * 64.0), round_index))
        _, selected_round = min(candidates)
        rounds[selected_round].append(bucket)
        round_bytes[selected_round] += float(bucket.payload_bytes)
        round_rows[selected_round] += float(bucket.payload_rows)
        round_rank_penalty[selected_round][bucket.destination_rank] = round_rank_penalty[selected_round].get(bucket.destination_rank, 0.0) + float(bucket.payload_bytes)
        round_expert_penalty[selected_round][bucket.expert_id] = round_expert_penalty[selected_round].get(bucket.expert_id, 0.0) + float(bucket.payload_rows)
    release_rounds = [[bucket.bucket_id for bucket in round_buckets] for round_buckets in rounds if round_buckets]
    diagnostics = evaluate_release_rounds(plan, release_rounds, None, strategy="lookahead-greedy")
    return {
        "release_rounds": release_rounds,
        "evaluation": diagnostics,
        "pressure": diagnostics.per_round,
        "peak_bottleneck_pressure": diagnostics.objective,
        "objective": diagnostics.objective,
        "optimality_proven": False,
        "diagnostic_kind": "lookahead-greedy-upper-heuristic",
    }


def exact_round_packer(plan: WorkloadPlan, round_size: int) -> dict[str, Any]:
    buckets = sorted(plan.bucket_workloads, key=_stable_bucket_order_key)
    if len(buckets) > 12:
        grouped = [
            buckets[index : index + round_size]
            for index in range(0, len(buckets), round_size)
        ]
        grouped_rounds = [[bucket.bucket_id for bucket in group] for group in grouped]
        diagnostics = evaluate_release_rounds(plan, grouped_rounds, None, strategy="best-known-grouped")
        return {
            "release_rounds": grouped_rounds,
            "evaluation": diagnostics,
            "pressure": diagnostics.per_round,
            "peak_bottleneck_pressure": diagnostics.objective,
            "objective": diagnostics.objective,
            "diagnostic_kind": "best-known-grouped",
            "optimality_gap_known": False,
            "optimality_proven": False,
        }

    best_rounds: list[list[str]] | None = None
    best_pressure = float("inf")

    def search(remaining: list[WorkloadBucket], current: list[list[str]]) -> None:
        nonlocal best_rounds, best_pressure
        if not remaining:
            diagnostics = evaluate_release_rounds(plan, current, None, strategy="exact")
            pressure = float(diagnostics.objective)
            if pressure < best_pressure:
                best_pressure = pressure
                best_rounds = [list(round_ids) for round_ids in current]
            return
        if best_rounds is not None and len(current) > len(best_rounds):
            return
        first = remaining[0]
        candidate_pool = remaining[1:]
        max_extra = min(round_size - 1, len(candidate_pool))
        for extra_count in range(max_extra + 1):
            for subset in combinations(candidate_pool, extra_count):
                chosen = [first, *subset]
                round_ids = [bucket.bucket_id for bucket in sorted(chosen, key=_stable_bucket_order_key)]
                diagnostics = evaluate_release_rounds(plan, current + [round_ids], None, strategy="exact")
                pressure = float(diagnostics.objective)
                if pressure > best_pressure:
                    continue
                used = {bucket.bucket_id for bucket in chosen}
                next_remaining = [bucket for bucket in remaining if bucket.bucket_id not in used]
                search(next_remaining, current + [round_ids])

    search(buckets, [])
    release_rounds = best_rounds or [[bucket.bucket_id for bucket in buckets]]
    diagnostics = evaluate_release_rounds(plan, release_rounds, None, strategy="exact")
    return {
        "release_rounds": release_rounds,
        "evaluation": diagnostics,
        "pressure": diagnostics.per_round,
        "peak_bottleneck_pressure": diagnostics.objective,
        "objective": diagnostics.objective,
        "diagnostic_kind": "exact" if len(buckets) <= 12 else "best-known-grouped",
        "optimality_gap_known": len(buckets) <= 12,
        "optimality_proven": len(buckets) <= 12,
    }


def _non_schedulable_mega_bucket(plan: WorkloadPlan, round_size: int) -> bool:
    if not plan.bucket_workloads:
        return False
    total_bytes = sum(bucket.payload_bytes for bucket in plan.bucket_workloads)
    heaviest = max(plan.bucket_workloads, key=lambda bucket: bucket.payload_bytes)
    has_alternative = sum(1 for bucket in plan.bucket_workloads if bucket.destination_rank != heaviest.destination_rank or bucket.expert_id != heaviest.expert_id) >= 2
    return bool(heaviest.payload_bytes >= 0.5 * max(1, total_bytes) and (not has_alternative or round_size <= 1))


def scenario_admissibility(
    plans: list[WorkloadPlan],
    *,
    round_size: int,
    seed: int,
) -> dict[str, Any]:
    per_plan = []
    schedulable_votes = 0
    for plan in plans:
        fifo_order = [bucket.bucket_id for bucket in sorted(plan.bucket_workloads, key=_stable_bucket_order_key)]
        fifo_rounds = release_rounds_for_order(fifo_order, round_size)
        fifo = evaluate_release_rounds(plan, fifo_rounds, None, strategy="fifo")
        random_rounds = release_rounds_for_order(random_order([bucket.bucket_id for bucket in plan.bucket_workloads], seed + plan.microbatch_id), round_size)
        random_diag = evaluate_release_rounds(plan, random_rounds, None, strategy="random-order")
        strong_sort_order = [
            bucket.bucket_id
            for bucket in sorted(
                plan.bucket_workloads,
                key=lambda bucket: (-strong_state_score_for_bucket(bucket, None),) + _stable_bucket_order_key(bucket),
            )
        ]
        strong_sort = evaluate_release_rounds(plan, release_rounds_for_order(strong_sort_order, round_size), None, strategy="strong-state-sort")
        strong_pack = _greedy_minimax_round_pack(
            plan.bucket_workloads,
            round_size=round_size,
            runtime_state=None,
            policy_seed=seed + plan.microbatch_id,
            arrival_jitter_mode=str(plan.routing_summary.get("arrival_jitter_mode", "none")),
            oracle=False,
        )
        lookahead = lookahead_greedy_round_packer(plan, round_size)
        exact = exact_round_packer(plan, round_size)
        fifo_pressure = float(fifo.objective)
        random_pressure = float(random_diag.objective)
        lookahead_pressure = float(lookahead["peak_bottleneck_pressure"])
        exact_pressure = float(exact["peak_bottleneck_pressure"])
        oracle_gap_vs_fifo = (fifo_pressure - exact_pressure) / max(fifo_pressure, 1e-9)
        oracle_gap_vs_random = (random_pressure - exact_pressure) / max(random_pressure, 1e-9)
        mega_bucket = _non_schedulable_mega_bucket(plan, round_size)
        alternative_flows = len({(bucket.destination_rank, bucket.expert_id) for bucket in plan.bucket_workloads}) >= 2
        schedulable = bool(
            oracle_gap_vs_fifo >= 0.15
            and oracle_gap_vs_random >= 0.10
            and not mega_bucket
            and alternative_flows
        )
        if schedulable:
            schedulable_votes += 1
        per_plan.append(
            {
                "plan_id": plan.plan_id,
                "fifo_pressure": fifo_pressure,
                "random_pressure": random_pressure,
                "strong_state_sort_pressure": float(strong_sort.objective),
                "strong_state_packer_pressure": float(strong_pack["round_pressure_score"]),
                "lookahead_greedy_pressure": lookahead_pressure,
                "exact_or_best_known_pressure": exact_pressure,
                "oracle_gap_vs_fifo": oracle_gap_vs_fifo,
                "oracle_gap_vs_random": oracle_gap_vs_random,
                "mega_bucket_dominated": mega_bucket,
                "alternative_flow_count": len({(bucket.destination_rank, bucket.expert_id) for bucket in plan.bucket_workloads}),
                "schedulable": schedulable,
            }
        )
    return {
        "per_plan": per_plan,
        "schedulable": bool(per_plan) and schedulable_votes >= max(1, len(per_plan) // 2),
        "round_size": round_size,
        "schedulable_fraction": (schedulable_votes / len(per_plan)) if per_plan else 0.0,
    }


def save_plan_snapshot_bundle(path: str | Path, plans: list[WorkloadPlan], manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "plans": [plan.to_dict() for plan in plans],
                "manifest": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
