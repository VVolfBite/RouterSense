#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.distributed_runtime import ProtocolConfig, create_workload_plans
from routesense_poc2.stress import (
    EPISODE_PHASE_COUNTS,
    TARGET_SKEW_RANGES,
    aggregate_scenario_metrics,
    build_burst_episode_schedule,
    choose_skewed_placement,
    clone_episode_plans,
    episode_schedule_hash,
    hash_json,
    plan_rank_metrics,
    placement_hash,
    scenario_admissibility,
    save_plan_snapshot_bundle,
    scenario_manifest,
    scenario_trace_bundle_hash,
    scenario_token_origin_map_hash,
    scenario_route_item_set_hash,
    scenario_payload_checksum,
    scenario_metadata_checksum,
    scenario_expert_weight_checksum,
    split_calibration_and_evaluation,
)


MODEL_PATHS = {
    "olmoe": "/root/autodl-tmp/models/OLMoE-1B-7B-0924",
    "qwen_moe": "/root/autodl-tmp/models/Qwen1.5-MoE-A2.7B",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build trace-preserving placement-skew and burst stress scenarios.")
    parser.add_argument("--model", choices=sorted(MODEL_PATHS), default="olmoe")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--artifact-dir", type=str, default=str(ROOT / "artifacts" / "poc2_stress_suite"))
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--microbatch-count", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--target-rank", type=int, default=2)
    parser.add_argument("--origin-sharding", choices=["contiguous", "round-robin"], default="round-robin")
    parser.add_argument("--arrival-jitter", choices=["none", "mild", "moderate"], default="none")
    parser.add_argument("--release-round-size", type=int, default=4)
    args = parser.parse_args(argv)

    protocol = ProtocolConfig(
        model_key=args.model,
        model_path=args.model_path or MODEL_PATHS[args.model],
        output_dir=str(ROOT / "outputs" / "poc2_stress_suite_tmp"),
        artifact_dir=args.artifact_dir,
        microbatch_size=args.microbatch_size,
        microbatch_count=args.microbatch_count,
        max_length=args.max_length,
        seed=args.seed,
        world_size=args.world_size,
        origin_sharding=args.origin_sharding,
        placement_policy="balanced",
    )
    plans, manifest = create_workload_plans_local(protocol)
    calibration, evaluation = split_calibration_and_evaluation(plans)
    if not calibration or not evaluation:
        raise RuntimeError("stress suite requires at least two plans to form calibration/evaluation split")
    target_rank = int(args.target_rank)
    base_placement = list(calibration[0].placement_mapping)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scenario_rows = []
    scenario_snapshots = {}
    for strength in ["balanced", "mild-skew", "moderate-skew", "high-skew"]:
        placement, observed = choose_skewed_placement(
            base_placement,
            calibration,
            evaluation,
            target_rank=target_rank,
            strength=strength,
        )
        scenario_id = f"{args.model}_{strength}"
        remapped_eval = clone_episode_plans(evaluation, placement, scenario_id=scenario_id)
        remapped_calibration = clone_episode_plans(calibration, placement, scenario_id=f"{scenario_id}_cal")
        phase_source_plans, episode_rows = build_burst_episode_schedule(remapped_eval, target_rank=target_rank)
        phase_lookup = {row["source_plan_id"]: row["phase"] for row in episode_rows}
        episode_plans = clone_episode_plans(phase_source_plans, placement, scenario_id=f"{scenario_id}_episode", phase_lookup=phase_lookup)
        for plan in episode_plans:
            plan.routing_summary["arrival_jitter_mode"] = args.arrival_jitter
        scenario_info = scenario_manifest(
            scenario_id=scenario_id,
            plans=episode_plans,
            placement=placement,
            episode_rows=episode_rows,
            target_rank=target_rank,
        )
        scenario_info["scenario_admissibility"] = scenario_admissibility(
            episode_plans,
            round_size=args.release_round_size,
            seed=args.seed,
        )
        scenario_info["strength_target_range"] = TARGET_SKEW_RANGES[strength]
        scenario_info["placement_policy"] = strength
        scenario_info["observed_skew_metrics"] = observed
        scenario_info["calibration_plan_ids"] = [plan.plan_id for plan in calibration]
        scenario_info["evaluation_plan_ids"] = [plan.plan_id for plan in evaluation]
        scenario_info["arrival_jitter_mode"] = args.arrival_jitter
        scenario_rows.append(scenario_info)
        snapshot_path = artifact_dir / f"{scenario_id}_plan_snapshot.json"
        save_plan_snapshot_bundle(
            snapshot_path,
            episode_plans,
            {
                **scenario_info,
                "episode_phase_counts": EPISODE_PHASE_COUNTS,
                "episode_rows": episode_rows,
            },
        )
        scenario_snapshots[scenario_id] = str(snapshot_path)
        with (artifact_dir / f"{scenario_id}_episode_schedule.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "scenario_id": scenario_id,
                    "episode_schedule_hash": episode_schedule_hash(episode_rows),
                    "episode_phase_counts": EPISODE_PHASE_COUNTS,
                    "rows": episode_rows,
                },
                handle,
                indent=2,
            )

    payload = {
        "model": args.model,
        "world_size": args.world_size,
        "target_rank": target_rank,
        "origin_sharding": args.origin_sharding,
        "trace_bundle_hash": scenario_trace_bundle_hash(plans),
        "token_origin_map_hash": scenario_token_origin_map_hash(plans),
        "route_item_set_hash": scenario_route_item_set_hash(plans),
        "payload_checksum": scenario_payload_checksum(plans),
        "metadata_checksum": scenario_metadata_checksum(plans),
        "expert_weight_checksum": scenario_expert_weight_checksum(plans),
        "calibration_split_plan_ids": [plan.plan_id for plan in calibration],
        "evaluation_split_plan_ids": [plan.plan_id for plan in evaluation],
        "scenario_snapshots": scenario_snapshots,
        "scenarios": scenario_rows,
    }
    (artifact_dir / "scenario_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(artifact_dir / "scenario_manifest.json")
    return 0


def create_workload_plans_local(protocol: ProtocolConfig):
    import torch.distributed as dist

    initialized = dist.is_initialized()
    if not initialized:
        import os

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind(("127.0.0.1", 0))
            port = handle.getsockname()[1]
        os.environ.setdefault("MASTER_PORT", str(port))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")
    try:
        return create_workload_plans(protocol)
    finally:
        if not initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
