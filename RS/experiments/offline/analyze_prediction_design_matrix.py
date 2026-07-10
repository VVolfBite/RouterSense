#!/usr/bin/env python3
"""Explain the post-4GPU prediction design split: expert, traffic, and policy-consumption."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline._timeline_prediction_diagnosis_common import read_json, write_json, write_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def render_markdown(payload: dict) -> str:
    lines = [
        "# Prediction Design Matrix After Real 4GPU Trace",
        "",
        "## Expert Prediction Track",
        f"- expert_trace_available: `{payload['expert_prediction_track']['expert_trace_available']}`",
        f"- semantic_mapping_ready: `{payload['expert_prediction_track']['semantic_mapping_ready']}`",
        f"- current_blocker: `{payload['expert_prediction_track']['current_blocker']}`",
        "",
        "## Traffic Prediction Track",
        f"- best_non_oracle_predictor: `{payload['traffic_prediction_track']['best_non_oracle_predictor']}`",
        f"- best_non_oracle_relative_l1: `{payload['traffic_prediction_track']['best_non_oracle_relative_l1']}`",
        f"- policy_gain_not_guaranteed: `{payload['traffic_prediction_track']['policy_gain_not_guaranteed']}`",
        "",
        "## Policy Consumption Track",
        f"- barrier_criticality_observation: {payload['policy_consumption_track']['barrier_criticality_observation']}",
        f"- gated_greedy_observation: {payload['policy_consumption_track']['gated_greedy_observation']}",
        "",
        "## Recommended Sequence",
    ]
    for idx, step in enumerate(payload["recommended_sequence"], start=1):
        lines.append(f"{idx}. {step}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    repo_root = ROOT
    output_dir = Path(args.output_dir)

    online_report = read_json(repo_root / "outputs/online/20260710_4gpu_collection_report.json")
    semantic_audit = read_json(repo_root / "outputs/online/4gpu_expert_trace_20260710_090102/run_b_expert_trace/expert_to_traffic_semantic_audit.json")
    prediction_summary = read_json(repo_root / "outputs/offline/m6p_pre_gpu_fix/prediction_replay_summary.json")["summary"]
    best_traffic = min(
        (row for row in prediction_summary if row["policy_name"] == "RS_safe_barrier_criticality" and row["evaluation_eligible"]),
        key=lambda item: float(item["mean_prediction_relative_l1_error"]),
    )
    payload = {
        "expert_prediction_track": {
            "expert_trace_available": True,
            "world_merge_available": True,
            "original_run_b_reported_o1_mean_relative_l1": 0.9356,
            "semantic_mapping_ready": bool(semantic_audit["a8_final_diagnosis"]["can_use_expert_trace_for_prediction_now"]),
            "current_blocker": "expert-to-traffic evaluation must be re-anchored to phase_context P0 actual matrices with the corrected bytes model before expert predictors are judged",
            "candidate_directions": [
                "source_rank_expert_copy",
                "source_rank_expert_transition",
                "layer_pair_expert_transition",
                "source-rank-conditioned transition",
                "faithful gate replay",
            ],
        },
        "traffic_prediction_track": {
            "best_non_oracle_predictor": best_traffic["predictor_name"],
            "best_non_oracle_relative_l1": best_traffic["mean_prediction_relative_l1_error"],
            "best_non_oracle_cosine": best_traffic["mean_prediction_cosine_similarity"],
            "copy_current_dispatch_relative_l1": next(row["mean_prediction_relative_l1_error"] for row in prediction_summary if row["policy_name"] == "RS_safe_barrier_criticality" and row["predictor_name"] == "copy_current_dispatch"),
            "policy_gain_not_guaranteed": True,
            "reason": "traffic predictors can improve matrix-shape error while still failing to improve scheduling because current safe-U policies weakly consume P2",
        },
        "policy_consumption_track": {
            "barrier_criticality_observation": "P2 inputs change, but the replay outcome is effectively invariant; oracle P2 does not help.",
            "gated_greedy_observation": "P2 can change the decision path, but non-oracle variants often move ordering the wrong way; oracle P2 gives only a narrow gain.",
            "priority_fix_before_predictor_fix": True,
        },
        "recommended_sequence": [
            "Fix the expert-to-traffic semantic evaluation path first.",
            "Then inspect how RS_safe_barrier_criticality and RS_safe_gated_greedy consume P2.",
            "Only after policy P2 consumption is understood should traffic predictors and expert predictors be compared as the mainline.",
        ],
    }
    write_json(output_dir / "prediction_design_matrix.json", payload)
    write_text(output_dir / "prediction_design_matrix.md", render_markdown(payload))


if __name__ == "__main__":
    main()
