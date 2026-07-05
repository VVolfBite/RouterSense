#!/usr/bin/env python3
"""Formal offline flow-window scheduling study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import random

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.artifact import collect_environment_snapshot, write_json
from rs.core.experiment_config import RunConfig, load_run_config
from rs.runtime.offline.traffic.matrix_builder import (
    build_owner_by_expert,
    build_predicted_traffic,
    build_sample_layer_matrices,
    combine_matrix_from_dispatch,
    load_trace_jsonl,
)
from rs.scheduling.multiphase.global_ready_set import (
    RUNTIME_LOOKAHEAD_MODE,
    schedule_global_ready_set,
    schedule_greedy,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def _resolve_trace_path(config: RunConfig) -> Path:
    trace_dir = Path(config.workload.trace_artifact_dir)
    trace_path = trace_dir / "trace.jsonl"
    if trace_path.exists():
        return trace_path
    return trace_dir


def _zero_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def _shuffle_matrix(matrix: list[list[int]]) -> list[list[int]]:
    flat = [value for row in matrix for value in row]
    rng = random.Random(42)
    rng.shuffle(flat)
    width = len(matrix[0]) if matrix else 0
    return [flat[index:index + width] for index in range(0, len(flat), width)] if width else []


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_run_config(
        config_path=args.config,
        overrides=list(args.override),
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    run_dir = Path(config.artifact.output_root) / config.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = _resolve_trace_path(config)
    records = load_trace_jsonl(trace_path)
    owner_by_expert = build_owner_by_expert(records, placement="round_robin", num_gpus=config.topology.ep_size)
    sample_layer_matrices = build_sample_layer_matrices(records, owner_by_expert=owner_by_expert, num_gpus=config.topology.ep_size)
    sample_id = sorted(sample_layer_matrices)[0]
    layer_ids = sorted(sample_layer_matrices[sample_id])
    dispatch_matrix = sample_layer_matrices[sample_id][layer_ids[0]]
    p1_return_matrix = combine_matrix_from_dispatch(dispatch_matrix)
    perfect_p2_matrix = sample_layer_matrices[sample_id][layer_ids[1]] if len(layer_ids) >= 2 else _zero_matrix(config.topology.ep_size)
    if config.policy.prediction_source == "perfect_trace":
        p2_matrix = perfect_p2_matrix
    elif config.policy.prediction_source == "zero_hint":
        p2_matrix = _zero_matrix(config.topology.ep_size)
    elif config.policy.prediction_source == "shuffled_hint":
        p2_matrix = _shuffle_matrix(perfect_p2_matrix)
    elif config.policy.prediction_source == "calibrated_artifact":
        raise ValueError("calibrated_artifact is unsupported until a real predictor artifact schema is implemented")
    else:
        raise ValueError(f"unsupported prediction_source={config.policy.prediction_source!r}")

    traffic_window = {
        "sample_id": sample_id,
        "layer_ids": layer_ids,
        "mode": config.runtime.scheduling_mode,
        "ready_flows": "p0_dispatch",
        "blocked_flows": "p1_return",
        "forecast_flows": "p2_next_dispatch_forecast",
        "prediction_source": config.policy.prediction_source,
    }
    write_json(run_dir / "run_manifest.json", {
        "run_id": config.run.name,
        "run_kind": config.run.kind,
        "trace_path": str(trace_path),
        "logical_num_gpus": config.topology.ep_size,
        "prediction_source": config.policy.prediction_source,
        "source_config_path": config.source_config_path,
    })
    write_json(run_dir / "environment.json", collect_environment_snapshot())
    write_json(run_dir / "placement.json", {"mode": "round_robin", "owner_by_expert": owner_by_expert})
    write_json(run_dir / "traffic_window.json", traffic_window)
    write_json(run_dir / "p0_dispatch_matrix.json", dispatch_matrix)
    write_json(run_dir / "p1_return_matrix.json", p1_return_matrix)
    write_json(run_dir / "p2_next_dispatch_forecast_matrix.json", p2_matrix)

    policies = config.policy.policies or ("global_ready_set", "greedy", "birkhoff", "oracle_guided_reference")
    comparison: list[dict[str, Any]] = []
    for policy_name in policies:
        if policy_name == "global_ready_set":
            result = schedule_global_ready_set(
                dispatch_matrix,
                p1_return_matrix,
                p2_matrix,
                config.topology.ep_size,
                scheduling_mode=config.runtime.scheduling_mode or RUNTIME_LOOKAHEAD_MODE,
                prediction_confidence=1.0 if any(any(v > 0 for v in row) for row in p2_matrix) else 0.0,
                expert_compute_delay=config.runtime.expert_compute_delay,
            )
            write_json(run_dir / f"schedule_{policy_name}.json", result)
            write_json(run_dir / f"audit_{policy_name}.json", result["audit"])
            planning_time_ms = float(result.get("solve_time_ms", result.get("audit", {}).get("planning_time_ms", 0.0)))
            write_json(run_dir / f"metrics_{policy_name}.json", {
                "policy_name": policy_name,
                "supported": True,
                "solver_status": "ready",
                "mode": config.runtime.scheduling_mode,
                "makespan": result["makespan"],
                "planning_time_ms": planning_time_ms,
                "replay_valid": result["audit"]["valid"],
                "prediction_used": result["prediction_used"],
                "prediction_source": config.policy.prediction_source,
                "certified_optimal": False,
                "optimality_gap": None,
            })
            comparison.append({
                "policy_name": policy_name,
                "supported": True,
                "solver_status": "ready",
                "mode": config.runtime.scheduling_mode,
                "makespan": result["makespan"],
                "planning_time_ms": planning_time_ms,
                "replay_valid": result["audit"]["valid"],
                "prediction_used": result["prediction_used"],
                "prediction_source": config.policy.prediction_source,
                "certified_optimal": False,
                "optimality_gap": None,
            })
        elif policy_name == "greedy":
            result = schedule_greedy(
                dispatch_matrix,
                p1_return_matrix,
                p2_matrix,
                config.topology.ep_size,
                scheduling_mode=config.runtime.scheduling_mode or RUNTIME_LOOKAHEAD_MODE,
                prediction_confidence=1.0 if any(any(v > 0 for v in row) for row in p2_matrix) else 0.0,
                expert_compute_delay=config.runtime.expert_compute_delay,
            )
            write_json(run_dir / f"schedule_{policy_name}.json", result)
            write_json(run_dir / f"audit_{policy_name}.json", result["audit"])
            planning_time_ms = float(result.get("solve_time_ms", result.get("audit", {}).get("planning_time_ms", 0.0)))
            write_json(run_dir / f"metrics_{policy_name}.json", {
                "policy_name": policy_name,
                "supported": True,
                "solver_status": "ready",
                "mode": config.runtime.scheduling_mode,
                "makespan": result["makespan"],
                "planning_time_ms": planning_time_ms,
                "replay_valid": result["audit"]["valid"],
                "prediction_used": result["prediction_used"],
                "prediction_source": config.policy.prediction_source,
                "certified_optimal": False,
                "optimality_gap": None,
            })
            comparison.append({
                "policy_name": policy_name,
                "supported": True,
                "solver_status": "ready",
                "mode": config.runtime.scheduling_mode,
                "makespan": result["makespan"],
                "planning_time_ms": planning_time_ms,
                "replay_valid": result["audit"]["valid"],
                "prediction_used": result["prediction_used"],
                "prediction_source": config.policy.prediction_source,
                "certified_optimal": False,
                "optimality_gap": None,
            })
        else:
            comparison.append({
                "policy_name": policy_name,
                "supported": False,
                "solver_status": "unsupported",
                "mode": config.runtime.scheduling_mode,
                "makespan": None,
                "planning_time_ms": None,
                "replay_valid": False,
                "prediction_used": False,
                "prediction_source": config.policy.prediction_source,
                "certified_optimal": False,
                "optimality_gap": None,
            })

    write_json(run_dir / "comparison.json", comparison)
    report_lines = [
        f"# Flow Schedule Study: {config.run.name}",
        "",
        f"- trace: `{trace_path}`",
        f"- sample_id: `{sample_id}`",
        f"- layer_ids: `{layer_ids}`",
    ]
    for row in comparison:
        report_lines.append(f"- {row['policy_name']}: supported={row['supported']} makespan={row['makespan']} replay_valid={row['replay_valid']}")
    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
