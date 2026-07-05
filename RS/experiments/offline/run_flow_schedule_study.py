#!/usr/bin/env python3
"""Formal offline flow-window scheduling study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.artifact import collect_environment_snapshot, write_json
from rs.core.experiment_config import RunConfig, load_run_config
from rs.runtime.offline.runner import (
    FlowWindowSelector,
    LogicalTopology,
    OfflineFlowStudyRequest,
    P2ForecastSource,
    PlacementConfig,
    build_flow_window,
    build_scheduling_problem,
    replay_and_audit_logical_plan,
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
    request = OfflineFlowStudyRequest(
        trace_artifact_dir=Path(config.workload.trace_artifact_dir),
        logical_topology=LogicalTopology(num_gpus=config.topology.ep_size),
        placement=PlacementConfig(mode="round_robin"),
        window=FlowWindowSelector(
            sample_selector=config.offline_study.window.sample_selector,
            start_layer_selector=config.offline_study.window.start_layer_selector,
        ),
        p2_source=P2ForecastSource(mode=config.offline_study.p2_source),
        policy_names=config.offline_study.policies,
        expert_compute_delay=config.runtime.expert_compute_delay,
        scheduling_mode=config.runtime.scheduling_mode,
    )
    flow_window, flow_metadata = build_flow_window(request)
    problem = build_scheduling_problem(request)
    trace_path = Path(flow_metadata["trace_path"])
    write_json(run_dir / "run_manifest.json", {
        "run_id": config.run.name,
        "run_kind": config.run.kind,
        "trace_path": str(trace_path),
        "logical_num_gpus": config.topology.ep_size,
        "prediction_source": config.offline_study.p2_source,
        "source_config_path": config.source_config_path,
    })
    write_json(run_dir / "environment.json", collect_environment_snapshot())
    write_json(run_dir / "placement.json", flow_metadata["placement"])
    write_json(
        run_dir / "traffic_window.json",
        {
            "sample_id": flow_metadata["sample_id"],
            "layer_ids": flow_metadata["layer_ids"],
            "mode": config.runtime.scheduling_mode,
            "forecast_source": flow_metadata["forecast_source"],
            "forecast_digest": flow_metadata["forecast_digest"],
            "ready_flows": [flow.to_dict() for flow in flow_window.ready_flows],
            "blocked_flows": [flow.to_dict() for flow in flow_window.blocked_flows],
            "forecast_flows": [flow.to_dict() for flow in flow_window.forecast_pressure],
        },
    )
    write_json(run_dir / "p0_dispatch_matrix.json", flow_metadata["dispatch_matrix"])
    write_json(run_dir / "p1_return_matrix.json", flow_metadata["p1_return_matrix"])
    write_json(run_dir / "p2_next_dispatch_forecast_matrix.json", flow_metadata["p2_next_dispatch_forecast_matrix"])

    policies = config.offline_study.policies or ("global_ready_set", "greedy")
    comparison: list[dict[str, Any]] = []
    for policy_name in policies:
        if policy_name == "global_ready_set":
            logical_plan = schedule_global_ready_set(problem)
            audit = replay_and_audit_logical_plan(problem, logical_plan)
            write_json(run_dir / f"schedule_{policy_name}.json", logical_plan.to_dict())
            write_json(run_dir / f"audit_{policy_name}.json", audit)
            planning_time_ms = float(logical_plan.diagnostics.get("solve_time_ms", 0.0))
            write_json(run_dir / f"metrics_{policy_name}.json", {
                "policy_name": policy_name,
                "supported": True,
                "solver_status": "ready",
                "mode": config.runtime.scheduling_mode,
                "makespan": logical_plan.diagnostics["makespan"],
                "planning_time_ms": planning_time_ms,
                "replay_valid": audit["valid"],
                "prediction_used": logical_plan.diagnostics["prediction_used"],
                "prediction_source": config.offline_study.p2_source,
                "certified_optimal": False,
                "optimality_gap": None,
            })
            comparison.append({
                "policy_name": policy_name,
                "supported": True,
                "solver_status": "ready",
                "mode": config.runtime.scheduling_mode,
                "makespan": logical_plan.diagnostics["makespan"],
                "planning_time_ms": planning_time_ms,
                "replay_valid": audit["valid"],
                "prediction_used": logical_plan.diagnostics["prediction_used"],
                "prediction_source": config.offline_study.p2_source,
                "certified_optimal": False,
                "optimality_gap": None,
            })
        elif policy_name == "greedy":
            logical_plan = schedule_greedy(problem)
            audit = replay_and_audit_logical_plan(problem, logical_plan)
            write_json(run_dir / f"schedule_{policy_name}.json", logical_plan.to_dict())
            write_json(run_dir / f"audit_{policy_name}.json", audit)
            planning_time_ms = float(logical_plan.diagnostics.get("solve_time_ms", 0.0))
            write_json(run_dir / f"metrics_{policy_name}.json", {
                "policy_name": policy_name,
                "supported": True,
                "solver_status": "ready",
                "mode": config.runtime.scheduling_mode,
                "makespan": logical_plan.diagnostics["makespan"],
                "planning_time_ms": planning_time_ms,
                "replay_valid": audit["valid"],
                "prediction_used": logical_plan.diagnostics["prediction_used"],
                "prediction_source": config.offline_study.p2_source,
                "certified_optimal": False,
                "optimality_gap": None,
            })
            comparison.append({
                "policy_name": policy_name,
                "supported": True,
                "solver_status": "ready",
                "mode": config.runtime.scheduling_mode,
                "makespan": logical_plan.diagnostics["makespan"],
                "planning_time_ms": planning_time_ms,
                "replay_valid": audit["valid"],
                "prediction_used": logical_plan.diagnostics["prediction_used"],
                "prediction_source": config.offline_study.p2_source,
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
                "prediction_source": config.offline_study.p2_source,
                "certified_optimal": False,
                "optimality_gap": None,
            })

    write_json(run_dir / "comparison.json", comparison)
    report_lines = [
        f"# Flow Schedule Study: {config.run.name}",
        "",
        f"- trace: `{trace_path}`",
        f"- sample_id: `{flow_metadata['sample_id']}`",
        f"- layer_ids: `{flow_metadata['layer_ids']}`",
    ]
    for row in comparison:
        report_lines.append(f"- {row['policy_name']}: supported={row['supported']} makespan={row['makespan']} replay_valid={row['replay_valid']}")
    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
