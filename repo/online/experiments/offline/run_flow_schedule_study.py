#!/usr/bin/env python3
"""Formal offline flow-window scheduling study."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    build_policy_logical_plan,
    build_scheduling_problem,
    replay_and_audit_logical_plan,
    summarize_schedule_tail_metrics,
)
from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
from rs.scheduling.registry import resolve_policy, supported_policies
from rs.scheduling.reference.exact_small_instance import solve_problem_exact
from rs.scheduling.validation import compare_plan_to_exact_reference, validate_bvn_fluid_certificate, validate_logical_plan


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


def _artifact_name(policy_name: str) -> str:
    return policy_name.replace(":", "__").replace("/", "_")


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

    policies = config.offline_study.policies or ("greedy_ready_set", "routersense_multiphase_lookahead:p0_p1_p2")
    policy_plan_dir = run_dir / "policy_plans"
    policy_plan_dir.mkdir(parents=True, exist_ok=True)
    exact_result = solve_problem_exact(problem, time_limit_ms=5000)
    write_json(run_dir / "exact_small_instance_reference.json", exact_result)
    write_json(
        run_dir / "logical_model_contract.json",
        {
            "online_phase_local": "discrete_bucket_phase_sync_wave",
            "offline_fluid_reference": "offline_fluid_crossbar",
            "runtime_latency_comparable": False,
            "notes": [
                "logical service horizons are not GPU runtime latency",
                "offline fluid references are not online executor policies",
            ],
        },
    )
    comparison: list[dict[str, Any]] = []
    reference_comparison: list[dict[str, Any]] = []
    for policy_name in policies:
        artifact_name = _artifact_name(policy_name)
        if policy_name in set(supported_policies()):
            policy = resolve_policy(
                policy_name=policy_name,
                bucket_rows=config.execution.bucket_rows,
                p0_weight=config.online_policy.parameters.p0_weight,
                p1_reservation_weight=config.online_policy.parameters.p1_reservation_weight,
                p2_hint_weight=config.online_policy.parameters.p2_hint_weight,
            )
            logical_plan = build_policy_logical_plan(
                problem=problem,
                policy_name=policy_name,
                bucket_rows=config.execution.bucket_rows,
                p0_weight=config.online_policy.parameters.p0_weight,
                p1_reservation_weight=config.online_policy.parameters.p1_reservation_weight,
                p2_hint_weight=config.online_policy.parameters.p2_hint_weight,
            )
            audit = replay_and_audit_logical_plan(problem, logical_plan)
            expected_flows = tuple(problem.flow_window.ready_flows + problem.flow_window.blocked_flows)
            logical_validation = validate_logical_plan(logical_plan, expected_flows=expected_flows)
            exact_comparison = compare_plan_to_exact_reference(logical_plan, exact_result)
            bvn_certificate_validation = None
            if policy_name == "birkhoff_von_neumann_fluid":
                bvn_certificate_validation = validate_bvn_fluid_certificate(logical_plan.diagnostics.get("certificate", {}))
            write_json(policy_plan_dir / f"{artifact_name}.json", logical_plan.to_dict())
            write_json(run_dir / f"schedule_{artifact_name}.json", logical_plan.to_dict())
            write_json(run_dir / f"audit_{artifact_name}.json", audit)
            write_json(run_dir / f"policy_diagnostics_{artifact_name}.json", logical_plan.diagnostics)
            planning_time_ms = float(logical_plan.diagnostics.get("solve_time_ms", 0.0))
            prediction_used = bool(logical_plan.diagnostics.get("p2_forecast_used", False))
            evaluation_eligible = bool(logical_plan.diagnostics.get("evaluation_eligible", True))
            makespan = float(logical_plan.diagnostics.get("makespan", audit.get("makespan", 0.0))) if audit.get("valid", False) else None
            reference_result = logical_plan.diagnostics.get("reference_result", {})
            solver_status = str(reference_result.get("solver_status", "ready")) if isinstance(reference_result, dict) else "ready"
            policy_supported = bool(reference_result.get("supported", True)) if isinstance(reference_result, dict) else True
            if isinstance(policy, RouterSenseMultiphaseLookaheadPolicy):
                prepared = policy.build_prepared_window_plan(
                    problem=problem,
                    created_at_layer_id=str(flow_metadata["start_layer"]),
                    applies_from_layer_id=str(flow_metadata["start_layer"]),
                )
                write_json(run_dir / f"prepared_window_plan_{artifact_name}.json", prepared.to_dict())
                actual_next = flow_metadata["actual_next_dispatch_matrix"]
                predicted_next = flow_metadata["p2_next_dispatch_forecast_matrix"]
                flat_actual = [int(v) for row in actual_next for v in row]
                flat_pred = [int(v) for row in predicted_next for v in row]
                l1_error = sum(abs(a - b) for a, b in zip(flat_actual, flat_pred, strict=False))
                dot = sum(a * b for a, b in zip(flat_actual, flat_pred, strict=False))
                actual_norm = sum(a * a for a in flat_actual) ** 0.5
                pred_norm = sum(b * b for b in flat_pred) ** 0.5
                cosine = 0.0 if actual_norm == 0.0 or pred_norm == 0.0 else float(dot / (actual_norm * pred_norm))
                endpoint_pressure_error = [
                    sum(abs(int(av) - int(pv)) for av, pv in zip(actual_row, pred_row, strict=False))
                    for actual_row, pred_row in zip(actual_next, predicted_next, strict=False)
                ]
                write_json(
                    run_dir / f"forecast_comparison_{artifact_name}.json",
                    {
                        "layer_id": str(flow_metadata["start_layer"]),
                        "actual_next_dispatch_matrix_digest": hashlib.sha256(json.dumps(actual_next).encode("utf-8")).hexdigest()[:16],
                        "predicted_next_dispatch_matrix_digest": problem.forecast.digest if problem.forecast is not None else "",
                        "prediction_source": config.offline_study.p2_source,
                        "matrix_l1_error": int(l1_error),
                        "matrix_cosine_similarity": float(cosine),
                        "endpoint_pressure_error": endpoint_pressure_error,
                        "oracle": bool(problem.forecast.oracle if problem.forecast is not None else False),
                        "evaluation_eligible": evaluation_eligible,
                    },
                )
            logical_model = str(logical_plan.diagnostics.get("logical_model", "discrete_bucket_phase_sync_wave"))
            online_executor_compatible = bool(getattr(policy.capabilities, "supports_online_phase_local_execution", False))
            runtime_latency_comparable = bool(logical_plan.diagnostics.get("runtime_latency_comparable", False))
            max_port_load_lower_bound = _max_port_load(problem)
            schedule_tail_metrics = summarize_schedule_tail_metrics(problem=problem, plan=logical_plan, audit=audit)
            makespan_to_lower_bound_ratio = (
                float(makespan) / float(max_port_load_lower_bound)
                if makespan is not None and max_port_load_lower_bound > 0
                else None
            )
            write_json(run_dir / f"schedule_metrics_{artifact_name}.json", schedule_tail_metrics)
            write_json(run_dir / f"metrics_{artifact_name}.json", {
                "policy_name": policy_name,
                "policy_version": getattr(policy, "policy_version", "v1"),
                "supported": policy_supported,
                "solver_status": solver_status,
                "logical_model": logical_model,
                "online_executor_compatible": online_executor_compatible,
                "runtime_latency_comparable": runtime_latency_comparable,
                "mode": config.runtime.scheduling_mode,
                "makespan": makespan if policy_supported else None,
                "logical_wave_count": len(logical_plan.waves),
                "logical_service_horizon": makespan if policy_supported else None,
                "max_port_load_lower_bound": max_port_load_lower_bound,
                "makespan_to_lower_bound_ratio": makespan_to_lower_bound_ratio,
                "planning_time_ms": planning_time_ms,
                "replay_valid": bool(audit["valid"]) if policy_supported else False,
                "logical_validation": logical_validation,
                "prediction_used": prediction_used,
                "prediction_source": config.offline_study.p2_source,
                "evaluation_eligible": evaluation_eligible,
                "exact_reference_available": bool(exact_comparison["available"]),
                "exact_optimality_gap": exact_comparison["optimality_gap"],
                "certified_optimal": bool(exact_comparison.get("policy_reaches_optimum", False)),
                "optimality_gap": exact_comparison["optimality_gap"],
                "bvn_certificate_validation": bvn_certificate_validation,
                "schedule_tail_metrics": schedule_tail_metrics,
            })
            row = {
                "policy_name": policy_name,
                "policy_version": getattr(policy, "policy_version", "v1"),
                "supported": policy_supported,
                "solver_status": solver_status,
                "logical_model": logical_model,
                "online_executor_compatible": online_executor_compatible,
                "runtime_latency_comparable": runtime_latency_comparable,
                "mode": config.runtime.scheduling_mode,
                "makespan": makespan if policy_supported else None,
                "logical_wave_count": len(logical_plan.waves),
                "logical_service_horizon": makespan if policy_supported else None,
                "max_port_load_lower_bound": max_port_load_lower_bound,
                "makespan_to_lower_bound_ratio": makespan_to_lower_bound_ratio,
                "planning_time_ms": planning_time_ms,
                "replay_valid": bool(audit["valid"]) if policy_supported else False,
                "logical_validation_passed": bool(logical_validation["valid"]),
                "prediction_used": prediction_used,
                "prediction_source": config.offline_study.p2_source,
                "evaluation_eligible": evaluation_eligible,
                "exact_reference_available": bool(exact_comparison["available"]),
                "exact_optimality_gap": exact_comparison["optimality_gap"],
                "certified_optimal": bool(exact_comparison.get("policy_reaches_optimum", False)),
                "optimality_gap": exact_comparison["optimality_gap"],
                "schedule_tail_metrics": schedule_tail_metrics,
            }
            comparison.append(row)
            reference_comparison.append({"policy_name": policy_name, "exact": exact_comparison, "bvn_certificate": bvn_certificate_validation})
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
                "evaluation_eligible": False,
                "certified_optimal": False,
                "optimality_gap": None,
            })

    write_json(run_dir / "comparison.json", comparison)
    write_json(run_dir / "policy_matrix_summary.json", comparison)
    write_json(run_dir / "reference_comparison.json", reference_comparison)
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


def _max_port_load(problem) -> int:
    matrices = (problem.p0_dispatch_matrix, problem.p1_return_matrix)
    max_load = 0
    for matrix in matrices:
        row_loads = [sum(int(value) for dst, value in enumerate(row) if src != dst) for src, row in enumerate(matrix)]
        col_loads = [sum(int(matrix[src][dst]) for src in range(len(matrix)) if src != dst) for dst in range(len(matrix))]
        max_load = max(max_load, max(row_loads, default=0), max(col_loads, default=0))
    return int(max_load)


if __name__ == "__main__":
    raise SystemExit(main())
