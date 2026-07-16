from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.hashing import stable_hash_dict

from .adapters.trace_adapter import capture_trace_from_config
from .aggregation import aggregate_results
from .capability_audit import apply_capability_evidence, baseline_capability_matrix, render_capability_markdown
from .configuration import consumed_config_payload, validate_paper_config
from .contracts import RecordMetadata
from .hiding_evaluation import evaluate_hiding_gap
from .oracle_evaluation import evaluate_oracle_control_set
from .prediction_evaluation import evaluate_prediction
from .result_bundle import build_result_bundle, write_json
from .runtime_evaluation import evaluate_materialization_contract_smoke, evaluate_runtime_correctness_with_gloo
from .scheduling_evaluation import evaluate_scheduling
from .traffic_builder import build_traffic_instances_from_trace_bundle, summarize_ownership_and_placement


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m experiments.paper.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "capture-trace", "build-traffic", "scheduling", "oracle-controls", "prediction", "hiding", "runtime-correctness", "aggregate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="")
        cmd.add_argument("--output-dir", default="")
        cmd.add_argument("--input", default="")
        cmd.add_argument("--model-path", default="")
        cmd.add_argument("--prompts", default="")
        cmd.add_argument("--evidence-dir", default="")
    return parser.parse_args(argv)


def _load_config(path: str, *, default_rel: str) -> tuple[dict[str, Any], Path]:
    config_path = ROOT / default_rel if not path else Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = dict(payload or {})
    validate_paper_config(config)
    return config, config_path


def _git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _load_json_path(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metadata(*, config: dict[str, Any]) -> RecordMetadata:
    config_digest = stable_hash_dict(config)
    branch = _git_text("branch", "--show-current")
    commit = _git_text("rev-parse", "HEAD")
    model = dict(config.get("models", {}))
    return RecordMetadata(
        branch=branch,
        commit=commit,
        config_digest=config_digest,
        run_id=str(config.get("run_id", f"paper-{datetime.now().strftime('%Y%m%d_%H%M%S')}")),
        seed=int(config.get("seeds", {}).get("default", 0)),
        model_id=str(model.get("model_id", "unknown-model")),
        model_revision=str(model.get("model_revision", "unknown-revision")),
    )


def _output_dir(args: argparse.Namespace, config: dict[str, Any], name: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    base = config.get("output", {}).get("dir", "outputs/paper")
    return ROOT / str(base) / name


def _selected_layers(config: dict[str, Any]) -> set[str] | None:
    layers = config.get("layers")
    if layers is None or layers == "auto" or layers == []:
        return None
    return {str(item) for item in layers}


def _resolved_input_path(config: dict[str, Any], *, cli_input: str = "") -> Path:
    source = cli_input or str(config.get("inputs", {}).get("source", ""))
    if not source:
        raise ValueError("inputs.source must be non-empty or --input must be provided")
    return ROOT / source if not Path(source).is_absolute() else Path(source)


def _write_common_output(output_dir: Path, config: dict[str, Any], input_path: str | None = None) -> RecordMetadata:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(config=config)
    consumed = consumed_config_payload(config, output_dir=output_dir, input_path=input_path)
    write_json(output_dir / "consumed_config.json", consumed)
    return metadata


def _remote_only_runtime_bundle(traffic_row: dict[str, Any]) -> dict[str, Any]:
    p0 = [[int(value) for value in row] for row in traffic_row["P0_matrix"]]
    p1 = [[int(value) for value in row] for row in traffic_row["P1_matrix"]]
    size = len(p0)
    p0_remote = [[0 if src == dst else int(p0[src][dst]) for dst in range(size)] for src in range(size)]
    p1_remote = [[0 if src == dst else int(p1[src][dst]) for dst in range(size)] for src in range(size)]
    ignored_self_rows = sum(int(p0[i][i]) for i in range(size))
    return {
        "p0_matrix": p0_remote,
        "p1_matrix": p1_remote,
        "full_p0_matrix": p0,
        "full_p1_matrix": p1,
        "ignored_self_rows": int(ignored_self_rows),
        "runtime_matrix_mode": "remote_only_from_real_trace",
    }


def cmd_capture_trace(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config, default_rel="configs/official/paper/trace_capture.yaml")
    output_dir = _output_dir(args, config, "capture_trace")
    metadata = _write_common_output(output_dir, config, input_path=args.prompts or config["inputs"].get("source"))
    model_path = args.model_path or os.environ.get("RS_MODEL_PATH", "")
    if not model_path:
        raise SystemExit("capture-trace requires --model-path or RS_MODEL_PATH")
    prompts_path = args.prompts or str(_resolved_input_path(config))
    bundle_dir = capture_trace_from_config(
        output_dir=output_dir,
        model_id=str(config["models"]["model_id"]),
        model_path=str(model_path),
        prompts_path=str(prompts_path),
        run_id=str(metadata.run_id),
        precision=str(config.get("measurement", {}).get("precision", "bf16")),
    )
    write_json(
        output_dir / "capture_trace_summary.json",
        {"status": "OK", "bundle_dir": str(bundle_dir), "model_path": str(model_path), "prompts_path": str(prompts_path)},
    )
    print(json.dumps({"output_dir": str(output_dir), "bundle_dir": str(bundle_dir)}, ensure_ascii=False, indent=2))
    return 0


def cmd_build_traffic(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config, default_rel="configs/official/paper/trace_capture.yaml")
    output_dir = _output_dir(args, config, "build_traffic")
    input_dir = Path(args.input) if args.input else _resolved_input_path(config)
    metadata = _write_common_output(output_dir, config, input_path=str(input_dir))
    if not input_dir.exists():
        summary = {"status": "ENVIRONMENT_BLOCKED", "reason": f"trace bundle missing: {input_dir}"}
        write_json(output_dir / "build_traffic_summary.json", summary)
        print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))
        return 0
    trace_samples, traffic_instances = build_traffic_instances_from_trace_bundle(
        bundle_dir=input_dir,
        virtual_ep_sizes=tuple(int(item) for item in config["virtual_ep_sizes"]),
        selected_layers=_selected_layers(config),
        metadata=metadata,
        cost_model_id=str(config["cost_model"]),
    )
    write_json(output_dir / "trace_samples.json", [item.to_dict() for item in trace_samples])
    write_json(output_dir / "compact_trace_samples.json", [item.to_dict() for item in trace_samples])
    write_json(output_dir / "traffic_instances.json", [item.to_dict() for item in traffic_instances])
    write_json(output_dir / "ownership_and_placement_summary.json", summarize_ownership_and_placement(traffic_instances=traffic_instances))
    veps_present = sorted({int(item.virtual_ep_size) for item in traffic_instances})
    summary = {
        "status": "OK",
        "trace_sample_count": len(trace_samples),
        "traffic_instance_count": len(traffic_instances),
        "virtual_ep_sizes_requested": list(config["virtual_ep_sizes"]),
        "virtual_ep_sizes_present": veps_present,
    }
    write_json(output_dir / "build_traffic_summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))
    return 0


def cmd_scheduling(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config, default_rel="configs/official/paper/scheduling_value.yaml")
    input_dir = _resolved_input_path(config, cli_input=args.input)
    output_dir = _output_dir(args, config, "scheduling")
    metadata = _write_common_output(output_dir, config, input_path=str(input_dir))
    result = evaluate_scheduling(
        fixture_dir=input_dir,
        metadata=metadata,
        model_id=metadata.model_id,
        model_revision=metadata.model_revision,
        policy_ids=tuple(config["policies"]),
        cost_model_id=str(config["cost_model"]),
    )
    write_json(output_dir / "scheduling_summary.json", result)
    if result.get("execution_window_bridge_summary") is not None:
        write_json(output_dir / "execution_window_bridge_summary.json", result["execution_window_bridge_summary"])
    if result.get("same_core_pair_summary") is not None:
        write_json(output_dir / "same_core_pair_summary.json", result["same_core_pair_summary"])
    print(json.dumps({"output_dir": str(output_dir), "status": result["status"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_prediction(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config, default_rel="configs/official/paper/prediction_value.yaml")
    input_dir = _resolved_input_path(config, cli_input=args.input)
    output_dir = _output_dir(args, config, "prediction")
    metadata = _write_common_output(output_dir, config, input_path=str(input_dir))
    result = evaluate_prediction(
        fixture_dir=input_dir,
        metadata=metadata,
        model_id=metadata.model_id,
        model_revision=metadata.model_revision,
        joint_policy_id=str(config.get("joint_policy", config["policies"][0])),
    )
    write_json(output_dir / "prediction_summary.json", result)
    print(json.dumps({"output_dir": str(output_dir), "status": result["status"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_oracle_controls(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config, default_rel="configs/official/paper/oracle_controls.yaml")
    fixture_dir = _resolved_input_path(config, cli_input=args.input)
    output_dir = _output_dir(args, config, "oracle_controls")
    metadata = _write_common_output(output_dir, config, input_path=str(fixture_dir))
    tolerance = float(config.get("measurement", {}).get("numeric_tolerance", 1.0e-9))
    result = evaluate_oracle_control_set(
        fixture_dir=fixture_dir,
        metadata=metadata,
        cost_model_id=str(config["cost_model"]),
        tolerance=tolerance,
    )
    write_json(output_dir / "oracle_control_summary.json", result)
    write_json(output_dir / "oracle_records.json", result["records"])
    for case in result["case_results"]:
        case_name = str(case["case_id"]).replace("oracle_", "").replace("_v1", "")
        case_dir = output_dir / case_name
        write_json(case_dir / "input.json", case["input"])
        write_json(case_dir / "o_local.json", case["o_local"])
        write_json(case_dir / "o_joint.json", case["o_joint"])
        write_json(case_dir / "comparison.json", case["comparison"])
    print(json.dumps({"output_dir": str(output_dir), "status": result["status"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "READY" else 2


def cmd_hiding(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config, default_rel="configs/official/paper/hiding_timeline.yaml")
    output_dir = _output_dir(args, config, "hiding")
    metadata = _write_common_output(output_dir, config)
    result = evaluate_hiding_gap(metadata=metadata, model_id=metadata.model_id)
    write_json(output_dir / "hiding_summary.json", result)
    print(json.dumps({"output_dir": str(output_dir), "status": result["status"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_runtime(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config, default_rel="configs/official/paper/runtime_correctness_gloo.yaml")
    output_dir = _output_dir(args, config, "runtime")
    input_dir = _resolved_input_path(config, cli_input=args.input)
    metadata = _write_common_output(output_dir, config, input_path=str(input_dir) if input_dir else None)
    materialization = evaluate_materialization_contract_smoke(metadata=metadata)
    matrix_bundle_path = ""
    trace_sample_id = None
    traffic_instance_id = None
    if input_dir is not None and input_dir.exists() and (input_dir / "traffic_instances.json").exists():
        traffic_instances = _load_json_path(input_dir / "traffic_instances.json")
        if traffic_instances:
            requested_vep = int(config.get("virtual_ep_sizes", [config.get("physical_world_size", 4)])[0])
            selected = next(
                (dict(row) for row in traffic_instances if int(row.get("virtual_ep_size", -1)) == requested_vep),
                dict(traffic_instances[0]),
            )
            generated = output_dir / "generated_matrix_bundle.json"
            write_json(generated, _remote_only_runtime_bundle(selected))
            matrix_bundle_path = str(generated)
            trace_sample_id = selected.get("trace_sample_id")
            traffic_instance_id = selected.get("instance_id")
    gloo = evaluate_runtime_correctness_with_gloo(
        metadata=metadata,
        policy_name=str(config["policies"][0]) if config.get("policies") else "U_barrier_criticality_global_matching",
        matrix_bundle_path=matrix_bundle_path,
        trace_sample_id=trace_sample_id,
        traffic_instance_id=traffic_instance_id,
    )
    overall_status = gloo["status"]
    if gloo["status"] == "ENVIRONMENT_BLOCKED":
        overall_status = "PARTIAL_GLOO_ENVIRONMENT_BLOCKED"
    result = {
        "status": overall_status,
        "records": materialization["records"] + gloo["records"],
        "tensor_parity_pass": bool(gloo.get("tensor_parity_pass", False)),
    }
    write_json(output_dir / "runtime_summary.json", result)
    if gloo.get("records"):
        for row in gloo["records"]:
            backend = str(row.get("execution_backend_id", "unknown"))
            evidence = dict(row.get("evidence", {}) or {})
            backend_dir = output_dir / backend
            rank_artifacts_dir = backend_dir / "rank_artifacts"
            write_json(backend_dir / "formal_runner_summary.json", evidence.get("runner_summary", {}))
            if evidence.get("runner_summary", {}).get("materialized_task_manifest_path"):
                materialized = Path(str(evidence["runner_summary"]["materialized_task_manifest_path"]))
                if materialized.exists():
                    write_json(backend_dir / "materialized_task_manifest.json", json.loads(materialized.read_text(encoding="utf-8")))
            if evidence.get("runner_summary", {}).get("executed_task_manifest_path"):
                executed = Path(str(evidence["runner_summary"]["executed_task_manifest_path"]))
                if executed.exists():
                    write_json(backend_dir / "executed_task_manifest.json", json.loads(executed.read_text(encoding="utf-8")))
            if evidence.get("parity_path"):
                parity_path = Path(str(evidence["parity_path"]))
                if parity_path.exists():
                    write_json(backend_dir / "parity.json", json.loads(parity_path.read_text(encoding="utf-8")))
            run_dir = Path(str(evidence.get("run_dir", "")))
            if run_dir.exists():
                for rank in range(int(evidence.get("runner_summary", {}).get("world_size", 4) or 4)):
                    summary_src = run_dir / f"rank{rank}.json"
                    materialized_src = run_dir / f"rank{rank}_materialized_task_manifest.json"
                    executed_src = run_dir / f"rank{rank}_executed_task_manifest.json"
                    parity_src = run_dir / f"rank{rank}_parity.json"
                    if summary_src.exists():
                        write_json(rank_artifacts_dir / f"rank{rank}_summary.json", json.loads(summary_src.read_text(encoding="utf-8")))
                    if materialized_src.exists():
                        write_json(rank_artifacts_dir / f"rank{rank}_materialized_task_manifest.json", json.loads(materialized_src.read_text(encoding="utf-8")))
                    if executed_src.exists():
                        write_json(rank_artifacts_dir / f"rank{rank}_executed_task_manifest.json", json.loads(executed_src.read_text(encoding="utf-8")))
                    if parity_src.exists():
                        write_json(rank_artifacts_dir / f"rank{rank}_parity.json", json.loads(parity_src.read_text(encoding="utf-8")))
        if trace_sample_id or traffic_instance_id:
            write_json(
                output_dir / "real_trace_runtime_summary.json",
                {
                    "status": result["status"],
                    "trace_sample_id": trace_sample_id,
                    "traffic_instance_id": traffic_instance_id,
                    "policy_id": str(config["policies"][0]) if config.get("policies") else "U_barrier_criticality_global_matching",
                },
            )
    print(json.dumps({"output_dir": str(output_dir), "status": result["status"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    input_dir = Path(args.input) if args.input else ROOT / "outputs/paper/audit"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    result = aggregate_results(input_dir=input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["paired_records"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / "aggregate_summary.json", result)
    print(json.dumps({"output_dir": str(output_dir), "sample_count": result["sample_count"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config, default_rel="configs/official/paper/capability_audit.yaml")
    output_dir = _output_dir(args, config, "audit")
    input_dir = _resolved_input_path(config, cli_input=args.input)
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None
    metadata = _write_common_output(output_dir, config, input_path=str(input_dir))
    if evidence_dir is not None and evidence_dir.exists():
        trace_summary = _load_json_path(evidence_dir / "trace" / "summary.json") if (evidence_dir / "trace" / "summary.json").exists() else {}
        scheduling = _load_json_path(evidence_dir / "scheduling" / "scheduling_summary.json") if (evidence_dir / "scheduling" / "scheduling_summary.json").exists() else {}
        oracle_control_summary = _load_json_path(evidence_dir / "oracle" / "oracle_control_summary.json") if (evidence_dir / "oracle" / "oracle_control_summary.json").exists() else None
        prediction = _load_json_path(evidence_dir / "results" / "prediction_summary.json") if (evidence_dir / "results" / "prediction_summary.json").exists() else {"status": "PARTIAL_MISSING_PREDICTED", "records": []}
        hiding = _load_json_path(evidence_dir / "results" / "hiding_summary.json") if (evidence_dir / "results" / "hiding_summary.json").exists() else {"status": "PARTIAL", "records": []}
        runtime_summary = _load_json_path(evidence_dir / "results" / "runtime_summary.json") if (evidence_dir / "results" / "runtime_summary.json").exists() else {"status": "ENVIRONMENT_BLOCKED", "records": [], "tensor_parity_pass": False}
        build_traffic_summary = _load_json_path(evidence_dir / "traffic" / "build_traffic_summary.json") if (evidence_dir / "traffic" / "build_traffic_summary.json").exists() else {"status": "ENVIRONMENT_BLOCKED", "reason": "build-traffic evidence missing"}
    else:
        trace_summary = {}
        scheduling = evaluate_scheduling(
            fixture_dir=input_dir,
            metadata=metadata,
            model_id=metadata.model_id,
            model_revision=metadata.model_revision,
            policy_ids=tuple(config["policies"]),
            cost_model_id=str(config["cost_model"]),
        )
        prediction = evaluate_prediction(
            fixture_dir=input_dir,
            metadata=metadata,
            model_id=metadata.model_id,
            model_revision=metadata.model_revision,
            joint_policy_id=str(config["policies"][-1]),
        )
        hiding = evaluate_hiding_gap(metadata=metadata, model_id=metadata.model_id)
        runtime = {"status": "MATERIALIZATION_CONTRACT_SMOKE", "records": []}
        try:
            runtime = evaluate_runtime_correctness_with_gloo(metadata=metadata)
        except Exception:
            runtime = {"status": "ENVIRONMENT_BLOCKED", "records": [], "tensor_parity_pass": False}
        materialization = evaluate_materialization_contract_smoke(metadata=metadata)
        runtime_status = runtime["status"]
        if runtime["status"] == "ENVIRONMENT_BLOCKED":
            runtime_status = "PARTIAL_GLOO_ENVIRONMENT_BLOCKED"
        runtime_summary = {
            "status": runtime_status,
            "records": materialization["records"] + runtime["records"],
            "tensor_parity_pass": bool(runtime.get("tensor_parity_pass", False)),
        }
        build_traffic_summary = {"status": "ENVIRONMENT_BLOCKED", "reason": "external trace bundle not provided during audit"}
        oracle_control_summary = None
    write_json(output_dir / "scheduling_summary.json", scheduling)
    write_json(output_dir / "prediction_summary.json", prediction)
    write_json(output_dir / "hiding_summary.json", hiding)
    write_json(output_dir / "runtime_summary.json", runtime_summary)
    matrix = apply_capability_evidence(
        baseline_capability_matrix(),
        trace_summary=trace_summary,
        scheduling_summary=scheduling,
        prediction_summary=prediction,
        runtime_summary=runtime_summary,
        build_traffic_summary=build_traffic_summary,
        oracle_control_summary=oracle_control_summary,
    )
    write_json(output_dir / "capability_matrix.json", matrix)
    (output_dir / "CAPABILITY_AUDIT.md").write_text(render_capability_markdown(matrix), encoding="utf-8")
    smoke = {
        "tiny_scheduling_status": scheduling["status"],
        "tiny_prediction_status": prediction["status"],
        "tiny_hiding_status": hiding["status"],
        "tiny_runtime_status": runtime_summary["status"],
        "real_trace_offline_smoke": build_traffic_summary["status"],
        "build_traffic_smoke": build_traffic_summary["status"],
    }
    write_json(output_dir / "smoke_summary.json", smoke)
    result_bundle = build_result_bundle(
        branch=metadata.branch,
        commit=metadata.commit,
        config_digest=metadata.config_digest,
        claim_scope=str(config["claim_scope"]),
        status="PAPER-EVAL-HARNESS-PARTIAL",
        scheduling_summary=scheduling,
        prediction_summary=prediction,
        hiding_summary=hiding,
        runtime_summary=runtime_summary,
        oracle_control_summary=oracle_control_summary,
        artifact_index={
            "audit/capability_matrix.json": "audit/capability_matrix.json",
            "results/scheduling_summary.json": "results/scheduling_summary.json",
            "results/prediction_summary.json": "results/prediction_summary.json",
            "results/hiding_summary.json": "results/hiding_summary.json",
            "results/runtime_summary.json": "results/runtime_summary.json",
            "tests/smoke_summary.json": "tests/smoke_summary.json",
            "trace/summary": "trace/summary.json" if evidence_dir is not None else "",
            "traffic/build_traffic_summary": "traffic/build_traffic_summary.json" if evidence_dir is not None else "",
            "traffic/traffic_instances.json": "traffic/traffic_instances.json" if evidence_dir is not None else "",
            "scheduling/strict_same_core_records.jsonl": "scheduling/strict_same_core_records.jsonl" if evidence_dir is not None else "",
            "oracle/oracle_control_summary.json": "oracle/oracle_control_summary.json" if evidence_dir is not None else "",
            "runtime/B/phase_sync/formal_runner_summary.json": "runtime/B/phase_sync/formal_runner_summary.json" if evidence_dir is not None else "",
        },
    )
    write_json(output_dir / "result_bundle.json", result_bundle)
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "capture-trace":
        return cmd_capture_trace(args)
    if args.command == "build-traffic":
        return cmd_build_traffic(args)
    if args.command == "scheduling":
        return cmd_scheduling(args)
    if args.command == "oracle-controls":
        return cmd_oracle_controls(args)
    if args.command == "prediction":
        return cmd_prediction(args)
    if args.command == "hiding":
        return cmd_hiding(args)
    if args.command == "runtime-correctness":
        return cmd_runtime(args)
    if args.command == "aggregate":
        return cmd_aggregate(args)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
