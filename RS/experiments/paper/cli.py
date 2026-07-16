from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.hashing import stable_hash_dict

from .aggregation import aggregate_results
from .capability_audit import render_capability_markdown, run_capability_audit
from .contracts import RecordMetadata
from .hiding_evaluation import evaluate_hiding_gap
from .prediction_evaluation import evaluate_prediction
from .result_bundle import write_json
from .runtime_evaluation import evaluate_runtime_correctness
from .scheduling_evaluation import evaluate_scheduling


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m experiments.paper.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "capture-trace", "build-traffic", "scheduling", "prediction", "hiding", "runtime-correctness", "aggregate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="")
        cmd.add_argument("--output-dir", default="")
        cmd.add_argument("--input", default="")
    return parser.parse_args(argv)


def _load_config(path: str, *, default_rel: str) -> tuple[dict[str, Any], Path]:
    config_path = ROOT / default_rel if not path else Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return dict(payload or {}), config_path


def _git_text(command: str) -> str:
    import subprocess

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _metadata(*, config: dict[str, Any], config_path: Path) -> RecordMetadata:
    config_digest = stable_hash_dict(config)
    branch = _git_text("git branch --show-current")
    commit = _git_text("git rev-parse HEAD")
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


def _write_smoke_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "smoke_summary.json", payload)


def cmd_audit(args: argparse.Namespace) -> int:
    config, config_path = _load_config(args.config, default_rel="configs/official/paper/capability_audit.yaml")
    output_dir = _output_dir(args, config, "audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(config=config, config_path=config_path)
    matrix = run_capability_audit(repo_root=ROOT)
    write_json(output_dir / "capability_matrix.json", matrix)
    (output_dir / "CAPABILITY_AUDIT.md").write_text(render_capability_markdown(matrix), encoding="utf-8")
    scheduling = evaluate_scheduling(
        fixture_dir=ROOT / "tests/fixtures/offline_replay_smoke",
        metadata=metadata,
        model_id=metadata.model_id,
        model_revision=metadata.model_revision,
        policy_ids=("birkhoff_von_neumann_fluid", "exact_small_instance_reference", "B_birkhoff", "U_barrier_criticality_global_matching"),
    )
    prediction = evaluate_prediction(
        fixture_dir=ROOT / "tests/fixtures/offline_replay_smoke",
        metadata=metadata,
        model_id=metadata.model_id,
        model_revision=metadata.model_revision,
        joint_policy_id="U_barrier_criticality_global_matching",
    )
    hiding = evaluate_hiding_gap(metadata=metadata, model_id=metadata.model_id)
    runtime = evaluate_runtime_correctness(metadata=metadata)
    write_json(output_dir / "scheduling_summary.json", scheduling)
    write_json(output_dir / "prediction_summary.json", prediction)
    write_json(output_dir / "hiding_summary.json", hiding)
    write_json(output_dir / "runtime_summary.json", runtime)
    _write_smoke_summary(
        output_dir,
        {
            "tiny_scheduling_status": scheduling["status"],
            "tiny_prediction_status": prediction["status"],
            "tiny_hiding_status": hiding["status"],
            "tiny_runtime_status": runtime["status"],
            "real_trace_offline_smoke": "MISSING_CAPABILITY",
            "reason": "clean frozen clone does not contain a replay-derived real trace bundle artifact",
        },
    )
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


def cmd_scheduling(args: argparse.Namespace) -> int:
    config, config_path = _load_config(args.config, default_rel="configs/official/paper/scheduling_value.yaml")
    metadata = _metadata(config=config, config_path=config_path)
    output_dir = _output_dir(args, config, "scheduling")
    result = evaluate_scheduling(
        fixture_dir=ROOT / "tests/fixtures/offline_replay_smoke",
        metadata=metadata,
        model_id=metadata.model_id,
        model_revision=metadata.model_revision,
        policy_ids=tuple(config.get("policies", ["birkhoff_von_neumann_fluid", "exact_small_instance_reference", "B_birkhoff", "U_barrier_criticality_global_matching"])),
    )
    write_json(output_dir / "scheduling_summary.json", result)
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


def cmd_prediction(args: argparse.Namespace) -> int:
    config, config_path = _load_config(args.config, default_rel="configs/official/paper/prediction_value.yaml")
    metadata = _metadata(config=config, config_path=config_path)
    output_dir = _output_dir(args, config, "prediction")
    result = evaluate_prediction(
        fixture_dir=ROOT / "tests/fixtures/offline_replay_smoke",
        metadata=metadata,
        model_id=metadata.model_id,
        model_revision=metadata.model_revision,
        joint_policy_id=str(config.get("joint_policy", "U_barrier_criticality_global_matching")),
    )
    write_json(output_dir / "prediction_summary.json", result)
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


def cmd_hiding(args: argparse.Namespace) -> int:
    config, config_path = _load_config(args.config, default_rel="configs/official/paper/hiding_timeline.yaml")
    metadata = _metadata(config=config, config_path=config_path)
    output_dir = _output_dir(args, config, "hiding")
    result = evaluate_hiding_gap(metadata=metadata, model_id=metadata.model_id)
    write_json(output_dir / "hiding_summary.json", result)
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


def cmd_runtime(args: argparse.Namespace) -> int:
    config, config_path = _load_config(args.config, default_rel="configs/official/paper/runtime_correctness_gloo.yaml")
    metadata = _metadata(config=config, config_path=config_path)
    output_dir = _output_dir(args, config, "runtime")
    result = evaluate_runtime_correctness(metadata=metadata)
    write_json(output_dir / "runtime_summary.json", result)
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    input_dir = Path(args.input) if args.input else ROOT / "outputs/paper/audit"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    result = aggregate_results(input_dir=input_dir)
    write_json(output_dir / "paired_records.jsonl", result["paired_records"])
    write_json(output_dir / "aggregate_summary.json", result)
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "scheduling":
        return cmd_scheduling(args)
    if args.command == "prediction":
        return cmd_prediction(args)
    if args.command == "hiding":
        return cmd_hiding(args)
    if args.command == "runtime-correctness":
        return cmd_runtime(args)
    if args.command == "aggregate":
        return cmd_aggregate(args)
    if args.command in {"capture-trace", "build-traffic"}:
        raise SystemExit(f"{args.command} is scaffolded but not executed in this frozen audit round")
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
