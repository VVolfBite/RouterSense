#!/usr/bin/env python3
"""Run online phase-local strategies and aggregate comparison metrics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from experiments.online.support.comparison_metrics import (
    aggregate_repetitions,
    build_comparison_report,
    metrics_from_rank_dir,
    render_markdown_report,
)
from rs.runtime.online.megatron_ep.trace_writer import write_json


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for strategy comparison configs")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"comparison config must be a mapping: {path}")
    return payload


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for strategy comparison configs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _model_config(model: dict[str, Any], output_dir: Path) -> str:
    path = output_dir / "generated_configs" / "model.yaml"
    payload = {
        "model_id": str(model.get("model_id", model.get("path", ""))),
        "local_path": str(model.get("path", model.get("local_path", ""))),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
    }
    _dump_yaml(path, payload)
    return str(path)


def _topology_config(topology: dict[str, Any], output_dir: Path) -> str:
    ep_size = int(topology.get("ep_size", 1))
    path = output_dir / "generated_configs" / "topology.yaml"
    payload = {
        "launcher": {"kind": "torchrun", "nnodes": 1, "nproc_per_node": ep_size, "standalone": True},
        "ep": {"size": ep_size},
        "network": {"scope": "single_node", "interface_hint": ""},
    }
    _dump_yaml(path, payload)
    return str(path)


def _single_strategy_config(
    *,
    comparison: dict[str, Any],
    strategy: dict[str, Any],
    repetition: int,
    output_dir: Path,
    model_config_path: str,
    topology_config_path: str,
) -> Path:
    execution = comparison.get("execution", {}) or {}
    runtime = comparison.get("runtime", {}) or {}
    workload = comparison.get("workload", {}) or {}
    run_name = f"rep{repetition}"
    config = {
        "run": {"kind": "online_policy_correctness", "name": run_name},
        "model": {"config": model_config_path},
        "topology": {"config": topology_config_path},
        "workload": {"prompts": str(workload.get("prompts", "configs/workload/smoke_prompts.json"))},
        "runtime": {
            "precision": str(runtime.get("precision", "fp16")),
            "dispatcher": str(runtime.get("dispatcher", "alltoall")),
            "control_mode": str(strategy.get("control_mode", "sync_before_phase")),
        },
        "online_policy": {
            "name": str(strategy.get("policy", "")),
            "parameters": {
                "p0_weight": float(execution.get("p0_weight", 1.0)),
                "p1_reservation_weight": float(execution.get("p1_reservation_weight", 1.0)),
                "p2_hint_weight": float(execution.get("p2_hint_weight", 1.0)),
            },
            "p2": {"mode": str(strategy.get("p2_hint_mode", "none")), "artifact": ""},
        },
        "offline_study": {"policies": []},
        "execution": {
            "mode": str(strategy.get("execution_mode", "phase_sync_wave")),
            "bucket_rows": int(execution.get("bucket_rows", 0)),
            "schedule": {
                "layer_selector": str(execution.get("schedule_layer_selector", "all")),
                "phase_selector": str(execution.get("schedule_phase_selector", "both")),
            },
        },
        "observation": {
            "profile": "execution",
            "capture_enabled": False,
            "capture_layer_selector": "",
            "capture_phase_selector": "",
            "heartbeat_enabled": False,
            "per_wave_timing_enabled": False,
        },
        "validation": {"save_logits": False, "stop_after_selected_layer": False},
        "artifact": {"artifact_root": str(output_dir / "per_strategy" / str(strategy["name"]))},
    }
    if bool(strategy.get("calibrated_p2", False)):
        config["online_policy"]["p2"]["mode"] = "calibrated_artifact"
    target = output_dir / "generated_configs" / f"{strategy['name']}_rep{repetition}.yaml"
    _dump_yaml(target, config)
    return target


def _torchrun_command(*, ep_size: int, config_path: Path, run_id: str, strategy_dir: Path) -> list[str]:
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={ep_size}",
        "-m",
        "experiments.online.run_policy_correctness",
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--output-dir",
        str(strategy_dir),
    ]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        env.pop(key, None)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not existing else f"src:{existing}"
    return env


def _copy_config(source: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output_dir / "config.yaml")


def _rank0_orchestrates_only() -> bool:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return True
    return str(local_rank) == "0"


def _collect_strategy_metrics(strategy_dir: Path, repetitions: int) -> list[dict[str, Any]]:
    rows = []
    for repetition in range(repetitions):
        run_dir = strategy_dir / f"rep{repetition}"
        metrics = metrics_from_rank_dir(run_dir, rank=0)
        summary = read_summary(run_dir)
        if "total_forward_us" in summary:
            metrics["total_forward_us"] = float(summary["total_forward_us"])
        rows.append(metrics)
    return rows


def read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    details = payload.get("details", {}) if isinstance(payload, dict) else {}
    if isinstance(details, dict):
        return details
    return {}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not _rank0_orchestrates_only():
        return 0
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    comparison = _load_yaml(config_path)
    _copy_config(config_path, output_dir)
    model_config_path = _model_config(comparison.get("model", {}) or {}, output_dir)
    topology_config_path = _topology_config(comparison.get("topology", {}) or {}, output_dir)
    strategies = list(comparison.get("strategies", []) or [])
    repetitions = int((comparison.get("execution", {}) or {}).get("repetitions", 1))
    ep_size = int((comparison.get("topology", {}) or {}).get("ep_size", 1))
    baseline = str((comparison.get("comparison", {}) or {}).get("baseline_strategy", strategies[0]["name"] if strategies else ""))
    timing: dict[str, Any] = {}
    strategy_entries: list[dict[str, Any]] = []
    for strategy in strategies:
        name = str(strategy["name"])
        strategy_dir = output_dir / "per_strategy" / name
        repetition_metrics = []
        timing[name] = []
        for repetition in range(repetitions):
            run_id = f"rep{repetition}"
            generated = _single_strategy_config(
                comparison=comparison,
                strategy=strategy,
                repetition=repetition,
                output_dir=output_dir,
                model_config_path=model_config_path,
                topology_config_path=topology_config_path,
            )
            cmd = _torchrun_command(ep_size=ep_size, config_path=generated, run_id=run_id, strategy_dir=strategy_dir)
            timing_start = time.monotonic_ns()
            if args.dry_run:
                (strategy_dir / run_id).mkdir(parents=True, exist_ok=True)
                (strategy_dir / run_id / "command.txt").write_text(" ".join(cmd), encoding="utf-8")
                return_code = 0
            else:
                proc = subprocess.run(cmd, cwd=ROOT, env=_child_env(), check=False)
                return_code = int(proc.returncode)
            elapsed_us = (time.monotonic_ns() - timing_start) / 1000.0
            timing[name].append({"repetition": repetition, "elapsed_us": elapsed_us, "return_code": return_code})
            if return_code != 0:
                raise SystemExit(return_code)
            if not args.dry_run:
                metrics = metrics_from_rank_dir(strategy_dir / run_id, rank=0)
                metrics["total_forward_us"] = elapsed_us
                repetition_metrics.append(metrics)
        if args.dry_run:
            repetition_metrics = []
        aggregated = aggregate_repetitions(repetition_metrics) if repetition_metrics else {}
        strategy_entries.append(
            {
                "name": name,
                "description": str(strategy.get("description", "")),
                "repetitions": repetitions,
                "metrics": aggregated,
            }
        )
    report = build_comparison_report(run_id=output_dir.name, baseline=baseline, strategies=strategy_entries)
    write_json(output_dir / "timing.json", timing)
    write_json(output_dir / "comparison_report.json", report)
    (output_dir / "comparison_report.md").write_text(render_markdown_report(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
