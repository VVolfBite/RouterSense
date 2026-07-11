#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distributed._gpu_runner_common import (
    available_cuda_count,
    build_policy_correctness_config,
    copy_config,
    dump_yaml,
    load_yaml,
    read_json,
    run_subprocess,
    torchrun_policy_command,
    write_json,
)


DEFAULT_STRATEGIES = (
    "native",
    "fifo_async_p2p",
    "greedy_async_p2p",
    "birkhoff_phase_local_sync",
    "birkhoff_phase_local_async_p2p",
    "routersense_joint_phase_sync",
    "routersense_joint_zero_hint_async_p2p",
    "routersense_joint_predicted_async_p2p",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4GPU A2 comparison body with one torchrun process per strategy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategies", nargs="*", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--selected-layers", default="all")
    parser.add_argument("--profile", default="perf", choices=("debug", "execution", "perf"))
    parser.add_argument("--preflight-mode", default="compact", choices=("full", "compact"))
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _fallback(output_dir: Path, *, world_size: int, config: str, strategies: list[str], warmup_iters: int, measure_iters: int, selected_layers: str, profile: str, preflight_mode: str, dry_run: bool) -> dict:
    gate_cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py",
    ]
    proc = run_subprocess(gate_cmd)
    payload = {
        "runner": "run_gpu_a2_strategy_compare",
        "config": str(config),
        "strategies": list(strategies),
        "warmup_iters": int(warmup_iters),
        "measure_iters": int(measure_iters),
        "selected_layers": str(selected_layers),
        "profile": str(profile),
        "preflight_mode": str(preflight_mode),
        "world_size": int(world_size),
        "dry_run": bool(dry_run),
        "status": "IMPLEMENTED_GPU_BLOCKED_BY_ENVIRONMENT",
        "fallback_used": True,
        "result_eligible_for_performance_comparison": False,
        "fallback_reason": "gpu_environment_insufficient_world_size",
        "fallback_command": gate_cmd,
        "fallback_returncode": int(proc.returncode),
    }
    (output_dir / "fallback_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "fallback_stderr.log").write_text(proc.stderr, encoding="utf-8")
    return payload


def _summary_stats(values: list[float]) -> dict[str, float | list[float]]:
    if not values:
        return {"raw": [], "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    ordered = sorted(float(v) for v in values)
    def _pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = (len(ordered) - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return ordered[lo]
        frac = idx - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac
    return {
        "raw": ordered,
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p25": float(_pct(0.25)),
        "p75": float(_pct(0.75)),
        "min": float(min(ordered)),
        "max": float(max(ordered)),
        "std": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def _safe_gain(by_name: dict[str, dict], left: str, right: str) -> float | None:
    left_row = by_name.get(left) or {}
    right_row = by_name.get(right) or {}
    left_value = (((left_row.get("metrics") or {}).get("total_forward_us") or {}).get("median"))
    right_value = (((right_row.get("metrics") or {}).get("total_forward_us") or {}).get("median"))
    if left_value in (None, 0) or right_value in (None, 0):
        return None
    return float((float(right_value) - float(left_value)) / float(right_value))


def _load_transport_epoch_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "rank0_transport_execution.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        if str(row.get("record_type", "")) != "async_phase_summary":
            continue
        epoch = int(row.get("forward_epoch", 0))
        phase = str(row.get("phase", "")).upper()
        bucket = result.setdefault(epoch, {"dispatch_transport_us": 0.0, "return_transport_us": 0.0, "p2p_enqueue_us": 0.0, "p2p_wait_us": 0.0})
        enqueue_us = max(0.0, float(row.get("enqueue_end_ns", 0) - row.get("enqueue_start_ns", 0)) / 1000.0)
        wait_us = max(0.0, float(row.get("wait_end_ns", 0) - row.get("wait_start_ns", 0)) / 1000.0)
        bucket["p2p_enqueue_us"] += enqueue_us
        bucket["p2p_wait_us"] += wait_us
        if phase == "P0":
            bucket["dispatch_transport_us"] += enqueue_us + wait_us
        elif phase == "P1":
            bucket["return_transport_us"] += enqueue_us + wait_us
    return result


def _metric_series(rank0_summary: dict, run_dir: Path) -> dict[str, list[float]]:
    repeat_records = [row for row in list(rank0_summary.get("repeat_records") or []) if not bool(row.get("warmup", False))]
    transport_by_epoch = _load_transport_epoch_metrics(run_dir)
    metrics: dict[str, list[float]] = {
        "total_forward_us": [],
        "dispatch_transport_us": [],
        "return_transport_us": [],
        "all_rank_transport_span_us": [],
        "p2p_enqueue_us": [],
        "p2p_wait_us": [],
        "prediction_us": [],
        "raw_u_build_us": [],
        "paired_b_build_us": [],
        "host_projection_us": [],
        "safe_selection_us": [],
        "plan_agreement_us": [],
        "local_materialization_us": [],
        "preflight_us": [],
        "control_overhead_us": [],
        "scheduling_overhead_us": [],
    }
    for row in repeat_records:
        epoch = int(row.get("forward_epoch", 0))
        transport = transport_by_epoch.get(epoch, {})
        dispatch_transport_us = float(transport.get("dispatch_transport_us", 0.0) or 0.0)
        return_transport_us = float(transport.get("return_transport_us", 0.0) or 0.0)
        p2p_enqueue_us = float(transport.get("p2p_enqueue_us", 0.0) or 0.0)
        p2p_wait_us = float(transport.get("p2p_wait_us", 0.0) or 0.0)
        if dispatch_transport_us <= 0.0:
            dispatch_transport_us = float(row.get("dispatch_hook_path_us", 0.0) or 0.0)
        if return_transport_us <= 0.0:
            return_transport_us = float(row.get("combine_hook_path_us", 0.0) or 0.0)
        control_overhead_us = sum(
            float(row.get(key, 0.0) or 0.0)
            for key in (
                "prediction_us",
                "raw_u_build_us",
                "paired_b_build_us",
                "host_projection_us",
                "safe_selection_us",
                "plan_agreement_us",
                "local_materialization_us",
                "preflight_us",
            )
        )
        metrics["total_forward_us"].append(float(row.get("global_max_forward_us", 0.0) or 0.0))
        metrics["dispatch_transport_us"].append(dispatch_transport_us)
        metrics["return_transport_us"].append(return_transport_us)
        metrics["all_rank_transport_span_us"].append(dispatch_transport_us + return_transport_us)
        metrics["p2p_enqueue_us"].append(p2p_enqueue_us)
        metrics["p2p_wait_us"].append(p2p_wait_us)
        metrics["prediction_us"].append(float(row.get("prediction_us", 0.0) or 0.0))
        metrics["raw_u_build_us"].append(float(row.get("raw_u_build_us", 0.0) or 0.0))
        metrics["paired_b_build_us"].append(float(row.get("paired_b_build_us", 0.0) or 0.0))
        metrics["host_projection_us"].append(float(row.get("host_projection_us", 0.0) or 0.0))
        metrics["safe_selection_us"].append(float(row.get("safe_selection_us", 0.0) or 0.0))
        metrics["plan_agreement_us"].append(float(row.get("plan_agreement_us", 0.0) or 0.0))
        metrics["local_materialization_us"].append(float(row.get("local_materialization_us", 0.0) or 0.0))
        metrics["preflight_us"].append(float(row.get("preflight_us", 0.0) or 0.0))
        metrics["control_overhead_us"].append(control_overhead_us)
        metrics["scheduling_overhead_us"].append(control_overhead_us)
    return metrics


def _build_strategy_result(*, strategy: str, run_dir: Path, summary_payload: dict) -> dict:
    details = summary_payload.get("details", {}) if isinstance(summary_payload, dict) else {}
    rank0_summary = read_json(run_dir / "rank0_summary.json")
    raw_series = _metric_series(rank0_summary, run_dir)
    metrics = {name: _summary_stats(values) for name, values in raw_series.items()}
    metrics["safe_selected_policy"] = rank0_summary.get("safe_selected_policy")
    metrics["fallback_count"] = int(rank0_summary.get("phase_sync_fallback_count", 0) or 0)
    metrics["async_executor_invocation_count"] = int(rank0_summary.get("async_executor_invocation_count", 0) or 0)
    metrics["batch_isend_irecv_call_count"] = int(rank0_summary.get("batch_isend_irecv_call_count", 0) or 0)
    metrics["real_send_op_count"] = int(rank0_summary.get("real_send_op_count", 0) or 0)
    metrics["real_recv_op_count"] = int(rank0_summary.get("real_recv_op_count", 0) or 0)
    metrics["local_copy_task_count"] = int(rank0_summary.get("local_copy_task_count", 0) or 0)
    return {
        "name": strategy,
        "status": "eligible" if int(metrics["fallback_count"]) == 0 else "fallback",
        "result_eligible_for_performance_comparison": int(metrics["fallback_count"]) == 0,
        "summary_status": str(summary_payload.get("status", "")),
        "metrics": metrics,
        "output_checksum": details.get("output_checksum"),
    }


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(Path(args.config), output_dir)
    base = load_yaml(Path(args.config))
    payload = {
        "runner": "run_gpu_a2_strategy_compare",
        "config": str(args.config),
        "strategies": list(args.strategies),
        "warmup_iters": int(args.warmup_iters),
        "measure_iters": int(args.measure_iters),
        "selected_layers": str(args.selected_layers),
        "profile": str(args.profile),
        "preflight_mode": str(args.preflight_mode),
        "world_size": int(args.world_size),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        payload["status"] = "dry_run_ready"
        write_json(output_dir / "a2_runner_summary.json", payload)
        print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
        return 0
    if available_cuda_count() < int(args.world_size):
        payload = _fallback(
            output_dir,
            world_size=int(args.world_size),
            config=str(args.config),
            strategies=list(args.strategies),
            warmup_iters=int(args.warmup_iters),
            measure_iters=int(args.measure_iters),
            selected_layers=str(args.selected_layers),
            profile=str(args.profile),
            preflight_mode=str(args.preflight_mode),
            dry_run=bool(args.dry_run),
        )
        write_json(output_dir / "a2_runner_summary.json", payload)
        print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
        return 0 if int(payload["fallback_returncode"]) == 0 else int(payload["fallback_returncode"])

    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    strategy_results: list[dict] = []
    for strategy in args.strategies:
        strategy_root = output_dir / "per_strategy" / str(strategy)
        run_name = f"a2_{strategy}"
        strategy_config = build_policy_correctness_config(
            base_comparison=base,
            strategy_name=str(strategy),
            run_name=run_name,
            output_root=strategy_root,
            profile=str(args.profile),
            selected_layers=str(args.selected_layers),
            save_logits=False,
        )
        config_path = generated_dir / f"{strategy}.yaml"
        dump_yaml(config_path, strategy_config)
        native = str(strategy) in {"native", "disabled"}
        cmd = torchrun_policy_command(
            config_path=config_path,
            run_id=run_name,
            output_dir=strategy_root,
            world_size=int(args.world_size),
            native=native,
        )
        proc = run_subprocess(
            cmd,
            extra_env={
                "ROUTERSENSE_WARMUP_ITERS": str(int(args.warmup_iters)),
                "ROUTERSENSE_MEASURE_ITERS": str(int(args.measure_iters)),
            },
        )
        (output_dir / f"{strategy}_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (output_dir / f"{strategy}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            payload.update({"status": f"{strategy}_failed", "failed_strategy": str(strategy), "returncode": int(proc.returncode), "failed_command": cmd})
            write_json(output_dir / "a2_runner_summary.json", payload)
            print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
            return int(proc.returncode)
        run_dir = strategy_root / run_name
        summary_payload = read_json(run_dir / "summary.json")
        strategy_results.append(_build_strategy_result(strategy=str(strategy), run_dir=run_dir, summary_payload=summary_payload))

    by_name = {row["name"]: row for row in strategy_results}
    payload.update(
        {
            "status": "executed",
            "strategies": strategy_results,
            "p2p_backend_gain": _safe_gain(by_name, "birkhoff_phase_local_async_p2p", "birkhoff_phase_local_sync"),
            "joint_gain": _safe_gain(by_name, "routersense_joint_zero_hint_async_p2p", "birkhoff_phase_local_async_p2p"),
            "prediction_gain": _safe_gain(by_name, "routersense_joint_predicted_async_p2p", "routersense_joint_zero_hint_async_p2p"),
            "full_system_gain": _safe_gain(by_name, "routersense_joint_predicted_async_p2p", "birkhoff_phase_local_sync"),
        }
    )
    write_json(output_dir / "a2_runner_summary.json", payload)
    print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
