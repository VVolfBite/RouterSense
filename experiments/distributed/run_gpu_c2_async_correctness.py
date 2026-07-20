#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distributed._gpu_runner_common import (
    available_cuda_count,
    build_policy_correctness_config,
    copy_config,
    dump_yaml,
    load_official_config,
    read_json,
    run_subprocess,
    torchrun_policy_command,
    write_json,
    write_runner_result_bundle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4GPU C2 parity body or an explicit no-4GPU fallback.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-strategy", default=None)
    parser.add_argument("--candidate-strategy", default=None)
    parser.add_argument("--candidate-strategies", nargs="*", default=None)
    parser.add_argument("--profile", default=None, choices=("debug", "execution", "perf"))
    parser.add_argument("--selected-layers", default=None)
    parser.add_argument("--forward-epochs", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    ref = reference.float()
    cand = candidate.float()
    diff = (ref - cand).abs()
    denom = ref.abs().clamp_min(1e-8)
    rel = diff / denom
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "max_rel_error": float(rel.max().item()) if rel.numel() else 0.0,
        "mismatch_count": int((diff > 1e-4).sum().item()) if diff.numel() else 0,
        "shape_parity": list(ref.shape) == list(cand.shape),
        "dtype_parity": str(reference.dtype) == str(candidate.dtype),
        "nan_count": int(torch.isnan(cand).sum().item()),
        "inf_count": int(torch.isinf(cand).sum().item()),
    }


def _fallback(output_dir: Path, *, world_size: int, config: str, reference_strategy: str, candidate_strategy: str, profile: str, selected_layers: str, forward_epochs: int, dry_run: bool) -> dict:
    cuda_count = available_cuda_count()
    payload = {
        "runner": "run_gpu_c2_async_correctness",
        "config": str(config),
        "reference_strategy": str(reference_strategy),
        "candidate_strategy": str(candidate_strategy),
        "profile": str(profile),
        "selected_layers": str(selected_layers),
        "forward_epochs": int(forward_epochs),
        "world_size": int(world_size),
        "dry_run": bool(dry_run),
        "status": "IMPLEMENTED_GPU_BLOCKED_BY_ENVIRONMENT",
        "fallback_used": True,
        "result_eligible_for_performance_comparison": False,
        "fallback_reason": "gpu_environment_insufficient_world_size",
        "fallback_command": [],
        "fallback_returncode": 247,
        "cuda_device_count": int(cuda_count),
    }
    (output_dir / "fallback_stdout.log").write_text("", encoding="utf-8")
    (output_dir / "fallback_stderr.log").write_text(
        f"cuda_device_count={cuda_count} world_size={world_size} strict_gpu_run_blocked\n",
        encoding="utf-8",
    )
    return payload


def _load_logits(run_dir: Path) -> dict[tuple[int, int], torch.Tensor]:
    result: dict[tuple[int, int], torch.Tensor] = {}
    for path in sorted(run_dir.glob("*-rank*-epoch*-measure*.pt")):
        stem = path.stem
        rank_str = stem.split("-rank")[-1].split("-")[0]
        epoch_str = stem.split("-epoch")[-1].split("-")[0]
        result[(int(rank_str), int(epoch_str))] = torch.load(path, map_location="cpu")
    for path in sorted(run_dir.glob("*-rank*-logits.pt")):
        stem = path.stem
        rank_str = stem.split("-rank")[-1].split("-")[0]
        result.setdefault((int(rank_str), 0), torch.load(path, map_location="cpu"))
    return result


def _load_rank_summaries(run_dir: Path) -> list[dict]:
    return [
        read_json(path)
        for path in sorted(run_dir.glob("rank*_summary.json"))
        if path.name.startswith("rank") and "_prepared_plan_" not in path.name
    ]


def _maybe_load_environment_block(run_dir: Path, stdout: str) -> dict | None:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = read_json(summary_path)
            details = dict(summary.get("details", {}) or {})
            if str(details.get("status", "")) == "blocked_environment":
                return details
        except Exception:
            pass
    text = str(stdout or "").strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = dict(__import__("json").loads(text))
        except Exception:
            return None
        if str(payload.get("status", "")) == "blocked_environment":
            return payload
    return None


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(Path(args.config), output_dir)
    base = load_official_config(Path(args.config))
    resolved_reference = str(args.reference_strategy or base.get("reference_strategy", "native"))
    resolved_candidates = (
        list(args.candidate_strategies)
        if args.candidate_strategies is not None
        else [str(args.candidate_strategy)]
        if args.candidate_strategy is not None
        else [str(item) for item in (base.get("candidate_strategies") or [base.get("candidate_strategy", "routersense_future_p012_joint_event_rscf_async")])]
    )
    resolved_profile = str(args.profile if args.profile is not None else base.get("profile", "execution"))
    resolved_selected_layers = str(args.selected_layers if args.selected_layers is not None else base.get("selected_layers", "selected"))
    resolved_forward_epochs = int(args.forward_epochs if args.forward_epochs is not None else base.get("forward_epochs", (base.get("evaluation", {}) or {}).get("repeats", 2)))
    resolved_world_size = int(args.world_size if args.world_size is not None else base.get("world_size", (base.get("topology", {}) or {}).get("world_size", 4)))
    payload = {
        "runner": "run_gpu_c2_async_correctness",
        "config": str(args.config),
        "reference_strategy": resolved_reference,
        "candidate_strategies": list(resolved_candidates),
        "profile": resolved_profile,
        "selected_layers": resolved_selected_layers,
        "forward_epochs": int(resolved_forward_epochs),
        "world_size": int(resolved_world_size),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        payload["status"] = "dry_run_ready"
        write_json(output_dir / "c2_runner_summary.json", payload)
        write_runner_result_bundle(output_dir, runner_name="run_gpu_c2_async_correctness", payload=payload, run_kind="GPU_CORRECTNESS")
        print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
        return 0
    if available_cuda_count() < int(resolved_world_size):
        payload = _fallback(
            output_dir,
            world_size=int(resolved_world_size),
            config=str(args.config),
            reference_strategy=resolved_reference,
            candidate_strategy=",".join(resolved_candidates),
            profile=resolved_profile,
            selected_layers=resolved_selected_layers,
            forward_epochs=int(resolved_forward_epochs),
            dry_run=bool(args.dry_run),
        )
        write_json(output_dir / "c2_runner_summary.json", payload)
        write_runner_result_bundle(output_dir, runner_name="run_gpu_c2_async_correctness", payload=payload, run_kind="GPU_CORRECTNESS")
        print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
        return 0 if int(payload["fallback_returncode"]) == 0 else int(payload["fallback_returncode"])

    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    reference_root = output_dir / "reference"
    candidates_root = output_dir / "candidates"

    for strategy_name, run_name, root_dir in (
        (resolved_reference, "c2_reference", reference_root),
    ):
        config_payload = build_policy_correctness_config(
            base_comparison=base,
            strategy_name=strategy_name,
            run_name=run_name,
            output_root=root_dir,
            profile=resolved_profile,
            selected_layers=resolved_selected_layers,
            save_logits=True,
            world_size=int(resolved_world_size),
        )
        config_path = generated_dir / f"{run_name}.yaml"
        dump_yaml(config_path, config_payload)
        native = strategy_name in {"native", "disabled"}
        cmd = torchrun_policy_command(
            config_path=config_path,
            run_id=run_name,
            output_dir=root_dir,
            world_size=int(resolved_world_size),
            native=native,
        )
        proc = run_subprocess(cmd, extra_env={"ROUTERSENSE_WARMUP_ITERS": "0", "ROUTERSENSE_MEASURE_ITERS": str(int(resolved_forward_epochs))})
        (output_dir / f"{run_name}_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (output_dir / f"{run_name}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            blocked = _maybe_load_environment_block(reference_root / "c2_reference", proc.stdout)
            if blocked is not None:
                payload.update(
                    {
                        "status": "blocked_environment",
                        "blocked_stage": run_name,
                        "blocked_environment": blocked,
                        "result_eligible_for_performance_comparison": False,
                        "fallback_used": True,
                        "failed_command": cmd,
                        "returncode": int(proc.returncode),
                    }
                )
                write_json(output_dir / "c2_runner_summary.json", payload)
                write_runner_result_bundle(output_dir, runner_name="run_gpu_c2_async_correctness", payload=payload, run_kind="GPU_CORRECTNESS")
                print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
                return 0
            payload.update({"status": f"{run_name}_failed", "failed_command": cmd, "returncode": int(proc.returncode)})
            write_json(output_dir / "c2_runner_summary.json", payload)
            write_runner_result_bundle(output_dir, runner_name="run_gpu_c2_async_correctness", payload=payload, run_kind="GPU_CORRECTNESS")
            print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
            return int(proc.returncode)
    reference_run = reference_root / "c2_reference"
    reference_logits = _load_logits(reference_run)
    reference_checksum = read_json(reference_run / "summary.json").get("details", {}).get("output_checksum")
    candidate_results: list[dict[str, object]] = []
    overall_pass = True
    for candidate_strategy in resolved_candidates:
        candidate_root = candidates_root / str(candidate_strategy)
        run_name = f"c2_{candidate_strategy}"
        config_payload = build_policy_correctness_config(
            base_comparison=base,
            strategy_name=str(candidate_strategy),
            run_name=run_name,
            output_root=candidate_root,
            profile=resolved_profile,
            selected_layers=resolved_selected_layers,
            save_logits=True,
            world_size=int(resolved_world_size),
        )
        config_path = generated_dir / f"{run_name}.yaml"
        dump_yaml(config_path, config_payload)
        native = candidate_strategy in {"native", "disabled"}
        cmd = torchrun_policy_command(
            config_path=config_path,
            run_id=run_name,
            output_dir=candidate_root,
            world_size=int(resolved_world_size),
            native=native,
        )
        proc = run_subprocess(cmd, extra_env={"ROUTERSENSE_WARMUP_ITERS": "0", "ROUTERSENSE_MEASURE_ITERS": str(int(resolved_forward_epochs))})
        (output_dir / f"{run_name}_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (output_dir / f"{run_name}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            blocked = _maybe_load_environment_block(candidate_root / run_name, proc.stdout)
            if blocked is not None:
                payload.update(
                    {
                        "status": "blocked_environment",
                        "blocked_stage": run_name,
                        "blocked_environment": blocked,
                        "result_eligible_for_performance_comparison": False,
                        "fallback_used": True,
                        "failed_command": cmd,
                        "returncode": int(proc.returncode),
                    }
                )
                write_json(output_dir / "c2_runner_summary.json", payload)
                write_runner_result_bundle(output_dir, runner_name="run_gpu_c2_async_correctness", payload=payload, run_kind="GPU_CORRECTNESS")
                print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
                return 0
            payload.update({"status": f"{run_name}_failed", "failed_command": cmd, "returncode": int(proc.returncode)})
            write_json(output_dir / "c2_runner_summary.json", payload)
            print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
            return int(proc.returncode)
        candidate_run = candidate_root / run_name
        candidate_logits = _load_logits(candidate_run)
        per_rank = []
        compared_epochs: set[int] = set()
        for key, ref_tensor in sorted(reference_logits.items()):
            if key not in candidate_logits:
                payload.update({"status": "candidate_missing_logits", "candidate_strategy": candidate_strategy, "missing_key": list(key)})
                write_json(output_dir / "c2_runner_summary.json", payload)
                write_runner_result_bundle(output_dir, runner_name="run_gpu_c2_async_correctness", payload=payload, run_kind="GPU_CORRECTNESS")
                print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
                return 1
            rank, epoch = key
            cand_tensor = candidate_logits[key]
            compared_epochs.add(int(epoch))
            per_rank.append({"rank": rank, "forward_epoch": epoch, **_compare_tensors(ref_tensor, cand_tensor)})
        max_abs = max((float(item["max_abs_error"]) for item in per_rank), default=0.0)
        max_rel = max((float(item["max_rel_error"]) for item in per_rank), default=0.0)
        mean_abs = float(sum(float(item["max_abs_error"]) for item in per_rank) / len(per_rank)) if per_rank else 0.0
        mismatch_count = sum(int(item["mismatch_count"]) for item in per_rank)
        candidate_summary = read_json(candidate_run / "summary.json").get("details", {})
        candidate_rank_summaries = _load_rank_summaries(candidate_run)
        if len(candidate_rank_summaries) != int(resolved_world_size):
            payload.update({"status": "candidate_rank_summary_missing", "candidate_strategy": candidate_strategy})
            write_json(output_dir / "c2_runner_summary.json", payload)
            write_runner_result_bundle(output_dir, runner_name="run_gpu_c2_async_correctness", payload=payload, run_kind="GPU_CORRECTNESS")
            print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
            return 1
        candidate_rank_summary = candidate_rank_summaries[0]
        checks = {
            "transport_invoked_if_mutating": (
                str(candidate_strategy) == "native"
                or all(int(item.get("selected_transport_execution_count", 0) or 0) > 0 for item in candidate_rank_summaries)
            ),
            "p2p_call_count_if_async": (
                str(candidate_strategy) in {"native", "birkhoff_phase_local_sync"}
                or all(int(item.get("batch_isend_irecv_call_count", 0) or 0) > 0 for item in candidate_rank_summaries)
            ),
            "real_send_op_count_if_async": (
                str(candidate_strategy) in {"native", "birkhoff_phase_local_sync"}
                or all(int(item.get("real_send_op_count", 0) or 0) > 0 for item in candidate_rank_summaries)
            ),
            "real_recv_op_count_if_async": (
                str(candidate_strategy) in {"native", "birkhoff_phase_local_sync"}
                or all(int(item.get("real_recv_op_count", 0) or 0) > 0 for item in candidate_rank_summaries)
            ),
            "fallback_count_zero": all(int(item.get("phase_sync_fallback_count", item.get("fallback_count", 0)) or 0) == 0 for item in candidate_rank_summaries),
            "timeout_count_zero": all("timeout_count" in item and int(item.get("timeout_count", 0) or 0) == 0 for item in candidate_rank_summaries),
            "all_work_completed": all("all_work_completed" in item and bool(item.get("all_work_completed", False)) for item in candidate_rank_summaries),
            "all_ranks_completed": all(bool(item.get("forward_completed") or item.get("forward_partial_stop")) for item in candidate_rank_summaries),
            "selected_layer_match_count": int(candidate_rank_summary.get("selected_layer_match_count", 0) or 0) > 0,
            "selected_transport_execution_count": (
                str(candidate_strategy) == "native"
                or int(candidate_rank_summary.get("selected_transport_execution_count", 0) or 0) > 0
            ),
            "shape_parity": all(bool(item["shape_parity"]) for item in per_rank),
            "dtype_parity": all(bool(item["dtype_parity"]) for item in per_rank),
            "nan_inf_zero": (sum(int(item["nan_count"]) for item in per_rank) == 0 and sum(int(item["inf_count"]) for item in per_rank) == 0),
            "tolerance": max_abs <= 5e-3 and max_rel <= 5e-2 and mismatch_count == 0,
            "two_forward_epochs": len(compared_epochs) >= 2,
        }
        candidate_pass = all(bool(v) for v in checks.values())
        overall_pass = overall_pass and candidate_pass
        candidate_results.append(
            {
                "candidate_strategy": str(candidate_strategy),
                "status": "passed" if candidate_pass else "failed",
                "reference_checksum": reference_checksum,
                "candidate_checksum": candidate_summary.get("output_checksum"),
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "mean_abs_error": mean_abs,
                "mismatch_count": mismatch_count,
                "shape_parity": all(bool(item["shape_parity"]) for item in per_rank),
                "dtype_parity": all(bool(item["dtype_parity"]) for item in per_rank),
                "nan_count": sum(int(item["nan_count"]) for item in per_rank),
                "inf_count": sum(int(item["inf_count"]) for item in per_rank),
                "per_rank": per_rank,
                "async_executor_invocation_count": candidate_rank_summary.get("async_executor_invocation_count", 0),
                "p2p_call_count": candidate_rank_summary.get("batch_isend_irecv_call_count", 0),
                "real_send_op_count": candidate_rank_summary.get("real_send_op_count", 0),
                "real_recv_op_count": candidate_rank_summary.get("real_recv_op_count", 0),
                "fallback_count": candidate_rank_summary.get("phase_sync_fallback_count", candidate_rank_summary.get("fallback_count", 0)),
                "timeout_count": candidate_rank_summary.get("timeout_count", 0),
                "all_work_completed": candidate_rank_summary.get("all_work_completed", True),
                "selected_layer_match_count": candidate_rank_summary.get("selected_layer_match_count", 0),
                "selected_transport_execution_count": candidate_rank_summary.get("selected_transport_execution_count", 0),
                "rank_summaries_present": len(candidate_rank_summaries),
                "forward_epochs_compared": sorted(compared_epochs),
                "parity_passed": bool(checks["tolerance"]),
                "checks": checks,
            }
        )
    payload.update(
        {
            "status": "passed" if overall_pass else "failed",
            "reference_checksum": reference_checksum,
            "candidates": candidate_results,
        }
    )
    write_json(output_dir / "c2_runner_summary.json", payload)
    write_runner_result_bundle(output_dir, runner_name="run_gpu_c2_async_correctness", payload=payload, run_kind="GPU_CORRECTNESS")
    print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
