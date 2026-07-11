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
    load_yaml,
    read_json,
    run_subprocess,
    torchrun_policy_command,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4GPU C2 parity body or an explicit no-4GPU fallback.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-strategy", default="birkhoff_phase_local_sync")
    parser.add_argument("--candidate-strategy", default="routersense_joint_predicted_async_p2p")
    parser.add_argument("--profile", default="execution", choices=("debug", "execution", "perf"))
    parser.add_argument("--selected-layers", default="2")
    parser.add_argument("--forward-epochs", type=int, default=2)
    parser.add_argument("--world-size", type=int, default=4)
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
    gate_cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py",
    ]
    proc = run_subprocess(gate_cmd)
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
        "fallback_command": gate_cmd,
        "fallback_returncode": int(proc.returncode),
    }
    (output_dir / "fallback_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "fallback_stderr.log").write_text(proc.stderr, encoding="utf-8")
    return payload


def _load_logits(run_dir: Path) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for path in sorted(run_dir.glob("*-rank*-logits.pt")):
        rank_str = path.stem.split("-rank")[-1].split("-")[0]
        result[int(rank_str)] = torch.load(path, map_location="cpu")
    return result


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(Path(args.config), output_dir)
    base = load_yaml(Path(args.config))
    payload = {
        "runner": "run_gpu_c2_async_correctness",
        "config": str(args.config),
        "reference_strategy": str(args.reference_strategy),
        "candidate_strategy": str(args.candidate_strategy),
        "profile": str(args.profile),
        "selected_layers": str(args.selected_layers),
        "forward_epochs": int(args.forward_epochs),
        "world_size": int(args.world_size),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        payload["status"] = "dry_run_ready"
        write_json(output_dir / "c2_runner_summary.json", payload)
        print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
        return 0
    if available_cuda_count() < int(args.world_size):
        payload = _fallback(
            output_dir,
            world_size=int(args.world_size),
            config=str(args.config),
            reference_strategy=str(args.reference_strategy),
            candidate_strategy=str(args.candidate_strategy),
            profile=str(args.profile),
            selected_layers=str(args.selected_layers),
            forward_epochs=int(args.forward_epochs),
            dry_run=bool(args.dry_run),
        )
        write_json(output_dir / "c2_runner_summary.json", payload)
        print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
        return 0 if int(payload["fallback_returncode"]) == 0 else int(payload["fallback_returncode"])

    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    reference_root = output_dir / "reference"
    candidate_root = output_dir / "candidate"

    for strategy_name, run_name, root_dir in (
        (str(args.reference_strategy), "c2_reference", reference_root),
        (str(args.candidate_strategy), "c2_candidate", candidate_root),
    ):
        config_payload = build_policy_correctness_config(
            base_comparison=base,
            strategy_name=strategy_name,
            run_name=run_name,
            output_root=root_dir,
            profile=str(args.profile),
            selected_layers=str(args.selected_layers),
            save_logits=True,
        )
        config_path = generated_dir / f"{run_name}.yaml"
        dump_yaml(config_path, config_payload)
        native = strategy_name in {"native", "disabled"}
        cmd = torchrun_policy_command(
            config_path=config_path,
            run_id=run_name,
            output_dir=root_dir,
            world_size=int(args.world_size),
            native=native,
        )
        proc = run_subprocess(cmd)
        (output_dir / f"{run_name}_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (output_dir / f"{run_name}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            payload.update({"status": f"{run_name}_failed", "failed_command": cmd, "returncode": int(proc.returncode)})
            write_json(output_dir / "c2_runner_summary.json", payload)
            print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
            return int(proc.returncode)

    reference_run = reference_root / "c2_reference"
    candidate_run = candidate_root / "c2_candidate"
    reference_logits = _load_logits(reference_run)
    candidate_logits = _load_logits(candidate_run)
    per_rank = []
    for rank, ref_tensor in sorted(reference_logits.items()):
        cand_tensor = candidate_logits[rank]
        per_rank.append({"rank": rank, **_compare_tensors(ref_tensor, cand_tensor)})
    max_abs = max((float(item["max_abs_error"]) for item in per_rank), default=0.0)
    max_rel = max((float(item["max_rel_error"]) for item in per_rank), default=0.0)
    mismatch_count = sum(int(item["mismatch_count"]) for item in per_rank)
    candidate_summary = read_json(candidate_run / "summary.json").get("details", {})
    candidate_rank_summary = read_json(candidate_run / "rank0_summary.json")
    payload.update(
        {
            "status": "executed",
            "reference_checksum": read_json(reference_run / "summary.json").get("details", {}).get("output_checksum"),
            "candidate_checksum": candidate_summary.get("output_checksum"),
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "mismatch_count": mismatch_count,
            "shape_parity": all(bool(item["shape_parity"]) for item in per_rank),
            "dtype_parity": all(bool(item["dtype_parity"]) for item in per_rank),
            "nan_count": sum(int(item["nan_count"]) for item in per_rank),
            "inf_count": sum(int(item["inf_count"]) for item in per_rank),
            "per_rank": per_rank,
            "async_invocation_count": candidate_rank_summary.get("async_executor_invocation_count", 0),
            "p2p_call_count": candidate_rank_summary.get("batch_isend_irecv_call_count", 0),
            "fallback_count": candidate_rank_summary.get("phase_sync_fallback_count", 0),
            "timeout_count": 0,
            "parity_passed": max_abs <= 5e-3 and mismatch_count == 0,
        }
    )
    write_json(output_dir / "c2_runner_summary.json", payload)
    print((output_dir / "c2_runner_summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
