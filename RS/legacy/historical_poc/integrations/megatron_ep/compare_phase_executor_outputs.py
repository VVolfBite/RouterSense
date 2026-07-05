#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.megatron_ep.routersense.trace_writer import write_json, write_jsonl


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    cosine = torch.nn.functional.cosine_similarity(af.reshape(1, -1), bf.reshape(1, -1)).item()
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": float(cosine),
    }


def _collect_capture_index(run_dir: Path) -> dict[tuple[int, str, str, str], dict]:
    result: dict[tuple[int, str, str, str], dict] = {}
    for path in sorted((run_dir / "captured_phase_tensors").glob("rank*_layer*_*.pt")):
        stem = path.stem
        parts = stem.split("_")
        rank = int(parts[0].replace("rank", ""))
        layer = parts[1].replace("layer", "")
        phase = parts[2]
        role = "_".join(parts[3:])
        result[(rank, layer, phase, role)] = {"path": path}
    return result


def _resolve_artifact_path(run_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve() if not path.exists() else path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    args = parser.parse_args(argv)

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = _load_json(baseline_dir / "summary.json")
    candidate_summary = _load_json(candidate_dir / "summary.json")
    baseline_caps = _collect_capture_index(baseline_dir)
    candidate_caps = _collect_capture_index(candidate_dir)

    layout_rows: list[dict] = []
    overall_layout_pass = True
    for key in sorted(set(baseline_caps) & set(candidate_caps)):
        base = torch.load(baseline_caps[key]["path"], map_location="cpu")
        cand = torch.load(candidate_caps[key]["path"], map_location="cpu")
        metrics = _tensor_metrics(base, cand)
        rank, layer, phase, role = key
        row = {
            "rank": rank,
            "layer_id": layer,
            "phase": phase,
            "tensor_role": role,
            "shape_identical": list(base.shape) == list(cand.shape),
            "dtype_identical": str(base.dtype) == str(cand.dtype),
            **metrics,
        }
        row["passed"] = row["shape_identical"] and row["dtype_identical"] and metrics["max_abs_error"] == 0.0
        overall_layout_pass = overall_layout_pass and row["passed"]
        layout_rows.append(row)
    write_jsonl(output_dir / "layout_comparison.jsonl", layout_rows)

    baseline_ranks = baseline_summary["details"]["rank_summaries"]
    candidate_ranks = candidate_summary["details"]["rank_summaries"]
    overall_max = 0.0
    overall_mean = 0.0
    overall_cos = 1.0
    per_rank: list[dict] = []
    for base_rank, cand_rank in zip(baseline_ranks, candidate_ranks, strict=True):
        base_path = _resolve_artifact_path(baseline_dir, base_rank.get("logits_path"))
        cand_path = _resolve_artifact_path(candidate_dir, cand_rank.get("logits_path"))
        if base_path is None or cand_path is None:
            continue
        base = torch.load(base_path, map_location="cpu")
        cand = torch.load(cand_path, map_location="cpu")
        metrics = _tensor_metrics(base, cand)
        per_rank.append({"rank": int(base_rank["rank"]), **metrics})
        overall_max = max(overall_max, metrics["max_abs_error"])
        overall_mean += metrics["mean_abs_error"]
        overall_cos = min(overall_cos, metrics["cosine_similarity"])
    if per_rank:
        overall_mean /= len(per_rank)
    logits_status = "passed" if per_rank and overall_max <= args.atol and overall_cos >= args.cosine_threshold else (
        "failed" if per_rank else "not_applicable"
    )
    logits_payload = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "max_abs_error": overall_max,
        "mean_abs_error": overall_mean,
        "cosine_similarity": overall_cos,
        "status": logits_status,
        "per_rank": per_rank,
    }
    write_json(output_dir / "logit_comparison.json", logits_payload)
    write_json(
        output_dir / "comparison.json",
        {
            "layout_passed": overall_layout_pass,
            "logits_status": logits_payload["status"],
            "required_layout_checks": len(layout_rows),
            "required_logit_checks": len(per_rank),
        },
    )
    logits_ok = logits_payload["status"] in {"passed", "not_applicable"}
    return 0 if overall_layout_pass and logits_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
