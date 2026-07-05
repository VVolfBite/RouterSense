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

from rs.core.artifact import write_json


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor_path(rank_summary: dict) -> Path:
    value = rank_summary.get("logits_path")
    if not value:
        raise ValueError("Missing logits_path in rank summary; rerun smoke with --save-logits")
    return Path(str(value))


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    cosine = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "cosine_similarity": float(cosine),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    args = parser.parse_args(argv)

    baseline = _load_summary(Path(args.baseline_summary))
    candidate = _load_summary(Path(args.candidate_summary))

    rank_pairs = zip(
        baseline["details"]["rank_summaries"],
        candidate["details"]["rank_summaries"],
        strict=True,
    )
    per_rank = []
    overall_max = 0.0
    overall_mean = 0.0
    overall_cos = 1.0
    count = 0
    for base_rank, cand_rank in rank_pairs:
        base_tensor = torch.load(_tensor_path(base_rank), map_location="cpu")
        cand_tensor = torch.load(_tensor_path(cand_rank), map_location="cpu")
        metrics = _compare_tensors(base_tensor, cand_tensor)
        per_rank.append(
            {
                "rank": int(base_rank["rank"]),
                "baseline_device": base_rank.get("device"),
                "candidate_device": cand_rank.get("device"),
                **metrics,
            }
        )
        overall_max = max(overall_max, metrics["max_abs_error"])
        overall_mean += metrics["mean_abs_error"]
        overall_cos = min(overall_cos, metrics["cosine_similarity"])
        count += 1

    overall_mean = overall_mean / max(count, 1)
    passed = overall_max <= args.atol and overall_cos >= args.cosine_threshold
    payload = {
        "baseline_summary": str(Path(args.baseline_summary)),
        "candidate_summary": str(Path(args.candidate_summary)),
        "candidate_name": args.candidate_name,
        "atol": args.atol,
        "cosine_threshold": args.cosine_threshold,
        "facade_noop_equivalence_passed": passed,
        "max_abs_error": overall_max,
        "mean_abs_error": overall_mean,
        "cosine_similarity": overall_cos,
        "per_rank": per_rank,
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
