#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_SUMMARY = ROOT / "outputs" / "poc1_trace_batch" / "trace_batch_summary.json"
DEFAULT_PROXY_SUMMARY = ROOT / "outputs" / "poc1_proxy_compare" / "proxy_comparison_summary.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "poc1_proxy_compare" / "proxy_result_verification.json"


def main() -> int:
    trace_summary_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACE_SUMMARY
    proxy_summary_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PROXY_SUMMARY
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUTPUT

    trace_summary = _load_json(trace_summary_path)
    proxy_summary = _load_json(proxy_summary_path)
    samples = proxy_summary.get("samples", [])
    recomputed = _recompute(samples)
    reported = proxy_summary.get("aggregate", {})
    checks = _compare(recomputed, reported)
    verified = all(item["matches"] for item in checks.values())

    payload = {
        "input_files": {
            "trace_batch_summary": str(trace_summary_path),
            "proxy_comparison_summary": str(proxy_summary_path),
        },
        "trace_sample_count": trace_summary.get("sample_count"),
        "proxy_sample_count": proxy_summary.get("sample_count"),
        "recomputed_metrics": recomputed,
        "reported_metrics": {key: reported.get(key) for key in recomputed},
        "checks": checks,
        "conclusion": "verified" if verified else "mismatch",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")
    print(payload["conclusion"])
    return 0 if verified else 1


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _recompute(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(samples)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "state_dependency_top1_different_rate": 0.0,
            "full_state_top1_different_rate": 0.0,
            "mean_state_dependency_topk_jaccard": None,
            "mean_state_full_topk_jaccard": None,
            "divergence_larger_in_high_skew_samples": False,
        }
    skew_values = [float(sample.get("max_to_avg_bucket_ratio") or 0.0) for sample in samples]
    skew_threshold = sorted(skew_values)[len(skew_values) // 2] if len(skew_values) % 2 else (
        sorted(skew_values)[len(skew_values) // 2 - 1] + sorted(skew_values)[len(skew_values) // 2]
    ) / 2.0
    high_skew = [sample for sample in samples if float(sample.get("max_to_avg_bucket_ratio") or 0.0) >= skew_threshold]
    low_skew = [sample for sample in samples if float(sample.get("max_to_avg_bucket_ratio") or 0.0) < skew_threshold]
    high_rate = _top1_diff_rate(high_skew)
    low_rate = _top1_diff_rate(low_skew)
    return {
        "sample_count": sample_count,
        "state_dependency_top1_different_rate": _top1_diff_rate(samples),
        "full_state_top1_different_rate": sum(1 for sample in samples if not sample["full_state_top1_same"]) / sample_count,
        "mean_state_dependency_topk_jaccard": fmean(sample["state_dependency_topk_jaccard"] for sample in samples),
        "mean_state_full_topk_jaccard": fmean(sample["state_full_topk_jaccard"] for sample in samples),
        "high_skew_state_dependency_top1_different_rate": high_rate,
        "low_skew_state_dependency_top1_different_rate": low_rate,
        "divergence_larger_in_high_skew_samples": high_rate > low_rate,
    }


def _top1_diff_rate(samples: list[dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    return sum(1 for sample in samples if not sample["state_dependency_top1_same"]) / len(samples)


def _compare(recomputed: dict[str, Any], reported: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for key, value in recomputed.items():
        reported_value = reported.get(key) if key != "sample_count" else reported.get(key)
        if reported_value is None and key == "sample_count":
            reported_value = value
        if isinstance(value, bool):
            delta = None
            matches = value == reported_value
        elif isinstance(value, (int, float)) and isinstance(reported_value, (int, float)):
            delta = abs(float(value) - float(reported_value))
            matches = delta <= 1e-9
        else:
            delta = None
            matches = value == reported_value
        checks[key] = {
            "recomputed": value,
            "reported": reported_value,
            "delta": delta,
            "matches": matches,
        }
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
