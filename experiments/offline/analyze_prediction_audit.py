#!/usr/bin/env python3
"""Summarize online prediction-audit JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", default=[])
    parser.add_argument("--audit-dir", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _audit_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.audit if value]
    if args.audit_dir:
        paths.extend(sorted(Path(args.audit_dir).glob("rank*_prediction_audit.jsonl")))
    if not paths:
        raise SystemExit("at least one --audit or an --audit-dir containing rank*_prediction_audit.jsonl is required")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize_prediction_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_predictor[str(row.get("predictor_name", "unknown"))].append(row)
        by_layer[str(row.get("predicted_layer_id", ""))].append(row)
    record_count = len(rows)
    result = {
        "record_count": record_count,
        "mean_relative_l1_error": _mean([float(row.get("relative_l1_error", 0.0) or 0.0) for row in rows]),
        "median_relative_l1_error": statistics.median([float(row.get("relative_l1_error", 0.0) or 0.0) for row in rows]) if rows else 0.0,
        "mean_cosine_similarity": _mean([float(row.get("cosine_similarity", 0.0) or 0.0) for row in rows]),
        "mean_topk_edge_overlap": _mean([float(row.get("topk_edge_overlap", 0.0) or 0.0) for row in rows]),
        "mean_nonzero_precision": _mean([float(row.get("nonzero_edge_precision", 0.0) or 0.0) for row in rows]),
        "mean_nonzero_recall": _mean([float(row.get("nonzero_edge_recall", 0.0) or 0.0) for row in rows]),
        "mean_predicted_remote_bytes": _mean([float(row.get("predicted_remote_bytes", 0.0) or 0.0) for row in rows]),
        "mean_actual_remote_bytes": _mean([float(row.get("actual_remote_bytes", 0.0) or 0.0) for row in rows]),
        "mean_predicted_self_bytes": _mean([float(row.get("predicted_self_bytes", 0.0) or 0.0) for row in rows]),
        "mean_actual_self_bytes": _mean([float(row.get("actual_self_bytes", 0.0) or 0.0) for row in rows]),
        "per_predictor": {},
        "per_layer": {},
    }
    for predictor_name, predictor_rows in by_predictor.items():
        result["per_predictor"][predictor_name] = {
            "record_count": len(predictor_rows),
            "mean_relative_l1_error": _mean([float(row.get("relative_l1_error", 0.0) or 0.0) for row in predictor_rows]),
            "mean_cosine_similarity": _mean([float(row.get("cosine_similarity", 0.0) or 0.0) for row in predictor_rows]),
            "mean_topk_edge_overlap": _mean([float(row.get("topk_edge_overlap", 0.0) or 0.0) for row in predictor_rows]),
        }
    for layer_id, layer_rows in by_layer.items():
        result["per_layer"][layer_id] = {
            "record_count": len(layer_rows),
            "mean_relative_l1_error": _mean([float(row.get("relative_l1_error", 0.0) or 0.0) for row in layer_rows]),
            "mean_cosine_similarity": _mean([float(row.get("cosine_similarity", 0.0) or 0.0) for row in layer_rows]),
            "mean_topk_edge_overlap": _mean([float(row.get("topk_edge_overlap", 0.0) or 0.0) for row in layer_rows]),
        }
    return result


def main() -> None:
    args = _parse_args()
    paths = _audit_paths(args)
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    payload = {
        "audit_paths": [str(path) for path in paths],
        **summarize_prediction_audits(rows),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
