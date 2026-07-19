#!/usr/bin/env python3
"""Evaluate lightweight offline traffic predictors over replay fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.offline.prediction import rolling_predictor_records, summarize_prediction_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-summary-md", required=True)
    parser.add_argument("--predictor", choices=("zero_hint", "copy_current_dispatch", "fate_style_history", "fate_style_linear"), required=True)
    return parser.parse_args()


def _render_md(predictor: str, summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Predictor Evaluation",
            "",
            f"- predictor: `{predictor}`",
            f"- record_count: {summary['record_count']}",
            f"- mean_relative_l1_error: {summary['mean_relative_l1_error']}",
            f"- mean_cosine_similarity: {summary['mean_cosine_similarity']}",
            f"- mean_topk_edge_overlap: {summary['mean_topk_edge_overlap']}",
            f"- mean_nonzero_precision: {summary['mean_nonzero_precision']}",
            f"- mean_nonzero_recall: {summary['mean_nonzero_recall']}",
            "",
        ]
    ) + "\n"


def main() -> None:
    args = _parse_args()
    records = rolling_predictor_records(fixture_dir=Path(args.fixture_dir), predictor_name=str(args.predictor))
    summary = summarize_prediction_records(records)
    payload = {"predictor_name": str(args.predictor), "summary": summary, "records": [record.to_dict() for record in records]}
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_summary_md).write_text(_render_md(str(args.predictor), summary), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
