#!/usr/bin/env python3
"""Render simple SVG bar charts from a strategy comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metric_mean(strategy: dict[str, Any], key: str) -> float:
    value = strategy.get("metrics", {}).get(key, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0) or 0.0)
    return float(value or 0.0)


def _svg_bar_chart(rows: list[tuple[str, float]], *, title: str, unit: str) -> str:
    width = 960
    row_h = 34
    label_w = 260
    bar_w = 560
    top = 70
    height = top + row_h * max(1, len(rows)) + 50
    max_v = max((value for _, value in rows), default=1.0) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="32" y="38" font-family="Georgia, serif" font-size="24" fill="#17211b">{_esc(title)}</text>',
        f'<text x="32" y="58" font-family="Verdana, sans-serif" font-size="12" fill="#5a655d">Unit: {_esc(unit)}. Generated from comparison_report.json.</text>',
    ]
    for idx, (name, value) in enumerate(rows):
        y = top + idx * row_h
        length = int((value / max_v) * bar_w) if max_v > 0 else 0
        color = "#1f6f5b" if idx == 0 else "#5f8f7d"
        parts.append(f'<text x="32" y="{y + 20}" font-family="Verdana, sans-serif" font-size="13" fill="#17211b">{_esc(name)}</text>')
        parts.append(f'<rect x="{label_w}" y="{y + 5}" width="{max(1, length)}" height="20" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{label_w + length + 8}" y="{y + 20}" font-family="Verdana, sans-serif" font-size="12" fill="#17211b">{value:.2f}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="comparison_report.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", action="append", default=None)
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    strategies = list(report.get("strategies", []) or [])
    metrics = args.metric or ["communication_makespan_us", "scheduling_overhead_us", "net_benefit_us"]
    for metric in metrics:
        rows = [(str(item.get("name", "unknown")), _metric_mean(item, metric)) for item in strategies]
        rows.sort(key=lambda item: item[1])
        unit = "us" if metric.endswith("_us") else "value"
        (out / f"{metric}.svg").write_text(_svg_bar_chart(rows, title=metric, unit=unit), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
