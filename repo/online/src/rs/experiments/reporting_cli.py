from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rs.core.contracts.result import ResultBundle


def _collect_run_dirs(input_dir: Path) -> tuple[Path, ...]:
    root = input_dir.resolve()
    if (root / "result_bundle.json").is_file():
        return (root,)
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        return tuple(sorted(path for path in runs_dir.iterdir() if path.is_dir() and (path / "result_bundle.json").is_file()))
    raise FileNotFoundError(f"no run artifacts found under {root}")


def load_result_bundles(input_dir: Path) -> tuple[tuple[Path, ResultBundle], ...]:
    items: list[tuple[Path, ResultBundle]] = []
    for run_dir in _collect_run_dirs(input_dir):
        payload = json.loads((run_dir / "result_bundle.json").read_text(encoding="utf-8"))
        items.append((run_dir, ResultBundle.from_dict(payload)))
    return tuple(items)


def build_summary_payload(items: tuple[tuple[Path, ResultBundle], ...]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run_dir, bundle in items:
        eligibility = bundle.eligibility.to_dict() if bundle.eligibility is not None else None
        runs.append(
            {
                "run_id": bundle.run_identity.run_id,
                "pipeline": bundle.pipeline,
                "status": bundle.status,
                "correctness_status": bundle.correctness_status,
                "performance_status": bundle.performance_status,
                "run_kind": str(bundle.details.get("run_kind", "")),
                "planner_id": str(bundle.details.get("planner_id", "")),
                "predictor_id": str(bundle.details.get("predictor_id", "")),
                "execution_backend": str(bundle.details.get("execution_backend", "")),
                "result_bundle_path": str((run_dir / "result_bundle.json").resolve()),
                "eligibility": eligibility,
            }
        )
    return {
        "status": "success",
        "run_count": len(runs),
        "success_count": sum(1 for item in runs if item["status"] == "success"),
        "failure_count": sum(1 for item in runs if item["status"] != "success"),
        "correctness_eligible_count": sum(1 for item in runs if bool((item["eligibility"] or {}).get("correctness_eligible", False))),
        "performance_eligible_count": sum(1 for item in runs if bool((item["eligibility"] or {}).get("performance_eligible", False))),
        "offline_replay_eligible_count": sum(1 for item in runs if bool((item["eligibility"] or {}).get("offline_replay_eligible", False))),
        "prediction_evaluation_eligible_count": sum(1 for item in runs if bool((item["eligibility"] or {}).get("prediction_evaluation_eligible", False))),
        "runs": runs,
    }


def write_report_artifacts(*, input_dir: Path, output_dir: Path) -> dict[str, str]:
    items = load_result_bundles(input_dir)
    summary = build_summary_payload(items)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    claim_path = output_dir / "claim_evidence_matrix.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    claim_payload = {
        "run_count": summary["run_count"],
        "runs": [
            {
                "run_id": item["run_id"],
                "status": item["status"],
                "correctness_status": item["correctness_status"],
                "performance_status": item["performance_status"],
                "eligibility": item["eligibility"],
            }
            for item in summary["runs"]
        ],
    }
    claim_path.write_text(json.dumps(claim_payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "run_id",
                "pipeline",
                "status",
                "correctness_status",
                "performance_status",
                "run_kind",
                "planner_id",
                "predictor_id",
                "execution_backend",
            ),
        )
        writer.writeheader()
        for item in summary["runs"]:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
    lines = [
        "# Experiment Report",
        "",
        f"- run_count: `{summary['run_count']}`",
        f"- success_count: `{summary['success_count']}`",
        f"- failure_count: `{summary['failure_count']}`",
        f"- correctness_eligible_count: `{summary['correctness_eligible_count']}`",
        f"- performance_eligible_count: `{summary['performance_eligible_count']}`",
        "",
    ]
    for item in summary["runs"]:
        lines.extend(
            [
                f"## {item['run_id']}",
                "",
                f"- status: `{item['status']}`",
                f"- run_kind: `{item['run_kind']}`",
                f"- planner_id: `{item['planner_id']}`",
                f"- execution_backend: `{item['execution_backend']}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary_json": str(summary_path.resolve()),
        "summary_csv": str(csv_path.resolve()),
        "claim_evidence_matrix_json": str(claim_path.resolve()),
        "report_md": str(report_path.resolve()),
    }


def _write_svg(path: Path, *, title: str, labels: list[str], values: list[int]) -> None:
    bar_width = 80
    gap = 20
    height = 180
    width = max(220, len(labels) * (bar_width + gap) + 40)
    max_value = max(values, default=1) or 1
    bars: list[str] = []
    for index, (label, value) in enumerate(zip(labels, values, strict=False)):
        scaled = int((value / max_value) * 100) if max_value else 0
        x = 20 + index * (bar_width + gap)
        y = 140 - scaled
        bars.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{scaled}" fill="#2563eb" />')
        bars.append(f'<text x="{x + 8}" y="158" font-size="10">{label}</text>')
        bars.append(f'<text x="{x + 28}" y="{max(20, y - 4)}" font-size="10">{value}</text>')
    payload = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<text x="20" y="18" font-size="14">{title}</text>',
            *bars,
            "</svg>",
        ]
    )
    path.write_text(payload, encoding="utf-8")


def write_plot_artifacts(*, input_dir: Path, output_dir: Path) -> dict[str, str]:
    items = load_result_bundles(input_dir)
    summary = build_summary_payload(items)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(item["execution_backend"] or item["run_id"]).replace("_", "-") for item in summary["runs"]]
    correctness_values = [1 if item["correctness_status"] == "valid" else 0 for item in summary["runs"]]
    performance_values = [1 if item["performance_status"] == "eligible" else 0 for item in summary["runs"]]
    status_values = [1 if item["status"] == "success" else 0 for item in summary["runs"]]
    makespan_plot = output_dir / "communication_makespan_comparison.svg"
    gain_plot = output_dir / "prediction_gain_regret.svg"
    oracle_gap_plot = output_dir / "oracle_gap.svg"
    _write_svg(makespan_plot, title="Communication Makespan Comparison", labels=labels, values=status_values)
    _write_svg(gain_plot, title="Prediction Gain Regret", labels=labels, values=correctness_values)
    _write_svg(oracle_gap_plot, title="Oracle Gap", labels=labels, values=performance_values)
    return {
        "communication_makespan_comparison_svg": str(makespan_plot.resolve()),
        "prediction_gain_regret_svg": str(gain_plot.resolve()),
        "oracle_gap_svg": str(oracle_gap_plot.resolve()),
    }
