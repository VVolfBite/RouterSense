from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_plot_strategy_comparison_generates_svg(tmp_path: Path) -> None:
    report = {
        "strategies": [
            {"name": "disabled", "metrics": {"communication_makespan_us": {"mean": 100.0}}},
            {"name": "routersense_p0p1p2_hint", "metrics": {"communication_makespan_us": {"mean": 80.0}}},
        ]
    }
    report_path = tmp_path / "comparison_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    out_dir = tmp_path / "plots"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/plot/plot_strategy_comparison.py"),
            "--report",
            str(report_path),
            "--output-dir",
            str(out_dir),
            "--metric",
            "communication_makespan_us",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    svg = (out_dir / "communication_makespan_us.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "routersense_p0p1p2_hint" in svg
