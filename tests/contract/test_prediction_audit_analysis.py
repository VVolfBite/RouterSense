from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_analyze_prediction_audit_cli(tmp_path: Path) -> None:
    audit_path = tmp_path / "rank0_prediction_audit.jsonl"
    audit_rows = [
        {
            "predictor_name": "copy_current_dispatch",
            "predicted_layer_id": "1",
            "relative_l1_error": 0.1,
            "cosine_similarity": 0.9,
            "topk_edge_overlap": 0.75,
            "nonzero_edge_precision": 1.0,
            "nonzero_edge_recall": 0.8,
        },
        {
            "predictor_name": "copy_current_dispatch",
            "predicted_layer_id": "2",
            "relative_l1_error": 0.2,
            "cosine_similarity": 0.8,
            "topk_edge_overlap": 0.5,
            "nonzero_edge_precision": 0.9,
            "nonzero_edge_recall": 0.7,
        },
    ]
    audit_path.write_text("\n".join(json.dumps(row) for row in audit_rows), encoding="utf-8")
    output_path = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.analyze_prediction_audit",
            "--audit",
            str(audit_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["record_count"] == 2
    assert payload["mean_relative_l1_error"] == 0.15000000000000002
    assert payload["mean_cosine_similarity"] == 0.8500000000000001
    assert "1" in payload["per_layer"]
    assert payload["per_predictor"]["copy_current_dispatch"]["record_count"] == 2
