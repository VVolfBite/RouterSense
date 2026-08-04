from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.online.run_noop_equivalence import main as compare_main


def test_noop_equivalence_contract(tmp_path: Path) -> None:
    baseline_tensor = tmp_path / "baseline.pt"
    candidate_tensor = tmp_path / "candidate.pt"
    torch.save(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), baseline_tensor)
    torch.save(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), candidate_tensor)

    baseline_summary = tmp_path / "baseline.json"
    candidate_summary = tmp_path / "candidate.json"
    baseline_summary.write_text(
        json.dumps(
            {
                "details": {
                    "rank_summaries": [
                        {"rank": 0, "device": "cuda:0", "logits_path": str(baseline_tensor)}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    candidate_summary.write_text(
        json.dumps(
            {
                "details": {
                    "rank_summaries": [
                        {"rank": 0, "device": "cuda:0", "logits_path": str(candidate_tensor)}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "equivalence.json"
    status = compare_main(
        [
            "--baseline-summary",
            str(baseline_summary),
            "--candidate-summary",
            str(candidate_summary),
            "--candidate-name",
            "native_order",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["facade_noop_equivalence_passed"] is True
    assert payload["max_abs_error"] == 0.0
