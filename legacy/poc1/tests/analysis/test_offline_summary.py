from __future__ import annotations

from pathlib import Path

from routesense_poc1.analysis.offline_summary import write_offline_summary


def test_write_offline_summary(tmp_path: Path) -> None:
    summary = {
        "pairwise_ranking_accuracy": 0.5,
        "pairwise_total": 10,
        "strategies": {
            "random": {"mean_delta_nll": 0.01, "p95_delta_nll": 0.1, "deferrable_ratio": {"0.01": 0.6}},
            "raw_routing": {"mean_delta_nll": 0.02, "p95_delta_nll": 0.2, "deferrable_ratio": {"0.01": 0.5}},
            "oracle": {"mean_delta_nll": -0.1, "p95_delta_nll": -0.01, "deferrable_ratio": {"0.01": 0.9}},
        },
        "conditional_analysis": {
            "context_entropy": {
                "high_entropy": {"pairwise_accuracy": 0.55},
                "low_entropy": {"pairwise_accuracy": 0.45},
            },
            "top1_dominance": {
                "small_gap": {"pairwise_accuracy": 0.52},
                "large_gap": {"pairwise_accuracy": 0.44},
            },
            "delta_magnitude": {
                "large_magnitude_groups": {"pairwise_accuracy": 0.41},
                "small_magnitude_groups": {"pairwise_accuracy": 0.52},
            },
        },
        "oracle_subset_analysis": {
            "pairwise_accuracy": 0.4,
            "oracle_mean_delta_nll": -0.2,
            "random_mean_delta_nll": -0.01,
        },
    }
    path = tmp_path / "offline_summary.md"
    write_offline_summary(summary, path)
    content = path.read_text(encoding="utf-8")
    assert "Conditional Findings" in content
    assert "Oracle Hard Subset" in content
