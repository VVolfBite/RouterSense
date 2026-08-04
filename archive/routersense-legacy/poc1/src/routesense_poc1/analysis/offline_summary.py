from __future__ import annotations

from pathlib import Path


def write_offline_summary(summary: dict, path: Path) -> None:
    lines = [
        "# RouteSense PoC1 Offline Summary",
        "",
        "## Core",
        "",
        f"- Pairwise ranking accuracy: {summary.get('pairwise_ranking_accuracy')}",
        f"- Pairwise comparisons: {summary.get('pairwise_total')}",
    ]

    strategies = summary.get("strategies", {})
    if strategies:
        lines.extend(["", "## Strategy Snapshot", ""])
        for key in ["random", "raw_routing", "calibrated", "oracle"]:
            payload = strategies.get(key)
            if not payload:
                continue
            lines.append(
                f"- {key}: mean={payload['mean_delta_nll']:.6f}, "
                f"p95={payload['p95_delta_nll']:.6f}, "
                f"safe@0.01={payload['deferrable_ratio']['0.01']:.4f}"
            )

    conditional = summary.get("conditional_analysis", {})
    if conditional:
        lines.extend(["", "## Conditional Findings", ""])
        entropy = conditional.get("context_entropy", {})
        if entropy:
            lines.append(
                f"- Entropy split: high={entropy['high_entropy']['pairwise_accuracy']}, "
                f"low={entropy['low_entropy']['pairwise_accuracy']}"
            )
        gap = conditional.get("top1_dominance", {})
        if gap:
            lines.append(
                f"- Gap split: small={gap['small_gap']['pairwise_accuracy']}, "
                f"large={gap['large_gap']['pairwise_accuracy']}"
            )
        magnitude = conditional.get("delta_magnitude", {})
        if magnitude:
            lines.append(
                f"- Delta magnitude split: large={magnitude['large_magnitude_groups']['pairwise_accuracy']}, "
                f"small={magnitude['small_magnitude_groups']['pairwise_accuracy']}"
            )

    oracle_subset = summary.get("oracle_subset_analysis")
    if oracle_subset:
        lines.extend(["", "## Oracle Hard Subset", ""])
        lines.append(
            f"- top32 pairwise={oracle_subset['pairwise_accuracy']}, "
            f"oracle_mean_delta={oracle_subset['oracle_mean_delta_nll']}, "
            f"random_mean_delta={oracle_subset['random_mean_delta_nll']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Current signal is weak globally.",
            "- Subgroup signal is stronger in high-entropy and small-gap cases.",
            "- Oracle still shows substantial headroom relative to current observable routing features.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
