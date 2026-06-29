from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from .policies import select_deferrable_expert
from .schemas import AblationRecord


def analyze_records(records: list[AblationRecord], calibrator=None) -> dict:
    grouped: dict[tuple[int, int], list[AblationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.window_id, record.layer_id)].append(record)

    strategies = ["full", "random", "raw_routing", "calibrated", "oracle"]
    deltas_by_strategy: dict[str, list[float]] = {strategy: [] for strategy in strategies}
    pairwise_total = 0
    pairwise_correct = 0
    epsilons = [0.01, 0.05, 0.10]

    for group_records in grouped.values():
        truth_order = sorted(group_records, key=lambda record: (record.delta_nll, record.topk_rank))
        predicted_order = sorted(group_records, key=lambda record: (record.effective_gate_weight, record.topk_rank))
        for left_index in range(len(truth_order)):
            for right_index in range(left_index + 1, len(truth_order)):
                left = truth_order[left_index]
                right = truth_order[right_index]
                if abs(left.delta_nll - right.delta_nll) < 1e-6:
                    continue
                pairwise_total += 1
                if predicted_order.index(left) < predicted_order.index(right):
                    pairwise_correct += 1

        for strategy in strategies:
            if strategy == "calibrated" and calibrator is None:
                continue
            if strategy == "full":
                deltas_by_strategy[strategy].append(0.0)
                continue
            if strategy == "random":
                deltas_by_strategy[strategy].append(mean(record.delta_nll for record in group_records))
                continue
            selected_expert_id = select_deferrable_expert(group_records, strategy, calibrator=calibrator)
            chosen = next(record for record in group_records if record.expert_id == selected_expert_id)
            deltas_by_strategy[strategy].append(chosen.delta_nll)

    strategy_summary: dict[str, dict] = {}
    for strategy, deltas in deltas_by_strategy.items():
        if not deltas:
            continue
        sorted_deltas = sorted(deltas)
        p95_index = min(len(sorted_deltas) - 1, int(round(0.95 * (len(sorted_deltas) - 1))))
        strategy_summary[strategy] = {
            "mean_delta_nll": mean(deltas),
            "median_delta_nll": median(deltas),
            "p95_delta_nll": sorted_deltas[p95_index],
            "deferrable_ratio": {
                str(epsilon): sum(1 for delta in deltas if delta <= epsilon) / len(deltas)
                for epsilon in epsilons
            },
            "num_groups": len(deltas),
        }

    return {
        "pairwise_ranking_accuracy": (pairwise_correct / pairwise_total) if pairwise_total else None,
        "pairwise_total": pairwise_total,
        "strategies": strategy_summary,
        "notes": {
            "quality_constrained_deferrable_ratio_semantics": (
                "ratio of window-layer groups where one selected route stays within epsilon delta_nll"
            )
        },
    }


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# RouteSense PoC1 Report",
        "",
        "## Summary",
        "",
        f"- Pairwise ranking accuracy: {summary.get('pairwise_ranking_accuracy')}",
        f"- Pairwise comparisons: {summary.get('pairwise_total')}",
        "",
        "## Strategies",
        "",
    ]
    for strategy, payload in summary.get("strategies", {}).items():
        lines.append(
            f"- {strategy}: mean={payload['mean_delta_nll']:.6f}, "
            f"median={payload['median_delta_nll']:.6f}, "
            f"p95={payload['p95_delta_nll']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 当前 PoC 每个 group 只取消一个 expert request，不直接代表最终通信节省比例。",
            "- 当前取消 route 代表请求暂缓后最终不释放的质量上限近似。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

