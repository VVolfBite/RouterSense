from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Callable

from .policies import SINGLE_FACTOR_STRATEGIES, select_deferrable_expert
from .schemas import AblationRecord


STRATEGIES = [
    "full",
    "random",
    "raw_routing",
    "router_probability_min",
    "effective_gate_weight_min",
    "router_logit_min",
    "topk_rank_max",
    "top1_top2_gap_min",
    "routing_entropy_max",
    "abs_router_logit_min",
    "calibrated",
    "oracle",
]

FEATURES: dict[str, Callable[[AblationRecord], float]] = {
    "effective_gate_weight": lambda record: float(record.effective_gate_weight),
    "router_probability": lambda record: float(record.router_probability),
    "router_logit": lambda record: float(record.router_logit),
    "abs_router_logit": lambda record: abs(float(record.router_logit)),
    "topk_rank": lambda record: float(record.topk_rank),
    "top1_top2_gap": lambda record: float(record.top1_top2_gap),
    "routing_entropy": lambda record: float(record.routing_entropy),
    "layer_id": lambda record: float(record.layer_id),
    "expert_id": lambda record: float(record.expert_id),
}


def _rankdata(values: list[float]) -> list[float]:
    pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(pairs):
        end = cursor + 1
        while end < len(pairs) and pairs[end][0] == pairs[cursor][0]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0 + 1.0
        for _, original_index in pairs[cursor:end]:
            ranks[original_index] = average_rank
        cursor = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / ((den_x * den_y) ** 0.5)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _sorted_deltas(deltas: list[float]) -> list[float]:
    return sorted(deltas)


def _p95(deltas: list[float]) -> float:
    sorted_deltas = _sorted_deltas(deltas)
    index = min(len(sorted_deltas) - 1, int(round(0.95 * (len(sorted_deltas) - 1))))
    return sorted_deltas[index]


def _group_records(records: list[AblationRecord]) -> dict[tuple[int, int], list[AblationRecord]]:
    grouped: dict[tuple[int, int], list[AblationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.window_id, record.layer_id)].append(record)
    return grouped


def _compute_pairwise_accuracy(
    grouped: dict[tuple[int, int], list[AblationRecord]],
    feature_fn: Callable[[AblationRecord], float],
    reverse: bool = False,
) -> dict[str, float | int | None]:
    total = 0
    correct = 0
    skipped_ties = 0
    for group_records in grouped.values():
        truth_order = sorted(group_records, key=lambda record: (record.delta_nll, record.topk_rank))
        predicted_order = sorted(
            group_records,
            key=lambda record: (feature_fn(record), record.topk_rank),
            reverse=reverse,
        )
        predicted_index = {id(record): index for index, record in enumerate(predicted_order)}
        for left_index in range(len(truth_order)):
            for right_index in range(left_index + 1, len(truth_order)):
                left = truth_order[left_index]
                right = truth_order[right_index]
                if abs(left.delta_nll - right.delta_nll) < 1e-6:
                    skipped_ties += 1
                    continue
                total += 1
                if predicted_index[id(left)] < predicted_index[id(right)]:
                    correct += 1
    return {
        "accuracy": (correct / total) if total else None,
        "pairwise_total": total,
        "pairwise_correct": correct,
        "pairwise_tie_skips": skipped_ties,
    }


def _strategy_summary(deltas: list[float], epsilons: list[float]) -> dict:
    return {
        "mean_delta_nll": mean(deltas),
        "median_delta_nll": median(deltas),
        "p95_delta_nll": _p95(deltas),
        "deferrable_ratio": {
            str(epsilon): sum(1 for delta in deltas if delta <= epsilon) / len(deltas)
            for epsilon in epsilons
        },
        "num_groups": len(deltas),
    }


def _factor_diagnostics(records: list[AblationRecord], grouped: dict[tuple[int, int], list[AblationRecord]]) -> dict:
    delta_values = [record.delta_nll for record in records]
    diagnostics: dict[str, dict] = {}
    orientation = {
        "effective_gate_weight": False,
        "router_probability": False,
        "router_logit": False,
        "abs_router_logit": False,
        "topk_rank": True,
        "top1_top2_gap": False,
        "routing_entropy": True,
        "layer_id": False,
        "expert_id": False,
    }
    for feature_name, feature_fn in FEATURES.items():
        feature_values = [feature_fn(record) for record in records]
        diagnostics[feature_name] = {
            "pearson_with_delta_nll": _pearson(feature_values, delta_values),
            "spearman_with_delta_nll": _spearman(feature_values, delta_values),
            "group_pairwise": _compute_pairwise_accuracy(
                grouped=grouped,
                feature_fn=feature_fn,
                reverse=orientation.get(feature_name, False),
            ),
        }
    return diagnostics


def analyze_records(records: list[AblationRecord], calibrator=None) -> dict:
    grouped = _group_records(records)
    deltas_by_strategy: dict[str, list[float]] = {strategy: [] for strategy in STRATEGIES}
    epsilons = [0.01, 0.05, 0.10]

    for group_records in grouped.values():
        for strategy in STRATEGIES:
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
        strategy_summary[strategy] = _strategy_summary(deltas, epsilons)

    factor_diagnostics = _factor_diagnostics(records, grouped)
    raw_pairwise = factor_diagnostics["effective_gate_weight"]["group_pairwise"]

    result = {
        "pairwise_ranking_accuracy": raw_pairwise["accuracy"],
        "pairwise_total": raw_pairwise["pairwise_total"],
        "strategies": strategy_summary,
        "routing_factor_diagnostics": factor_diagnostics,
        "routing_factor_strategy_subset": {
            strategy: strategy_summary[strategy]
            for strategy in strategy_summary
            if strategy in SINGLE_FACTOR_STRATEGIES or strategy in {"random", "raw_routing", "oracle"}
        },
        "notes": {
            "quality_constrained_deferrable_ratio_semantics": (
                "ratio of window-layer groups where one selected route stays within epsilon delta_nll"
            ),
            "pairwise_ranking_semantics": (
                "group-local pairwise agreement between a routing-context factor order and realized delta_nll order"
            ),
        },
    }
    return result


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
    lines.extend(["", "## Routing Factor Diagnostics", ""])
    for feature_name, payload in summary.get("routing_factor_diagnostics", {}).items():
        pairwise = payload["group_pairwise"]
        lines.append(
            f"- {feature_name}: pearson={payload['pearson_with_delta_nll']}, "
            f"spearman={payload['spearman_with_delta_nll']}, "
            f"pairwise={pairwise['accuracy']}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 当前 PoC 每个 group 只取消一个 expert request，不直接代表最终通信节省比例。",
            "- 当前取消 route 代表请求暂缓后最终不释放的质量上限近似。",
            "- routing factor diagnostics 用于控制变量看单个 routing context 因子与 realized criticality 的关系。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
