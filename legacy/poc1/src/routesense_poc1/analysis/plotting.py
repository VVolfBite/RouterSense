from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_ranking_accuracy(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    value = summary.get("pairwise_ranking_accuracy")
    if value is None:
        return
    figure, axis = plt.subplots(figsize=(4, 3))
    axis.bar(["raw_routing"], [value])
    axis.axhline(0.5, linestyle="--", color="red")
    axis.set_ylabel("Accuracy")
    axis.set_title("Pairwise Ranking Accuracy")
    figure.tight_layout()
    figure.savefig(output_dir / "ranking_accuracy.png")
    plt.close(figure)


def plot_policy_delta_nll(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = summary.get("strategies", {})
    if not strategies:
        return
    names = list(strategies.keys())
    values = [strategies[name]["mean_delta_nll"] for name in names]
    figure, axis = plt.subplots(figsize=(6, 3))
    axis.bar(names, values)
    axis.set_ylabel("Mean delta_nll")
    axis.set_title("Policy delta_nll")
    figure.tight_layout()
    figure.savefig(output_dir / "policy_delta_nll.png")
    plt.close(figure)


def plot_deferrable_ratio(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = summary.get("strategies", {})
    if not strategies:
        return
    figure, axis = plt.subplots(figsize=(6, 3))
    for strategy, payload in strategies.items():
        epsilons = [float(key) for key in payload["deferrable_ratio"].keys()]
        values = list(payload["deferrable_ratio"].values())
        axis.plot(epsilons, values, marker="o", label=strategy)
    axis.set_xlabel("epsilon")
    axis.set_ylabel("Deferrable ratio")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "deferrable_ratio.png")
    plt.close(figure)
