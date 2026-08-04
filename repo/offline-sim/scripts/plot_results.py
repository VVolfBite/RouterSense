#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("plotting requires: pip install -e '.[plot]'") from exc

from rs_sim.reporting.plot_data import PreparedResults, prepare_results, write_prepared


METRICS = {
    "mean_stall": ("mean_stall_reduction_percent", "Mean communication stall reduction (%)"),
    "p95_stall": ("p95_stall_reduction_percent", "P95 communication stall reduction (%)"),
    "comm_makespan": ("comm_makespan_reduction_percent", "Communication makespan reduction (%)"),
    "window_makespan": ("window_makespan_reduction_percent", "P12 window reduction (%)"),
}

DEFAULT_MAIN = [
    "FIFO-Local", "Greedy-Local", "Birkhoff-Local", "Residual-MWM-Local",
    "FAST-Local", "RSCF-Local", "RSCF-Joint-FATE", "Oracle-Local", "Oracle-Joint",
]


def _split(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip() for item in value.split(",") if item.strip()]
    return result or None


def _configure_style(width: str) -> tuple[float, float]:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    })
    return (3.45, 2.45) if width == "single" else (7.1, 2.55)


def _filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = df.copy()
    for column, values in (
        ("model", _split(args.models)),
        ("treatment", _split(args.treatments)),
    ):
        if values:
            result = result[result[column].astype(str).isin(values)]
    eps = _split(args.eps)
    if eps:
        result = result[pd.to_numeric(result["ep"], errors="coerce").isin([float(v) for v in eps])]
    seqs = _split(args.sequences)
    if seqs:
        result = result[pd.to_numeric(result["sequence_length"], errors="coerce").isin([float(v) for v in seqs])]
    return result


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def _aggregate_reduction(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for treatment, group in df.groupby("treatment", sort=False):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        ci = 0.0 if len(values) < 2 else float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))
        rows.append({"treatment": treatment, "mean": mean, "ci95": ci, "n": len(values)})
    return pd.DataFrame(rows)


def plot_main_bars(runtime: pd.DataFrame, output: Path, size: tuple[float, float], requested: list[str] | None) -> None:
    treatments = requested or [name for name in DEFAULT_MAIN if name in set(runtime["treatment"])]
    if not treatments:
        treatments = list(dict.fromkeys(runtime["treatment"].astype(str)))
    metrics = [
        (METRICS["mean_stall"][0], "Mean stall"),
        (METRICS["p95_stall"][0], "P95 stall"),
        (METRICS["comm_makespan"][0], "Comm. makespan"),
    ]
    figure_height = max(size[1], 0.34 * len(treatments) + 0.85)
    fig, axes = plt.subplots(1, 3, figsize=(size[0], figure_height), sharey=True)
    order = {name: index for index, name in enumerate(treatments)}
    for axis, (metric, title) in zip(axes, metrics):
        data = _aggregate_reduction(runtime[runtime["treatment"].isin(treatments)], metric)
        data["order"] = data["treatment"].map(order)
        data = data.sort_values("order")
        positions = np.arange(len(data))
        bars = axis.barh(positions, data["mean"], xerr=data["ci95"], capsize=2,
                         linewidth=0.6, edgecolor="black")
        for index, bar in enumerate(bars):
            bar.set_hatch(("", "//", "..", "xx")[index % 4])
        axis.axvline(0, linewidth=0.8, color="black")
        axis.set_title(title)
        axis.set_xlabel("Reduction vs. FIFO-Local (%)")
        axis.set_yticks(positions)
        axis.set_yticklabels(data["treatment"])
        axis.invert_yaxis()
    fig.tight_layout(w_pad=0.9)
    _save(fig, output / "fig_main_metrics")


def plot_local_joint(prepared: PreparedResults, output: Path, size: tuple[float, float]) -> None:
    paired = prepared.paired_summary
    if paired.empty:
        return
    metric_columns = [
        ("mean_stall_ns__joint_improvement_percent", "Mean stall"),
        ("p95_stall_ns__joint_improvement_percent", "P95 stall"),
        ("comm_makespan_mean_ns__joint_improvement_percent", "Comm. makespan"),
    ]
    cores = list(dict.fromkeys(paired["core_pair"].astype(str)))
    x = np.arange(len(cores))
    width = 0.24
    fig, axis = plt.subplots(figsize=size)
    for offset, (column, label) in enumerate(metric_columns):
        values = [pd.to_numeric(paired.loc[paired["core_pair"] == core, column], errors="coerce").mean() for core in cores]
        axis.bar(x + (offset - 1) * width, values, width=width, label=label, edgecolor="black", linewidth=0.5,
                 hatch=("", "//", "..") [offset])
    axis.axhline(0, linewidth=0.8, color="black")
    axis.set_ylabel("Joint improvement over matched Local (%)")
    axis.set_xticks(x)
    axis.set_xticklabels(cores, rotation=35, ha="right")
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    _save(fig, output / "fig_local_vs_joint")


def plot_prediction(runtime: pd.DataFrame, output: Path, size: tuple[float, float]) -> None:
    names = ["RSCF-Joint-Zero", "RSCF-Joint-FATE", "RSCF-Joint-Perfect"]
    subset = runtime[runtime["treatment"].isin(names)]
    if subset["treatment"].nunique() < 2:
        return
    metrics = [
        ("mean_stall_ns", "Mean stall"),
        ("p95_stall_ns", "P95 stall"),
        ("comm_makespan_mean_ns", "Comm. makespan"),
    ]
    pair_columns = ["model", "ep", "sequence_length", "fixture_id", "repeat_index", "task_kib"]
    pivoted = {}
    for metric, label in metrics:
        pivot = subset.pivot_table(index=pair_columns, columns="treatment", values=metric, aggfunc="last")
        if "RSCF-Joint-Zero" not in pivot:
            continue
        reductions = pd.DataFrame(index=pivot.index)
        for name in names:
            if name in pivot:
                reductions[name] = 100.0 * (pivot["RSCF-Joint-Zero"] - pivot[name]) / pivot["RSCF-Joint-Zero"]
        pivoted[label] = reductions.mean()
    if not pivoted:
        return
    x = np.arange(len(pivoted))
    width = 0.25
    fig, axis = plt.subplots(figsize=size)
    for index, name in enumerate(names):
        values = [pivoted[label].get(name, np.nan) for label in pivoted]
        axis.bar(x + (index - 1) * width, values, width=width, label=name.replace("RSCF-Joint-", ""),
                 edgecolor="black", linewidth=0.5, hatch=("", "//", "..")[index])
    axis.axhline(0, linewidth=0.8, color="black")
    axis.set_ylabel("Reduction vs. Zero-P2 (%)")
    axis.set_xticks(x)
    axis.set_xticklabels(list(pivoted))
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    _save(fig, output / "fig_prediction_zero_fate_perfect")


def plot_scaling(runtime: pd.DataFrame, output: Path, size: tuple[float, float], metric_key: str) -> None:
    metric, ylabel = METRICS[metric_key]
    if runtime["ep"].nunique() < 2:
        return
    for model, model_df in runtime.groupby("model", sort=False):
        fig, axis = plt.subplots(figsize=size)
        for treatment, group in model_df.groupby("treatment", sort=False):
            reduced = group.groupby(["ep", "evidence"], as_index=False)[metric].mean()
            for evidence, evidence_df in reduced.groupby("evidence"):
                evidence_df = evidence_df.sort_values("ep")
                axis.plot(
                    evidence_df["ep"], evidence_df[metric], marker="o",
                    linestyle="--" if evidence == "PROJECTED" else "-",
                    markerfacecolor="none" if evidence == "PROJECTED" else None,
                    label=f"{treatment}{' (projected)' if evidence == 'PROJECTED' else ''}",
                )
        axis.axhline(0, linewidth=0.8, color="black")
        axis.set_xlabel("Expert-parallel size")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted(pd.to_numeric(model_df["ep"], errors="coerce").dropna().unique()))
        axis.legend(frameon=False, ncol=2, fontsize=6.8)
        axis.set_title(str(model))
        fig.tight_layout()
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(model)).strip("_")
        _save(fig, output / f"fig_scaling_{metric_key}_{safe}")


def plot_cdf(rank_samples: pd.DataFrame, output: Path, size: tuple[float, float], treatments: list[str] | None) -> None:
    if rank_samples.empty:
        return
    available = set(rank_samples["treatment"].astype(str))
    if treatments is None:
        preferred = [
            "FIFO-Local", "Birkhoff-Local", "RSCF-Local", "ReleaseFrontier-Local",
            "RSCF-Joint-FATE", "ReleaseFrontier-Joint", "Oracle-Joint",
        ]
        treatments = [name for name in preferred if name in available]
    subset = rank_samples
    if treatments:
        subset = subset[subset["treatment"].isin(treatments)]
    if subset.empty:
        return
    fig, axis = plt.subplots(figsize=size)
    for treatment, group in subset.groupby("treatment", sort=False):
        values = np.sort(pd.to_numeric(group["stall_us"], errors="coerce").dropna().to_numpy())
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1) / values.size
        axis.plot(values, y, label=treatment)
    axis.set_xlabel("Rank communication stall (μs)")
    axis.set_ylabel("CDF")
    axis.set_ylim(0, 1.01)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output / "fig_rank_stall_cdf")


def plot_overhead(runtime: pd.DataFrame, output: Path, size: tuple[float, float]) -> None:
    subset = runtime.copy()
    components = [
        ("prediction_exposed_ns", "Prediction"),
        ("control_exposed_ns", "Planning/control"),
        ("binding_exposed_ns", "Binding/repair"),
    ]
    means = subset.groupby("treatment", sort=False)[[column for column, _ in components]].mean() / 1_000.0
    if means.fillna(0).to_numpy().sum() == 0:
        return
    fig, axis = plt.subplots(figsize=size)
    bottom = np.zeros(len(means))
    for index, (column, label) in enumerate(components):
        values = means[column].fillna(0).to_numpy()
        axis.bar(np.arange(len(means)), values, bottom=bottom, label=label, edgecolor="black", linewidth=0.4,
                 hatch=("", "//", "..")[index])
        bottom += values
    axis.set_ylabel("Visible overhead (μs)")
    axis.set_xticks(np.arange(len(means)))
    axis.set_xticklabels(means.index, rotation=45, ha="right")
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    _save(fig, output / "fig_visible_overhead")


def export_table(summary: pd.DataFrame, output: Path) -> None:
    columns = [
        "model", "ep", "sequence_length", "task_kib", "evidence", "treatment",
        "mean_stall_us", "mean_stall_reduction_percent",
        "p95_stall_us", "p95_stall_reduction_percent",
        "comm_makespan_us", "comm_makespan_reduction_percent",
        "window_makespan_ms", "window_makespan_reduction_percent",
        "ttft_proxy_ms", "ttft_reduction_percent", "sample_count",
    ]
    table = summary[[column for column in columns if column in summary.columns]].copy()
    table.to_csv(output / "table_main.csv", index=False)
    with (output / "table_main.tex").open("w", encoding="utf-8") as handle:
        handle.write(table.to_latex(index=False, float_format=lambda value: f"{value:.3f}", escape=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="prepare and plot RouterSense durable sweep CSV results")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="FIFO-Local")
    parser.add_argument("--figures", default="main,local-joint,prediction,scaling,cdf,overhead,table")
    parser.add_argument("--models")
    parser.add_argument("--eps")
    parser.add_argument("--sequences")
    parser.add_argument("--treatments")
    parser.add_argument("--width", choices=("single", "double"), default="double")
    parser.add_argument("--scaling-metric", choices=tuple(METRICS), default="comm_makespan")
    return parser


def main() -> int:
    args = _parser().parse_args()
    size = _configure_style(args.width)
    prepared = prepare_results(args.input_csv, baseline=args.baseline)
    filtered_runtime = _filter(prepared.runtime, args)
    if filtered_runtime.empty:
        raise SystemExit("filters removed every runtime row")
    selected_run_keys = set(filtered_runtime["run_key"].astype(str))
    filtered = PreparedResults(
        runtime=filtered_runtime,
        per_window=prepared.per_window[prepared.per_window["run_key"].astype(str).isin(selected_run_keys)],
        rank_samples=prepared.rank_samples[prepared.rank_samples["run_key"].astype(str).isin(selected_run_keys)],
        summary=prepared.summary,
        paired_summary=prepared.paired_summary,
    )
    output = args.output_dir.expanduser().resolve()
    data_dir = output / "data"
    figure_dir = output / "figures"
    write_prepared(filtered, data_dir)
    requested = set(_split(args.figures) or [])
    treatments = _split(args.treatments)
    if "main" in requested:
        plot_main_bars(filtered.runtime, figure_dir, size, treatments)
    if "local-joint" in requested:
        plot_local_joint(filtered, figure_dir, size)
    if "prediction" in requested:
        plot_prediction(filtered.runtime, figure_dir, size)
    if "scaling" in requested:
        plot_scaling(filtered.runtime, figure_dir, size, args.scaling_metric)
    if "cdf" in requested:
        plot_cdf(filtered.rank_samples, figure_dir, size, treatments)
    if "overhead" in requested:
        plot_overhead(filtered.runtime, figure_dir, size)
    if "table" in requested:
        export_table(filtered.summary, output)
    print(f"prepared data: {data_dir}")
    print(f"figures: {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
