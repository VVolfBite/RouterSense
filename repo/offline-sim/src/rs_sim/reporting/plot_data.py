from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "record_type": ("description__record_type", "record_type"),
    "status": ("description__status", "status"),
    "run_key": ("description__run_key", "run_key"),
    "trace_key": ("description__trace_key", "trace_key"),
    "experiment_name": ("description__experiment_name", "experiment_name"),
    "trace_root": ("description__trace_root", "trace_root"),
    "trace_relative_path": ("description__trace_relative_path", "trace_relative_path"),
    "fixture_id": ("description__fixture_id", "fixture_id"),
    "fixture_truth_digest": ("description__fixture_truth_digest", "fixture_truth_digest"),
    "model": ("description__trace_model", "trace_model", "model"),
    "ep": ("description__trace_ep", "trace_ep", "world_size", "description__world_size"),
    "sequence_length": ("description__trace_sequence_length", "trace_sequence_length", "sequence_length"),
    "treatment": ("description__treatment_name", "treatment_name", "treatment"),
    "repeat_index": ("description__repeat_index", "repeat_index"),
    "warmup": ("description__warmup", "warmup"),
    "paired_instance_id": ("description__paired_instance_id", "paired_instance_id"),
    "core": ("setting__treatment__core", "setting__algorithm_core", "algorithm_core", "core"),
    "scope": ("setting__treatment__scope", "setting__scope", "scope"),
    "planning": ("setting__treatment__planning", "setting__planning", "planning"),
    "information": ("setting__treatment__information", "setting__information", "information"),
    "release_mode": ("setting__treatment__release_mode", "setting__config__simulation__release_mode", "release_mode"),
    "experiment_role": ("setting__treatment__experiment_role", "setting__experiment_role", "experiment_role"),
    "task_bytes": ("setting__max_task_bytes", "setting__config__simulation__max_task_bytes", "max_task_bytes"),
    "window_count": ("observation__window_count", "window_count"),
    "mean_stall_ns": ("observation__mean_communication_stall_ns", "mean_communication_stall_ns"),
    "p95_stall_ns": ("observation__p95_communication_stall_ns", "p95_communication_stall_ns"),
    "max_stall_ns": ("observation__max_communication_stall_ns", "max_communication_stall_ns"),
    "rank_stall_by_window": ("observation__rank_communication_exposed_ns_by_window", "rank_communication_exposed_ns_by_window"),
    "stall_by_window": ("observation__communication_stall_ns_by_window", "communication_stall_ns_by_window"),
    "comm_makespan_mean_ns": ("metric__compute_excluded_communication_makespan_ns_mean", "compute_excluded_communication_makespan_ns_mean"),
    "comm_makespan_sum_ns": ("metric__compute_excluded_communication_makespan_ns_sum", "compute_excluded_communication_makespan_ns_sum"),
    "comm_makespan_values": ("metric__compute_excluded_communication_makespan_ns_values", "compute_excluded_communication_makespan_ns_values"),
    "window_makespan_mean_ns": ("metric__window_makespan_ns_mean", "window_makespan_ns_mean"),
    "window_makespan_sum_ns": ("metric__window_makespan_ns_sum", "window_makespan_ns_sum"),
    "window_makespan_values": ("observation__window_makespan_ns_values", "window_makespan_ns_values"),
    "ttft_proxy_ns": ("metric__ttft_proxy_ns", "ttft_proxy_ns"),
    "network_active_union_sum_ns": ("metric__network_active_union_ns_sum", "network_active_union_ns_sum"),
    "network_active_union_values": ("observation__network_active_union_ns_values", "network_active_union_ns_values"),
    "p2_first_release_values": ("observation__p2_first_rank_release_offset_ns_values", "p2_first_rank_release_offset_ns_values"),
    "p2_last_release_values": ("observation__p2_last_rank_release_offset_ns_values", "p2_last_rank_release_offset_ns_values"),
    "p2_release_spread_values": ("metric__p2_rank_release_spread_ns_values", "p2_rank_release_spread_ns_values"),
    "prediction_overlap_ppm": ("metric__prediction_matrix_overlap_ppm_mean", "prediction_matrix_overlap_ppm_mean"),
    "prediction_rae_ppm": ("metric__prediction_relative_absolute_error_ppm_mean", "prediction_relative_absolute_error_ppm_mean"),
    "prediction_top_dest_ppm": ("metric__prediction_top_destination_accuracy_ppm_mean", "prediction_top_destination_accuracy_ppm_mean"),
    "prediction_exposed_ns": ("observation__prediction_exposed_ns", "prediction_exposed_ns"),
    "control_exposed_ns": ("observation__control_exposed_ns", "control_exposed_ns"),
    "binding_exposed_ns": ("observation__binding_exposed_ns", "binding_exposed_ns"),
    "prediction_hidden_ns": ("observation__prediction_hidden_ns", "prediction_hidden_ns"),
    "control_hidden_ns": ("observation__control_hidden_ns", "control_hidden_ns"),
    "binding_hidden_ns": ("observation__binding_hidden_ns", "binding_hidden_ns"),
    "worker_elapsed_ns": ("observation__worker_elapsed_ns", "worker_elapsed_ns"),
    "wall_clock_ns": ("observation__wall_clock_ns", "wall_clock_ns"),
    "metric_status": ("observation__communication_stall_metric_status", "communication_stall_metric_status"),
    "formal_zero_comm_supported": ("observation__formal_zero_comm_supported", "formal_zero_comm_supported"),
    "performance_claim_allowed": ("setting__performance_claim_allowed", "performance_claim_allowed"),
}


@dataclass(frozen=True)
class PreparedResults:
    runtime: pd.DataFrame
    per_window: pd.DataFrame
    rank_samples: pd.DataFrame
    summary: pd.DataFrame
    paired_summary: pd.DataFrame


def _first_existing(columns: Iterable[str], aliases: Sequence[str]) -> str | None:
    present = set(columns)
    for name in aliases:
        if name in present:
            return name
    return None


def _series(df: pd.DataFrame, canonical: str, default: Any = "") -> pd.Series:
    column = _first_existing(df.columns, _CANONICAL_ALIASES[canonical])
    if column is None:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[column]


def _as_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _as_number(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _parse_json(value: Any, default: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _infer_evidence(trace_root: str, trace_relative_path: str, source_kind: str = "") -> str:
    text = " ".join((trace_root, trace_relative_path, source_kind)).lower()
    if "projected" in text or "synthetic_ep" in text:
        return "PROJECTED"
    return "MEASURED"


def load_runtime_rows(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    raw = pd.read_csv(source, dtype=str, keep_default_na=False, low_memory=False)
    normalized = pd.DataFrame(index=raw.index)
    for canonical in _CANONICAL_ALIASES:
        normalized[canonical] = _series(raw, canonical)

    record_type = normalized["record_type"].str.upper()
    status = normalized["status"].str.upper()
    warmup = normalized["warmup"].map(_as_bool)
    runtime = normalized.loc[(record_type == "RUNTIME") & (status == "PASS") & (~warmup)].copy()
    if runtime.empty:
        raise ValueError(f"no PASS non-warmup RUNTIME rows found in {source}")

    for column in (
        "ep", "sequence_length", "repeat_index", "task_bytes", "window_count",
        "mean_stall_ns", "p95_stall_ns", "max_stall_ns",
        "comm_makespan_mean_ns", "comm_makespan_sum_ns",
        "window_makespan_mean_ns", "window_makespan_sum_ns", "ttft_proxy_ns",
        "network_active_union_sum_ns", "prediction_overlap_ppm", "prediction_rae_ppm",
        "prediction_top_dest_ppm", "prediction_exposed_ns", "control_exposed_ns",
        "binding_exposed_ns", "prediction_hidden_ns", "control_hidden_ns",
        "binding_hidden_ns", "worker_elapsed_ns", "wall_clock_ns",
    ):
        runtime[column] = runtime[column].map(_as_number)

    source_kind_column = _first_existing(raw.columns, ("description__trace_source_kind", "trace_source_kind"))
    source_kind = raw.loc[runtime.index, source_kind_column] if source_kind_column else pd.Series("", index=runtime.index)
    runtime["evidence"] = [
        _infer_evidence(str(root), str(relative), str(kind))
        for root, relative, kind in zip(runtime["trace_root"], runtime["trace_relative_path"], source_kind)
    ]
    runtime["source_csv"] = str(source)
    runtime["task_kib"] = runtime["task_bytes"] / 1024.0
    runtime["mean_stall_us"] = runtime["mean_stall_ns"] / 1_000.0
    runtime["p95_stall_us"] = runtime["p95_stall_ns"] / 1_000.0
    runtime["max_stall_us"] = runtime["max_stall_ns"] / 1_000.0
    runtime["comm_makespan_us"] = runtime["comm_makespan_mean_ns"] / 1_000.0
    runtime["window_makespan_ms"] = runtime["window_makespan_mean_ns"] / 1_000_000.0
    runtime["ttft_proxy_ms"] = runtime["ttft_proxy_ns"] / 1_000_000.0
    runtime["visible_overhead_us"] = (
        runtime["prediction_exposed_ns"].fillna(0)
        + runtime["control_exposed_ns"].fillna(0)
        + runtime["binding_exposed_ns"].fillna(0)
    ) / 1_000.0
    runtime["hidden_overhead_us"] = (
        runtime["prediction_hidden_ns"].fillna(0)
        + runtime["control_hidden_ns"].fillna(0)
        + runtime["binding_hidden_ns"].fillna(0)
    ) / 1_000.0
    runtime["prediction_overlap_percent"] = runtime["prediction_overlap_ppm"] / 10_000.0
    runtime["prediction_rae_percent"] = runtime["prediction_rae_ppm"] / 10_000.0
    runtime["prediction_top_dest_percent"] = runtime["prediction_top_dest_ppm"] / 10_000.0
    return runtime.reset_index(drop=True)


def _safe_list(values: Any) -> list[Any]:
    parsed = _parse_json(values, [])
    return list(parsed) if isinstance(parsed, list) else []


def build_per_window(runtime: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for _, row in runtime.iterrows():
        rank_windows = _safe_list(row["rank_stall_by_window"])
        stall_windows = _safe_list(row["stall_by_window"])
        comm_values = _safe_list(row["comm_makespan_values"])
        window_values = _safe_list(row["window_makespan_values"])
        network_values = _safe_list(row["network_active_union_values"])
        p2_first = _safe_list(row["p2_first_release_values"])
        p2_last = _safe_list(row["p2_last_release_values"])
        p2_spread = _safe_list(row["p2_release_spread_values"])
        lengths = [
            len(rank_windows), len(stall_windows), len(comm_values), len(window_values),
            len(network_values), len(p2_first), len(p2_last), len(p2_spread),
        ]
        window_count = max(lengths + [int(row["window_count"]) if not math.isnan(row["window_count"]) else 0])
        common = {
            "run_key": row["run_key"], "trace_key": row["trace_key"],
            "fixture_id": row["fixture_id"], "paired_instance_id": row["paired_instance_id"],
            "model": row["model"], "ep": row["ep"], "sequence_length": row["sequence_length"],
            "repeat_index": row["repeat_index"], "treatment": row["treatment"],
            "core": row["core"], "scope": row["scope"], "planning": row["planning"],
            "information": row["information"], "release_mode": row["release_mode"],
            "evidence": row["evidence"], "task_kib": row["task_kib"],
            "metric_status": row["metric_status"],
        }
        for window_index in range(window_count):
            ranks_raw = rank_windows[window_index] if window_index < len(rank_windows) else []
            ranks = [float(value) for value in ranks_raw if _as_number(value) == _as_number(value)] if isinstance(ranks_raw, list) else []
            if not ranks and window_index < len(stall_windows):
                stall_value = _as_number(stall_windows[window_index])
                ranks = [] if math.isnan(stall_value) else [stall_value]
            mean_ns = float(np.mean(ranks)) if ranks else math.nan
            p95_ns = float(np.percentile(ranks, 95)) if ranks else math.nan
            max_ns = float(np.max(ranks)) if ranks else math.nan
            wr = {
                **common,
                "window_index": window_index,
                "mean_stall_ns": mean_ns,
                "p95_stall_ns": p95_ns,
                "max_stall_ns": max_ns,
                "mean_stall_us": mean_ns / 1_000.0 if not math.isnan(mean_ns) else math.nan,
                "p95_stall_us": p95_ns / 1_000.0 if not math.isnan(p95_ns) else math.nan,
                "max_stall_us": max_ns / 1_000.0 if not math.isnan(max_ns) else math.nan,
                "comm_makespan_ns": _as_number(comm_values[window_index]) if window_index < len(comm_values) else math.nan,
                "window_makespan_ns": _as_number(window_values[window_index]) if window_index < len(window_values) else math.nan,
                "network_active_union_ns": _as_number(network_values[window_index]) if window_index < len(network_values) else math.nan,
                "p2_first_release_ns": _as_number(p2_first[window_index]) if window_index < len(p2_first) else math.nan,
                "p2_last_release_ns": _as_number(p2_last[window_index]) if window_index < len(p2_last) else math.nan,
                "p2_release_spread_ns": _as_number(p2_spread[window_index]) if window_index < len(p2_spread) else math.nan,
            }
            wr["comm_makespan_us"] = wr["comm_makespan_ns"] / 1_000.0
            wr["window_makespan_ms"] = wr["window_makespan_ns"] / 1_000_000.0
            window_rows.append(wr)
            for rank, stall_ns in enumerate(ranks):
                rank_rows.append({**common, "window_index": window_index, "rank": rank,
                                  "stall_ns": stall_ns, "stall_us": stall_ns / 1_000.0})
    return pd.DataFrame(window_rows), pd.DataFrame(rank_rows)


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan, math.nan
    mean = float(numeric.mean())
    if len(numeric) < 2:
        return mean, 0.0
    return mean, float(1.96 * numeric.std(ddof=1) / math.sqrt(len(numeric)))


def add_baseline_reductions(runtime: pd.DataFrame, baseline: str) -> pd.DataFrame:
    result = runtime.copy()
    pair_columns = ["model", "ep", "sequence_length", "fixture_id", "repeat_index", "task_kib"]
    metrics = {
        "mean_stall_ns": "mean_stall_reduction_percent",
        "p95_stall_ns": "p95_stall_reduction_percent",
        "comm_makespan_mean_ns": "comm_makespan_reduction_percent",
        "window_makespan_mean_ns": "window_makespan_reduction_percent",
        "ttft_proxy_ns": "ttft_reduction_percent",
    }
    baseline_rows = result.loc[result["treatment"] == baseline, pair_columns + list(metrics)].copy()
    baseline_rows = baseline_rows.drop_duplicates(pair_columns, keep="last")
    baseline_rows = baseline_rows.rename(columns={metric: f"baseline__{metric}" for metric in metrics})
    result = result.merge(baseline_rows, on=pair_columns, how="left")
    for metric, output in metrics.items():
        denominator = pd.to_numeric(result[f"baseline__{metric}"], errors="coerce")
        numerator = denominator - pd.to_numeric(result[metric], errors="coerce")
        result[output] = np.where(denominator > 0, 100.0 * numerator / denominator, np.nan)
    return result


def build_summary(runtime: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["model", "ep", "sequence_length", "task_kib", "evidence", "treatment", "core", "scope", "information", "release_mode"]
    metrics = [
        "mean_stall_us", "p95_stall_us", "max_stall_us", "comm_makespan_us",
        "window_makespan_ms", "ttft_proxy_ms", "visible_overhead_us", "hidden_overhead_us",
        "prediction_overlap_percent", "prediction_rae_percent", "prediction_top_dest_percent",
        "mean_stall_reduction_percent", "p95_stall_reduction_percent",
        "comm_makespan_reduction_percent", "window_makespan_reduction_percent", "ttft_reduction_percent",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in runtime.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, keys))
        row["sample_count"] = len(group)
        for metric in metrics:
            mean, ci = _mean_ci(group[metric])
            row[metric] = mean
            row[f"{metric}__ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_summary(runtime: pd.DataFrame) -> pd.DataFrame:
    pairs: list[tuple[str, str, str]] = []
    treatments = set(runtime["treatment"].astype(str))
    for local in sorted(name for name in treatments if name.endswith("-Local")):
        core_name = local[:-6]
        candidates = [f"{core_name}-Joint", f"{core_name}-Joint-FATE"]
        joint = next((name for name in candidates if name in treatments), None)
        if joint:
            pairs.append((core_name, local, joint))
    metrics = ["mean_stall_ns", "p95_stall_ns", "comm_makespan_mean_ns", "window_makespan_mean_ns"]
    pair_columns = ["model", "ep", "sequence_length", "fixture_id", "repeat_index", "task_kib"]
    rows: list[dict[str, Any]] = []
    for core, local_name, joint_name in pairs:
        local = runtime.loc[runtime["treatment"] == local_name, pair_columns + metrics].copy()
        joint = runtime.loc[runtime["treatment"] == joint_name, pair_columns + metrics].copy()
        merged = local.merge(joint, on=pair_columns, suffixes=("__local", "__joint"))
        for _, item in merged.iterrows():
            row: dict[str, Any] = {column: item[column] for column in pair_columns}
            row.update({"core_pair": core, "local_treatment": local_name, "joint_treatment": joint_name})
            for metric in metrics:
                local_value = _as_number(item[f"{metric}__local"])
                joint_value = _as_number(item[f"{metric}__joint"])
                row[f"{metric}__local"] = local_value
                row[f"{metric}__joint"] = joint_value
                row[f"{metric}__joint_improvement_percent"] = (
                    100.0 * (local_value - joint_value) / local_value if local_value > 0 else math.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_results(path: str | Path, *, baseline: str = "FIFO-Local") -> PreparedResults:
    runtime = add_baseline_reductions(load_runtime_rows(path), baseline)
    per_window, rank_samples = build_per_window(runtime)
    summary = build_summary(runtime)
    paired = build_paired_summary(runtime)
    return PreparedResults(runtime=runtime, per_window=per_window, rank_samples=rank_samples, summary=summary, paired_summary=paired)


def write_prepared(prepared: PreparedResults, output_dir: str | Path) -> None:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prepared.runtime.to_csv(destination / "runtime_normalized.csv", index=False)
    prepared.per_window.to_csv(destination / "per_window.csv", index=False)
    prepared.rank_samples.to_csv(destination / "rank_stall_samples.csv", index=False)
    prepared.summary.to_csv(destination / "summary.csv", index=False)
    prepared.paired_summary.to_csv(destination / "local_joint_paired.csv", index=False)
