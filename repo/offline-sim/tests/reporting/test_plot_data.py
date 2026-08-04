from __future__ import annotations

import csv
import json
from pathlib import Path

from rs_sim.reporting.plot_data import prepare_results


def test_prepare_results_parses_durable_wide_csv(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    header = [
        "description__record_type", "description__status", "description__run_key",
        "description__trace_key", "description__fixture_id", "description__trace_model",
        "description__trace_ep", "description__trace_sequence_length",
        "description__treatment_name", "description__repeat_index", "description__warmup",
        "description__paired_instance_id", "setting__treatment__core",
        "setting__treatment__scope", "setting__treatment__information",
        "setting__treatment__release_mode", "setting__max_task_bytes",
        "observation__window_count", "observation__mean_communication_stall_ns",
        "observation__p95_communication_stall_ns", "observation__max_communication_stall_ns",
        "observation__rank_communication_exposed_ns_by_window",
        "metric__compute_excluded_communication_makespan_ns_mean",
        "metric__compute_excluded_communication_makespan_ns_values",
        "metric__window_makespan_ns_mean", "observation__window_makespan_ns_values",
        "metric__ttft_proxy_ns",
    ]
    rows = []
    for treatment, core, mean, p95, comm in (
        ("FIFO-Local", "fifo", 100, 140, 200),
        ("RSCF-Local", "rscf", 100, 140, 200),
        ("RSCF-Joint-FATE", "rscf", 80, 110, 150),
    ):
        rows.append([
            "RUNTIME", "PASS", treatment, "trace", "fixture", "OLMoE", "8", "128",
            treatment, "0", "False", "paired", core,
            "WINDOW_JOINT" if "Joint" in treatment else "PHASE_LOCAL",
            "FATE_P2" if "Joint" in treatment else "ZERO_P2",
            "RANK_LOCAL" if "Joint" in treatment else "PHASE_BARRIER", "262144",
            "2", str(mean), str(p95), "160", json.dumps([[80, 120], [90, 110]]),
            str(comm), json.dumps([190, 210]), "1000", json.dumps([900, 1100]), "2000",
        ])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    prepared = prepare_results(path)
    assert len(prepared.runtime) == 3
    joint = prepared.runtime.loc[prepared.runtime["treatment"] == "RSCF-Joint-FATE"].iloc[0]
    assert joint["mean_stall_reduction_percent"] == 20.0
    assert len(prepared.per_window) == 6
    assert len(prepared.rank_samples) == 12
    assert prepared.paired_summary.iloc[0]["mean_stall_ns__joint_improvement_percent"] == 20.0
