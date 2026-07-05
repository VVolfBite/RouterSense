from __future__ import annotations

"""Collection-stage evaluation helpers.

This module provides a pipeline-oriented import surface without changing the
legacy implementation files. Existing callers can keep using the old modules;
new code can import collection helpers from here.
"""

from pathlib import Path
from typing import Any

from .artifacts import collect_environment_snapshot
from .cross_layer import load_gate_weight_bundle, load_hidden_state_bundle
from .dc_asymmetry import load_pairwise_results_index
from .traffic_matrix import load_trace_jsonl


def collect_scheduling_results(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    return load_pairwise_results_index(path)


def collect_traffic_records(path: str | Path):
    return load_trace_jsonl(path)


def collect_hidden_state_bundle(path: str | Path):
    return load_hidden_state_bundle(path)


def collect_gate_weight_bundle(path: str | Path):
    return load_gate_weight_bundle(path)


__all__ = [
    "collect_environment_snapshot",
    "collect_gate_weight_bundle",
    "collect_hidden_state_bundle",
    "collect_scheduling_results",
    "collect_traffic_records",
]
