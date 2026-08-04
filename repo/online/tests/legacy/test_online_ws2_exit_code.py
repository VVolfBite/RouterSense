from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / relative_path
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_native_ep_ws2_metadata_success_exits_zero() -> None:
    module = _load_script_module("bench_native_ep_module", "experiments/online/bench_native_ep.py")
    assert (
        module._result_exit_code(  # type: ignore[attr-defined]
            {
                "execution_mode": "online_ws2_hidden_dispatch_only",
                "correctness_status": "metadata_passed",
                "numerical_correctness_pass": None,
            }
        )
        == 0
    )


def test_collect_native_ep_trace_ws2_metadata_failure_exits_nonzero() -> None:
    module = _load_script_module("collect_native_ep_trace_module", "experiments/online/collect_native_ep_trace.py")
    assert (
        module._result_exit_code(  # type: ignore[attr-defined]
            {
                "execution_mode": "online_ws2_route_partition_only",
                "correctness_status": "metadata_failed",
                "numerical_correctness_pass": None,
            }
        )
        == 2
    )
