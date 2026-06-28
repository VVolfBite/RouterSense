from pathlib import Path
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_trace_and_ablate_use_safe_output_names(tmp_path):
    env = os.environ.copy()
    env["ROUTESENSE_OUTPUT_DIR"] = str(tmp_path)
    script = PROJECT_ROOT / "scripts" / "run_action.py"

    trace_result = subprocess.run(
        [sys.executable, str(script), "trace", "--text", "hello", "--mock"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert trace_result.returncode == 0
    assert (tmp_path / "one_sample_trace.json").exists()
    assert (tmp_path / "batch_routing_summary.json").exists()
    assert not (tmp_path / "trace_not_run.json").exists()

    ablate_result = subprocess.run(
        [sys.executable, str(script), "ablate", "--text", "hello", "--rank", "0"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert ablate_result.returncode == 0
    assert (tmp_path / "ablation_not_run.json").exists()
    assert not (tmp_path / "one_route_ablation.json").exists()
    assert (tmp_path / "mock_one_sample_trace.json").exists()
    assert (tmp_path / "mock_one_sample_trace.json").read_text(encoding="utf-8").find('"status": "mock"') >= 0
