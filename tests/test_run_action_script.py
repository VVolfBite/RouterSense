from pathlib import Path
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_action_script_dispatches_trace_and_ablate(tmp_path):
    script = PROJECT_ROOT / "scripts" / "run_action.py"
    assert script.exists()
    env = os.environ.copy()
    env["ROUTESENSE_OUTPUT_DIR"] = str(tmp_path)

    for args in [
        ["trace", "--text", "hello"],
        ["ablate", "--text", "hello", "--rank", "0"],
    ]:
        result = subprocess.run(
            [sys.executable, str(script), *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stdout + result.stderr
