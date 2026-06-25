from pathlib import Path
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_check_adapter_ready_returns_2_without_inspection(tmp_path):
    script = PROJECT_ROOT / "scripts" / "check_adapter_ready.py"
    assert script.exists()
    env = os.environ.copy()
    env["ROUTESENSE_OUTPUT_DIR"] = str(tmp_path)
    result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == 2
