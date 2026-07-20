from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_verify_env_contract() -> None:
    script = REPO_ROOT / "experiments/online/support/environment_validation.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--model", "/definitely/missing/model"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["pipeline"] == "host_runtime_native_ep"
    assert payload["host_runtime"] == "megatron_core"
    assert payload["status"] in {"ready", "blocked_environment"}
    assert "reason" in payload
    assert "missing" in payload
    assert "python_version" in payload
