from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_verify_env_contract() -> None:
    script = Path("integrations/megatron_ep/verify_env.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--model", "/root/autodl-tmp/models/OLMoE-1B-7B-0125"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["pipeline"] == "host_runtime_native_ep"
    assert payload["host_runtime"] == "megatron_core"
    assert payload["status"] in {"ready", "blocked_environment"}
