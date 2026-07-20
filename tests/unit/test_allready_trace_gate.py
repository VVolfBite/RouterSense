from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_gate_module():
    path = ROOT / "scripts" / "verify" / "run_allready_gate.py"
    spec = importlib.util.spec_from_file_location("rs_allready_gate_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_mode_matrix_rejects_zero_instance_reports(tmp_path: Path, monkeypatch) -> None:
    gate = _load_gate_module()

    def fake_run_command(command, *, log_path, timeout_seconds):
        del log_path, timeout_seconds
        report = Path(command[command.index("--output") + 1])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"aggregate": {"instances": 0}, "rows": []}), encoding="utf-8")
        return {"returncode": 0, "timed_out": False, "duration_seconds": 0.0, "command": command}

    monkeypatch.setattr(gate, "run_command", fake_run_command)
    result = gate.trace_mode_matrix(output_dir=tmp_path, trace_root=tmp_path / "empty", timeout_seconds=5)

    assert result["status"] == "FAIL"
    assert result["instance_count_consistent"] is False
    assert all(row["status"] == "FAIL" for row in result["rows"])
    assert all(row["aggregate"]["validation_error"] for row in result["rows"])
