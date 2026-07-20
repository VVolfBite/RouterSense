from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_trace_compare_entrypoint_matches_current_target_planning_contract(tmp_path: Path) -> None:
    root = tmp_path / "RouterSense_fate_trace_fixture_single_gpu_20260718"
    (root / "traffic").mkdir(parents=True)
    (root / "fate").mkdir(parents=True)

    instance_id = "val-prompt-0-layer-0"
    traffic = [
        {
            "instance_id": instance_id,
            "trace_sample_id": "prompt:0",
            "P0_matrix": [[0, 4], [2, 0]],
            "P1_matrix": [[0, 2], [4, 0]],
            "P2_truth_matrix": [[0, 3], [1, 0]],
        }
    ]
    hint = {
        "instance_id": instance_id,
        "target_dispatch_rows": [[0, 3], [1, 0]],
        "confidence": 0.9,
    }
    (root / "traffic" / "traffic_instances.json").write_text(json.dumps(traffic), encoding="utf-8")
    (root / "fate" / "fate_hints.jsonl").write_text(json.dumps(hint) + "\n", encoding="utf-8")

    report = tmp_path / "report.json"
    env = dict(os.environ)
    pythonpath = os.pathsep.join((str(ROOT), str(ROOT / "src")))
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.compare_p012_p0123_future",
            "--bundle",
            str(tmp_path),
            "--output",
            str(report),
            "--core",
            "rscf",
            "--branch",
            "global",
            "--split-prefix",
            "val-",
            "--p3-weight",
            "0.01",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["aggregate"]["instances"] == 1
    assert result["aggregate"]["future_plan_equivalence_rate"] == 1.0
    assert result["by_model"]["fixture"]["instances"] == 1
    assert result["rows"][0]["future_reconcile_status"] == "exact"


def test_trace_compare_entrypoint_accepts_directory_of_model_archives(tmp_path: Path) -> None:
    import zipfile

    source = tmp_path / "fixture_source"
    (source / "traffic").mkdir(parents=True)
    (source / "fate").mkdir(parents=True)
    instance_id = "val-prompt-0-layer-0"
    (source / "traffic" / "traffic_instances.json").write_text(
        json.dumps(
            [
                {
                    "instance_id": instance_id,
                    "trace_sample_id": "prompt:0",
                    "P0_matrix": [[0, 4], [2, 0]],
                    "P1_matrix": [[0, 2], [4, 0]],
                    "P2_truth_matrix": [[0, 3], [1, 0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    (source / "fate" / "fate_hints.jsonl").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "target_dispatch_rows": [[0, 3], [1, 0]],
                "confidence": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    archive = bundle_dir / "RouterSense_fate_trace_fixture_single_gpu_20260718.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(source))

    report = tmp_path / "archive_report.json"
    env = dict(os.environ)
    pythonpath = os.pathsep.join((str(ROOT), str(ROOT / "src")))
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.compare_p012_p0123_future",
            "--bundle",
            str(bundle_dir),
            "--output",
            str(report),
            "--core",
            "rscf",
            "--branch",
            "global",
            "--split-prefix",
            "val-",
            "--p3-weight",
            "0.01",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["aggregate"]["instances"] == 1
    assert result["aggregate"]["future_plan_equivalence_rate"] == 1.0
