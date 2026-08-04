from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from rs_sim.trace.collection.config import example_config, load_pipeline_config
from rs_sim.trace.collection.pipeline import (
    bundle_pipeline_artifacts,
    inspect_capture_artifacts,
    launch_collection,
)


def _write_config(tmp_path: Path) -> tuple[Path, dict]:
    config = example_config()
    config["output_dir"] = str(tmp_path / "output")
    config["capture"].update(
        {
            "backend": "EXPLICIT_API",
            "capture_id": "subprocess-explicit-smoke",
            "request_id": "req",
            "sample_id_prefix": "sample",
            "model_path": None,
            "rank_to_node": [0],
            "expert_to_rank": [0, 0],
            "capture_compute": False,
        }
    )
    config["launcher"].update(
        {
            "command": [],
            "clean_output_before_collect": True,
            "timeout_seconds": 30,
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


def test_launch_collection_requires_real_nonempty_capture(tmp_path: Path):
    config_path, _ = _write_config(tmp_path)
    script = tmp_path / "runner.py"
    script.write_text(
        "from rs_sim.trace.collection import capture_routing\n"
        "for layer in (0, 1):\n"
        "    capture_routing(layer_id=layer, routing_map=[[True, False], [False, True]])\n",
        encoding="utf-8",
    )
    config = load_pipeline_config(config_path)
    result = launch_collection(config, command_override=[sys.executable, str(script)])
    assert result["status"] == "PASS"
    assert result["capture_artifacts"]["routing_record_count"] == 2
    assert result["capture_artifacts"]["layer_ids"] == [0, 1]
    assert result["capture_artifacts"]["source_ranks"] == [0]


def test_capture_acceptance_and_bundle_manifest(tmp_path: Path):
    config_path, _ = _write_config(tmp_path)
    config = load_pipeline_config(config_path)
    output = Path(config["output_dir"])
    raw = output / "raw"
    raw.mkdir(parents=True)
    path = raw / "rank0000-global0000_source_expert_counts.jsonl"
    path.write_text(
        json.dumps({"sample_id": "sample:step0", "source_rank": 0, "layer_id": 0}) + "\n"
        + json.dumps({"sample_id": "sample:step0", "source_rank": 0, "layer_id": 1}) + "\n",
        encoding="utf-8",
    )
    assert inspect_capture_artifacts(config)["status"] == "PASS"
    (output / "pipeline_summary.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    bundle, sha_file = bundle_pipeline_artifacts(config)
    assert bundle.is_file()
    assert sha_file.is_file()
    assert (output / "ARTIFACT_MANIFEST.sha256").is_file()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "ARTIFACT_MANIFEST.sha256" in names
    assert "pipeline_summary.json" in names
    assert path.relative_to(output).as_posix() in names
