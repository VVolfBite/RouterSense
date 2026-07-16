from __future__ import annotations

import json

from experiments.paper.cli import main


def test_aggregate_writes_jsonl_objects(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "scheduling_summary.json").write_text(
        json.dumps({"records": [{"instance_id": "fixture:vep2", "objective": 1.0, "comparable": True, "metadata": {"model_id": "m"}}]}),
        encoding="utf-8",
    )
    (input_dir / "prediction_summary.json").write_text(json.dumps({"records": []}), encoding="utf-8")
    (input_dir / "hiding_summary.json").write_text(json.dumps({"records": []}), encoding="utf-8")
    (input_dir / "runtime_summary.json").write_text(json.dumps({"records": []}), encoding="utf-8")
    rc = main(["aggregate", "--input", str(input_dir), "--output-dir", str(output_dir)])
    assert rc == 0
    lines = (output_dir / "paired_records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["instance_id"] == "fixture:vep2"


def test_selected_layers_accepts_explicit_list_without_type_error(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "schema_version: paper_eval_config.v1",
                "claim_scope: trace_capture",
                "models:",
                "  model_id: m",
                "  model_revision: r",
                "inputs:",
                "  source: trace_bundle",
                "layers: [1, 2]",
                "virtual_ep_sizes: [2, 4]",
                "physical_world_size: 1",
                "policies: []",
                "predictors: []",
                "cost_model: x",
                "seeds:",
                "  default: 0",
                "splits:",
                "  development: []",
                "  validation: []",
                "  frozen_evaluation: []",
                "measurement:",
                "  mode: trace_only",
                "eligibility:",
                "  runtime_performance: false",
                "output:",
                "  dir: outputs/paper",
            ]
        ),
        encoding="utf-8",
    )
    missing_bundle = tmp_path / "missing_bundle"
    output_dir = tmp_path / "out"
    rc = main(["build-traffic", "--config", str(config), "--input", str(missing_bundle), "--output-dir", str(output_dir)])
    assert rc == 0
    summary = json.loads((output_dir / "build_traffic_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ENVIRONMENT_BLOCKED"
