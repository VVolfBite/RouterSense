from __future__ import annotations

import json
import subprocess
from pathlib import Path

from experiments.paper import cli
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


def test_git_text_uses_argument_list_not_powershell(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="value\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli._git_text("rev-parse", "HEAD") == "value"
    assert calls == [["git", "rev-parse", "HEAD"]]


def test_audit_can_start_with_monkeypatched_git(monkeypatch, tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "schema_version: paper_eval_config.v1",
                "claim_scope: capability_audit",
                "models:",
                "  model_id: m",
                "  model_revision: r",
                "inputs:",
                "  source: tests/fixtures/offline_replay_smoke",
                "layers: [1, 2]",
                "virtual_ep_sizes: [2, 4]",
                "physical_world_size: 1",
                "policies: [exact_small_instance_reference]",
                "predictors: []",
                "cost_model: formal_replay_makespan",
                "seeds:",
                "  default: 0",
                "splits:",
                "  development: []",
                "  validation: []",
                "  frozen_evaluation: []",
                "measurement:",
                "  mode: audit_only",
                "eligibility:",
                "  runtime_performance: false",
                "output:",
                "  dir: outputs/paper",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "branch"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="convergence/m123-integration\n", stderr="")
        if args[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="b403433e1803288ce872e41dc0ce4e36384536c2\n", stderr="")
        raise AssertionError(f"unexpected subprocess args: {args!r}")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "evaluate_hiding_gap", lambda **kwargs: {"status": "PARTIAL", "records": []})
    monkeypatch.setattr(cli, "evaluate_runtime_correctness_with_gloo", lambda **kwargs: {"status": "ENVIRONMENT_BLOCKED", "records": [], "tensor_parity_pass": False})
    monkeypatch.setattr(cli, "evaluate_materialization_contract_smoke", lambda **kwargs: {"status": "MATERIALIZATION_CONTRACT_SMOKE", "records": []})
    output_dir = tmp_path / "out"
    rc = main(["audit", "--config", str(config), "--output-dir", str(output_dir)])
    assert rc == 0
    assert (output_dir / "result_bundle.json").exists()


def test_oracle_controls_cli_reads_all_cases(tmp_path) -> None:
    output_dir = tmp_path / "out"
    rc = main(
        [
            "oracle-controls",
            "--config",
            str(Path(__file__).resolve().parents[3] / "configs" / "official" / "paper" / "oracle_controls.yaml"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    summary = json.loads((output_dir / "oracle_control_summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 3
    assert summary["cases"]["joint_advantage"]["status"] == "PASS"
