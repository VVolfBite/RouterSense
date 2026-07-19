from __future__ import annotations

import json
from pathlib import Path

import yaml

from experiments.offline.collect_router_trace import main as collect_router_trace_main


def _temp_formal_trace_config(
    *,
    output_dir: Path,
    model_id: str,
    model_path: str,
    prompts_path: str,
    run_id: str,
    precision: str = "bf16",
) -> Path:
    payload = {
        "run": {"kind": "offline_trace", "name": str(run_id)},
        "model": {
            "config": "configs/model/olmoe_1b_7b_instruct.yaml",
            "model_id": str(model_id),
            "local_path": str(model_path),
        },
        "topology": {"launcher": {"kind": "python"}, "ep": {"size": 1}},
        "workload": {"prompts": str(prompts_path), "num_prompts": 1},
        "runtime": {"precision": str(precision)},
        "online_policy": {"name": "disabled"},
        "offline_study": {"policies": []},
        "execution": {"mode": "native_passthrough"},
        "observation": {"profile": "minimal"},
        "validation": {"save_logits": False, "stop_after_selected_layer": False},
        "artifact": {"artifact_root": str(output_dir)},
    }
    config_path = output_dir / "paper_trace_capture_formal_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def capture_trace_from_config(
    *,
    output_dir: Path,
    model_id: str,
    model_path: str,
    prompts_path: str,
    run_id: str,
    precision: str = "bf16",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = _temp_formal_trace_config(
        output_dir=output_dir,
        model_id=model_id,
        model_path=model_path,
        prompts_path=prompts_path,
        run_id=run_id,
        precision=precision,
    )
    rc = int(
        collect_router_trace_main(
            [
                "--config",
                str(config_path),
                "--run-id",
                str(run_id),
                "--output-dir",
                str(output_dir),
            ]
        )
    )
    if rc != 0:
        raise RuntimeError(f"formal trace collector exited with code {rc}")
    bundle_dir = output_dir / run_id
    if not bundle_dir.exists():
        raise FileNotFoundError(f"expected trace bundle dir not produced: {bundle_dir}")
    manifest = {
        "bundle_schema_version": "paper_trace_bundle.v1",
        "collector": "experiments.offline.collect_router_trace",
        "bundle_dir": str(bundle_dir),
        "trace_path": str(bundle_dir / "trace.jsonl"),
        "summary_path": str(bundle_dir / "summary.json"),
        "architecture_probe_path": str(bundle_dir / "architecture_probe.json"),
        "environment_path": str(bundle_dir / "environment.json"),
    }
    (bundle_dir / "paper_trace_bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle_dir
