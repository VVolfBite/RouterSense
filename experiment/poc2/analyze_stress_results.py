#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 3/4 stress-suite artifacts.")
    parser.add_argument("--predictiveness", type=str, required=True)
    parser.add_argument("--artifact-dir", action="append", default=[])
    parser.add_argument("--output", type=str, default=str(ROOT / "artifacts" / "poc2_stress_suite_stage4" / "summary.json"))
    args = parser.parse_args(argv)

    predictiveness = json.loads(Path(args.predictiveness).read_text(encoding="utf-8"))
    stage4 = []
    for artifact in [Path(item) for item in args.artifact_dir]:
        summary = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
        protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
        stage4.append(
            {
                "artifact_dir": str(artifact),
                "scenario_id": protocol.get("stress_scenario_id"),
                "strategies": summary.get("strategies", {}),
                "paired_deltas": summary.get("paired_deltas", {}),
                "position_effects": summary.get("position_effects", {}),
            }
        )

    payload = {
        "predictiveness_and_oracle": predictiveness,
        "stage4": stage4,
        "mechanism_lever_present": any(item.get("mechanism_lever_present") for item in predictiveness.get("scenarios", [])),
        "dependency_predictiveness_gate": {
            item["scenario_id"]: item["dependency_predictiveness_gate"]["gate"]
            for item in predictiveness.get("scenarios", [])
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

