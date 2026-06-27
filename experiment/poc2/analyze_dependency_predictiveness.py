#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.distributed_runtime import WorkloadPlan
from routesense_poc2.stress import dependency_predictiveness_gate, oracle_minimax_round_packer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run dependency predictiveness gate and oracle round-pack diagnostic.")
    parser.add_argument("--scenario-snapshot", action="append", required=True)
    parser.add_argument("--output", type=str, default=str(ROOT / "artifacts" / "poc2_stress_suite" / "predictiveness_and_oracle.json"))
    parser.add_argument("--round-size", type=int, default=4)
    parser.add_argument("--target-rank", type=int, default=2)
    args = parser.parse_args(argv)

    scenarios = []
    for snapshot_path in [Path(item) for item in args.scenario_snapshot]:
        bundle = json.loads(snapshot_path.read_text(encoding="utf-8"))
        plans = [WorkloadPlan.from_dict(item) for item in bundle["plans"]]
        manifest = dict(bundle["manifest"])
        split_index = max(1, len(plans) // 2)
        calibration = plans[:split_index]
        evaluation = plans[split_index:]
        gate = dependency_predictiveness_gate(calibration, evaluation, target_rank=args.target_rank)
        oracle = oracle_minimax_round_packer(evaluation[0] if evaluation else plans[0], args.round_size)
        scenarios.append(
            {
                "snapshot_path": str(snapshot_path),
                "scenario_id": manifest["scenario_id"],
                "dependency_predictiveness_gate": gate,
                "oracle_round_diagnostic": oracle,
                "mechanism_lever_present": bool(
                    oracle["peak_bottleneck_pressure"]
                    < max(
                        sum(bucket.payload_bytes for bucket in (evaluation[0] if evaluation else plans[0]).bucket_workloads),
                        1,
                    ) * 0.85
                ),
            }
        )
    payload = {
        "target_rank": args.target_rank,
        "round_size": args.round_size,
        "scenarios": scenarios,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

