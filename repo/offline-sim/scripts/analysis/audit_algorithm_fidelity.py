from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rs_sim.scheduler.registry import (
    formal_algorithm_audit_table,
    formal_internal_rscf_core_audit_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump RouterSense algorithm fidelity registry")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    external = []
    for row in formal_algorithm_audit_table():
        item = asdict(row)
        item["fidelity_grade"] = row.fidelity_grade.value
        item["comparison_class"] = row.comparison_class.value
        external.append(item)
    payload = {
        "external_algorithms": external,
        "internal_rscf_cores": [asdict(row) for row in formal_internal_rscf_core_audit_table()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
