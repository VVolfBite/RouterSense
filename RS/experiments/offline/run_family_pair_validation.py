"""Run strict Local(f)/Joint(f) family comparisons on a replay fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper.adapters.scheduling_adapter import replay_window_from_matrices
from experiments.paper.family_evaluation import evaluate_family_pairs
from rs.scheduling.families import STRICT_FAMILY_IDS, family_inventory


def _matrix(payload, key: str):
    return tuple(tuple(int(value) for value in row) for row in payload[key])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--families", default=",".join(STRICT_FAMILY_IDS))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--expert-compute-delay", type=float, default=0.0)
    parser.add_argument("--bucket-rows", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    layer_id = int(dict(fixture.get("metadata", {})).get("layer_id", 0))
    window = replay_window_from_matrices(
        fixture_id=fixture_path.stem,
        layer_id=layer_id,
        p0_matrix=_matrix(fixture, "p0_dispatch_matrix"),
        p1_matrix=_matrix(fixture, "p1_return_matrix"),
        p2_matrix=_matrix(fixture, "p2_next_dispatch_matrix"),
    )
    families = tuple(item.strip() for item in str(args.families).split(",") if item.strip())
    result = evaluate_family_pairs(
        replay_window=window,
        family_ids=families,
        repeats=int(args.repeats),
        warmups=int(args.warmups),
        expert_compute_delay=float(args.expert_compute_delay),
        bucket_rows=int(args.bucket_rows),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"inventory": family_inventory(), "evaluation": result}, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
