#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.megatron_ep.routersense.trace_writer import write_json


def _load_last(path: Path) -> dict | None:
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--status", default="timeout")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    last_progress: dict[str, dict] = {}
    for path in sorted(run_dir.glob("heartbeat-rank*.jsonl")):
        last = _load_last(path)
        if last is not None:
            last_progress[path.stem.replace("heartbeat-", "")] = last
    payload = {
        "status": args.status,
        "timeout_seconds": int(args.timeout_seconds),
        "last_progress_by_rank": last_progress,
        "last_layer_by_rank": {rank: row.get("layer_id") for rank, row in last_progress.items()},
        "last_phase_by_rank": {rank: row.get("phase") for rank, row in last_progress.items()},
        "last_wave_by_rank": {rank: row.get("wave_id") for rank, row in last_progress.items()},
        "last_tensor_role_by_rank": {rank: row.get("tensor_role") for rank, row in last_progress.items()},
        "last_event_seq_by_rank": {rank: row.get("event_seq") for rank, row in last_progress.items()},
    }
    write_json(run_dir / "watchdog_report.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
