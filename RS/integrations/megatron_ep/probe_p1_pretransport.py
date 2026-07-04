#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.megatron_ep.probe_dispatch_boundary import main as probe_main  # noqa: E402
from integrations.megatron_ep.routersense.trace_writer import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True)
    args, rest = parser.parse_known_args(argv)
    status = probe_main(rest)
    probe_dir = Path(args.probe_dir)
    timeline_path = probe_dir / "dispatcher_call_timeline.jsonl"
    capability = {
        "p1_pretransport_capability": "unavailable",
        "missing_fields": ["packed buffer", "offsets", "per-peer segment boundaries", "restore metadata", "safe host callback point"],
    }
    if timeline_path.exists():
        rows = [json.loads(line) for line in timeline_path.read_text(encoding="utf-8").splitlines()]
        seen = {row.get("event") for row in rows}
        if {"expert_compute_boundary", "token_combine_enter", "actual_combine_transport_enter"}.issubset(seen):
            capability = {"p1_pretransport_capability": "available", "missing_fields": []}
    write_json(probe_dir / "p1_pretransport_capability.json", capability)
    return status


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    raise SystemExit(main())
