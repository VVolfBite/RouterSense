from __future__ import annotations

import argparse
import json
from pathlib import Path
from rs.runtime.offline.control_replay import (
    collect_trace_rows,
    read_jsonl,
    summarize_control_replay_trace,
    trace_paths_from_args,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize lightweight online control replay traces.")
    parser.add_argument("--trace", action="append", default=[], help="Path to rank*_control_replay_trace.jsonl; may be repeated")
    parser.add_argument("--trace-dir", default="", help="Directory containing rank*_control_replay_trace.jsonl files")
    parser.add_argument("--output", default="", help="Optional path to write JSON summary")
    args = parser.parse_args()
    trace_paths = trace_paths_from_args(trace_values=list(args.trace), trace_dir=str(args.trace_dir))
    if not trace_paths:
        raise SystemExit("at least one --trace or a --trace-dir with rank*_control_replay_trace.jsonl files is required")
    rows, rank_rows = collect_trace_rows(trace_paths=trace_paths)
    payload = summarize_control_replay_trace(rows, rank_rows=rank_rows)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
