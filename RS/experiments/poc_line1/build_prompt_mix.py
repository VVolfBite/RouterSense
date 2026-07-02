#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a mixed unique prompt set for POC-line1 traces.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL prompt files.")
    parser.add_argument("--per-input", type=int, default=50)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args(argv)

    merged: list[dict] = []
    next_document_id = 0
    seen_text: set[str] = set()
    for input_index, raw_path in enumerate(args.inputs):
        path = Path(raw_path)
        rows = _load_jsonl(path)[: args.per_input]
        for row_index, row in enumerate(rows):
            text = str(row["text"])
            if text in seen_text:
                continue
            seen_text.add(text)
            payload = dict(row)
            payload["document_id"] = next_document_id
            payload["source_file"] = path.name
            payload["source_index"] = row_index
            payload["mix_group"] = input_index
            merged.append(payload)
            next_document_id += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "count": len(merged)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
