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
    parser.add_argument("--limit-total", type=int, default=None)
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument(
        "--require-field",
        action="append",
        default=[],
        help="Field filter in key=value form; may be repeated.",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args(argv)

    merged: list[dict] = []
    next_document_id = 0
    seen_text: set[str] = set()
    required: dict[str, str] = {}
    for item in args.require_field:
        if "=" not in item:
            raise SystemExit(f"invalid --require-field {item!r}; expected key=value")
        key, value = item.split("=", 1)
        required[key] = value
    for input_index, raw_path in enumerate(args.inputs):
        path = Path(raw_path)
        rows = _load_jsonl(path)[: args.per_input]
        for row_index, row in enumerate(rows):
            if any(str(row.get(key, "")) != value for key, value in required.items()):
                continue
            text = str(row[args.text_key])
            if text in seen_text:
                continue
            seen_text.add(text)
            payload = dict(row)
            payload["text"] = text
            payload["document_id"] = next_document_id
            payload["source_file"] = path.name
            payload["source_index"] = row_index
            payload["mix_group"] = input_index
            merged.append(payload)
            next_document_id += 1
            if args.limit_total is not None and len(merged) >= args.limit_total:
                break
        if args.limit_total is not None and len(merged) >= args.limit_total:
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "count": len(merged)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
