#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.online import build_online_unimplemented_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect online native EP trace.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="artifacts/online/native_ep_trace")
    args = parser.parse_args(argv)
    run_id = f"online-native-trace-{uuid.uuid4().hex[:12]}"
    payload = build_online_unimplemented_result(
        run_id=run_id,
        world_size=args.world_size,
        transport_backend="online_native_a2a_ep",
        extra={
            "entrypoint": "collect_native_ep_trace",
            "output_dir": args.output_dir,
            "implemented_reason": "online observer is not implemented yet",
        },
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}_metadata.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
