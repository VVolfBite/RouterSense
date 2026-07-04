#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.contracts import FutureInformationMode
from rs.online import build_online_unimplemented_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark scheduled online EP runtime.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="artifacts/online/bench_scheduled_ep")
    args = parser.parse_args(argv)
    run_id = f"online-scheduled-bench-{uuid.uuid4().hex[:12]}"
    payload = build_online_unimplemented_result(
        run_id=run_id,
        world_size=args.world_size,
        transport_backend="scheduled_p2p",
        future_information_mode=FutureInformationMode.PREDICTED,
        extra={
            "entrypoint": "bench_scheduled_ep",
            "output_dir": args.output_dir,
            "implemented_reason": "scheduled_p2p backend is not implemented yet",
        },
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
