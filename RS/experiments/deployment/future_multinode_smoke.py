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

from routesense.topology import load_inventory, render_torchrun_dry_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run a future 2-node x 2-GPU torchrun contract.")
    parser.add_argument("--inventory", type=str, default=str(ROOT / "deploy" / "inventory" / "hosts.local.yaml"))
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)
    inventory = load_inventory(Path(args.inventory))
    payload = render_torchrun_dry_run(inventory, nnodes=2, nproc_per_node=2)
    print(json.dumps(payload, indent=2))
    if not args.dry_run:
        raise RuntimeError("this entrypoint is dry-run only in phase 0B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

