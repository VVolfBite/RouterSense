#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.topology import load_inventory, render_torchrun_dry_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run a multi-node torchrun contract from inventory.")
    parser.add_argument("--inventory", type=str, default=str(ROOT / "deploy" / "inventory" / "hosts.local.yaml"))
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)
    inventory = load_inventory(Path(args.inventory))
    payload = render_torchrun_dry_run(inventory)
    print(json.dumps(payload, indent=2))
    if not args.dry_run:
        raise RuntimeError("this entrypoint is dry-run only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
