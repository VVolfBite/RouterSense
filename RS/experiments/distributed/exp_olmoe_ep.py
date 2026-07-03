#!/usr/bin/env python3
from __future__ import annotations

# Compatibility shim. Keep the shorter historical entrypoint name stable while
# the canonical OLMoE EP smoke entrypoint lives in distributed_olmoe_ep_smoke.py.
from distributed_olmoe_ep_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
