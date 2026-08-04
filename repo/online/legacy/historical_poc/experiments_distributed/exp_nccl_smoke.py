#!/usr/bin/env python3
from __future__ import annotations

# Compatibility shim. Keep the older filename stable while the canonical
# NCCL smoke entrypoint lives in distributed_nccl_smoke.py.
from distributed_nccl_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
