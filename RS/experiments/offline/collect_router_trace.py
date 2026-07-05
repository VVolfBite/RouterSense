"""Formal offline router-trace collection entrypoint.

Round-1 keeps the historical POC implementation and exposes it under the new
formal experiments namespace.
"""

from __future__ import annotations

from experiments.poc_line1.exp_trace import main


if __name__ == "__main__":
    raise SystemExit(main())
