"""Formal online policy-benchmark entrypoint.

Round-1 keeps benchmark execution mapped to the frozen phase executor. Benchmark
config and semantics can be specialized in later rounds without changing the
formal entrypoint name.
"""

from __future__ import annotations

from integrations.megatron_ep.exp_phase_executor import main


if __name__ == "__main__":
    raise SystemExit(main())
