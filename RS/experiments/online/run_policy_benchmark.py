"""Formal online policy-benchmark entrypoint.

The pre-evaluation tree keeps benchmark execution mapped to the same frozen
phase-executor implementation as policy-correctness runs, but the canonical
entrypoint now lives under ``experiments/online`` instead of ``integrations``.
"""

from __future__ import annotations

from experiments.online.run_policy_correctness import main


if __name__ == "__main__":
    raise SystemExit(main())
