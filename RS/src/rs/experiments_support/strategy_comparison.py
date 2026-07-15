"""Shared helpers for strategy comparison runners."""

from __future__ import annotations

import os


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src:." if not existing else f"src:.:{existing}"
    omp = env.get("OMP_NUM_THREADS", "").strip()
    if not omp or not omp.isdigit() or int(omp) <= 0:
        env["OMP_NUM_THREADS"] = "1"
    return env


__all__ = ["child_env"]

