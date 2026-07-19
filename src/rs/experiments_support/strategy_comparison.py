"""Shared helpers for strategy comparison runners."""

from __future__ import annotations

import os


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    pythonpath_entries = ["src", "."]
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    omp = env.get("OMP_NUM_THREADS", "").strip()
    if not omp or not omp.isdigit() or int(omp) <= 0:
        env["OMP_NUM_THREADS"] = "1"
    if not str(env.get("USE_LIBUV", "")).strip():
        env["USE_LIBUV"] = "0"
    return env


__all__ = ["child_env"]
