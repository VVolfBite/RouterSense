"""Dependency-aware flow release contracts for multiphase scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseConstraint:
    phase: str
    release_state: str
    dependency: str | None = None
