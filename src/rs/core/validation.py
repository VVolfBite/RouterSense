"""Minimal validation helpers used by the formal RouteSense package layout."""

from __future__ import annotations


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
