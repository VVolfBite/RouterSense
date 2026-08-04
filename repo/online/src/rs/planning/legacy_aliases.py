from __future__ import annotations


LEGACY_PLANNER_ALIASES = {
    "B": "local",
    "U": "joint",
    "O_local": "exact_local",
    "O_joint": "exact_joint",
}


def normalize_family_name(value: str) -> str:
    normalized = str(value).strip()
    return LEGACY_PLANNER_ALIASES.get(normalized, normalized)


__all__ = ["LEGACY_PLANNER_ALIASES", "normalize_family_name"]
