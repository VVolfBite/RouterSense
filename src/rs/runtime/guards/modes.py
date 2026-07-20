from __future__ import annotations

INVARIANT_MODE_EVALUATION_STRICT = "evaluation_strict"
INVARIANT_MODE_RUNTIME_SAFE = "runtime_safe"
INVARIANT_MODE_DIAGNOSTIC = "diagnostic"

_ALL = {
    INVARIANT_MODE_EVALUATION_STRICT,
    INVARIANT_MODE_RUNTIME_SAFE,
    INVARIANT_MODE_DIAGNOSTIC,
}


def normalize_invariant_mode(value: str | None) -> str:
    mode = str(value or INVARIANT_MODE_DIAGNOSTIC).strip() or INVARIANT_MODE_DIAGNOSTIC
    if mode not in _ALL:
        raise ValueError(f"unsupported invariant_mode {mode!r}")
    return mode


def invariant_mode_is_strict(value: str | None) -> bool:
    return normalize_invariant_mode(value) == INVARIANT_MODE_EVALUATION_STRICT


def invariant_mode_allows_dirty_git(value: str | None) -> bool:
    return normalize_invariant_mode(value) == INVARIANT_MODE_DIAGNOSTIC


def invariant_mode_allows_fallback(value: str | None) -> bool:
    return normalize_invariant_mode(value) != INVARIANT_MODE_EVALUATION_STRICT


def invariant_mode_allows_diagnostic_bridge(value: str | None) -> bool:
    return normalize_invariant_mode(value) == INVARIANT_MODE_DIAGNOSTIC


def invariant_mode_forbids_diagnostic_bridge(value: str | None) -> bool:
    return normalize_invariant_mode(value) != INVARIANT_MODE_DIAGNOSTIC

