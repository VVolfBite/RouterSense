from __future__ import annotations


class FutureHintMode:
    NONE = "none"
    PREDICTED_M2 = "predicted_m2"


def normalize_online_future_hint_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {FutureHintMode.NONE, FutureHintMode.PREDICTED_M2}:
        return normalized
    if normalized == "oracle_full_trace":
        raise RuntimeError("online scheduler bridge must not consume oracle_full_trace")
    raise RuntimeError(f"unsupported online future hint mode: {mode!r}")
