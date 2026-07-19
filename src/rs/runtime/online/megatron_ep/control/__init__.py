"""Control-plane package exports.

Keep package import light. Heavy control modules are resolved lazily so that
runtime subprocesses importing only a narrow helper do not eagerly import the
entire planning stack.
"""

from __future__ import annotations

__all__ = [
    "P2HintMetadata",
    "P2HintRequest",
    "build_p2_hint_provider",
    "extract_prepared_plan_priority",
    "run_phase_plan_agreement",
]


def __getattr__(name: str):
    if name in {"P2HintMetadata", "P2HintRequest"}:
        from .p2_contracts import P2HintMetadata, P2HintRequest

        return {"P2HintMetadata": P2HintMetadata, "P2HintRequest": P2HintRequest}[name]
    if name in {"build_p2_hint_provider", "extract_prepared_plan_priority"}:
        from .p2_provider import build_p2_hint_provider, extract_prepared_plan_priority

        return {
            "build_p2_hint_provider": build_p2_hint_provider,
            "extract_prepared_plan_priority": extract_prepared_plan_priority,
        }[name]
    if name == "run_phase_plan_agreement":
        from .plan_agreement import run_phase_plan_agreement

        return run_phase_plan_agreement
    raise AttributeError(name)
