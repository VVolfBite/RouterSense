from __future__ import annotations

"""Single formal extension authority for deployable prediction assets."""

from collections.abc import Mapping

from .bridge_future import bridge_predictor_specs, create_bridge_predictor
from .fate_future import FATE_PREDICTOR_SPEC, create_fate_predictor


def predictor_specs():
    return (*bridge_predictor_specs(), FATE_PREDICTOR_SPEC)


def _names(spec) -> set[str]:
    return {str(spec.predictor_id), *(str(alias) for alias in spec.historical_aliases)}



def resolve_predictor(predictor_id: str) -> str:
    name = str(predictor_id).strip()
    for spec in predictor_specs():
        if name in _names(spec):
            return str(spec.predictor_id)
    raise ValueError(f"unknown extension predictor {predictor_id!r}")


def resolves_predictor(predictor_id: str) -> bool:
    name = str(predictor_id)
    return any(name in _names(spec) for spec in predictor_specs())


def create_predictor(predictor_id: str, config: Mapping[str, object] | None = None):
    name = str(predictor_id)
    if name in _names(FATE_PREDICTOR_SPEC):
        return create_fate_predictor(config)
    for spec in bridge_predictor_specs():
        if name in _names(spec):
            return create_bridge_predictor(name, config)
    raise ValueError(f"unknown extension predictor {predictor_id!r}")


def merge_predictor_specs(existing_specs):
    rows = list(existing_specs)
    occupied: set[str] = set()
    for spec in rows:
        occupied.update(_names(spec))
    for spec in predictor_specs():
        if not (_names(spec) & occupied):
            rows.append(spec)
            occupied.update(_names(spec))
    return tuple(rows)


__all__ = [
    "FATE_PREDICTOR_SPEC",
    "create_predictor",
    "merge_predictor_specs",
    "predictor_specs",
    "resolve_predictor",
    "resolves_predictor",
]
