from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ResolvedLayerSelector:
    requested_selector: str
    resolved_selector: str
    resolved_layer_ids: tuple[str, ...]
    matches_all: bool = False


def _normalize_layer_ids(values: Iterable[object]) -> tuple[str, ...]:
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text not in ordered:
            ordered.append(text)
    return tuple(ordered)


def resolve_layer_selector(
    selector: str,
    *,
    selected_layer_ids: Iterable[object] = (),
    available_layer_ids: Iterable[object] = (),
    invariant_mode: str = "diagnostic",
) -> ResolvedLayerSelector:
    requested = str(selector or "all").strip() or "all"
    selected_ids = _normalize_layer_ids(selected_layer_ids)
    available_ids = _normalize_layer_ids(available_layer_ids)

    if requested == "all":
        return ResolvedLayerSelector(
            requested_selector=requested,
            resolved_selector="all",
            resolved_layer_ids=(),
            matches_all=True,
        )

    if requested == "selected":
        if not selected_ids:
            if str(invariant_mode) == "evaluation_strict":
                raise ValueError("selected layer selector requires non-empty selected_layer_ids")
            return ResolvedLayerSelector(
                requested_selector=requested,
                resolved_selector="selected",
                resolved_layer_ids=(),
                matches_all=False,
            )
        return ResolvedLayerSelector(
            requested_selector=requested,
            resolved_selector="explicit",
            resolved_layer_ids=selected_ids,
            matches_all=False,
        )

    if requested in {"first", "middle", "last"}:
        if not available_ids:
            raise ValueError(f"{requested} layer selector requires available_layer_ids")
        index = 0
        if requested == "middle":
            index = len(available_ids) // 2
        elif requested == "last":
            index = len(available_ids) - 1
        return ResolvedLayerSelector(
            requested_selector=requested,
            resolved_selector="explicit",
            resolved_layer_ids=(available_ids[index],),
            matches_all=False,
        )

    explicit_ids = _normalize_layer_ids(part for part in requested.split(","))
    if not explicit_ids:
        raise ValueError(f"unsupported layer selector {requested!r}")
    return ResolvedLayerSelector(
        requested_selector=requested,
        resolved_selector="explicit",
        resolved_layer_ids=explicit_ids,
        matches_all=False,
    )


def layer_selected(
    layer_id: str,
    *,
    selector: ResolvedLayerSelector,
) -> bool:
    if selector.matches_all:
        return True
    return str(layer_id) in selector.resolved_layer_ids


__all__ = [
    "ResolvedLayerSelector",
    "layer_selected",
    "resolve_layer_selector",
]
