from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


PHASE_COMPONENT_FIELDS = (
    "context_us",
    "observation_us",
    "local_matrix_us",
    "global_matrix_us",
    "prediction_us",
    "plan_input_us",
    "scheduler_solve_us",
    "plan_postprocess_us",
    "plan_store_us",
    "agreement_us",
    "materialization_us",
    "preflight_us",
    "pack_us",
    "executor_wall_us",
    "unpack_us",
    "state_update_us",
    "summary_us",
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def attribution_tolerance_us(total_us: float) -> float:
    return max(50.0, abs(float(total_us)) * 0.02)


def _sum_present(values: Iterable[Any]) -> float:
    total = 0.0
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            total += parsed
    return total


def _ratio(known_us: float, total_us: float) -> float | None:
    if total_us <= 0.0:
        return None
    return float(known_us) / float(total_us)


@dataclass(frozen=True)
class PhaseCostTree:
    strategy: str
    rank: int
    forward_epoch: int
    layer_id: str
    phase: str
    hook_total_us: float
    components: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        total = float(self.hook_total_us)
        known = _sum_present(self.components.get(field) for field in PHASE_COMPONENT_FIELDS)
        unattributed = total - known
        tolerance = attribution_tolerance_us(total)
        valid = unattributed >= -tolerance
        return {
            "strategy": str(self.strategy),
            "rank": int(self.rank),
            "forward_epoch": int(self.forward_epoch),
            "layer_id": str(self.layer_id),
            "phase": str(self.phase),
            "hook_total_us": total,
            **{field: _as_float(self.components.get(field)) for field in PHASE_COMPONENT_FIELDS},
            "known_component_us": known,
            "unattributed_us": unattributed,
            "unattributed_measurement_status": "derived",
            "explained_ratio": _ratio(known, total),
            "tolerance_us": tolerance,
            "phase_tree_valid": bool(valid),
            "validation_failure": "" if valid else "component sum exceeds hook_total_us beyond tolerance",
        }


@dataclass(frozen=True)
class SelectedLayerCostTree:
    strategy: str
    rank: int
    forward_epoch: int
    layer_id: str
    selected_layer_total_us: float
    p0_total_routerSense_us: float
    p0_to_expert_us: float | None
    expert_module_wall_us: float | None
    expert_to_p1_us: float | None
    p1_total_routerSense_us: float

    def to_dict(self) -> dict[str, Any]:
        total = float(self.selected_layer_total_us)
        known = _sum_present(
            (
                self.p0_total_routerSense_us,
                self.p0_to_expert_us,
                self.expert_module_wall_us,
                self.expert_to_p1_us,
                self.p1_total_routerSense_us,
            )
        )
        unattributed = total - known
        tolerance = attribution_tolerance_us(total)
        valid = unattributed >= -tolerance
        return {
            "strategy": str(self.strategy),
            "rank": int(self.rank),
            "forward_epoch": int(self.forward_epoch),
            "layer_id": str(self.layer_id),
            "selected_layer_total_us": total,
            "p0_total_routerSense_us": float(self.p0_total_routerSense_us),
            "p0_to_expert_us": self.p0_to_expert_us,
            "expert_module_wall_us": self.expert_module_wall_us,
            "expert_to_p1_us": self.expert_to_p1_us,
            "p1_total_routerSense_us": float(self.p1_total_routerSense_us),
            "known_component_us": known,
            "selected_layer_unattributed_us": unattributed,
            "selected_layer_unattributed_status": "derived",
            "selected_layer_explained_ratio": _ratio(known, total),
            "tolerance_us": tolerance,
            "selected_layer_tree_valid": bool(valid),
            "validation_failure": "" if valid else "selected layer component sum exceeds layer total beyond tolerance",
        }


@dataclass(frozen=True)
class ForwardCostTree:
    strategy: str
    rank: int
    forward_epoch: int
    full_forward_wall_us: float
    outside_selected_layers_us: float
    selected_layer_0_total_us: float
    inter_selected_layer_gap_us: float
    selected_layer_1_total_us: float

    def to_dict(self) -> dict[str, Any]:
        total = float(self.full_forward_wall_us)
        known = _sum_present(
            (
                self.outside_selected_layers_us,
                self.selected_layer_0_total_us,
                self.inter_selected_layer_gap_us,
                self.selected_layer_1_total_us,
            )
        )
        unattributed = total - known
        tolerance = attribution_tolerance_us(total)
        valid = abs(unattributed) <= tolerance
        return {
            "strategy": str(self.strategy),
            "rank": int(self.rank),
            "forward_epoch": int(self.forward_epoch),
            "full_forward_wall_us": total,
            "outside_selected_layers_us": float(self.outside_selected_layers_us),
            "selected_layer_0_total_us": float(self.selected_layer_0_total_us),
            "inter_selected_layer_gap_us": float(self.inter_selected_layer_gap_us),
            "selected_layer_1_total_us": float(self.selected_layer_1_total_us),
            "known_component_us": known,
            "forward_unattributed_us": unattributed,
            "forward_unattributed_status": "derived",
            "forward_explained_ratio": _ratio(known, total),
            "tolerance_us": tolerance,
            "forward_tree_valid": bool(valid),
            "validation_failure": "" if valid else "forward component sum does not match forward wall time",
        }


def legacy_outside_measured_hooks(
    *,
    full_forward_us: float,
    dispatch_hook_us: float,
    combine_hook_us: float,
) -> dict[str, Any]:
    return {
        "outside_measured_hooks_us": float(full_forward_us) - float(dispatch_hook_us) - float(combine_hook_us),
        "measurement_status": "derived_legacy",
        "deprecated": True,
        "replacement": "selected-layer non-overlapping cost tree",
    }


def aggregate_sync_callsite_cost(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        callsite_id = str(event.get("callsite_id", "unknown"))
        row = grouped.setdefault(
            callsite_id,
            {
                "callsite_id": callsite_id,
                "call_count": 0,
                "wall_us": 0.0,
                "execution_required": bool(event.get("execution_required", False)),
                "measurement_induced": bool(event.get("measurement_induced", False)),
            },
        )
        row["call_count"] = int(row["call_count"]) + int(event.get("call_count", 1) or 0)
        row["wall_us"] = float(row["wall_us"]) + float(event.get("wall_us", 0.0) or 0.0)
        row["execution_required"] = bool(row["execution_required"] or event.get("execution_required", False))
        row["measurement_induced"] = bool(row["measurement_induced"] or event.get("measurement_induced", False))
    return [grouped[key] for key in sorted(grouped)]


def attribution_schema() -> dict[str, Any]:
    return {
        "profile": "attribution_light",
        "forward_cost_tree": [
            "full_forward_wall_us",
            "outside_selected_layers_us",
            "selected_layer_0_total_us",
            "inter_selected_layer_gap_us",
            "selected_layer_1_total_us",
            "forward_unattributed_us",
            "forward_tree_valid",
        ],
        "selected_layer_cost_tree": [
            "selected_layer_total_us",
            "p0_total_routerSense_us",
            "p0_to_expert_us",
            "expert_module_wall_us",
            "expert_to_p1_us",
            "p1_total_routerSense_us",
            "selected_layer_unattributed_us",
            "selected_layer_tree_valid",
        ],
        "phase_cost_tree": ["hook_total_us", *PHASE_COMPONENT_FIELDS, "unattributed_us", "phase_tree_valid"],
        "legacy_fields": {
            "outside_measured_hooks_us": {
                "measurement_status": "derived_legacy",
                "deprecated": True,
                "note": "This replaces the invalid full_forward - dispatch_hook - combine_hook expert-compute label.",
            },
            "selected_window_span_us": {
                "deprecated_as_communication": True,
                "note": "This is a range reference and may include compute/control gaps; it is not network busy time.",
            },
        },
    }


__all__ = [
    "PHASE_COMPONENT_FIELDS",
    "ForwardCostTree",
    "PhaseCostTree",
    "SelectedLayerCostTree",
    "aggregate_sync_callsite_cost",
    "attribution_schema",
    "attribution_tolerance_us",
    "legacy_outside_measured_hooks",
]
