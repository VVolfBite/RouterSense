"""Observation profile contracts for formal online runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ObservationProfile = Literal["minimal", "execution", "debug"]
ExecutionAuditStatus = Literal["passed", "failed", "not_applicable"]


@dataclass(frozen=True)
class ObservationConfig:
    profile: ObservationProfile
    capture_enabled: bool = False
    capture_layer_selector: str = ""
    capture_phase_selector: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionAudit:
    status: ExecutionAuditStatus
    policy_name: str
    plan_hash: str
    phase: str
    layer_id: str
    planned_wave_count: int
    executed_wave_count: int
    planned_task_ids: tuple[str, ...] = ()
    executed_task_ids: tuple[str, ...] = ()
    missing_tasks: tuple[str, ...] = ()
    unexpected_tasks: tuple[str, ...] = ()
    duplicate_tasks: tuple[str, ...] = ()
    order_mismatches: tuple[str, ...] = ()
    planned_rows: int = 0
    actual_rows: int = 0
    planned_bytes: int = 0
    actual_bytes: int = 0
    native_fallback_events: int = 0
    contract_violation_events: int = 0
    p0_bundle_atomicity_preserved: bool = True
    local_copy_coverage_passed: bool = True
    remote_flow_coverage_passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservationEmitter:
    """Small helper for profile-gated event collection."""

    def __init__(self, config: ObservationConfig) -> None:
        self.config = config

    def includes_execution(self) -> bool:
        return self.config.profile in {"execution", "debug"}

    def includes_debug(self) -> bool:
        return self.config.profile == "debug"
