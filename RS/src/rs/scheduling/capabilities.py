from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyCapabilities:
    supports_offline: bool
    supports_online_phase_local_execution: bool
    supports_online_multiphase_execution: bool
    uses_current_ready_flows: bool
    uses_blocked_p1_dependency: bool
    uses_p2_forecast: bool
    requires_fixed_placement: bool
    evaluation_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "supports_offline": self.supports_offline,
            "supports_online_phase_local_execution": self.supports_online_phase_local_execution,
            "supports_online_multiphase_execution": self.supports_online_multiphase_execution,
            "uses_current_ready_flows": self.uses_current_ready_flows,
            "uses_blocked_p1_dependency": self.uses_blocked_p1_dependency,
            "uses_p2_forecast": self.uses_p2_forecast,
            "requires_fixed_placement": self.requires_fixed_placement,
            "evaluation_eligible": self.evaluation_eligible,
            # Compatibility projection for the frozen online runtime artifact schema.
            "uses_p0": self.uses_current_ready_flows,
            "uses_p1": True,
            "uses_p2": self.uses_p2_forecast,
            "cross_phase": self.uses_blocked_p1_dependency or self.uses_p2_forecast or self.supports_online_multiphase_execution,
            "requires_topology": False,
            "supports_sync_before_phase": self.supports_online_phase_local_execution,
            "supports_default_continue": False,
        }

    @property
    def uses_p2(self) -> bool:
        return bool(self.uses_p2_forecast)
