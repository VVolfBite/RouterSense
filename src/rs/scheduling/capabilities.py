from __future__ import annotations

from dataclasses import dataclass, replace
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
    supports_current_window_joint_planning: bool = False
    supports_cross_layer_prediction: bool = False
    supports_two_horizon_prediction: bool = False
    supports_target_layer_preplanning: bool = False
    supports_p1_plan_reuse: bool = False
    supports_late_suffix_splice: bool = False
    supports_rank_release_batch: bool = False

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
            "supports_current_window_joint_planning": self.supports_current_window_joint_planning,
            "supports_cross_layer_prediction": self.supports_cross_layer_prediction,
            "supports_two_horizon_prediction": self.supports_two_horizon_prediction,
            "supports_target_layer_preplanning": self.supports_target_layer_preplanning,
            "supports_p1_plan_reuse": self.supports_p1_plan_reuse,
            "supports_late_suffix_splice": self.supports_late_suffix_splice,
            "supports_rank_release_batch": self.supports_rank_release_batch,
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

    def with_runtime_flags(
        self,
        *,
        supports_current_window_joint_planning: bool | None = None,
        supports_cross_layer_prediction: bool | None = None,
        supports_two_horizon_prediction: bool | None = None,
        supports_target_layer_preplanning: bool | None = None,
        supports_p1_plan_reuse: bool | None = None,
        supports_late_suffix_splice: bool | None = None,
        supports_rank_release_batch: bool | None = None,
    ) -> "PolicyCapabilities":
        return replace(
            self,
            supports_current_window_joint_planning=(
                self.supports_current_window_joint_planning
                if supports_current_window_joint_planning is None
                else bool(supports_current_window_joint_planning)
            ),
            supports_cross_layer_prediction=(
                self.supports_cross_layer_prediction
                if supports_cross_layer_prediction is None
                else bool(supports_cross_layer_prediction)
            ),
            supports_two_horizon_prediction=(
                self.supports_two_horizon_prediction
                if supports_two_horizon_prediction is None
                else bool(supports_two_horizon_prediction)
            ),
            supports_target_layer_preplanning=(
                self.supports_target_layer_preplanning
                if supports_target_layer_preplanning is None
                else bool(supports_target_layer_preplanning)
            ),
            supports_p1_plan_reuse=(
                self.supports_p1_plan_reuse
                if supports_p1_plan_reuse is None
                else bool(supports_p1_plan_reuse)
            ),
            supports_late_suffix_splice=(
                self.supports_late_suffix_splice
                if supports_late_suffix_splice is None
                else bool(supports_late_suffix_splice)
            ),
            supports_rank_release_batch=(
                self.supports_rank_release_batch
                if supports_rank_release_batch is None
                else bool(supports_rank_release_batch)
            ),
        )
