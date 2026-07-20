"""Joint shadow policy。

作用：只做影子联合规划/观测，不真正修改执行面 transport。
主要用于 smoke 和控制面验证。
"""

from __future__ import annotations

from dataclasses import replace

from rs.scheduling.observation_contracts import PeerFlow, PlanWave, PolicyContext, RouterSensePlan, RuntimeObservation
from rs.scheduling.validation import (
    build_phase_demands,
    build_remote_flows,
    flow_priority,
    stable_hash,
    summarize_plan_metrics,
    validate_global_observations,
    validate_shadow_plan,
)


class JointShadowP0P1Policy:
    policy_name = "joint_shadow_p0p1"
    policy_version = "v1"

    def build_plan(
        self,
        context: PolicyContext,
        global_observation: tuple[RuntimeObservation, ...],
    ) -> RouterSensePlan:
        validate_global_observations(context, global_observation)
        remaining = list(build_remote_flows(global_observation))
        remaining.sort(key=flow_priority)
        phase_demands = build_phase_demands(remaining)
        ready_flows = [flow for flow in remaining if flow.release_state == "ready"]
        blocked_flows = [flow for flow in remaining if flow.release_state == "blocked"]

        def _pack_waves(flows: list[PeerFlow], *, release_state: str, start_wave_id: int) -> tuple[list[PlanWave], int]:
            waves: list[PlanWave] = []
            wave_id = start_wave_id
            remaining_local = list(flows)
            while remaining_local:
                outgoing_used: set[int] = set()
                incoming_used: set[int] = set()
                chosen: list[PeerFlow] = []
                next_remaining: list[PeerFlow] = []
                for flow in remaining_local:
                    if flow.src_rank in outgoing_used or flow.dst_rank in incoming_used:
                        next_remaining.append(flow)
                        continue
                    outgoing_used.add(flow.src_rank)
                    incoming_used.add(flow.dst_rank)
                    chosen.append(flow)
                if not chosen:
                    raise ValueError("joint_shadow_p0p1 could not schedule remaining flows")
                waves.append(PlanWave(wave_id=wave_id, release_state=release_state, flows=tuple(chosen)))
                wave_id += 1
                remaining_local = next_remaining
            return waves, wave_id

        ready_waves, next_wave_id = _pack_waves(ready_flows, release_state="ready", start_wave_id=0)
        blocked_future_waves, _ = _pack_waves(blocked_flows, release_state="blocked", start_wave_id=next_wave_id)
        plan = RouterSensePlan(
            run_id=context.run_id,
            step_id=context.step_id,
            microbatch_id=context.microbatch_id,
            layer_id=context.layer_id,
            ep_group_hash=context.ep_group_hash,
            request_table_hash=context.request_table_hash,
            model_revision_hash=context.model_revision_hash,
            expert_placement_hash=context.expert_placement_hash,
            observation_digest=stable_hash([obs.observation_digest for obs in global_observation]),
            plan_hash="",
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            execution_mode="shadow_only",
            transport_mutation=False,
            future_hint_mode=context.future_hint_mode,
            control_mode=context.control_mode,
            is_shadow_only=True,
            phase_demands=phase_demands,
            ready_waves=tuple(ready_waves),
            blocked_future_waves=tuple(blocked_future_waves),
            metrics={},
        )
        validate_shadow_plan(context, plan)
        metrics = summarize_plan_metrics(plan)
        plan = replace(plan, metrics=metrics, plan_hash=stable_hash({"plan": plan.to_dict(), "metrics": metrics}))
        return plan
