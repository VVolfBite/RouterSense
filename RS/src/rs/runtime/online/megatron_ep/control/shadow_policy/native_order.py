from __future__ import annotations

from dataclasses import replace

from rs.scheduling.observation_contracts import PolicyContext, RouterSensePlan, RuntimeObservation
from rs.scheduling.validation import (
    build_phase_demands,
    build_remote_flows,
    stable_hash,
    summarize_plan_metrics,
    validate_global_observations,
)


class NativeOrderPolicy:
    policy_name = "native_order"
    policy_version = "v1"

    def build_plan(
        self,
        context: PolicyContext,
        global_observation: tuple[RuntimeObservation, ...],
    ) -> RouterSensePlan:
        validate_global_observations(context, global_observation)
        flows = build_remote_flows(global_observation)
        phase_demands = build_phase_demands(flows)
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
            execution_mode="native_passthrough",
            transport_mutation=False,
            future_hint_mode=context.future_hint_mode,
            control_mode=context.control_mode,
            is_shadow_only=False,
            phase_demands=phase_demands,
            ready_waves=(),
            blocked_future_waves=(),
            metrics={},
        )
        metrics = summarize_plan_metrics(plan)
        plan = replace(plan, metrics=metrics, plan_hash=stable_hash({"plan": plan.to_dict(), "metrics": metrics}))
        return plan
