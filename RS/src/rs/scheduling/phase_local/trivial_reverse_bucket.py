from __future__ import annotations

from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext

from .fifo import build_bucket_execution_plan, reverse_bucket_task_key
from ..capabilities import PolicyCapabilities


class TrivialReverseBucketPolicy:
    policy_name = "trivial_reverse_bucket"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        uses_p0=True,
        uses_p1=True,
        uses_p2=False,
        cross_phase=False,
        requires_topology=False,
        supports_sync_before_phase=True,
        supports_default_continue=False,
    )

    def __init__(self, *, bucket_rows: int) -> None:
        self.bucket_rows = int(bucket_rows)

    def build_plan(
        self,
        *,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> PhaseExecutionPlan:
        return build_bucket_execution_plan(
            local_context=local_context,
            global_contexts=global_contexts,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            task_sort_key=reverse_bucket_task_key,
        )
