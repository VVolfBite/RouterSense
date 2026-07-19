from __future__ import annotations

from typing import Protocol

from integrations.megatron_ep.routersense.contracts import PolicyContext, RouterSensePlan, RuntimeObservation
from integrations.megatron_ep.routersense.phase import PhaseExecutionPlan, PhaseReadyContext
from .capabilities import PolicyCapabilities


class RouterSensePolicy(Protocol):
    def build_plan(
        self,
        context: PolicyContext,
        global_observation: tuple[RuntimeObservation, ...],
    ) -> RouterSensePlan:
        ...


class RouterSensePhasePolicy(Protocol):
    policy_name: str
    policy_version: str
    capabilities: PolicyCapabilities

    def build_plan(
        self,
        *,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> PhaseExecutionPlan:
        ...
