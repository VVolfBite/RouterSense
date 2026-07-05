from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.phase import PhaseExecutionPlan, PhaseReadyContext

from .sync_wave_executor import PhaseExecutionResult, execute_scheduled_phase_tensor


class HostAPIDriftError(RuntimeError):
    pass


@dataclass
class ActivePhaseTransport:
    layer_name: str
    phase: str
    context: PhaseReadyContext
    plan: PhaseExecutionPlan
    call_index: int = 0
    expected_roles: tuple[str, ...] = ()


class MegatronPhaseTransportAdapter:
    """Version-locked transport adapter that replaces only the monolithic transport primitive."""

    def __init__(self, *, dispatcher_class: str, dispatcher_module_sha256: str | None) -> None:
        self.dispatcher_class = dispatcher_class
        self.dispatcher_module_sha256 = dispatcher_module_sha256
        self._active: ActivePhaseTransport | None = None
        self._latest_results: list[dict[str, Any]] = []

    def activate(self, *, layer_name: str, phase: str, context: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
        expected_roles = ("hidden_states", "routing_probs") if phase == "P0" else ("hidden_states",)
        self._active = ActivePhaseTransport(
            layer_name=layer_name,
            phase=phase,
            context=context,
            plan=plan,
            call_index=0,
            expected_roles=expected_roles,
        )

    def deactivate(self, *, layer_name: str, phase: str) -> None:
        if self._active is None:
            return
        if self._active.layer_name == layer_name and self._active.phase == phase:
            if self._active.call_index != len(self._active.expected_roles):
                raise HostAPIDriftError(
                    f"incomplete transport consumption for {layer_name} {phase}: "
                    f"expected {len(self._active.expected_roles)} payloads, saw {self._active.call_index}"
                )
            self._active = None

    def current(self) -> ActivePhaseTransport | None:
        return self._active

    def export_results(self) -> list[dict[str, Any]]:
        return list(self._latest_results)

    def maybe_execute(
        self,
        *,
        group: dist.ProcessGroup | None,
        input_tensor: torch.Tensor,
        output_split_sizes: Any,
        input_split_sizes: Any,
        original_all_to_all: Any,
    ) -> torch.Tensor:
        state = self._active
        if state is None or not state.plan.transport_mutation:
            return original_all_to_all(group, input_tensor, output_split_sizes, input_split_sizes)

        if state.call_index >= len(state.expected_roles):
            raise HostAPIDriftError(
                f"extra transport call for {state.layer_name} {state.phase}: call_index={state.call_index}"
            )
        tensor_role = state.expected_roles[state.call_index]
        expected_send = tuple(int(v) for v in state.context.send_splits)
        expected_recv = tuple(int(v) for v in state.context.recv_splits)
        actual_recv = self._normalize_splits(output_split_sizes)
        actual_send = self._normalize_splits(input_split_sizes)
        if actual_send != expected_send or actual_recv != expected_recv:
            raise HostAPIDriftError(
                f"split mismatch for {state.layer_name} {state.phase} {tensor_role}: "
                f"expected send={expected_send} recv={expected_recv}, got send={actual_send} recv={actual_recv}"
            )
        output, result, execution_entries = execute_scheduled_phase_tensor(
            context=state.context,
            plan=state.plan,
            tensor_role=tensor_role,
            input_tensor=input_tensor,
            group=group,
            timeline_hook=getattr(self, "timeline_hook", None),
        )
        state.call_index += 1
        base_ordinal = len(self._latest_results)
        for index, entry in enumerate(execution_entries, start=1):
            self._latest_results.append(
                {
                    "layer_name": state.layer_name,
                    "layer_id": str(state.plan.plan_key.get("layer_id", "unknown")),
                    "phase": state.phase,
                    "tensor_role": tensor_role,
                    "policy_name": state.plan.policy_name,
                    "plan_hash": state.plan.plan_hash,
                    "execution_ordinal": base_ordinal + index,
                    **entry,
                }
            )
        self._latest_results.append(
            {
                "layer_name": state.layer_name,
                "layer_id": str(state.plan.plan_key.get("layer_id", "unknown")),
                "phase": state.phase,
                "tensor_role": tensor_role,
                "result": result.to_dict(),
                "plan_hash": state.plan.plan_hash,
                "policy_name": state.plan.policy_name,
                "execution_ordinal": len(self._latest_results) + 1,
                "record_type": "result_summary",
            }
        )
        return output

    @staticmethod
    def _normalize_splits(value: Any) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(int(v) for v in value)
        if hasattr(value, "tolist"):
            return MegatronPhaseTransportAdapter._normalize_splits(value.tolist())
        return ()
