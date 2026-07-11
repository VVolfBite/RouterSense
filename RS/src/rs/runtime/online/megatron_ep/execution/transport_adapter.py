"""Megatron transport adapter。

主要职责：
- 接住被 hook 的 all_to_all
- 根据当前激活的 PhaseExecutionPlan 决定走原生还是自定义执行
它是执行面最关键的 runtime 适配器之一。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.phase import PhaseExecutionPlan, PhaseReadyContext

from .async_p2p_executor import execute_async_phase_tensor, validate_async_phase_preflight
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

    def __init__(
        self,
        *,
        dispatcher_class: str,
        dispatcher_module_sha256: str | None,
        p2p_group: dist.ProcessGroup | None = None,
    ) -> None:
        self.dispatcher_class = dispatcher_class
        self.dispatcher_module_sha256 = dispatcher_module_sha256
        self.p2p_group = p2p_group
        self._active: ActivePhaseTransport | None = None
        self._latest_results: list[dict[str, Any]] = []
        self.async_executor_invocation_count = 0
        self.batch_isend_irecv_call_count = 0
        self.real_send_op_count = 0
        self.real_recv_op_count = 0
        self.local_copy_task_count = 0
        self.phase_sync_fallback_count = 0

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
        use_nccl_stream: bool = False,
    ) -> torch.Tensor:
        state = self._active
        if state is None or not state.plan.transport_mutation:
            return original_all_to_all(group, input_tensor, output_split_sizes, input_split_sizes, use_nccl_stream=use_nccl_stream)

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
        if str(state.plan.execution_mode) == "joint_window_async_p2p":
            preflight = validate_async_phase_preflight(
                context=state.context,
                plan=state.plan,
                tensor_role=tensor_role,
                process_group=self.p2p_group if self.p2p_group is not None else group,
                rank_context={
                    "global_rank": int(state.context.global_rank),
                    "local_rank": int(state.context.local_rank),
                },
                mode=str((state.plan.metrics or {}).get("preflight_mode", "full")),
            )
            if not preflight.all_ranks_ok:
                output, result, execution_entries = execute_scheduled_phase_tensor(
                    context=state.context,
                    plan=replace(state.plan, execution_mode="phase_sync_wave"),
                    tensor_role=tensor_role,
                    input_tensor=input_tensor,
                    group=group,
                    timeline_hook=getattr(self, "timeline_hook", None),
                )
                self.phase_sync_fallback_count += 1
                execution_entries = list(execution_entries) + [
                    {
                        "record_type": "async_preflight_fallback",
                        "fallback_before_p2p": True,
                        "preflight_failure_reason": str(preflight.reason),
                        "all_ranks_preflight_ok": bool(preflight.all_ranks_ok),
                        "preflight_collective_count": int(preflight.collective_count),
                        "preflight_mode": str(preflight.preflight_mode),
                    }
                ]
            else:
                self.async_executor_invocation_count += 1
                async_result = execute_async_phase_tensor(
                    context=state.context,
                    plan=state.plan,
                    tensor_role=tensor_role,
                    input_tensor=input_tensor,
                    process_group=self.p2p_group if self.p2p_group is not None else group,
                    rank_context={
                        "global_rank": int(state.context.global_rank),
                        "local_rank": int(state.context.local_rank),
                    },
                    timeline_hook=getattr(self, "timeline_hook", None),
                )
                output = async_result.output
                result = async_result.summary
                execution_entries = async_result.execution_entries
                summary_entry = next(
                    (row for row in execution_entries if row.get("record_type") == "async_phase_summary"),
                    {},
                )
                self.batch_isend_irecv_call_count += 1 if (int(summary_entry.get("send_op_count", 0)) + int(summary_entry.get("recv_op_count", 0))) > 0 else 0
                self.real_send_op_count += int(summary_entry.get("send_op_count", 0) or 0)
                self.real_recv_op_count += int(summary_entry.get("recv_op_count", 0) or 0)
                self.local_copy_task_count += int(result.local_copy_rows)
                execution_entries.append(
                    {
                        "record_type": "async_preflight_summary",
                        "preflight_mode": str(preflight.preflight_mode),
                        "preflight_collective_count": int(preflight.collective_count),
                        "all_ranks_preflight_ok": bool(preflight.all_ranks_ok),
                    }
                )
        else:
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
                "async_executor_invocation_count": int(self.async_executor_invocation_count),
                "batch_isend_irecv_call_count": int(self.batch_isend_irecv_call_count),
                "real_send_op_count": int(self.real_send_op_count),
                "real_recv_op_count": int(self.real_recv_op_count),
                "local_copy_task_count": int(self.local_copy_task_count),
                "phase_sync_fallback_count": int(self.phase_sync_fallback_count),
                "use_nccl_stream_requested": bool(use_nccl_stream),
                "use_nccl_stream_effective": bool(use_nccl_stream),
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
