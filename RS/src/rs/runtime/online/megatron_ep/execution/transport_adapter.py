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

from .async_p2p_executor import validate_async_phase_preflight
from .executor_facade import ExecutionRequest, execute_transport
from .sync_wave_executor import PhaseExecutionResult


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
    shared_session: dict[str, Any] | None = None


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
        self.local_copy_row_count = 0
        self.phase_sync_fallback_count = 0
        self._async_phase_sessions: dict[tuple[str, str], dict[str, Any]] = {}

    def activate(self, *, layer_name: str, phase: str, context: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
        expected_roles = ("hidden_states", "routing_probs") if phase == "P0" else ("hidden_states",)
        shared_session = None
        if str(plan.execution_mode) == "joint_window_async_p2p":
            session_key = (str(layer_name), str(phase))
            shared_session = {
                "session_key": [str(layer_name), str(phase)],
                "primary_tensor_role": "hidden_states",
                "suffix_splice_count": 0,
                "execution_origin": "",
                "final_task_order": [],
                "lineage": [],
            }
            self._async_phase_sessions[session_key] = shared_session
        self._active = ActivePhaseTransport(
            layer_name=layer_name,
            phase=phase,
            context=context,
            plan=plan,
            call_index=0,
            expected_roles=expected_roles,
            shared_session=shared_session,
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
            self._async_phase_sessions.pop((layer_name, phase), None)
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
            should_collective_preflight = bool(state.phase == "P0" and tensor_role == "hidden_states")
            if should_collective_preflight:
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
            else:
                preflight = replace(
                    validate_async_phase_preflight(
                        context=state.context,
                        plan=state.plan,
                        tensor_role=tensor_role,
                        process_group=None,
                        rank_context={
                            "global_rank": int(state.context.global_rank),
                            "local_rank": int(state.context.local_rank),
                        },
                        mode="local_only",
                    ),
                    all_ranks_ok=True,
                    collective_count=0,
                    preflight_mode="local_only",
                )
            if not preflight.all_ranks_ok:
                facade_result = execute_transport(
                    ExecutionRequest(
                        execution_plan=replace(state.plan, execution_mode="phase_sync_wave"),
                        phase_context=state.context,
                        tensor_role=tensor_role,
                        input_tensor=input_tensor,
                        process_group=group,
                        rank_context={
                            "global_rank": int(state.context.global_rank),
                            "local_rank": int(state.context.local_rank),
                            "late_suffix_provider": getattr(self, "late_suffix_provider", None),
                            "async_phase_session": state.shared_session,
                            "is_primary_payload": bool(tensor_role == "hidden_states"),
                            "on_release_batch_completed": getattr(self, "on_release_batch_completed", None),
                        },
                        event_sink=getattr(self, "timeline_hook", None),
                        requested_backend_id="async_release",
                    ),
                    backend="phase_sync",
                )
                facade_result = replace(
                    facade_result,
                    fallback_used=True,
                    fallback_reason=str(preflight.reason),
                    requested_backend_id="async_release",
                    preflight_collective_count=int(preflight.collective_count),
                    preflight_passed=bool(preflight.all_ranks_ok),
                    all_work_completed=bool(facade_result.all_work_completed),
                )
                output = facade_result.output_tensor
                result = PhaseExecutionResult.from_dict(facade_result.raw_summary)
                execution_entries = list(facade_result.execution_entries)
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
                self.batch_isend_irecv_call_count += int(facade_result.batch_isend_irecv_call_count)
                self.real_send_op_count += int(facade_result.send_op_count)
                self.real_recv_op_count += int(facade_result.recv_op_count)
                self.local_copy_task_count += int(facade_result.local_copy_task_count)
                self.local_copy_row_count += int(facade_result.local_copy_row_count)
            else:
                self.async_executor_invocation_count += 1
                facade_result = execute_transport(
                    ExecutionRequest(
                        execution_plan=state.plan,
                        phase_context=state.context,
                        tensor_role=tensor_role,
                        input_tensor=input_tensor,
                        process_group=self.p2p_group if self.p2p_group is not None else group,
                        rank_context={
                            "global_rank": int(state.context.global_rank),
                            "local_rank": int(state.context.local_rank),
                            "late_suffix_provider": getattr(self, "late_suffix_provider", None),
                            "async_phase_session": state.shared_session,
                            "is_primary_payload": bool(tensor_role == "hidden_states"),
                            "precomputed_task_order": list((state.shared_session or {}).get("final_task_order", [])),
                            "on_release_batch_completed": getattr(self, "on_release_batch_completed", None),
                        },
                        event_sink=getattr(self, "timeline_hook", None),
                        requested_backend_id="async_release",
                    ),
                    backend="async_release",
                )
                facade_result = replace(
                    facade_result,
                    preflight_collective_count=int(preflight.collective_count),
                    preflight_passed=bool(preflight.all_ranks_ok),
                )
                output = facade_result.output_tensor
                result = PhaseExecutionResult.from_dict(facade_result.raw_summary)
                execution_entries = list(facade_result.execution_entries)
                self.batch_isend_irecv_call_count += int(facade_result.batch_isend_irecv_call_count)
                self.real_send_op_count += int(facade_result.send_op_count)
                self.real_recv_op_count += int(facade_result.recv_op_count)
                self.local_copy_task_count += int(facade_result.local_copy_task_count)
                self.local_copy_row_count += int(facade_result.local_copy_row_count)
                execution_entries.append(
                    {
                        "record_type": "async_preflight_summary",
                        "preflight_mode": str(preflight.preflight_mode),
                        "preflight_collective_count": int(preflight.collective_count),
                        "all_ranks_preflight_ok": bool(preflight.all_ranks_ok),
                        "shared_session_suffix_splice_count": int((state.shared_session or {}).get("suffix_splice_count", 0) or 0),
                    }
                )
        else:
            facade_result = execute_transport(
                ExecutionRequest(
                    execution_plan=state.plan,
                    phase_context=state.context,
                    tensor_role=tensor_role,
                    input_tensor=input_tensor,
                    process_group=group,
                    rank_context={
                        "global_rank": int(state.context.global_rank),
                        "local_rank": int(state.context.local_rank),
                        "late_suffix_provider": getattr(self, "late_suffix_provider", None),
                    },
                    event_sink=getattr(self, "timeline_hook", None),
                    requested_backend_id="phase_sync",
                ),
                backend="phase_sync",
            )
            output = facade_result.output_tensor
            result = PhaseExecutionResult.from_dict(facade_result.raw_summary)
            execution_entries = list(facade_result.execution_entries)
            self.batch_isend_irecv_call_count += int(facade_result.batch_isend_irecv_call_count)
            self.real_send_op_count += int(facade_result.send_op_count)
            self.real_recv_op_count += int(facade_result.recv_op_count)
            self.local_copy_task_count += int(facade_result.local_copy_task_count)
            self.local_copy_row_count += int(facade_result.local_copy_row_count)
        state.call_index += 1
        base_ordinal = len(self._latest_results)
        for index, entry in enumerate(execution_entries, start=1):
            self._latest_results.append(
                {
                    "layer_name": state.layer_name,
                    "layer_id": str(state.plan.plan_key.get("layer_id", "unknown")),
                    "forward_epoch": int(getattr(state.context, "forward_epoch", 0)),
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
                "forward_epoch": int(getattr(state.context, "forward_epoch", 0)),
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
                "local_copy_row_count": int(self.local_copy_row_count),
                "phase_sync_fallback_count": int(self.phase_sync_fallback_count),
                "executed_backend_id": str(facade_result.executed_backend_id),
                "requested_backend_id": str(facade_result.requested_backend_id),
                "fallback_used": bool(facade_result.fallback_used),
                "fallback_reason": str(facade_result.fallback_reason),
                "timeout": bool(facade_result.timeout),
                "preflight_collective_count": int(facade_result.preflight_collective_count),
                "preflight_passed": bool(facade_result.preflight_passed),
                "all_work_completed": bool(facade_result.all_work_completed),
                "first_transport_submit_ns": int(facade_result.first_transport_submit_ns),
                "last_transport_complete_ns": int(facade_result.last_transport_complete_ns),
                "p0_first_submit_ns": int(facade_result.p0_first_submit_ns),
                "p0_last_complete_ns": int(facade_result.p0_last_complete_ns),
                "p1_first_submit_ns": int(facade_result.p1_first_submit_ns),
                "p1_last_complete_ns": int(facade_result.p1_last_complete_ns),
                "timing_us": dict(facade_result.timing_us or {}),
                "phase_metrics": dict(facade_result.phase_metrics or {}),
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
