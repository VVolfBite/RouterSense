from __future__ import annotations

from typing import Any

from rs.core.contracts.execution import (
    ActualPhaseContext,
    ExecutionBatch,
    MaterializedPlan,
    PayloadSpec,
    PublishedPlan,
    TransferSlice,
    ValidationResult,
)
from rs.core.contracts.planning import WindowPlan
from rs.runtime.online.megatron_ep.materialization.layout import validate_materialized_layout
from rs.runtime.online.megatron_ep.phase import PhaseReadyContext


_PHASE_TO_FLOW_PHASE = {
    "P0": "p0_dispatch",
    "P1": "p1_return",
    "P2": "p2_dispatch",
}


def _coerce_phase_ready_context(payload: object) -> PhaseReadyContext:
    if isinstance(payload, PhaseReadyContext):
        return payload
    if isinstance(payload, dict):
        return PhaseReadyContext.from_dict(dict(payload))
    raise ValueError("phase_ready_context metadata is required")


def _expected_rows_for_roles(context: PhaseReadyContext, *, use_send: bool) -> dict[str, dict[int, int]]:
    splits = context.send_splits if use_send else context.recv_splits
    return {
        str(spec.tensor_role): {int(index): int(value) for index, value in enumerate(splits)}
        for spec in context.payload_specs
    }


def _payload_specs(context: PhaseReadyContext) -> tuple[PayloadSpec, ...]:
    send_rows = int(sum(context.send_splits))
    payloads: list[PayloadSpec] = []
    for spec in context.payload_specs:
        elements_per_row = 1
        for dim in spec.shape_suffix:
            elements_per_row *= int(dim)
        payloads.append(
            PayloadSpec(
                payload_role=str(spec.tensor_role),
                row_count=send_rows,
                element_count=int(send_rows * elements_per_row),
                byte_count=int(send_rows * elements_per_row * int(spec.element_size_bytes)),
                bytes_per_row=int(elements_per_row * int(spec.element_size_bytes)),
                dtype=str(spec.dtype),
                shape_suffix=tuple(int(dim) for dim in spec.shape_suffix),
            )
        )
    return tuple(payloads)


def _window_plan(plan: PublishedPlan) -> WindowPlan:
    plan.validate()
    return plan.window_plan


def _dependency_layer_id(*, flow_phase: str, context_layer_id: str, target_layer_id: str) -> str:
    if str(flow_phase) == "p2_dispatch":
        return str(target_layer_id)
    return str(context_layer_id)


def _release_dependency_ids(
    *,
    flow_phase: str,
    dependency_layer_id: str,
    src_group_rank: int,
    src_global_rank: int,
    local_global_rank: int,
) -> tuple[str, ...]:
    if int(src_global_rank) != int(local_global_rank):
        return ()
    if str(flow_phase) == "p1_return":
        return (f"release:{dependency_layer_id}:p0_inbound_complete:{src_group_rank}",)
    if str(flow_phase) == "p2_dispatch":
        return (f"release:{dependency_layer_id}:p1_inbound_complete:{src_group_rank}",)
    return ()

class CommonPlanMaterializer:
    def materialize(self, plan: PublishedPlan, context: ActualPhaseContext) -> MaterializedPlan:
        plan.validate()
        context.validate()
        phase_ready_context = _coerce_phase_ready_context(dict(context.metadata).get("phase_ready_context"))
        if str(phase_ready_context.layer_id) != str(context.layer_id):
            raise ValueError("phase_ready_context layer_id does not match ActualPhaseContext")
        if str(phase_ready_context.phase) != str(context.phase):
            raise ValueError("phase_ready_context phase does not match ActualPhaseContext")
        flow_phase = _PHASE_TO_FLOW_PHASE.get(str(context.phase).upper())
        if flow_phase is None:
            raise ValueError(f"unsupported phase {context.phase!r}")
        window_plan = _window_plan(plan)
        payload_specs = _payload_specs(phase_ready_context)
        expected_outgoing = _expected_rows_for_roles(phase_ready_context, use_send=True)
        expected_incoming = _expected_rows_for_roles(phase_ready_context, use_send=False)
        local_global_rank = int(phase_ready_context.global_rank)
        local_group_rank = int(phase_ready_context.ep_group_ranks.index(int(local_global_rank)))
        send_offsets = {
            str(role): [0 for _ in phase_ready_context.ep_group_ranks]
            for role in expected_outgoing
        }
        recv_offsets = {
            str(role): [0 for _ in phase_ready_context.ep_group_ranks]
            for role in expected_incoming
        }
        batches: list[ExecutionBatch] = []
        next_transfer_tag = 1
        for wave in window_plan.waves:
            phase_flows = [
                flow
                for flow in wave.flows
                if str(flow.phase) == str(flow_phase) and int(flow.row_count) > 0
            ]
            if not phase_flows:
                continue
            slices: list[TransferSlice] = []
            for flow in phase_flows:
                src_group_rank = int(flow.src_rank)
                dst_group_rank = int(flow.dst_rank)
                src_global_rank = int(plan.rank_map.group_rank_to_global_rank(src_group_rank))
                dst_global_rank = int(plan.rank_map.group_rank_to_global_rank(dst_group_rank))
                dependency_layer_id = _dependency_layer_id(
                    flow_phase=str(flow.phase),
                    context_layer_id=str(context.layer_id),
                    target_layer_id=str(plan.publication_slot["target_layer_id"]),
                )
                if int(src_global_rank) != local_global_rank and int(dst_global_rank) != local_global_rank:
                    next_transfer_tag += int(len(payload_specs))
                    continue
                for spec in payload_specs:
                    dependency_ids = _release_dependency_ids(
                        flow_phase=str(flow.phase),
                        dependency_layer_id=dependency_layer_id,
                        src_group_rank=src_group_rank,
                        src_global_rank=src_global_rank,
                        local_global_rank=local_global_rank,
                    )
                    send_offset = int(send_offsets[str(spec.payload_role)][dst_group_rank]) if int(src_global_rank) == local_global_rank else 0
                    recv_offset = int(recv_offsets[str(spec.payload_role)][src_group_rank]) if int(dst_global_rank) == local_global_rank else 0
                    task_id = f"{flow.flow_id}:{spec.payload_role}"
                    slice_ = TransferSlice(
                        task_id=task_id,
                        flow_id=str(flow.flow_id),
                        payload_role=str(spec.payload_role),
                        src_group_rank=int(src_group_rank),
                        dst_group_rank=int(dst_group_rank),
                        src_global_rank=int(src_global_rank),
                        dst_global_rank=int(dst_global_rank),
                        row_count=int(flow.row_count),
                        send_offset_rows=int(send_offset),
                        recv_offset_rows=int(recv_offset),
                        transfer_tag=int(next_transfer_tag),
                        dependency_ids=dependency_ids,
                    )
                    next_transfer_tag += 1
                    slices.append(slice_)
                    if int(src_global_rank) == local_global_rank:
                        send_offsets[str(spec.payload_role)][dst_group_rank] += int(flow.row_count)
                    if int(dst_global_rank) == local_global_rank:
                        recv_offsets[str(spec.payload_role)][src_group_rank] += int(flow.row_count)
            batches.append(
                ExecutionBatch(
                    batch_id=f"{context.phase}:wave:{wave.wave_id}",
                    wave_id=int(wave.wave_id),
                    phase=str(context.phase),
                    slices=tuple(slices),
                    collective_required=True,
                    metadata={
                        "phase": str(context.phase),
                        "phase_flow_count": int(len(phase_flows)),
                        "local_slice_count": int(len(slices)),
                    },
                )
            )
        draft = MaterializedPlan(
            publication_slot=dict(plan.publication_slot),
            local_global_rank=local_global_rank,
            local_group_rank=local_group_rank,
            phase=str(context.phase),
            payload_specs=payload_specs,
            batches=tuple(batches),
            rank_map=plan.rank_map,
            expected_outgoing_rows=expected_outgoing,
            expected_incoming_rows=expected_incoming,
            logical_plan_digest=str(plan.logical_plan_digest),
            published_plan_digest=str(plan.published_plan_digest),
            layout_digest=str(context.layout_digest),
            materialized_plan_digest="pending",
            metadata={
                "phase_ready_context": phase_ready_context.to_dict(),
                "layer_id": str(context.layer_id),
            },
        )
        finalized = MaterializedPlan(
            publication_slot=draft.publication_slot,
            local_global_rank=draft.local_global_rank,
            local_group_rank=draft.local_group_rank,
            phase=draft.phase,
            payload_specs=draft.payload_specs,
            batches=draft.batches,
            rank_map=draft.rank_map,
            expected_outgoing_rows=draft.expected_outgoing_rows,
            expected_incoming_rows=draft.expected_incoming_rows,
            logical_plan_digest=draft.logical_plan_digest,
            published_plan_digest=draft.published_plan_digest,
            layout_digest=draft.layout_digest,
            materialized_plan_digest=draft.recompute_materialized_plan_digest(),
            metadata=draft.metadata,
        )
        validation = validate_materialized_layout(finalized, phase_ready_context)
        if not validation.valid:
            raise ValueError(validation.reason or "materialized layout invalid")
        return finalized


class CommonPlanValidator:
    def validate(self, plan: MaterializedPlan, context: ActualPhaseContext) -> ValidationResult:
        try:
            plan.validate()
            context.validate()
            phase_ready_context = _coerce_phase_ready_context(dict(plan.metadata).get("phase_ready_context"))
        except Exception as exc:
            return ValidationResult(valid=False, stage="materialized_plan", reason=str(exc))
        if str(plan.layout_digest) != str(context.layout_digest):
            return ValidationResult(valid=False, stage="layout_digest", reason="layout_digest_mismatch")
        if str(dict(plan.metadata).get("layer_id", "")) != str(context.layer_id):
            return ValidationResult(valid=False, stage="identity", reason="layer_id_mismatch")
        if str(plan.phase) != str(context.phase):
            return ValidationResult(valid=False, stage="identity", reason="phase_mismatch")
        if str(plan.materialized_plan_digest) != str(plan.recompute_materialized_plan_digest()):
            return ValidationResult(valid=False, stage="materialized_digest", reason="materialized_digest_mismatch")
        return validate_materialized_layout(plan, phase_ready_context)
