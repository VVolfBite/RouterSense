from __future__ import annotations

from dataclasses import asdict
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
from rs.runtime.online.megatron_ep.materialization.layout import validate_materialized_layout
from rs.runtime.online.megatron_ep.phase import AbstractPhaseExecutionPlan, PhaseExecutionPlan, PhaseReadyContext
from rs.scheduling.phase_execution_utils import materialize_local_execution_plan, validate_phase_execution_plan
from rs.scheduling.validation import stable_hash


def _coerce_phase_ready_context(payload: object) -> PhaseReadyContext:
    if isinstance(payload, PhaseReadyContext):
        return payload
    if isinstance(payload, dict):
        return PhaseReadyContext.from_dict(dict(payload))
    raise ValueError("phase_ready_context metadata is required")


def _coerce_execution_plan(payload: object) -> PhaseExecutionPlan:
    if isinstance(payload, PhaseExecutionPlan):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("logical_plan must be a mapping payload")
    waves = payload.get("waves", [])
    if waves and isinstance(waves[0], dict) and "bucket_tasks" in waves[0]:
        return PhaseExecutionPlan.from_dict(dict(payload))
    return materialize_local_execution_plan(
        local_context=_coerce_phase_ready_context(payload.get("_phase_ready_context", {})),
        abstract_plan=AbstractPhaseExecutionPlan.from_dict(dict(payload)),
    )


class CommonPlanMaterializer:
    def materialize(self, plan: PublishedPlan, context: ActualPhaseContext) -> MaterializedPlan:
        plan.validate()
        context.validate()
        phase_ready_context = _coerce_phase_ready_context(dict(context.metadata).get("phase_ready_context"))
        if str(phase_ready_context.layer_id) != str(context.layer_id):
            raise ValueError("phase_ready_context layer_id does not match ActualPhaseContext")
        if str(phase_ready_context.phase) != str(context.phase):
            raise ValueError("phase_ready_context phase does not match ActualPhaseContext")
        logical_payload = dict(plan.logical_plan)
        if "plan_key" not in logical_payload:
            raise ValueError("logical_plan payload is missing plan_key")
        logical_payload["_phase_ready_context"] = phase_ready_context.to_dict()
        execution_plan = _coerce_execution_plan(logical_payload)
        validate_phase_execution_plan(phase_ready_context, execution_plan)
        payload_specs: list[PayloadSpec] = []
        total_rows = int(sum(phase_ready_context.recv_splits))
        for spec in phase_ready_context.payload_specs:
            elements_per_row = 1
            for dim in spec.shape_suffix:
                elements_per_row *= int(dim)
            payload_specs.append(
                PayloadSpec(
                    payload_role=str(spec.tensor_role),
                    row_count=int(total_rows),
                    element_count=int(total_rows * elements_per_row),
                    byte_count=int(total_rows * elements_per_row * int(spec.element_size_bytes)),
                    bytes_per_row=int(elements_per_row * int(spec.element_size_bytes)),
                    dtype=str(spec.dtype),
                    shape_suffix=tuple(int(dim) for dim in spec.shape_suffix),
                )
            )
        batches: list[ExecutionBatch] = []
        for wave in execution_plan.waves:
            slices: list[TransferSlice] = []
            for task in wave.bucket_tasks:
                for payload in task.payload_slices:
                    slices.append(
                        TransferSlice(
                            flow_id=f"{task.task_id}:{payload.tensor_role}",
                            task_id=str(task.task_id),
                            payload_role=str(payload.tensor_role),
                            src_rank=int(task.src_rank),
                            dst_rank=int(task.dst_rank),
                            row_count=int(payload.row_count),
                            send_offset_rows=int(payload.sender_offset_rows),
                            recv_offset_rows=int(payload.receiver_offset_rows),
                            dependency_ids=(),
                        )
                    )
            batches.append(
                ExecutionBatch(
                    batch_id=f"{context.phase}:wave:{wave.wave_id}",
                    wave_id=int(wave.wave_id),
                    slices=tuple(slices),
                    metadata={"phase": str(context.phase)},
                )
            )
        materialized = MaterializedPlan(
            published_plan_digest=str(plan.published_plan_digest),
            materialized_plan_digest="pending",
            layout_digest=str(context.layout_digest),
            payload_specs=tuple(payload_specs),
            batches=tuple(batches),
            local_rank=int(dict(context.metadata).get("local_rank", 0)),
            layer_id=str(context.layer_id),
            phase=str(context.phase),
            logical_plan_digest=str(plan.logical_plan_digest),
            expected_payload_roles=tuple(str(spec.tensor_role) for spec in phase_ready_context.payload_specs),
            metadata={
                "phase_ready_context": phase_ready_context.to_dict(),
                "phase_execution_plan": execution_plan.to_dict(),
            },
        )
        digest = stable_hash(materialized.to_dict() | {"published_plan_digest": str(plan.published_plan_digest)})
        finalized = MaterializedPlan(
            published_plan_digest=materialized.published_plan_digest,
            materialized_plan_digest=str(digest),
            layout_digest=materialized.layout_digest,
            payload_specs=materialized.payload_specs,
            batches=materialized.batches,
            local_rank=materialized.local_rank,
            layer_id=materialized.layer_id,
            phase=materialized.phase,
            logical_plan_digest=materialized.logical_plan_digest,
            expected_payload_roles=materialized.expected_payload_roles,
            metadata=materialized.metadata,
        )
        validation = validate_materialized_layout(finalized, phase_ready_context)
        if not validation.valid:
            raise ValueError(validation.reason or "materialized layout invalid")
        return finalized


class CommonPlanValidator:
    def validate(self, plan: MaterializedPlan, context: ActualPhaseContext) -> ValidationResult:
        try:
            phase_ready_context = _coerce_phase_ready_context(dict(plan.metadata).get("phase_ready_context"))
        except Exception as exc:
            return ValidationResult(valid=False, stage="materialized_plan", reason=str(exc))
        if str(plan.layout_digest) != str(context.layout_digest):
            return ValidationResult(valid=False, stage="layout_digest", reason="layout_digest_mismatch")
        if str(plan.layer_id) != str(context.layer_id):
            return ValidationResult(valid=False, stage="identity", reason="layer_id_mismatch")
        if str(plan.phase) != str(context.phase):
            return ValidationResult(valid=False, stage="identity", reason="phase_mismatch")
        return validate_materialized_layout(plan, phase_ready_context)
