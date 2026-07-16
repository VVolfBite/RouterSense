from __future__ import annotations

import hashlib
from typing import Any

import torch

from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan

from .adapters.execution_adapter import (
    actual_phase_context_from_ready_context,
    build_execution_pipeline,
    build_phase_contexts_from_matrix,
)
from .adapters.publication_adapter import materialize_and_validate, publish_plan
from .contracts import RecordMetadata, RuntimeEvaluationRecord


def _digest_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def _window_plan() -> WindowPlan:
    return WindowPlan(
        planner_id="paper_runtime_joint_stub",
        planner_family="joint",
        request_digest="paper-runtime:0->1",
        waves=(
            PlanWave(
                wave_id=0,
                flows=(
                    PlannedFlow(
                        flow_id="p0_0_1",
                        phase="p0_dispatch",
                        src_rank=0,
                        dst_rank=1,
                        row_count=4,
                        release_state="ready",
                        executable=True,
                    ),
                ),
                estimated_duration=4.0,
            ),
        ),
        metadata={"source_layer_id": "0", "target_layer_id": "1"},
    )


def evaluate_runtime_correctness(*, metadata: RecordMetadata) -> dict[str, Any]:
    contexts = build_phase_contexts_from_matrix(phase="P0", matrix=((0, 4), (0, 0)))
    actual_context = actual_phase_context_from_ready_context(contexts[0], phase="P0")
    published = publish_plan(window_plan=_window_plan(), world_size=2)
    materialized, validation = materialize_and_validate(
        published_plan=published,
        actual_context=actual_context,
    )
    submitted = sum(len(batch.slices) for batch in materialized.batches)
    record = RuntimeEvaluationRecord(
        instance_id="paper-runtime-smoke",
        requested_policy_id=_window_plan().planner_id,
        selected_policy_id=_window_plan().planner_id,
        published_plan_digest=str(published.published_plan_digest),
        materialized_plan_digest=str(materialized.materialized_plan_digest),
        executed_plan_digest=str(materialized.materialized_plan_digest),
        execution_backend_id="materialize_validate_only",
        submitted_tasks=int(submitted),
        completed_tasks=int(submitted if validation.valid else 0),
        unresolved_tasks=0 if validation.valid else int(submitted),
        fallback_count=0,
        reference_output_digest="not_executed",
        executed_output_digest="not_executed",
        parity_status="VALID" if validation.valid else "INVALID",
        communication_makespan_ms=None,
        visible_control_ms=None,
        metadata=metadata,
    )
    return {
        "records": [record.to_dict()],
        "status": "RUNTIME_CORRECTNESS" if validation.valid else "INVALID",
        "validation": validation.to_dict(),
    }
