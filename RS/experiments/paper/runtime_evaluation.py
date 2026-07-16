from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan

from .adapters.execution_adapter import actual_phase_context_from_ready_context, build_phase_contexts_from_matrix
from .adapters.publication_adapter import materialize_and_validate, publish_plan
from .contracts import RecordMetadata, RuntimeEvaluationRecord


def _hash_jsonable(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _window_plan() -> WindowPlan:
    return WindowPlan(
        planner_id="paper_materialization_contract_smoke",
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


def evaluate_materialization_contract_smoke(*, metadata: RecordMetadata) -> dict[str, Any]:
    contexts = build_phase_contexts_from_matrix(phase="P0", matrix=((0, 4), (0, 0)))
    actual_context = actual_phase_context_from_ready_context(contexts[0], phase="P0")
    published = publish_plan(window_plan=_window_plan(), world_size=2)
    materialized, validation = materialize_and_validate(
        published_plan=published,
        actual_context=actual_context,
    )
    submitted = sum(len(batch.slices) for batch in materialized.batches)
    record = RuntimeEvaluationRecord(
        instance_id="paper-materialization-contract-smoke",
        requested_policy_id=None,
        selected_policy_id=None,
        published_plan_digest=str(published.published_plan_digest),
        materialized_plan_digest=str(materialized.materialized_plan_digest),
        executed_plan_digest=None,
        execution_backend_id=None,
        submitted_tasks=int(submitted),
        completed_tasks=None,
        unresolved_tasks=None,
        fallback_count=None,
        reference_output_digest=None,
        executed_output_digest=None,
        parity_status="NOT_EXECUTED",
        communication_makespan_ms=None,
        visible_control_ms=None,
        runtime_status="MATERIALIZATION_CONTRACT_SMOKE" if validation.valid else "INVALID",
        metadata=metadata,
        evidence={"validation": validation.to_dict()},
    )
    return {
        "status": "MATERIALIZATION_CONTRACT_SMOKE" if validation.valid else "INVALID",
        "records": [record.to_dict()],
        "validation": validation.to_dict(),
    }


def evaluate_runtime_correctness_with_gloo(*, metadata: RecordMetadata) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[1] / "distributed" / "run_m123_integrated_publication_execution_gloo.py"
    with tempfile.TemporaryDirectory(prefix="rs_paper_gloo_") as tmpdir:
        tmp = Path(tmpdir)
        summary_path = tmp / "summary.json"
        output_dir = tmp / "runner"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--quiet",
                "--output-dir",
                str(output_dir),
                "--summary-path",
                str(summary_path),
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            record = RuntimeEvaluationRecord(
                instance_id="paper-gloo-runtime-wrapper",
                requested_policy_id=None,
                selected_policy_id=None,
                published_plan_digest=None,
                materialized_plan_digest=None,
                executed_plan_digest=None,
                execution_backend_id=None,
                submitted_tasks=None,
                completed_tasks=None,
                unresolved_tasks=None,
                fallback_count=None,
                reference_output_digest=None,
                executed_output_digest=None,
                parity_status="NOT_EXECUTED",
                communication_makespan_ms=None,
                visible_control_ms=None,
                runtime_status="ENVIRONMENT_BLOCKED",
                metadata=metadata,
                evidence={"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode},
            )
            return {"status": "ENVIRONMENT_BLOCKED", "records": [record.to_dict()], "tensor_parity_pass": False}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ranks = list(summary.get("ranks", []))
        published_digests = sorted({str(item.get("published_plan_digest")) for item in ranks})
        materialized_pairs = [
            (str(item.get("p0_materialized_plan_digest")), str(item.get("p1_materialized_plan_digest")))
            for item in ranks
        ]
        materialized_bundle_digest = _hash_jsonable(materialized_pairs)
        submitted = int(sum(int(item.get("submitted_task_count", 0) or 0) for item in ranks))
        completed = int(sum(int(item.get("completed_task_count", 0) or 0) for item in ranks))
        unresolved = int(sum(int(item.get("unresolved_task_count", 0) or 0) for item in ranks))
        observed_outputs = []
        expected_outputs = []
        for rank_payload in ranks:
            for role_name, role_payload in dict(rank_payload.get("full_group", {}).get("results", {})).items():
                observed_outputs.append(role_payload.get("sync", {}).get("output_payload", {}))
                expected_outputs.append(role_payload.get("expected_output"))
        executed_output_digest = _hash_jsonable(observed_outputs) if observed_outputs else None
        reference_output_digest = _hash_jsonable(expected_outputs) if expected_outputs else None
        tensor_parity_pass = executed_output_digest == reference_output_digest and bool(observed_outputs)
        executed_plan_digest = None
        status = "PARTIAL_EXECUTED_DIGEST_MISSING"
        if executed_plan_digest is not None and executed_plan_digest == materialized_bundle_digest and submitted == completed and unresolved == 0 and tensor_parity_pass:
            status = "RUNTIME_CORRECTNESS"
        record = RuntimeEvaluationRecord(
            instance_id="paper-gloo-runtime-wrapper",
            requested_policy_id="formal_m123_integrated_publication_execution_gloo",
            selected_policy_id="formal_m123_integrated_publication_execution_gloo",
            published_plan_digest=published_digests[0] if len(published_digests) == 1 else None,
            materialized_plan_digest=materialized_bundle_digest,
            executed_plan_digest=executed_plan_digest,
            execution_backend_id=str(summary.get("execution_backend")),
            submitted_tasks=submitted,
            completed_tasks=completed,
            unresolved_tasks=unresolved,
            fallback_count=None,
            reference_output_digest=reference_output_digest,
            executed_output_digest=executed_output_digest,
            parity_status="PASS" if tensor_parity_pass else "FAILED",
            communication_makespan_ms=None,
            visible_control_ms=None,
            runtime_status=status,
            metadata=metadata,
            evidence={"summary_path": str(summary_path), "runner_summary": summary},
        )
        return {"status": status, "records": [record.to_dict()], "tensor_parity_pass": tensor_parity_pass}
