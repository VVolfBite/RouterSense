from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from rs.scheduling.phase_execution import AbstractPhaseExecutionPlan, PhaseReadyContext
from rs.scheduling.phase_execution_utils import materialize_local_execution_plan
from rs.scheduling.registry import resolve_phase_policy
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phase_ready_context_roundtrip_from_agreement_payload() -> None:
    ctx = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 32), (16, 0)),
        p2_hint_mode="deterministic_stub",
    )[0]
    rebuilt = PhaseReadyContext.from_agreement_payload(ctx.to_agreement_payload())
    assert rebuilt.to_agreement_payload() == ctx.to_agreement_payload()


def test_phase_ready_context_roundtrip_from_planning_summary() -> None:
    ctx = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 32), (16, 0)),
        p2_hint_mode="deterministic_stub",
    )[0]
    summary = ctx.to_planning_summary()
    rebuilt = PhaseReadyContext.from_planning_summary(summary)
    rebuilt_summary = rebuilt.to_planning_summary()
    assert rebuilt_summary.to_dict() == summary.to_dict()
    assert rebuilt.local_rank == ctx.local_rank
    assert rebuilt.per_peer_bytes == ctx.per_peer_bytes
    assert rebuilt.p2_hint.hint_mode == "none"


def test_phase_ready_context_roundtrip_from_artifact_dict() -> None:
    ctx = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 32), (16, 0)),
        p2_hint_mode="deterministic_stub",
    )[0]
    rebuilt = PhaseReadyContext.from_dict(ctx.to_dict())
    assert rebuilt.to_dict() == ctx.to_dict()


def test_abstract_plan_materializes_to_local_execution_plan() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 48), (32, 0)),
        p2_hint_mode="deterministic_stub",
    )
    policy = resolve_phase_policy(policy_name="greedy_bucket", bucket_rows=16)
    full_plan = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
    abstract = AbstractPhaseExecutionPlan.from_dict(full_plan.to_abstract_plan().to_dict())
    local_plan = materialize_local_execution_plan(local_context=contexts[0], abstract_plan=abstract)
    assert local_plan.plan_hash == full_plan.plan_hash
    assert len(local_plan.waves) == len(full_plan.waves)
    assert sum(len(wave.bucket_tasks) for wave in local_plan.waves) >= 1
    for key in (
        "build_transfer_layouts_and_tasks_time_us",
        "sort_tasks_time_us",
        "pack_phase_tasks_time_us",
        "finalize_execution_plan_time_us",
    ):
        assert key in full_plan.metrics
        assert float(full_plan.metrics[key]) >= 0.0
    for key in (
        "materialize_local_execution_plan_total_time_us",
        "local_outgoing_catalog_build_time_us",
        "local_incoming_catalog_build_time_us",
        "local_wave_materialize_time_us",
        "local_materialize_validate_time_us",
    ):
        assert key in local_plan.metrics
        assert float(local_plan.metrics[key]) >= 0.0
    assert int(full_plan.metrics["remote_row_count"]) > 0
    assert int(full_plan.metrics["remote_byte_count"]) > 0
    assert int(full_plan.metrics["task_count"]) >= int(full_plan.metrics["flow_count"]) >= 1


def test_abstract_plan_materialization_matches_receiver_side_wave_participation() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=(
            (0, 48, 0, 16),
            (32, 0, 24, 0),
            (0, 40, 0, 8),
            (12, 0, 20, 0),
        ),
        p2_hint_mode="deterministic_stub",
    )
    contexts = tuple(replace(context, plan_key={"layer_id": "0", "phase": "P0"}) for context in contexts)
    policy = resolve_phase_policy(policy_name="greedy_bucket", bucket_rows=16)
    full_plan = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
    abstract = AbstractPhaseExecutionPlan.from_dict(full_plan.to_abstract_plan().to_dict())

    for context in contexts:
        local_plan = materialize_local_execution_plan(local_context=context, abstract_plan=abstract)
        incoming_offsets = {
            (int(slot.src_rank), int(slot.dst_rank)): int(slot.receive_offset_rows)
            for slot in context.incoming_slots
            if int(slot.row_count) > 0 and not bool(slot.is_local)
        }
        for wave in local_plan.waves:
            expected_participation = any(
                int(task.src_rank) == int(context.global_rank) or int(task.dst_rank) == int(context.global_rank)
                for task in full_plan.waves[int(wave.wave_id)].bucket_tasks
            )
            assert bool(wave.bucket_tasks) == bool(expected_participation)
            for task in wave.bucket_tasks:
                if int(task.dst_rank) == int(context.global_rank):
                    expected_offset = incoming_offsets[(int(task.src_rank), int(task.dst_rank))]
                    assert int(task.receiver_offset_rows) >= int(expected_offset)


def test_replay_online_planner_runs_from_phase_context_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 48), (32, 0)),
        p2_hint_mode="deterministic_stub",
    )
    for rank, ctx in enumerate(contexts):
        (run_dir / f"rank{rank}_phase_contexts.jsonl").write_text(
            json.dumps(ctx.to_dict(), ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    output_path = tmp_path / "replay.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.diagnostics.replay_online_planner",
            "--run-dir",
            str(run_dir),
            "--policy",
            "greedy_bucket",
            "--bucket-rows",
            "16",
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["policy"] == "greedy_bucket"
    assert payload["phase_count"] == 1
    assert payload["records"][0]["wave_count"] >= 1
    assert payload["records"][0]["elapsed_us"] >= 0.0
    assert payload["records"][0]["task_count"] >= 1
    assert payload["records"][0]["bucket_row_histogram"]
    assert payload["records"][0]["flow_bucket_count_summary"]["flow_count"] >= 1
