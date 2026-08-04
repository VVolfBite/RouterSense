from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from rs.runtime.online.megatron_ep.execution.async_p2p_executor import (
    _digest_sequence_items,
    _pair_index,
    _sequence_entry,
    validate_async_phase_preflight,
)
from rs.scheduling.phase_execution import BucketTask, PayloadSlice, PhaseExecutionPlan, PlanWave
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


REPO_ROOT = Path(__file__).resolve().parents[3]


def _plan_for_context(*, context, tasks: list[BucketTask], preflight_mode: str = "full") -> PhaseExecutionPlan:
    return PhaseExecutionPlan(
        plan_key=dict(context.plan_key),
        phase=context.phase,
        policy_name="test",
        policy_version="v1",
        control_mode="sync_before_phase",
        execution_mode="joint_window_async_p2p",
        transport_mutation=True,
        is_shadow_only=False,
        future_hint_mode="none",
        root_rank=0,
        observation_digest="obs",
        plan_hash="plan",
        waves=(PlanWave(wave_id=0, phase=context.phase, bucket_tasks=tuple(tasks)),),
        metrics={"preflight_mode": preflight_mode, "emit_detailed_task_artifacts": True},
    )


def _task(*, context, src: int, dst: int, sender_offset: int, receiver_offset: int, row_count: int, tensor_role: str = "hidden_states", task_id: str = "t") -> BucketTask:
    payload = PayloadSlice(
        bundle_id=f"{task_id}:bundle",
        tensor_role=tensor_role,
        src_rank=src,
        dst_rank=dst,
        segment_ordinal=0,
        sender_offset_rows=sender_offset,
        receiver_offset_rows=receiver_offset,
        row_count=row_count,
        dtype="torch.float32",
        shape_suffix=(4,),
        element_size_bytes=4,
        payload_byte_count=row_count * 16,
        packed_layout_id="packed",
    )
    return BucketTask(
        task_id=task_id,
        bundle_id=f"{task_id}:bundle",
        phase=context.phase,
        src_rank=src,
        dst_rank=dst,
        source_peer_index=context.ep_group_ranks.index(src),
        destination_peer_index=context.ep_group_ranks.index(dst),
        segment_ordinal=0,
        bucket_ordinal=0,
        sender_offset_rows=sender_offset,
        receiver_offset_rows=receiver_offset,
        row_count=row_count,
        byte_count=row_count * 16,
        packed_send_layout_id="packed",
        canonical_receive_layout_id="recv",
        payload_slices=(payload,),
    )


def _patch_all_gather_same_rank(monkeypatch, module):
    def _all_gather(outputs, tensor, group=None):
        del group
        for out in outputs:
            out.copy_(tensor)
    monkeypatch.setattr(module.dist, "all_gather", _all_gather)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)


def test_local_copy_coverage_is_counted(monkeypatch) -> None:
    from rs.runtime.online.megatron_ep.execution import async_p2p_executor as module

    context = make_contexts_from_matrix(phase="P0", matrix=((2, 3), (1, 0)))[0]
    tasks = [
        _task(context=context, src=0, dst=0, sender_offset=0, receiver_offset=0, row_count=2, task_id="local"),
        _task(context=context, src=1, dst=0, sender_offset=0, receiver_offset=2, row_count=1, task_id="remote"),
    ]
    plan = _plan_for_context(context=context, tasks=tasks)
    _patch_all_gather_same_rank(monkeypatch, module)
    result = validate_async_phase_preflight(
        context=context,
        plan=plan,
        tensor_role="hidden_states",
        process_group=None,
        rank_context={"global_rank": 0, "local_rank": 0},
        mode="full",
    )
    assert result.ok is True
    assert result.reason != "recv_coverage_invalid"


def test_local_copy_coverage_gap_fails(monkeypatch) -> None:
    from rs.runtime.online.megatron_ep.execution import async_p2p_executor as module

    context = make_contexts_from_matrix(phase="P0", matrix=((0, 3), (2, 0)))[0]
    tasks = [
        _task(context=context, src=1, dst=0, sender_offset=0, receiver_offset=0, row_count=1, task_id="remote"),
    ]
    plan = _plan_for_context(context=context, tasks=tasks)
    _patch_all_gather_same_rank(monkeypatch, module)
    result = validate_async_phase_preflight(
        context=context,
        plan=plan,
        tensor_role="hidden_states",
        process_group=None,
        rank_context={"global_rank": 0, "local_rank": 0},
        mode="full",
    )
    assert result.ok is False
    assert result.reason == "recv_coverage_invalid"


def test_sequence_digest_is_order_sensitive() -> None:
    a = _digest_sequence_items([(1, 2, 3), (4, 5, 6)])
    b = _digest_sequence_items([(4, 5, 6), (1, 2, 3)])
    assert a != b


def test_pair_index_uses_group_rank_not_global_rank() -> None:
    assert _pair_index(src_rank=2, dst_rank=3, ep_group_ranks=(2, 3)) == 1


def test_sequence_entry_includes_microbatch_and_forward_epoch() -> None:
    context = make_contexts_from_matrix(phase="P0", matrix=((0, 2), (1, 0)))[0]
    task = _task(context=context, src=0, dst=1, sender_offset=0, receiver_offset=0, row_count=2)
    seq0 = _sequence_entry(
        context=context,
        phase="P0",
        tensor_role="hidden_states",
        wave_id=0,
        task=task,
        row_count=2,
        dtype="torch.float32",
        shape_suffix=(4,),
    )
    other = replace(context, forward_epoch=1, plan_key={**context.plan_key, "microbatch_id": "mb1"})
    seq1 = _sequence_entry(
        context=other,
        phase="P0",
        tensor_role="hidden_states",
        wave_id=0,
        task=task,
        row_count=2,
        dtype="torch.float32",
        shape_suffix=(4,),
    )
    assert seq0 != seq1


def test_gpu_runners_support_help_and_dry_run(tmp_path: Path) -> None:
    scripts = [
        "experiments/distributed/run_gpu_b2_lifecycle.py",
        "experiments/distributed/run_gpu_c2_async_correctness.py",
        "experiments/distributed/run_gpu_a2_strategy_compare.py",
    ]
    for script in scripts:
        help_proc = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=str(REPO_ROOT),
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        assert help_proc.returncode == 0
        out_dir = tmp_path / Path(script).stem
        dry_proc = subprocess.run(
            [sys.executable, script, "--config", "configs/comparison/natural_256x128_4gpu.yaml", "--output-dir", str(out_dir), "--dry-run"],
            cwd=str(REPO_ROOT),
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        assert dry_proc.returncode == 0
        payload = json.loads(dry_proc.stdout)
        assert payload["dry_run"] is True
