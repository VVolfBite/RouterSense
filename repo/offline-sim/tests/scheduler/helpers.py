from __future__ import annotations

from rs_sim import (
    EdgeKey,
    ExpectationOrigin,
    PhaseKey,
    PhaseKind,
    ReceiveExpectation,
    WindowKey,
)
from rs_sim.scheduler.stable import stable_digest


def phase(layer: int = 1, kind: str = "P2_DISPATCH") -> PhaseKey:
    phase_kind = PhaseKind.COMBINE if "COMBINE" in str(kind).upper() else PhaseKind.DISPATCH
    return PhaseKey(
        run_id="run",
        sample_id="scheduler-test",
        layer_index=int(layer),
        phase_kind=phase_kind,
    )


def window(index: int = 0) -> WindowKey:
    return WindowKey(run_id="run", sample_id="scheduler-test", window_index=int(index))


def expectation(
    phase_key: PhaseKey,
    src: int,
    dst: int,
    total_bytes: int,
    *,
    created_at_ns: int = 10,
) -> ReceiveExpectation:
    edge = EdgeKey(phase_key=phase_key, src_rank=int(src), dst_rank=int(dst))
    digest = stable_digest(
        {
            "phase": phase_key,
            "src": int(src),
            "dst": int(dst),
            "total_bytes": int(total_bytes),
        }
    )
    dispatch = phase_key.phase_kind is PhaseKind.DISPATCH
    return ReceiveExpectation(
        edge_key=edge,
        phase_key=phase_key,
        src_rank=int(src),
        dst_rank=int(dst),
        total_expected_payload_bytes=int(total_bytes),
        expectation_digest=digest,
        origin=(
            ExpectationOrigin.DISPATCH_DESCRIPTOR
            if dispatch
            else ExpectationOrigin.COMBINE_REALIZED
        ),
        created_at_ns=int(created_at_ns),
        zero_edge=int(total_bytes) == 0,
        descriptor_digest_or_none=(digest if dispatch else None),
    )


def make_ready(stack, task_ids, *, permit_at: int = 20, payload_at: int = 30):
    for task_id in task_ids:
        stack.runtime.note_permit(task_id, at_ns=permit_at)
        stack.runtime.note_source_payload_ready(task_id, at_ns=payload_at)
