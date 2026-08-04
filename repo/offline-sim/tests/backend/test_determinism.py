from __future__ import annotations

import hashlib
import json

from tests.backend.conftest import Phase, make_system, make_task


def _run_once() -> str:
    system = make_system(
        world_size=1,
        capacity=64,
        posting_fixed_ns=2,
        drain_fixed_ns=4,
    )
    c0 = Phase(0, 0, "COMBINE")
    d1 = Phase(0, 1, "DISPATCH")
    system.backend.register_local_path_spec(
        combine_phase_key=c0,
        next_dispatch_phase_key=d1,
        rank_id=0,
        combine_release_to_router_ready_ns=7,
        router_and_pack_ns=11,
    )
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[64],
        created_at_ns=0,
    )
    task = make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=64)
    system.backend.register_canonical_task_catalogue([task])
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=0)
    system.kernel.run_until(3)
    system.backend.on_transfer_completed(task_id=task.task_id, at_ns=5)
    system.kernel.run_until()
    normalized = [
        {
            "kind": kind,
            "at_ns": at_ns,
            "payload": {
                key: str(value)
                for key, value in sorted(payload.items(), key=lambda item: item[0])
            },
        }
        for kind, at_ns, payload in system.observer.rows
    ]
    data = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def test_backend_timeline_is_deterministic_across_100_runs():
    digests = {_run_once() for _ in range(100)}
    assert len(digests) == 1
