from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.distributed as dist

from .context import InvariantContext
from .errors import InvariantFailure, RouterSenseInvariantError


_ERROR_IDS = {
    "": 0,
    "RS-STARTUP-001": 1001,
    "RS-STATE-001": 2001,
    "RS-PLANNING-001": 3001,
    "RS-COMPILER-001": 4001,
    "RS-COMPILER-MISSING-TASKS": 4002,
    "RS-TRANSPORT-001": 5001,
    "RS-OFFLINE-001": 6001,
    "RS-REPORT-001": 7001,
}
_STAGE_IDS = {
    "": 0,
    "startup": 1,
    "planning": 2,
    "compiler": 3,
    "transport": 4,
    "offline": 5,
    "report": 6,
}


def distributed_invariant_gate(
    *,
    local_failure: InvariantFailure | None,
    process_group: Any,
    context: InvariantContext,
) -> None:
    if not dist.is_available() or not dist.is_initialized():
        if local_failure is not None:
            raise RouterSenseInvariantError(local_failure)
        return
    group = process_group if process_group is not None else dist.group.WORLD
    rank = int(dist.get_rank(group))
    device = torch.device("cpu")
    local = torch.tensor(
        [
            1 if local_failure is not None else 0,
            int(_ERROR_IDS.get(local_failure.error_code if local_failure else "", -1)),
            int(_STAGE_IDS.get(local_failure.stage if local_failure else "", -1)),
            int(rank),
            int(local_failure.layer_id if local_failure and local_failure.layer_id is not None else -1),
            0 if not local_failure or not local_failure.phase else (1 if str(local_failure.phase) == "P0" else 2),
        ],
        dtype=torch.long,
        device=device,
    )
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    failed_rows = [row.tolist() for row in gathered if int(row[0].item()) != 0]
    if not failed_rows:
        return
    first = failed_rows[0]
    failure = local_failure or InvariantFailure(
        error_code=next((key for key, value in _ERROR_IDS.items() if value == int(first[1])), "RS-TRANSPORT-001"),
        stage=next((key for key, value in _STAGE_IDS.items() if value == int(first[2])), str(context.stage)),
        message="distributed invariant gate observed a peer failure",
        rank=int(first[3]),
        layer_id=None if int(first[4]) < 0 else int(first[4]),
        phase="" if int(first[5]) <= 0 else ("P0" if int(first[5]) == 1 else "P1"),
        logical_plan_digest=str(context.logical_plan_digest),
        compiled_plan_digest=str(context.compiled_plan_digest),
        execution_plan_digest=str(context.execution_plan_digest),
    )
    raise RouterSenseInvariantError(failure)


def invariant_failure_to_dict(failure: InvariantFailure) -> dict[str, Any]:
    return asdict(failure)

