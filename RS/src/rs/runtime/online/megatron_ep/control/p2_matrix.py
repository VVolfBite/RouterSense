"""Helpers for building global prepared-plan matrices from rank-local observations."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch.distributed as dist

from rs.runtime.online.megatron_ep.contracts import RuntimeObservation


@dataclass(frozen=True)
class PreparedPlanMatrixBundle:
    dispatch_matrix: tuple[tuple[int, ...], ...]
    p1_return_matrix: tuple[tuple[int, ...], ...]
    forecast_matrix: tuple[tuple[int, ...], ...]
    p2_matrix_source: str
    p2_matrix_is_replicated_local_row: bool
    row_sums: tuple[int, ...]
    col_sums: tuple[int, ...]
    total_bytes: int
    shape: tuple[int, int]
    gather_time_us: float
    gather_status: str
    gather_call_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch_matrix": [list(row) for row in self.dispatch_matrix],
            "p1_return_matrix": [list(row) for row in self.p1_return_matrix],
            "forecast_matrix": [list(row) for row in self.forecast_matrix],
            "p2_matrix_source": self.p2_matrix_source,
            "p2_matrix_is_replicated_local_row": self.p2_matrix_is_replicated_local_row,
            "row_sums": list(self.row_sums),
            "col_sums": list(self.col_sums),
            "total_bytes": self.total_bytes,
            "shape": list(self.shape),
            "gather_time_us": self.gather_time_us,
            "gather_status": self.gather_status,
            "gather_call_count": self.gather_call_count,
        }


def build_prepared_plan_matrices(
    *,
    rank: int,
    ep_group_ranks: tuple[int, ...],
    ep_process_group: Any | None,
    layer_name: str,
    observation_p1: RuntimeObservation,
    observation_p0: RuntimeObservation | None,
) -> PreparedPlanMatrixBundle:
    num_peers = len(tuple(int(value) for value in observation_p1.per_peer_bytes))
    if num_peers <= 0:
        raise ValueError(f"{layer_name}: empty per_peer_bytes")
    local_p1 = tuple(int(value) for value in observation_p1.per_peer_bytes)
    local_p0 = (
        tuple(int(value) for value in observation_p0.per_peer_bytes)
        if observation_p0 is not None
        else local_p1
    )
    gather_start_ns = time.monotonic_ns()
    gathered_p1 = _gather_rows(
        local_row=local_p1,
        rank=rank,
        ep_group_ranks=ep_group_ranks,
        ep_process_group=ep_process_group,
    )
    gathered_p0 = _gather_rows(
        local_row=local_p0,
        rank=rank,
        ep_group_ranks=ep_group_ranks,
        ep_process_group=ep_process_group,
    )
    gather_end_ns = time.monotonic_ns()
    gather_time_us = max(0.0, float(gather_end_ns - gather_start_ns) / 1000.0)
    gather_call_count = 2
    if gathered_p1 is not None and gathered_p0 is not None:
        forecast_matrix = gathered_p1
        dispatch_matrix = gathered_p0
        p1_return_matrix = gathered_p1
        source = "gathered_global_matrix"
        replicated = False
        gather_status = "ok"
    else:
        forecast_matrix = _replicate_local_row(local_p1)
        dispatch_matrix = _replicate_local_row(local_p0)
        p1_return_matrix = forecast_matrix
        source = "replicated_local_row"
        replicated = True
        if not ep_group_ranks or len(ep_group_ranks) != len(local_p1):
            gather_status = "skipped_ep_group_size_mismatch"
        elif not dist.is_available():
            gather_status = "skipped_dist_unavailable"
        elif not dist.is_initialized():
            gather_status = "skipped_dist_uninitialized"
        else:
            gather_status = "fallback_after_gather_failure"
    row_sums = tuple(int(sum(row)) for row in forecast_matrix)
    col_sums = tuple(
        int(sum(forecast_matrix[row_idx][col_idx] for row_idx in range(num_peers)))
        for col_idx in range(num_peers)
    )
    total_bytes = int(sum(row_sums))
    return PreparedPlanMatrixBundle(
        dispatch_matrix=dispatch_matrix,
        p1_return_matrix=p1_return_matrix,
        forecast_matrix=forecast_matrix,
        p2_matrix_source=source,
        p2_matrix_is_replicated_local_row=replicated,
        row_sums=row_sums,
        col_sums=col_sums,
        total_bytes=total_bytes,
        shape=(num_peers, num_peers),
        gather_time_us=gather_time_us,
        gather_status=gather_status,
        gather_call_count=gather_call_count,
    )


def _replicate_local_row(local_row: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    num_peers = len(local_row)
    return tuple(
        tuple(int(local_row[j]) if i != j else 0 for j in range(num_peers))
        for i in range(num_peers)
    )


def _gather_rows(
    *,
    local_row: tuple[int, ...],
    rank: int,
    ep_group_ranks: tuple[int, ...],
    ep_process_group: Any | None,
) -> tuple[tuple[int, ...], ...] | None:
    if not ep_group_ranks or len(ep_group_ranks) != len(local_row):
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    try:
        ep_rank = ep_group_ranks.index(int(rank))
    except ValueError:
        return None
    payload = {
        "ep_rank": ep_rank,
        "row": list(int(value) for value in local_row),
    }
    gathered: list[dict[str, Any] | None] = [None for _ in ep_group_ranks]
    dist.all_gather_object(gathered, payload, group=ep_process_group)
    rows: list[list[int] | None] = [None for _ in ep_group_ranks]
    for item in gathered:
        if not isinstance(item, dict):
            return None
        idx = int(item.get("ep_rank", -1))
        row = [int(value) for value in item.get("row", [])]
        if idx < 0 or idx >= len(ep_group_ranks) or len(row) != len(local_row):
            return None
        rows[idx] = row
    if any(row is None for row in rows):
        return None
    return tuple(tuple(int(value) for value in row or []) for row in rows)
