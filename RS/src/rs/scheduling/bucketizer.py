from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CanonicalBucketTask:
    phase: str
    src_group_rank: int
    dst_group_rank: int
    row_offset: int
    row_count: int
    task_id: str
    release_dependency: str

    def to_tuple(self) -> tuple[Any, ...]:
        return (
            str(self.phase),
            int(self.src_group_rank),
            int(self.dst_group_rank),
            int(self.row_offset),
            int(self.row_count),
            str(self.task_id),
            str(self.release_dependency),
        )


class ReplayWindowLike(Protocol):
    p0_truth_rows: tuple[tuple[int, ...], ...]
    p1_truth_rows: tuple[tuple[int, ...], ...]
    p2_truth_rows: tuple[tuple[int, ...], ...]


class CanonicalBucketizer:
    def __init__(self, *, bucket_rows: int) -> None:
        self.bucket_rows = int(bucket_rows)

    def bucketize(self, replay_window: ReplayWindowLike) -> tuple[CanonicalBucketTask, ...]:
        tasks: list[CanonicalBucketTask] = []
        for phase, matrix, dependency in (
            ("P0", replay_window.p0_truth_rows, "none"),
            ("P1", replay_window.p1_truth_rows, "after_p0"),
            ("P2", replay_window.p2_truth_rows, "after_p1"),
        ):
            for src_rank, row in enumerate(matrix):
                for dst_rank, value in enumerate(row):
                    row_count = int(value)
                    if src_rank == dst_rank or row_count <= 0:
                        continue
                    step = row_count if self.bucket_rows <= 0 else self.bucket_rows
                    offset = 0
                    bucket_ordinal = 0
                    while offset < row_count:
                        current = min(step, row_count - offset)
                        tasks.append(
                            CanonicalBucketTask(
                                phase=phase,
                                src_group_rank=int(src_rank),
                                dst_group_rank=int(dst_rank),
                                row_offset=int(offset),
                                row_count=int(current),
                                task_id=f"{phase}:{src_rank}->{dst_rank}:bucket:{bucket_ordinal}",
                                release_dependency=dependency,
                            )
                        )
                        offset += current
                        bucket_ordinal += 1
        return tuple(tasks)

    @staticmethod
    def digest(tasks: tuple[CanonicalBucketTask, ...]) -> str:
        digest = hashlib.sha256()
        for task in tasks:
            digest.update(repr(task.to_tuple()).encode("utf-8"))
        return digest.hexdigest()


__all__ = [
    "CanonicalBucketTask",
    "CanonicalBucketizer",
]
