from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankMap:
    group_ranks: tuple[int, ...]
    root_rank: int

    def validate(self) -> None:
        if not self.group_ranks:
            raise ValueError("group_ranks must be non-empty")
        if len(set(int(rank) for rank in self.group_ranks)) != len(self.group_ranks):
            raise ValueError("group_ranks must be unique")
        if any(int(rank) < 0 for rank in self.group_ranks):
            raise ValueError("group_ranks must be >= 0")
        if int(self.root_rank) not in {int(rank) for rank in self.group_ranks}:
            raise ValueError("root_rank must belong to group_ranks")

    @property
    def world_size(self) -> int:
        return len(self.group_ranks)

    @property
    def root_group_rank(self) -> int:
        self.validate()
        return self.group_ranks.index(int(self.root_rank))

    def group_rank_to_global_rank(self, group_rank: int) -> int:
        self.validate()
        if int(group_rank) < 0 or int(group_rank) >= len(self.group_ranks):
            raise ValueError(f"group_rank out of range: {group_rank}")
        return int(self.group_ranks[int(group_rank)])

    def global_rank_to_group_rank(self, global_rank: int) -> int:
        self.validate()
        try:
            return self.group_ranks.index(int(global_rank))
        except ValueError as exc:
            raise ValueError(f"global_rank not part of group: {global_rank}") from exc

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "group_ranks": [int(rank) for rank in self.group_ranks],
            "root_rank": int(self.root_rank),
            "root_group_rank": int(self.root_group_rank),
            "world_size": int(self.world_size),
        }
