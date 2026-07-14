from __future__ import annotations

from rs.core.contracts import ExpertRoutePrediction, MatrixRows, RankedExpertRoutes


class RouteToTrafficMapper:
    def map(
        self,
        route_prediction: ExpertRoutePrediction,
        *,
        source_rank: int,
        expert_owner_by_id: tuple[int, ...],
        world_size: int,
    ) -> MatrixRows:
        self._validate_world_size(world_size)
        self._validate_source_rank(source_rank=source_rank, world_size=world_size)
        route_prediction.validate(expert_count=len(expert_owner_by_id))
        self._validate_owners(expert_owner_by_id=expert_owner_by_id, world_size=world_size)
        traffic = [[0 for _ in range(int(world_size))] for _ in range(int(world_size))]
        for expert_row in route_prediction.expert_ids:
            for expert_id in expert_row:
                expert_index = int(expert_id)
                if expert_index < 0 or expert_index >= len(expert_owner_by_id):
                    raise ValueError(f"expert_id {expert_index} missing from expert_owner_by_id")
                dst_rank = int(expert_owner_by_id[expert_index])
                self._validate_owner_rank(owner_rank=dst_rank, world_size=world_size)
                if int(source_rank) == dst_rank:
                    continue
                traffic[int(source_rank)][dst_rank] += 1
        return tuple(tuple(int(value) for value in row) for row in traffic)

    def map_ranked(
        self,
        ranked_routes: RankedExpertRoutes,
        *,
        expert_owner_by_id: tuple[int, ...],
        world_size: int,
    ) -> MatrixRows:
        self._validate_world_size(world_size)
        ranked_routes.validate(world_size=world_size, expert_count=len(expert_owner_by_id))
        self._validate_owners(expert_owner_by_id=expert_owner_by_id, world_size=world_size)
        traffic = [[0 for _ in range(int(world_size))] for _ in range(int(world_size))]
        for source_rank, route_prediction in enumerate(ranked_routes.routes_by_source_rank):
            source_matrix = self.map(
                route_prediction,
                source_rank=int(source_rank),
                expert_owner_by_id=expert_owner_by_id,
                world_size=world_size,
            )
            for src_rank, row in enumerate(source_matrix):
                for dst_rank, value in enumerate(row):
                    traffic[src_rank][dst_rank] += int(value)
        return tuple(tuple(int(value) for value in row) for row in traffic)

    @staticmethod
    def _validate_world_size(world_size: int) -> None:
        if int(world_size) <= 0:
            raise ValueError("world_size must be > 0")

    @staticmethod
    def _validate_source_rank(*, source_rank: int, world_size: int) -> None:
        if int(source_rank) < 0 or int(source_rank) >= int(world_size):
            raise ValueError(f"source_rank {source_rank} outside world_size {world_size}")

    @staticmethod
    def _validate_owner_rank(*, owner_rank: int, world_size: int) -> None:
        if int(owner_rank) < 0 or int(owner_rank) >= int(world_size):
            raise ValueError(f"owner_rank {owner_rank} outside world_size {world_size}")

    def _validate_owners(self, *, expert_owner_by_id: tuple[int, ...], world_size: int) -> None:
        if not expert_owner_by_id:
            raise ValueError("expert_owner_by_id must be non-empty")
        for owner_rank in expert_owner_by_id:
            self._validate_owner_rank(owner_rank=int(owner_rank), world_size=world_size)


__all__ = ["RouteToTrafficMapper"]
