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
        traffic = [[0 for _ in range(int(world_size))] for _ in range(int(world_size))]
        for expert_row in route_prediction.expert_ids:
            for expert_id in expert_row:
                expert_index = int(expert_id)
                if expert_index < 0 or expert_index >= len(expert_owner_by_id):
                    raise ValueError(f"expert_id {expert_index} missing from expert_owner_by_id")
                dst_rank = int(expert_owner_by_id[expert_index])
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


__all__ = ["RouteToTrafficMapper"]
