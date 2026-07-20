from __future__ import annotations

import math
from collections.abc import Sequence

from rs.core.contracts import (
    ExpertRoutePrediction,
    ExpertScoreDistribution,
    MatrixRows,
    RankedExpertRoutes,
    TrafficForecastEnvelope,
)

from .traffic_envelope import build_traffic_forecast_envelope


class RouteToTrafficMapper:
    def map(
        self,
        route_prediction: ExpertRoutePrediction,
        *,
        source_rank: int,
        expert_owner_by_id: tuple[int, ...],
        world_size: int,
        include_self: bool = False,
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
                if not include_self and int(source_rank) == dst_rank:
                    continue
                traffic[int(source_rank)][dst_rank] += 1
        return tuple(tuple(int(value) for value in row) for row in traffic)

    def map_ranked(
        self,
        ranked_routes: RankedExpertRoutes,
        *,
        expert_owner_by_id: tuple[int, ...],
        world_size: int,
        include_self: bool = False,
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
                include_self=include_self,
            )
            for src_rank, row in enumerate(source_matrix):
                for dst_rank, value in enumerate(row):
                    traffic[src_rank][dst_rank] += int(value)
        return tuple(tuple(int(value) for value in row) for row in traffic)

    def map_score_distribution(
        self,
        score_distribution: ExpertScoreDistribution,
        *,
        expert_owner_by_id: tuple[int, ...],
        world_size: int,
        include_self: bool = True,
    ) -> MatrixRows:
        """Map full expert scores to an integer expected traffic matrix.

        Each token's scores are converted to capped inclusion mass ``q_e`` with
        ``0 <= q_e <= 1`` and ``sum_e q_e = top_k``.  Aggregation and largest-
        remainder rounding preserve every source rank's known total assignment
        count exactly when ``include_self=True``.
        """

        self._validate_world_size(world_size)
        score_distribution.validate(
            world_size=world_size,
            expert_count=len(expert_owner_by_id),
        )
        self._validate_owners(expert_owner_by_id=expert_owner_by_id, world_size=world_size)
        expected = [[0.0 for _ in range(world_size)] for _ in range(world_size)]
        for source_rank, token_rows in enumerate(score_distribution.scores_by_source_rank):
            for scores in token_rows:
                inclusion = self._inclusion_mass(
                    scores,
                    top_k=int(score_distribution.top_k),
                    score_domain=str(score_distribution.score_domain),
                )
                for expert_id, mass in enumerate(inclusion):
                    destination = int(expert_owner_by_id[expert_id])
                    if not include_self and source_rank == destination:
                        continue
                    expected[source_rank][destination] += float(mass)
        output: list[tuple[int, ...]] = []
        for source_rank, row in enumerate(expected):
            if include_self:
                target_total = (
                    len(score_distribution.scores_by_source_rank[source_rank])
                    * int(score_distribution.top_k)
                )
            else:
                target_total = int(round(sum(row)))
            output.append(self._largest_remainder_round(row, target_total=target_total))
        return tuple(output)

    def map_score_distribution_to_envelope(
        self,
        score_distribution: ExpertScoreDistribution,
        *,
        predictor_id: str,
        expert_owner_by_id: tuple[int, ...],
        world_size: int,
        relative_error_bound: float | Sequence[float] | None,
        calibration_id: str = "",
        source_layer_id: str | None = None,
        target_layer_id: str | None = None,
        precedence_margin: float = 0.0,
        precedence_confidence_floor: float = 0.5,
        metadata: dict[str, object] | None = None,
    ) -> TrafficForecastEnvelope:
        mean = self.map_score_distribution(
            score_distribution,
            expert_owner_by_id=expert_owner_by_id,
            world_size=world_size,
            include_self=True,
        )
        return build_traffic_forecast_envelope(
            predictor_id=predictor_id,
            mean_rows=mean,
            relative_error_bound=relative_error_bound,
            precedence_margin=precedence_margin,
            precedence_confidence_floor=precedence_confidence_floor,
            calibration_id=calibration_id,
            source_layer_id=source_layer_id,
            target_layer_id=target_layer_id,
            metadata={
                "adapter": "fixed_topk_capped_inclusion_v1",
                "score_domain": score_distribution.score_domain,
                "top_k": int(score_distribution.top_k),
                "routed_expert_count": len(expert_owner_by_id),
                **dict(metadata or {}),
            },
        )

    @staticmethod
    def _normalized_scores(scores: Sequence[float], *, score_domain: str) -> tuple[float, ...]:
        values = [float(value) for value in scores]
        if score_domain == "logits":
            maximum = max(values)
            weights = [math.exp(value - maximum) for value in values]
        else:
            weights = [max(0.0, value) for value in values]
        total = sum(weights)
        if total <= 0.0:
            return tuple(1.0 / len(weights) for _ in weights)
        return tuple(value / total for value in weights)

    @classmethod
    def _inclusion_mass(
        cls,
        scores: Sequence[float],
        *,
        top_k: int,
        score_domain: str,
    ) -> tuple[float, ...]:
        probabilities = cls._normalized_scores(scores, score_domain=score_domain)
        if top_k >= len(probabilities):
            return tuple(1.0 for _ in probabilities)
        # Capped proportional projection: q_e=min(1, alpha*p_e), sum(q)=k.
        low = 0.0
        high = max(1.0, float(top_k) / max(min(probabilities), 1e-15))
        for _ in range(80):
            middle = (low + high) / 2.0
            mass = sum(min(1.0, middle * value) for value in probabilities)
            if mass < float(top_k):
                low = middle
            else:
                high = middle
        alpha = high
        projected = [min(1.0, alpha * value) for value in probabilities]
        # Correct floating error without violating the cap.
        residual = float(top_k) - sum(projected)
        if abs(residual) > 1e-10:
            order = sorted(range(len(projected)), key=lambda index: (-probabilities[index], index))
            for index in order:
                room = 1.0 - projected[index] if residual > 0.0 else projected[index]
                delta = math.copysign(min(abs(residual), room), residual)
                projected[index] += delta
                residual -= delta
                if abs(residual) <= 1e-10:
                    break
        return tuple(projected)

    @staticmethod
    def _largest_remainder_round(values: Sequence[float], *, target_total: int) -> tuple[int, ...]:
        floors = [max(0, int(math.floor(float(value)))) for value in values]
        residual = int(target_total) - sum(floors)
        if residual < 0:
            order = sorted(range(len(values)), key=lambda index: (float(values[index]) - floors[index], index))
            for index in order[: -residual]:
                if floors[index] <= 0:
                    raise ValueError("cannot preserve target total during rounding")
                floors[index] -= 1
        elif residual > 0:
            order = sorted(
                range(len(values)),
                key=lambda index: (-(float(values[index]) - floors[index]), index),
            )
            for offset in range(residual):
                floors[order[offset % len(order)]] += 1
        if sum(floors) != int(target_total):
            raise ValueError("largest-remainder rounding failed to preserve row total")
        return tuple(floors)

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
