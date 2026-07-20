from __future__ import annotations

from rs.scheduling.catalog import algorithm_specs, resolve_algorithm_id

from .api import PlannerSpec, planner_family_for_spec
from .asset_registry import (
    create_planner as create_extended_planner,
    merge_planner_specs,
    resolves_planner as resolves_extended_planner,
)
from .runtime_adapter import FormalRuntimePlanner


class PlannerRegistry:
    @staticmethod
    def specs() -> tuple[PlannerSpec, ...]:
        rows: list[PlannerSpec] = []
        for spec in algorithm_specs():
            rows.append(
                PlannerSpec(
                    planner_id=str(spec.canonical_id),
                    planner_family=planner_family_for_spec(
                        family=str(spec.family),
                        scheduling_scope=str(spec.scheduling_scope),
                        reference_only=bool(spec.reference_only),
                        deployable=bool(spec.deployable),
                        supports_p2_hint=bool(spec.supports_p2_hint),
                        canonical_id=str(spec.canonical_id),
                        execution_model=str(spec.execution_model),
                    ),
                    deployable=bool(spec.deployable),
                    reference_only=bool(spec.reference_only),
                    requires_prediction=bool(spec.supports_p2_hint),
                    exact=bool(str(spec.execution_model) == "exact_reference"),
                    historical_aliases=tuple(spec.aliases + spec.deprecated_aliases),
                )
            )
        return merge_planner_specs(tuple(rows))

    @staticmethod
    def resolve(planner_id: str) -> PlannerSpec:
        """Resolve a canonical planner ID or one of its compatibility aliases."""

        requested = str(planner_id)
        for spec in PlannerRegistry.specs():
            if requested == str(spec.planner_id) or requested in tuple(str(alias) for alias in spec.historical_aliases):
                return spec
        raise ValueError(f"unknown planner {planner_id!r}")

    @staticmethod
    def canonical_id(planner_id: str) -> str:
        return str(PlannerRegistry.resolve(planner_id).planner_id)

    @staticmethod
    def create(planner_id: str, config=None, *, usage: str | None = None):
        if resolves_extended_planner(planner_id):
            values = config if isinstance(config, dict) else ({} if config is None else vars(config))
            return create_extended_planner(
                planner_id,
                values,
                child_factory=lambda child_id, child_config: PlannerRegistry.create(
                    child_id, child_config, usage=usage
                ),
            )
        resolved = resolve_algorithm_id(planner_id)
        spec = resolved.spec
        normalized_usage = None if usage is None else str(usage)
        if normalized_usage == "runtime":
            if not bool(spec.deployable) or bool(spec.reference_only) or str(spec.execution_model) == "exact_reference":
                raise ValueError(f"planner {spec.canonical_id} is not runtime-deployable")
        elif normalized_usage == "offline_exact":
            if str(spec.execution_model) != "exact_reference":
                raise ValueError(f"planner {spec.canonical_id} is not an offline exact planner")
            raise ValueError("exact algorithms must be invoked through M4 OracleRegistry")
        elif normalized_usage == "reporting":
            raise ValueError("reporting aliases are not executable planners")
        family = planner_family_for_spec(
            family=str(spec.family),
            scheduling_scope=str(spec.scheduling_scope),
            reference_only=bool(spec.reference_only),
            deployable=bool(spec.deployable),
            supports_p2_hint=bool(spec.supports_p2_hint),
            canonical_id=str(spec.canonical_id),
            execution_model=str(spec.execution_model),
        )
        return FormalRuntimePlanner(
            _planner_id=str(spec.canonical_id),
            _planner_family=family,
        )


__all__ = ["PlannerRegistry"]
