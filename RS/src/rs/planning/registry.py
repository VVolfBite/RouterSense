from __future__ import annotations

from rs.scheduling.catalog import algorithm_specs, resolve_algorithm_id

from .api import LegacyPlannerAdapter, PlannerSpec, planner_family_for_spec


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
                    ),
                    deployable=bool(spec.deployable),
                    reference_only=bool(spec.reference_only),
                    requires_prediction=bool(spec.supports_p2_hint),
                    exact=bool(spec.reference_only),
                    historical_aliases=tuple(spec.aliases + spec.deprecated_aliases),
                )
            )
        return tuple(rows)

    @staticmethod
    def create(planner_id: str, config=None):
        resolved = resolve_algorithm_id(planner_id)
        spec = resolved.spec
        family = planner_family_for_spec(
            family=str(spec.family),
            scheduling_scope=str(spec.scheduling_scope),
            reference_only=bool(spec.reference_only),
            deployable=bool(spec.deployable),
            supports_p2_hint=bool(spec.supports_p2_hint),
            canonical_id=str(spec.canonical_id),
        )
        return LegacyPlannerAdapter(
            _planner_id=str(spec.canonical_id),
            _planner_family=family,
            _builder_key=str(spec.builder_key),
        )


__all__ = ["PlannerRegistry"]
