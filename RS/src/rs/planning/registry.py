from __future__ import annotations

from rs.scheduling.catalog import algorithm_specs, resolve_algorithm_id

from ._legacy_runtime import LegacyPlannerAdapter
from .api import PlannerSpec, planner_family_for_spec


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
        return tuple(rows)

    @staticmethod
    def create(planner_id: str, config=None, *, usage: str | None = None):
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
        return LegacyPlannerAdapter(
            _planner_id=str(spec.canonical_id),
            _planner_family=family,
            _builder_key=str(spec.builder_key),
        )


__all__ = ["PlannerRegistry"]
