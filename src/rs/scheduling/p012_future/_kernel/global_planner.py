from __future__ import annotations

import time

import numpy as np

from .artifacts import ForecastArtifact
from .contracts import ForecastPlanningRequest, P2RevealRequest
from .event_core import bind_template, forecast_plan, plan_event, rank_hint_plan
from .p01_aware import p01_aware_hint, release_delay
from .plan import tuple_to_compact_plan


class GlobalPlanSelector:
    """One-shot complete-plan selector over forecast-only candidates.

    This is deliberately named a selector rather than a monolithic global
    optimizer. It generates complete candidate plans once, applies a P01 release
    envelope, selects one plan before execution, and never re-scores per wave.
    """

    def __init__(
        self,
        *,
        planner_id: str,
        planner_family: str,
        weights: tuple[float, float, float, float],
        margin: float = 0.001,
        min_release_slack: float = 0.0,
        max_release_slack: float = 0.08,
        tie_band: float = 0.12,
        release_weight: float = 0.15,
    ) -> None:
        self.planner_id = str(planner_id)
        self.planner_family = str(planner_family)
        self.weights = tuple(float(x) for x in weights)
        self.margin = float(margin)
        self.min_release_slack = float(min_release_slack)
        self.max_release_slack = float(max_release_slack)
        self.tie_band = float(tie_band)
        self.release_weight = float(release_weight)

    def _resource(self, request: ForecastPlanningRequest):
        slope, intercept = request.cost_matrices()
        return slope, intercept, request.constraints

    def plan_forecast(self, request: ForecastPlanningRequest) -> ForecastArtifact:
        request.validate()
        start = time.perf_counter_ns()
        p0, p1, hint = request.matrices()
        zero = np.zeros_like(p0)
        slope, intercept, constraints = self._resource(request)
        kwargs = dict(
            weights=self.weights,
            edge_slope=slope,
            edge_intercept=intercept,
            expert_compute_delay=constraints.expert_compute_delay,
            wave_launch_b=request.cost_model.wave_launch_b,
            max_waves=constraints.max_waves,
        )

        # Candidate 0 keeps the P01 event ordering and binds P2 conservatively.
        p01_template, p01_proxy = forecast_plan([p0, p1, hint], zero, **kwargs)
        confidence = float(request.prediction_hint.confidence)
        candidates = [("p01_event_p2", p01_template, p01_proxy, zero)]
        adjustment_meta = {"enabled": False}
        if not np.any(hint):
            # Zero/P01 mode has no prediction information. Generating a second
            # identical rank-hint candidate only adds control-path latency.
            adjustment_meta = {"enabled": False, "reason": "zero_hint_short_circuit"}
        elif request.prediction_hint.oracle or confidence >= 0.90:
            # With a high-confidence/perfect hint, a complete forecast plan is
            # the useful alternative; an extra rank-hint candidate only adds
            # latency without improving the upper-bound path.
            event_template = plan_event(
                [p0, p1, hint], hint=hint, scope="joint", full_truth_geometry=True, **kwargs
            )
            # Proxy P2 equals the planned hint, so template and proxy plan are
            # identical; avoid a redundant binder invocation.
            event_proxy = event_template
            candidates.append(("event_forecast", event_template, event_proxy, hint))
        else:
            # For ordinary prediction, consult the P01 release vector when
            # predicted rank pressures are close.
            adjusted, adjustment_meta = p01_aware_hint(
                p0, p1, hint, weights=self.weights, tie_band=self.tie_band,
                release_weight=self.release_weight, edge_slope=slope, edge_intercept=intercept,
                expert_compute_delay=constraints.expert_compute_delay,
                wave_launch_b=request.cost_model.wave_launch_b, max_waves=constraints.max_waves,
                p01_release2=np.asarray(p01_proxy[11], dtype=np.float64),
                p01_horizon=float(p01_proxy[0]),
            )
            adjusted_int = np.rint(np.maximum(adjusted, 0.0)).astype(np.int32)
            aware_template, aware_proxy = rank_hint_plan([p0, p1, hint], adjusted_int, **kwargs)
            candidates.append(("p01_aware_rank_hint", aware_template, aware_proxy, adjusted_int))
        confidence = float(request.prediction_hint.confidence)
        release_slack = self.min_release_slack + (self.max_release_slack - self.min_release_slack) * confidence
        baseline = p01_proxy
        eligible: list[tuple[tuple[float, float, float, int, int], int]] = []
        audit: list[dict] = []
        for index, (name, template, proxy, binding_hint) in enumerate(candidates):
            dmax, dmean = release_delay(proxy, baseline)
            improvement = (float(baseline[0]) - float(proxy[0])) / max(float(baseline[0]), 1e-12)
            allowed = name == "p01_event_p2" or (improvement >= self.margin and dmax <= release_slack)
            objective = (float(proxy[0]), dmax, dmean, int(proxy[1]), index)
            audit.append({
                "name": name,
                "proxy_makespan": float(proxy[0]),
                "proxy_wave_count": int(proxy[1]),
                "proxy_improvement_over_p01": float(improvement),
                "release_delay_max": float(dmax),
                "release_delay_mean": float(dmean),
                "eligible": bool(allowed),
            })
            if allowed:
                eligible.append((objective, index))
        eligible.sort(key=lambda item: item[0])
        selected_index = eligible[0][1]
        selected_name, selected_template, selected_proxy, selected_hint = candidates[selected_index]
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        metadata = {
            "selector_semantic_version": "global_plan_selector_v2",
            "selected_candidate": selected_name,
            "selected_index": int(selected_index),
            "candidate_audit": audit,
            "forecast_only_selection": True,
            "weights": list(self.weights),
            "margin": self.margin,
            "effective_release_slack": float(release_slack),
            "tie_band": self.tie_band,
            "release_weight": self.release_weight,
            "p01_aware_adjustment": adjustment_meta,
            "planning_ms": float(elapsed_ms),
            "predictor_id": request.prediction_hint.predictor_id,
            "prediction_confidence": confidence,
            "cost_model": request.cost_model.to_dict(),
            "topology": request.topology.to_dict(),
            "constraints": request.constraints.to_dict(),
        }
        plan = tuple_to_compact_plan(
            selected_template,
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            branch="global_ordering",
            request_digest=request.semantic_digest(),
            forecast=True,
            metadata=metadata,
        )
        return ForecastArtifact(
            request_digest=request.semantic_digest(),
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            branch="global_ordering",
            raw_template=selected_template,
            plan=plan,
            hint_rows=np.asarray(selected_hint, dtype=np.int32).tolist(),
            metadata=metadata,
        )

    def bind(self, artifact: ForecastArtifact, request: ForecastPlanningRequest, reveal: P2RevealRequest):
        request.validate(); reveal.validate(world_size=request.topology.world_size)
        if reveal.request_id != request.request_id:
            raise ValueError("P2 reveal request_id mismatch")
        if artifact.request_digest != request.semantic_digest():
            raise ValueError("forecast artifact does not belong to this planning request")
        if reveal.forecast_request_digest != artifact.semantic_digest():
            raise ValueError("P2 reveal references a different forecast artifact")
        p0, p1, _ = request.matrices(); p2 = reveal.matrix(world_size=request.topology.world_size)
        slope, intercept = request.cost_matrices()
        bound = bind_template(
            [p0, p1, p2], np.asarray(artifact.hint_rows, dtype=np.float64), artifact.raw_template,
            edge_slope=slope, edge_intercept=intercept,
            expert_compute_delay=request.constraints.expert_compute_delay,
            wave_launch_b=request.cost_model.wave_launch_b,
            max_waves=request.constraints.max_waves,
        )
        metadata = dict(artifact.metadata)
        metadata.update({
            "bound_from_forecast_digest": artifact.semantic_digest(),
            "truth_binding_only": True,
        })
        return tuple_to_compact_plan(
            bound,
            planner_id=artifact.planner_id,
            planner_family=artifact.planner_family,
            branch=artifact.branch,
            request_digest=request.semantic_digest(),
            forecast=False,
            metadata=metadata,
        )


__all__ = ["GlobalPlanSelector"]
