"""Lifecycle Export stage methods."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


class LifecycleExportMixin:
    def _export_list(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(rows)

    def _export_observation_rows(self, method_name: str) -> list[dict[str, Any]]:
        if self.observation_recorder is None:
            return []
        export_fn = getattr(self.observation_recorder, method_name)
        return list(export_fn())

    def export_records(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.completed]

    def export_control_timeline(self) -> list[dict[str, Any]]:
        return self._export_list(self.control_timeline)

    def export_control_commands(self) -> list[dict[str, Any]]:
        return self._export_list(self.control_commands)

    def export_plan_arrival_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.plan_arrival_records)

    def export_window_state_records(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.window_state_records)

    def export_prepared_plan_bindings(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.prepared_plan_bindings)

    def export_release_events(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.release_events)

    def export_window_schedule_shadows(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.window_schedule_shadows)

    def export_prepared_phase_plan_shadows(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.prepared_phase_plan_shadows)

    def export_planning_timing_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.planning_timing_records)

    def export_control_replay_traces(self) -> list[dict[str, Any]]:
        if not self._replay_trace_enabled():
            return []
        return self._export_list(self.control_replay_traces)

    def export_prediction_audits(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.prediction_audits)

    def export_expert_route_traces(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_expert_route_traces")

    def export_source_expert_counts(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_source_expert_counts")

    def export_expert_to_traffic_audits(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_expert_to_traffic_audits")

    def export_expert_trace_warnings(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_expert_trace_warnings")

    def export_assertions(self) -> dict[str, Any]:
        return dict(self.assertion_state)

    def export_prepared_plan_summary(self) -> dict[str, Any]:
        return build_prepared_plan_summary(runtime_state=self._runtime_state)

    def export_phase_contexts(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_phase_contexts")

    def export_transport_bundles(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_transport_bundles")

    def export_scheduled_phase_plans(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_scheduled_phase_plans")

    def export_transport_execution_results(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_transport_execution")

    def export_captured_phase_tensors(self) -> list[dict[str, Any]]:
        rows = self._export_observation_rows("export_captured_phase_tensors")
        return [{key: value for key, value in item.items() if key != "tensor"} for item in rows]

    def export_captured_phase_tensors_with_payload(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_captured_phase_tensors")


__all__ = ["LifecycleExportMixin"]
