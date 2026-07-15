"""观测面 artifact 记录器。

主要职责：
- 写 run manifest / summary / source provenance
- 把 RuntimeObservationSnapshot 刷成 json/jsonl 文件
不参与调度决策，只负责落盘。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rs.evidence.artifact_writer import FilesystemArtifactWriter

from .contracts import RuntimeObservationSnapshot


@dataclass
class RuntimeArtifactRecorder:
    run_dir: Path
    producer: str = "runtime_observation_recorder"

    def __post_init__(self) -> None:
        self._writer = FilesystemArtifactWriter(root_dir=self.run_dir)

    def _write_json(self, relative_path: str, payload: object, *, schema: str, claim_role: str) -> None:
        self._writer.write_json(
            relative_path=relative_path.replace("\\", "/"),
            payload=payload,
            schema=schema,
            producer=self.producer,
            claim_role=claim_role,
        )

    def _write_jsonl(self, relative_path: str, rows: list[dict[str, Any]], *, schema: str, claim_role: str) -> None:
        self._writer.write_jsonl(
            relative_path=relative_path.replace("\\", "/"),
            payload=rows,
            schema=schema,
            producer=self.producer,
            claim_role=claim_role,
        )

    def write_run_manifest(self, payload: dict[str, Any]) -> None:
        self._write_json("run_manifest.json", payload, schema="runtime_run_manifest.v1", claim_role="diagnostic")

    def write_summary(self, payload: dict[str, Any]) -> None:
        self._write_json("summary.json", payload, schema="runtime_summary.v1", claim_role="diagnostic")

    def write_source_provenance(self, payload: dict[str, Any]) -> None:
        self._write_json("source_provenance.json", payload, schema="source_provenance.v1", claim_role="diagnostic")

    def flush_snapshot(self, snapshot: RuntimeObservationSnapshot) -> None:
        counters = snapshot.counters
        self._write_json("observation_counters.json", counters, schema="runtime_observation_counters.v1", claim_role="diagnostic")
        if snapshot.phase_contexts:
            self._write_jsonl("phase_contexts.jsonl", list(snapshot.phase_contexts), schema="runtime_phase_contexts.v1", claim_role="diagnostic")
        if snapshot.transport_bundles:
            self._write_jsonl("transport_bundles.jsonl", list(snapshot.transport_bundles), schema="runtime_transport_bundles.v1", claim_role="diagnostic")
        if snapshot.scheduled_phase_plans:
            self._write_jsonl("scheduled_phase_plans.jsonl", list(snapshot.scheduled_phase_plans), schema="runtime_scheduled_phase_plans.v1", claim_role="diagnostic")
        if snapshot.transport_execution:
            self._write_jsonl("transport_execution.jsonl", list(snapshot.transport_execution), schema="runtime_transport_execution.v1", claim_role="diagnostic")
        if snapshot.execution_audits:
            self._write_json("execution_audit.json", list(snapshot.execution_audits), schema="runtime_execution_audit.v1", claim_role="diagnostic")
        if snapshot.expert_route_traces:
            self._write_jsonl("expert_route_trace.jsonl", list(snapshot.expert_route_traces), schema="runtime_expert_route_trace.v1", claim_role="diagnostic")
        if snapshot.source_expert_counts:
            self._write_jsonl("source_expert_counts.jsonl", list(snapshot.source_expert_counts), schema="runtime_source_expert_counts.v1", claim_role="diagnostic")
        if snapshot.expert_to_traffic_audits:
            self._write_jsonl("expert_to_traffic_audit.jsonl", list(snapshot.expert_to_traffic_audits), schema="runtime_expert_to_traffic_audit.v1", claim_role="diagnostic")
        if snapshot.expert_trace_warnings:
            self._write_jsonl("expert_trace_warnings.jsonl", list(snapshot.expert_trace_warnings), schema="runtime_expert_trace_warnings.v1", claim_role="diagnostic")
        if snapshot.heartbeats:
            self._write_jsonl("heartbeats.jsonl", list(snapshot.heartbeats), schema="runtime_heartbeats.v1", claim_role="diagnostic")
        if snapshot.failures:
            self._write_jsonl("failures.jsonl", list(snapshot.failures), schema="runtime_failures.v1", claim_role="diagnostic")
        if snapshot.captured_phase_tensors:
            self._write_jsonl(
                "captured_phase_tensors.jsonl",
                [{key: value for key, value in row.items() if key != "tensor"} for row in snapshot.captured_phase_tensors],
                schema="runtime_captured_phase_tensors.v1",
                claim_role="diagnostic",
            )

    def write_failure_placeholder(self) -> None:
        self._write_json("failure_report.json", {"status": "not_triggered"}, schema="runtime_failure_report.v1", claim_role="diagnostic")

    def write_watchdog_placeholder(self) -> None:
        self._write_json("watchdog_report.json", {"status": "not_triggered"}, schema="runtime_watchdog_report.v1", claim_role="diagnostic")


__all__ = ["RuntimeArtifactRecorder"]
