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

from rs.core.artifact import write_json, write_jsonl

from .contracts import RuntimeObservationSnapshot


@dataclass
class RuntimeArtifactRecorder:
    run_dir: Path

    def write_run_manifest(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "run_manifest.json", payload)

    def write_summary(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "summary.json", payload)

    def write_source_provenance(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "source_provenance.json", payload)

    def flush_snapshot(self, snapshot: RuntimeObservationSnapshot) -> None:
        counters = snapshot.counters
        write_json(self.run_dir / "observation_counters.json", counters)
        if snapshot.phase_contexts:
            write_jsonl(self.run_dir / "phase_contexts.jsonl", list(snapshot.phase_contexts))
        if snapshot.transport_bundles:
            write_jsonl(self.run_dir / "transport_bundles.jsonl", list(snapshot.transport_bundles))
        if snapshot.scheduled_phase_plans:
            write_jsonl(self.run_dir / "scheduled_phase_plans.jsonl", list(snapshot.scheduled_phase_plans))
        if snapshot.transport_execution:
            write_jsonl(self.run_dir / "transport_execution.jsonl", list(snapshot.transport_execution))
        if snapshot.execution_audits:
            write_json(self.run_dir / "execution_audit.json", list(snapshot.execution_audits))
        if snapshot.heartbeats:
            write_jsonl(self.run_dir / "heartbeats.jsonl", list(snapshot.heartbeats))
        if snapshot.failures:
            write_jsonl(self.run_dir / "failures.jsonl", list(snapshot.failures))
        if snapshot.captured_phase_tensors:
            write_jsonl(
                self.run_dir / "captured_phase_tensors.jsonl",
                [{key: value for key, value in row.items() if key != "tensor"} for row in snapshot.captured_phase_tensors],
            )

    def write_failure_placeholder(self) -> None:
        write_json(self.run_dir / "failure_report.json", {"status": "not_triggered"})

    def write_watchdog_placeholder(self) -> None:
        write_json(self.run_dir / "watchdog_report.json", {"status": "not_triggered"})


__all__ = ["RuntimeArtifactRecorder", "write_json", "write_jsonl"]
