from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs_sim.contracts.schema import LinkClass, PhaseKey
from rs_sim.contracts.digest import stable_digest


@dataclass(frozen=True, slots=True)
class PhysicalTaskMetric:
    task_id: str
    phase_key: PhaseKey
    batch_id: str
    link_class: LinkClass
    lane_id: str
    tx_nic_id: str
    rx_nic_id: str
    committed_at_ns: int
    start_at_ns: int
    complete_at_ns: int
    payload_bytes: int
    completed: bool


@dataclass(frozen=True, slots=True)
class PhysicalLaunchMetric:
    batch_id: str
    phase_key: PhaseKey
    link_class: LinkClass
    physical_batch_task_ids: tuple[str, ...]
    selected_task_ids: tuple[str, ...]
    committed_at_ns: int
    start_at_ns: int
    complete_at_ns: int
    launch_delay_ns: int


@dataclass(frozen=True, slots=True)
class PhysicalBusyInterval:
    resource_kind: str
    resource_id: str
    phase_key: PhaseKey
    task_id: str
    batch_id: str
    start_at_ns: int
    complete_at_ns: int

    @property
    def duration_ns(self) -> int:
        return self.complete_at_ns - self.start_at_ns


@dataclass(frozen=True, slots=True)
class PhysicalMetricsView:
    requested_phase_keys: tuple[PhaseKey, ...]
    requested_task_ids: tuple[str, ...]
    requested_window_task_ids: tuple[str, ...]
    selected_task_ids: tuple[str, ...]
    task_metrics: tuple[PhysicalTaskMetric, ...]
    launch_metrics: tuple[PhysicalLaunchMetric, ...]
    busy_intervals: tuple[PhysicalBusyInterval, ...]
    outstanding_prepared_receipt_ids: tuple[str, ...]
    outstanding_confirmed_receipt_ids: tuple[str, ...]
    physical_completed_bytes: int
    launch_count: int
    launch_delay_total_ns: int
    all_resources_free: bool
    terminal: bool

    @property
    def metrics_digest(self) -> str:
        return stable_digest(self, domain="TRANSPORT_FILTERED_PHYSICAL_METRICS")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "TRANSPORT_FILTERED_PHYSICAL_METRICS",
            "requested_phase_keys": self.requested_phase_keys,
            "requested_task_ids": self.requested_task_ids,
            "requested_window_task_ids": self.requested_window_task_ids,
            "selected_task_ids": self.selected_task_ids,
            "task_metrics": self.task_metrics,
            "launch_metrics": self.launch_metrics,
            "busy_intervals": self.busy_intervals,
            "outstanding_prepared_receipt_ids": (
                self.outstanding_prepared_receipt_ids
            ),
            "outstanding_confirmed_receipt_ids": (
                self.outstanding_confirmed_receipt_ids
            ),
            "physical_completed_bytes": self.physical_completed_bytes,
            "launch_count": self.launch_count,
            "launch_delay_total_ns": self.launch_delay_total_ns,
            "all_resources_free": self.all_resources_free,
            "terminal": self.terminal,
            "metrics_digest": self.metrics_digest,
        }


__all__ = [
    "PhysicalBusyInterval",
    "PhysicalLaunchMetric",
    "PhysicalMetricsView",
    "PhysicalTaskMetric",
]
