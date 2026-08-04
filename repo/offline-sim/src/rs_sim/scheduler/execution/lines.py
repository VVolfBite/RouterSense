from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs_sim.scheduler.stable import stable_digest


@dataclass(frozen=True, slots=True)
class PlanningCostModel:
    """Deterministic integer-ns complexity model for the three service lines."""

    prediction_base_ns: int = 0
    prediction_per_observation_ns: int = 0
    prediction_per_task_ns: int = 0
    control_base_ns: int = 0
    control_per_observation_ns: int = 0
    control_per_task_ns: int = 0
    control_per_phase_ns: int = 0
    binding_base_ns: int = 0
    binding_per_task_ns: int = 0
    binding_per_phase_ns: int = 0
    zero_cost_mode: bool = False

    def __post_init__(self) -> None:
        for name in (
            "prediction_base_ns",
            "prediction_per_observation_ns",
            "prediction_per_task_ns",
            "control_base_ns",
            "control_per_observation_ns",
            "control_per_task_ns",
            "control_per_phase_ns",
            "binding_base_ns",
            "binding_per_task_ns",
            "binding_per_phase_ns",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")

    @property
    def model_digest(self) -> str:
        return stable_digest(self)

    def prediction_duration_ns(self, *, observation_count: int, task_count: int) -> int:
        if self.zero_cost_mode:
            return 0
        return (
            self.prediction_base_ns
            + self.prediction_per_observation_ns * int(observation_count)
            + self.prediction_per_task_ns * int(task_count)
        )

    def control_duration_ns(
        self, *, observation_count: int, task_count: int, phase_count: int
    ) -> int:
        if self.zero_cost_mode:
            return 0
        return (
            self.control_base_ns
            + self.control_per_observation_ns * int(observation_count)
            + self.control_per_task_ns * int(task_count)
            + self.control_per_phase_ns * int(phase_count)
        )

    def binding_duration_ns(self, *, task_count: int, phase_count: int) -> int:
        if self.zero_cost_mode:
            return 0
        return (
            self.binding_base_ns
            + self.binding_per_task_ns * int(task_count)
            + self.binding_per_phase_ns * int(phase_count)
        )


@dataclass(frozen=True, slots=True)
class LineReservation:
    line_name: str
    job_id: str
    enqueue_ordinal: int
    arrival_at_ns: int
    start_at_ns: int
    finish_at_ns: int
    duration_ns: int
    queue_wait_ns: int
    hidden_service_ns: int
    exposed_service_ns: int
    exposed_delay_ns: int
    hide_until_ns: int
    payload_digest: str
    reservation_digest: str


@dataclass(frozen=True, slots=True)
class ServiceLineMetrics:
    line_name: str
    job_count: int
    total_queue_wait_ns: int
    total_service_ns: int
    hidden_service_ns: int
    exposed_service_ns: int
    exposed_delay_ns: int
    first_arrival_at_ns: int | None
    last_finish_at_ns: int | None
    metrics_digest: str


class SingleServerFIFOLine:
    """Deterministic single-server FIFO non-preemptive service line."""

    def __init__(self, name: str) -> None:
        self.name = str(name)
        self.available_at_ns = 0
        self._last_arrival_ns = -1
        self._ordinal = 0
        self._reservations: list[LineReservation] = []

    def submit(
        self,
        *,
        job_id: str,
        arrival_at_ns: int,
        duration_ns: int,
        payload: Any,
        hide_until_ns: int | None = None,
    ) -> LineReservation:
        arrival = int(arrival_at_ns)
        duration = int(duration_ns)
        if arrival < self._last_arrival_ns:
            raise ValueError(
                f"{self.name} received out-of-order arrival {arrival} < {self._last_arrival_ns}"
            )
        if duration < 0:
            raise ValueError("line service duration cannot be negative")
        hidden_until = arrival if hide_until_ns is None else int(hide_until_ns)
        if hidden_until < 0:
            raise ValueError("hide_until_ns cannot be negative")
        start = max(arrival, int(self.available_at_ns))
        finish = start + duration
        hidden = max(0, min(finish, hidden_until) - start)
        exposed = duration - hidden
        queue_wait = start - arrival
        exposed_delay = max(0, finish - hidden_until)
        payload_digest = stable_digest(payload)
        reservation_digest = stable_digest(
            {
                "line": self.name,
                "job_id": str(job_id),
                "ordinal": self._ordinal,
                "arrival_at_ns": arrival,
                "start_at_ns": start,
                "finish_at_ns": finish,
                "duration_ns": duration,
                "queue_wait_ns": queue_wait,
                "hidden_service_ns": hidden,
                "exposed_service_ns": exposed,
                "exposed_delay_ns": exposed_delay,
                "hide_until_ns": hidden_until,
                "payload_digest": payload_digest,
            }
        )
        reservation = LineReservation(
            line_name=self.name,
            job_id=str(job_id),
            enqueue_ordinal=int(self._ordinal),
            arrival_at_ns=arrival,
            start_at_ns=start,
            finish_at_ns=finish,
            duration_ns=duration,
            queue_wait_ns=queue_wait,
            hidden_service_ns=hidden,
            exposed_service_ns=exposed,
            exposed_delay_ns=exposed_delay,
            hide_until_ns=hidden_until,
            payload_digest=payload_digest,
            reservation_digest=reservation_digest,
        )
        self._ordinal += 1
        self._last_arrival_ns = arrival
        self.available_at_ns = finish
        self._reservations.append(reservation)
        return reservation


    def reclassify_hide_until(
        self, *, job_ids: set[str] | frozenset[str], hide_until_ns: int
    ) -> None:
        """Recompute hidden/exposed attribution against a causal deadline.

        This never changes service ordering or completion time; it only replaces
        the provisional P0-time attribution once the first real consumer-ready
        timestamp becomes observable.
        """

        deadline = int(hide_until_ns)
        selected = frozenset(str(item) for item in job_ids)
        if not selected:
            return
        rebuilt: list[LineReservation] = []
        for item in self._reservations:
            if item.job_id not in selected:
                rebuilt.append(item)
                continue
            hidden = max(0, min(int(item.finish_at_ns), deadline) - int(item.start_at_ns))
            exposed = int(item.duration_ns) - hidden
            exposed_delay = max(0, int(item.finish_at_ns) - deadline)
            payload = {
                "line": item.line_name,
                "job_id": item.job_id,
                "ordinal": item.enqueue_ordinal,
                "arrival_at_ns": item.arrival_at_ns,
                "start_at_ns": item.start_at_ns,
                "finish_at_ns": item.finish_at_ns,
                "duration_ns": item.duration_ns,
                "queue_wait_ns": item.queue_wait_ns,
                "hidden_service_ns": hidden,
                "exposed_service_ns": exposed,
                "exposed_delay_ns": exposed_delay,
                "hide_until_ns": deadline,
                "payload_digest": item.payload_digest,
            }
            rebuilt.append(
                LineReservation(
                    line_name=item.line_name,
                    job_id=item.job_id,
                    enqueue_ordinal=item.enqueue_ordinal,
                    arrival_at_ns=item.arrival_at_ns,
                    start_at_ns=item.start_at_ns,
                    finish_at_ns=item.finish_at_ns,
                    duration_ns=item.duration_ns,
                    queue_wait_ns=item.queue_wait_ns,
                    hidden_service_ns=hidden,
                    exposed_service_ns=exposed,
                    exposed_delay_ns=exposed_delay,
                    hide_until_ns=deadline,
                    payload_digest=item.payload_digest,
                    reservation_digest=stable_digest(payload),
                )
            )
        self._reservations = rebuilt

    @property
    def reservations(self) -> tuple[LineReservation, ...]:
        return tuple(self._reservations)

    def metrics(self) -> ServiceLineMetrics:
        reservations = self.reservations
        payload = {
            "line_name": self.name,
            "job_count": len(reservations),
            "total_queue_wait_ns": sum(item.queue_wait_ns for item in reservations),
            "total_service_ns": sum(item.duration_ns for item in reservations),
            "hidden_service_ns": sum(item.hidden_service_ns for item in reservations),
            "exposed_service_ns": sum(item.exposed_service_ns for item in reservations),
            "exposed_delay_ns": sum(item.exposed_delay_ns for item in reservations),
            "first_arrival_at_ns": reservations[0].arrival_at_ns if reservations else None,
            "last_finish_at_ns": reservations[-1].finish_at_ns if reservations else None,
        }
        return ServiceLineMetrics(**payload, metrics_digest=stable_digest(payload))

    def digest(self) -> str:
        return stable_digest(self._reservations)


class ThreeLineServices:
    """Exactly one global Prediction, Control and ExecutionBinding line."""

    def __init__(self) -> None:
        self.prediction = SingleServerFIFOLine("PredictionLine")
        self.control = SingleServerFIFOLine("ControlLine")
        self.execution_binding = SingleServerFIFOLine("ExecutionBindingLine")

    def metrics(self) -> tuple[ServiceLineMetrics, ...]:
        return (
            self.prediction.metrics(),
            self.control.metrics(),
            self.execution_binding.metrics(),
        )

    def digest(self) -> str:
        return stable_digest(
            {
                "prediction": self.prediction.reservations,
                "control": self.control.reservations,
                "execution_binding": self.execution_binding.reservations,
            }
        )
