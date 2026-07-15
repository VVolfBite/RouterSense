from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Protocol


@dataclass(frozen=True)
class MeasurementEvent:
    run_id: str
    rank: int
    forward_generation: int
    microbatch_id: str
    event_type: str
    started_at_ns: int
    ended_at_ns: int
    layer_id: str | None = None
    phase: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.rank) < 0:
            raise ValueError("measurement event rank must be non-negative")
        if int(self.forward_generation) < 0:
            raise ValueError("measurement event forward_generation must be non-negative")
        if int(self.started_at_ns) < 0 or int(self.ended_at_ns) < int(self.started_at_ns):
            raise ValueError("measurement event timestamps must be non-negative and ordered")
        if not str(self.run_id).strip():
            raise ValueError("measurement event run_id must be non-empty")
        if not str(self.microbatch_id).strip():
            raise ValueError("measurement event microbatch_id must be non-empty")
        if not str(self.event_type).strip():
            raise ValueError("measurement event event_type must be non-empty")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class MeasurementRequirement:
    mode: str
    required_event_types: tuple[str, ...] = ()
    performance_claim_requested: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementCapability:
    mode: str
    emits_measurements: bool
    performance_claim_allowed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementCompleteness:
    complete: bool
    missing_required_event_types: tuple[str, ...] = ()
    unknown_event_count: int = 0
    malformed_event_count: int = 0
    dropped_event_count: int = 0
    overflowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementSnapshot:
    event_count: int
    dropped_event_count: int
    instrumentation_mode: str
    events: tuple[MeasurementEvent, ...] = ()
    summary: Mapping[str, object] = field(default_factory=dict)
    capability: MeasurementCapability = field(
        default_factory=lambda: MeasurementCapability(
            mode="off",
            emits_measurements=False,
            performance_claim_allowed=False,
        )
    )
    completeness: MeasurementCompleteness = field(
        default_factory=lambda: MeasurementCompleteness(complete=False)
    )
    unknown_event_count: int = 0
    malformed_event_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "event_count": int(self.event_count),
            "dropped_event_count": int(self.dropped_event_count),
            "instrumentation_mode": str(self.instrumentation_mode),
            "events": [event.to_dict() for event in self.events],
            "summary": dict(self.summary),
            "capability": self.capability.to_dict(),
            "completeness": self.completeness.to_dict(),
            "unknown_event_count": int(self.unknown_event_count),
            "malformed_event_count": int(self.malformed_event_count),
        }


class MeasurementSink(Protocol):
    def record(self, event: MeasurementEvent) -> None:
        ...

    def snapshot(self) -> MeasurementSnapshot:
        ...

    def reset(self) -> None:
        ...
