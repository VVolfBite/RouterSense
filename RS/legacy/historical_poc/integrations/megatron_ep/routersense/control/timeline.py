from __future__ import annotations

from dataclasses import dataclass, field

from integrations.megatron_ep.routersense.control.contracts import ControlTimelineEvent


@dataclass
class ControlTimeline:
    events: list[ControlTimelineEvent] = field(default_factory=list)

    def record(self, event: ControlTimelineEvent) -> None:
        self.events.append(event)

    def to_rows(self) -> list[dict[str, object]]:
        return [item.to_dict() for item in self.events]
