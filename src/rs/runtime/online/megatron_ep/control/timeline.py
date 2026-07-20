"""早期 control timeline 容器。

当前正式 lifecycle 已经用自己的 timeline list 记录事件；
这个文件主要保留给历史控制面结构和测试使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rs.runtime.online.megatron_ep.control.contracts import ControlTimelineEvent


@dataclass
class ControlTimeline:
    events: list[ControlTimelineEvent] = field(default_factory=list)

    def record(self, event: ControlTimelineEvent) -> None:
        self.events.append(event)

    def to_rows(self) -> list[dict[str, object]]:
        return [item.to_dict() for item in self.events]
