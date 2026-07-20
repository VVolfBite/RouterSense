"""轻量原生 observer。

这个文件只提供一个简单容器 RouterSenseObserver，
用于在不接入正式 runtime 时记录原生 dispatcher 观测行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RouterSenseObserver:
    """Read-only observer for native Megatron EP runtime."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **payload: Any) -> None:
        self.records.append(dict(payload))

    def export_rows(self) -> list[dict[str, Any]]:
        return list(self.records)
