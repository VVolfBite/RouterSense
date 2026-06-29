from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkerLoopState:
    mode: str = "synchronous"
    processed_layers: list[int] = field(default_factory=list)


class WorkerLoop:
    def __init__(self, mode: str = "synchronous") -> None:
        self.state = WorkerLoopState(mode=mode)

    def record_layer(self, layer_id: int) -> None:
        self.state.processed_layers.append(layer_id)
