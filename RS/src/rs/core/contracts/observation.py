"""Canonical runtime observation configuration contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ObservationProfile = Literal["minimal", "perf", "execution", "debug"]


@dataclass(frozen=True)
class RuntimeObservationConfig:
    profile: ObservationProfile = "minimal"
    capture_enabled: bool = False
    capture_layer_selector: str = ""
    capture_phase_selector: str = ""
    heartbeat_enabled: bool = False
    per_wave_timing_enabled: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
