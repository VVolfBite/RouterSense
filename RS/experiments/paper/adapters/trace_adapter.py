from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.offline.collect_router_trace import main as collect_router_trace_main


def capture_trace_from_config(*, config_path: Path, output_dir: Path) -> int:
    return int(
        collect_router_trace_main(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ]
        )
    )


def trace_capture_capability() -> dict[str, Any]:
    return {
        "entrypoint": "experiments.offline.collect_router_trace",
        "mode": "formal_wrapper",
    }
