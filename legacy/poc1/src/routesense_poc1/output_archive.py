from __future__ import annotations

import shutil
from pathlib import Path


def archive_previous_outputs(output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    archive_dir = output_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived: list[Path] = []
    for name in [
        "model_inspection.json",
        "environment.json",
        "model_forward_summary.json",
        "selected_moe_layer_forward.txt",
        "adapter_status.json",
    ]:
        source = output_path / name
        if not source.exists():
            continue
        destination = archive_dir / name
        counter = 1
        while destination.exists():
            destination = archive_dir / f"{name}.{counter}"
            counter += 1
        shutil.move(str(source), str(destination))
        archived.append(destination)
    return archived
