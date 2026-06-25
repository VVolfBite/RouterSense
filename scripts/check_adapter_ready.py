#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc1.adapter_registry import get_adapter_for_layer
from routesense_poc1.serialization import load_json


def main() -> int:
    output_dir = Path(__import__("os").environ.get("ROUTESENSE_OUTPUT_DIR", str(ROOT / "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection_path = output_dir / "model_inspection.json"
    if not inspection_path.exists():
        payload = {
            "inspection_available": False,
            "selected_layer_available": False,
            "adapter_name": "UnsupportedOLMoEAdapter",
            "adapter_ready": False,
            "message": "inspection result not found; run bootstrap_and_inspect.sh first",
        }
        (output_dir / "adapter_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2

    data = load_json(inspection_path)
    selected_layer = None
    if isinstance(data, dict):
        selected_layer = data.get("selected_layer")
    adapter = get_adapter_for_layer(selected_layer)
    adapter_name = type(adapter).__name__ if adapter is not None else "UnsupportedOLMoEAdapter"
    adapter_ready = adapter is not None and getattr(adapter, "name", "UnsupportedOLMoEAdapter") != "UnsupportedOLMoEAdapter"
    if adapter_ready:
        payload = {
            "inspection_available": True,
            "selected_layer_available": bool(selected_layer),
            "adapter_name": adapter_name,
            "adapter_ready": True,
            "message": "adapter appears ready",
        }
        (output_dir / "adapter_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    payload = {
        "inspection_available": True,
        "selected_layer_available": bool(selected_layer),
        "adapter_name": adapter_name,
        "adapter_ready": False,
        "message": "inspection exists but adapter is still UnsupportedOLMoEAdapter; implement a real adapter before running trace/ablation",
    }
    (output_dir / "adapter_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
