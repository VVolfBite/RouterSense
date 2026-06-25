#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc1.cli import build_config, ensure_output_dir, parse_args
from routesense_poc1.model_loader import ModelLoadError, load_olmoe_model
from routesense_poc1.model_inspector import inspect_model, select_middle_moe_layer
from routesense_poc1.schemas import RunConfig
from routesense_poc1.serialization import save_json


def _safe_source(layer) -> dict[str, object]:
    payload = {"available": False, "error": None, "lines": []}
    try:
        source = inspect.getsource(layer.forward)
        payload["available"] = True
        payload["lines"] = source.splitlines()[:120]
    except Exception as exc:  # pragma: no cover - optional introspection
        payload["error"] = str(exc)
    return payload


def _module_summary(module) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for name, child in getattr(module, "named_children", lambda: [])():
        items.append({"name": name, "class": child.__class__.__name__})
    return items


def _module_tree(module, depth: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, child in getattr(module, "named_modules", lambda: [])():
        if depth is not None and name.count(".") >= depth:
            continue
        rows.append({"path": name, "class": child.__class__.__name__})
    return rows


def main() -> int:
    args = parse_args(["--config", str(ROOT / "configs" / "poc1.yaml")])
    config = build_config(args)
    output_dir = ensure_output_dir(config.output_dir)
    save_json(config.to_dict(), output_dir / "config.json")
    try:
        model, tokenizer = load_olmoe_model(config)
    except ModelLoadError as exc:
        save_json(
            {
                "status": "error",
                "message": str(exc),
                "model_id": config.model_id,
            },
            output_dir / "environment.json",
        )
        print(str(exc))
        return 1

    environment = {
        "model_id": config.model_id,
        "model_class": model.__class__.__name__,
        "device_map": getattr(model, "device_map", None),
        "config_moe_fields": [key for key in getattr(model.config, "__dict__", {}) if "moe" in key.lower() or "expert" in key.lower() or "router" in key.lower()],
    }
    save_json(environment, output_dir / "environment.json")

    layers = inspect_model(model)
    selected_layer = select_middle_moe_layer(layers)
    payload_layers = []
    for layer in layers:
        payload_layers.append(
            {
                "layer_path": layer.layer_path,
                "module_class": layer.module_class,
                "forward_signature": layer.forward_signature,
                "candidate_score": layer.candidate_score,
                "candidate_reason": layer.candidate_reason,
                "child_module_names": layer.child_module_names,
                "has_gate": layer.has_gate,
                "has_router": any(token in layer.layer_path.lower() for token in ("router", "gate", "moe", "expert", "sparse")),
                "has_experts": layer.has_experts,
            }
        )

    selected_payload = None
    if selected_layer is not None:
        selected_layer_obj = model.get_submodule(selected_layer.layer_path)
        selected_payload = {
            "layer_path": selected_layer.layer_path,
            "module_class": selected_layer.module_class,
            "forward_signature": selected_layer.forward_signature,
            "repr": repr(selected_layer_obj),
            "named_children": _module_summary(selected_layer_obj),
            "named_modules_depth_2": _module_tree(selected_layer_obj, depth=2),
            "forward_source": _safe_source(selected_layer_obj),
        }
        text_lines = [
            f"layer path: {selected_layer.layer_path}",
            f"layer class: {selected_layer.module_class}",
            f"forward signature: {selected_layer.forward_signature}",
        ]
        source = selected_payload["forward_source"]
        if source.get("available"):
            text_lines.append("source file: <available>")
            text_lines.append("source start line: <available>")
            text_lines.extend([f"{line}" for line in source.get("lines", [])[:120]])
        else:
            text_lines.append("source file: <unavailable>")
            text_lines.append("source start line: <unavailable>")
        (output_dir / "selected_moe_layer_forward.txt").write_text("\n".join(text_lines), encoding="utf-8")

    save_json(
        {
            "status": "ok",
            "selected_layer": selected_payload,
            "layers": payload_layers,
        },
        output_dir / "model_inspection.json",
    )

    text = "The history of science is a story of"
    model.eval()
    try:
        inputs = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            forward_kwargs = {"use_cache": False}
            if "output_router_logits" in inspect.signature(model.forward).parameters:
                forward_kwargs["output_router_logits"] = True
            try:
                outputs = model(**inputs, **forward_kwargs)
            except TypeError:
                outputs = model(**inputs, use_cache=False)
            logits = outputs.logits
            summary = {
                "input_ids_shape": list(inputs["input_ids"].shape),
                "logits_shape": list(logits.shape),
                "dtype": str(logits.dtype),
                "device": str(logits.device),
                "has_router_logits": hasattr(outputs, "router_logits"),
            }
            if hasattr(outputs, "router_logits"):
                router_logits = outputs.router_logits
                if isinstance(router_logits, (tuple, list)):
                    summary["router_logits_summary"] = [
                        {
                            "type": type(item).__name__,
                            "length": len(item) if hasattr(item, "__len__") else None,
                            "shape": list(item.shape) if hasattr(item, "shape") else None,
                            "dtype": str(item.dtype) if hasattr(item, "dtype") else None,
                        }
                        for item in router_logits
                    ]
                else:
                    summary["router_logits_summary"] = {
                        "type": type(router_logits).__name__,
                        "length": len(router_logits) if hasattr(router_logits, "__len__") else None,
                        "shape": list(router_logits.shape) if hasattr(router_logits, "shape") else None,
                        "dtype": str(router_logits.dtype) if hasattr(router_logits, "dtype") else None,
                    }
            save_json(summary, output_dir / "model_forward_summary.json")
    except Exception as exc:  # pragma: no cover - optional inspection robustness
        save_json({"status": "error", "message": str(exc)}, output_dir / "model_forward_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
