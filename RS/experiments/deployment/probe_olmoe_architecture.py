#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense.runtime import load_model_and_tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the real OLMoE architecture.")
    parser.add_argument("--model-id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "phase0c_architecture_probe"))
    args = parser.parse_args(argv)
    model, _, _, _, source = load_model_and_tokenizer(
        model_id=args.model_id,
        model_path=args.model_path,
        precision=args.precision,
        device_index=args.device_index,
    )
    moe_layer = model.model.layers[0].mlp
    experts = moe_layer.experts
    router = moe_layer.gate
    probe = {
        "model_class": model.__class__.__name__,
        "moe_block_class": moe_layer.__class__.__name__,
        "router_module_path": "model.layers[*].mlp.gate",
        "router_output_fields": ["router_logits", "router_scores", "router_indices"],
        "num_moe_layers": len(model.model.layers),
        "num_experts": int(model.config.num_experts),
        "top_k": int(model.config.num_experts_per_tok),
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "expert_parameter_names": [name for name, _ in experts.named_parameters()],
        "expert_parameter_shapes": {name: list(param.shape) for name, param in experts.named_parameters()},
        "packed_experts": True,
        "single_expert_weight_formula": "gate_up_proj[expert_id] and down_proj[expert_id] with silu(gate) * up",
        "source": source,
        "gate_weight_shape": list(router.weight.shape),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "architecture_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(json.dumps(probe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
