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
from routesense.runtime.distributed_ep.adapter.olmoe_adapter import probe_olmoe_adapter_config


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
    adapter = probe_olmoe_adapter_config(model)
    moe_layer = model.model.layers[0].mlp
    experts = moe_layer.experts
    router = moe_layer.gate
    probe = {
        "model_class": adapter.model_class,
        "moe_block_class": moe_layer.__class__.__name__,
        "router_module_path": "model.layers[*].mlp.gate",
        "router_output_fields": ["router_logits", "router_scores", "router_indices"],
        "num_moe_layers": len(model.model.layers),
        "num_experts": adapter.num_experts,
        "top_k": adapter.top_k,
        "hidden_size": adapter.hidden_size,
        "intermediate_size": adapter.intermediate_size,
        "expert_parameter_names": [name for name, _ in experts.named_parameters()],
        "expert_parameter_shapes": {name: list(param.shape) for name, param in experts.named_parameters()},
        "packed_experts": True,
        "single_expert_weight_formula": "gate_up_proj[expert_id] and down_proj[expert_id] with silu(gate) * up",
        "source": source,
        "gate_weight_shape": list(router.weight.shape),
        "adapter": adapter.to_dict(),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "architecture_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(json.dumps(probe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
