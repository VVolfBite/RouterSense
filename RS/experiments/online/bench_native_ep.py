#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.online import build_online_unimplemented_result
from rs.online.olmoe_ep import run_world_size_one_native_parity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark online native EP runtime.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Explain mixture-of-experts routing in one paragraph.")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--output-dir", type=str, default="artifacts/online/bench_native_ep")
    args = parser.parse_args(argv)
    run_id = f"online-native-bench-{uuid.uuid4().hex[:12]}"
    if int(args.world_size) == 1:
        parity = run_world_size_one_native_parity(
            model_id=args.model,
            model_path=args.model_path,
            prompt_text=args.prompt,
            layer_index=args.layer_index,
            precision=args.precision,
            device_index=args.device_index,
            atol=args.atol,
            rtol=args.rtol,
        )
        payload = {
            **build_online_unimplemented_result(
                run_id=run_id,
                world_size=args.world_size,
                transport_backend="online_native_a2a_ep",
                extra={
                    "entrypoint": "bench_native_ep",
                    "output_dir": args.output_dir,
                    "implemented": True,
                    "implemented_scope": "world_size_1_layer_parity_only",
                    "correctness_status": (
                        "passed" if parity["parity"]["numerical_correctness_pass"] else "failed"
                    ),
                },
            ),
            "execution_mode": "online_native_a2a_ep_world_size_1_parity",
            "expert_residency_mode": "full_model_local_weight_extract_for_parity",
            "correctness_status": "passed" if parity["parity"]["numerical_correctness_pass"] else "failed",
            "numerical_correctness_pass": parity["parity"]["numerical_correctness_pass"],
            "world_size_1_parity": parity,
        }
    else:
        payload = build_online_unimplemented_result(
            run_id=run_id,
            world_size=args.world_size,
            transport_backend="online_native_a2a_ep",
            extra={
                "entrypoint": "bench_native_ep",
                "output_dir": args.output_dir,
                "implemented_reason": "online native EP runtime is not implemented yet for world_size>1",
            },
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("numerical_correctness_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
