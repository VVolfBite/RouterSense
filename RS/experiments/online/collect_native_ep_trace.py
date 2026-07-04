#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.online import build_online_unimplemented_result
from rs.online.olmoe_ep import collect_world_size_one_observed_native_ep_trace, export_native_ep_trace_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect online native EP trace.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Explain mixture-of-experts routing in one paragraph.")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="artifacts/online/native_ep_trace")
    args = parser.parse_args(argv)
    run_id = f"online-native-trace-{uuid.uuid4().hex[:12]}"
    if int(args.world_size) == 1:
        observed = collect_world_size_one_observed_native_ep_trace(
            model_id=args.model,
            model_path=args.model_path,
            prompt_text=args.prompt,
            layer_index=args.layer_index,
            precision=args.precision,
            device_index=args.device_index,
        )
        jsonl_path, metadata_path = export_native_ep_trace_artifacts(
            output_dir=args.output_dir,
            run_id=run_id,
            trace=observed.execution_trace,
            extra_metadata={
                **observed.metadata,
                "entrypoint": "collect_native_ep_trace",
                "implemented": True,
                "implemented_scope": "world_size_1_observed_native_ep_trace",
                "expert_residency_mode": "full_model_local_weight_extract_for_parity",
                "correctness_status": (
                    "passed" if observed.parity.numerical_correctness_pass else "failed"
                ),
            },
        )
        payload = {
            "pipeline": "online",
            "execution_mode": "online_native_a2a_ep_world_size_1_observed_trace",
            "trace_origin": "observed_online_native_ep",
            "future_information_mode": "none",
            "correctness_status": "passed" if observed.parity.numerical_correctness_pass else "failed",
            "numerical_correctness_pass": observed.parity.numerical_correctness_pass,
            "performance_claim_eligible": False,
            "jsonl_path": str(jsonl_path),
            "metadata_path": str(metadata_path),
            "parity": observed.parity.to_dict(),
        }
    else:
        payload = build_online_unimplemented_result(
            run_id=run_id,
            world_size=args.world_size,
            transport_backend="online_native_a2a_ep",
            extra={
                "entrypoint": "collect_native_ep_trace",
                "output_dir": args.output_dir,
                "implemented_reason": "online observer is not implemented yet for world_size>1",
            },
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}_summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("numerical_correctness_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
