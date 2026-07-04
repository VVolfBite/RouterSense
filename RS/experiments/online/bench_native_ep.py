#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import torch.distributed as dist

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.online import build_online_unimplemented_result
from rs.online.olmoe_ep import (
    build_ws2_hidden_dispatch_trace,
    execute_ws2_hidden_dispatch_only,
    run_world_size_one_native_parity,
    run_world_size_two_route_partition_only,
)


def _resolve_run_id(prefix: str, explicit_run_id: str | None, world_size: int) -> str:
    if explicit_run_id:
        return explicit_run_id
    if int(world_size) > 1:
        return str(os.environ.get("TORCHELASTIC_RUN_ID", f"{prefix}-shared"))
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark online native EP runtime.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Explain mixture-of-experts routing in one paragraph.")
    parser.add_argument("--prompt-rank0", type=str, default=None)
    parser.add_argument("--prompt-rank1", type=str, default=None)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--route-partition-only", action="store_true")
    parser.add_argument("--hidden-dispatch-only", action="store_true")
    parser.add_argument("--validate-metadata", action="store_true")
    parser.add_argument("--allow-identical-prompts", action="store_true")
    parser.add_argument("--dist-backend", type=str, default="gloo")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/online/bench_native_ep")
    args = parser.parse_args(argv)
    run_id = _resolve_run_id("online-native-bench", args.run_id, args.world_size)
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
                    "implemented_scope": "world_size_1_local_moe_reconstruction_parity",
                    "correctness_status": (
                        "passed" if parity["parity"]["numerical_correctness_pass"] else "failed"
                    ),
                },
            ),
            "execution_mode": "world_size_1_local_moe_reconstruction_parity",
            "trace_origin": "observed_single_rank_local_moe",
            "is_real_ep_runtime": False,
            "expert_residency_mode": "full_model_local_weight_extract_for_parity",
            "correctness_status": "passed" if parity["parity"]["numerical_correctness_pass"] else "failed",
            "numerical_correctness_pass": parity["parity"]["numerical_correctness_pass"],
            "world_size_1_parity": parity,
        }
    elif int(args.world_size) == 2 and bool(args.route_partition_only):
        if not dist.is_initialized():
            dist.init_process_group(backend=args.dist_backend)
        try:
            observed = run_world_size_two_route_partition_only(
                run_id=run_id,
                model_id=args.model,
                model_path=args.model_path,
                prompts_by_rank=[
                    str(args.prompt_rank0 if args.prompt_rank0 is not None else args.prompt),
                    str(args.prompt_rank1 if args.prompt_rank1 is not None else args.prompt),
                ],
                layer_index=args.layer_index,
                precision=args.precision,
                device_index=args.device_index,
                validate_metadata=bool(args.validate_metadata),
                allow_identical_prompts=bool(args.allow_identical_prompts),
            )
            payload = {
                "pipeline": "online",
                "execution_mode": "online_ws2_route_partition_only",
                "trace_origin": "observed_online_ws2_route_partition",
                "future_information_mode": "none",
                "claim_scope": "distributed_route_partition_and_count_agreement_only",
                "is_real_ep_runtime": False,
                "is_real_ep_transport": False,
                "is_transport_calibration_trace": False,
                "correctness_status": observed.agreement.validation.correctness_status,
                "performance_claim_eligible": False,
                "world_size": 2,
                "placement_hash": observed.placement.placement_hash,
                "manifest_hash": observed.manifest.manifest_hash,
                "request_protocol_hash": observed.manifest.request_protocol_hash,
                "local_route_count": len(observed.partition.local_routes),
                "remote_route_count": len(observed.partition.remote_send_routes),
                "per_peer_send_rows": observed.partition.per_peer_send_rows,
                "transport_operation": observed.agreement.transport_record.to_dict(),
                "gathered_send_count_matrix": observed.agreement.gathered_send_count_matrix,
                "gathered_manifest_hashes": observed.agreement.gathered_manifest_hashes,
                "metadata_details": observed.agreement.validation.details,
            }
            if bool(args.hidden_dispatch_only):
                hidden_dispatch = execute_ws2_hidden_dispatch_only(
                    hidden_states=observed.hidden_states,
                    partition=observed.partition,
                    manifest=observed.manifest,
                    placement=observed.placement,
                    agreement=observed.agreement,
                )
                payload.update(
                    {
                        "execution_mode": "online_ws2_hidden_dispatch_only",
                        "trace_origin": "observed_online_ws2_hidden_dispatch",
                        "claim_scope": "distributed_hidden_dispatch_only",
                        "is_real_ep_transport": True,
                        "correctness_status": hidden_dispatch.validation.correctness_status,
                        "hidden_dispatch_transport": hidden_dispatch.transport_record.to_dict(),
                        "received_remote_route_count": len(hidden_dispatch.received_routes),
                    }
                )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
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
    output_name = f"{run_id}.json"
    if int(args.world_size) == 2 and bool(args.route_partition_only):
        rank_suffix = payload.get("metadata_details", {}).get("rank")
        if rank_suffix is not None:
            output_name = f"{run_id}-rank{rank_suffix}.json"
    output_path = output_dir / output_name
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("numerical_correctness_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
