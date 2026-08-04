#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.online import build_online_unimplemented_result
from rs.online.distributed_runtime import (
    DistributedStageError,
    assert_distinct_cuda_device_mapping,
    destroy_process_group_checked,
    init_process_group_checked,
    require_ws2_native_ep_backend,
    resolve_backend,
    resolve_distributed_device,
)
from rs.online.olmoe_ep import (
    build_ws2_hidden_dispatch_trace,
    execute_ws2_hidden_dispatch_only,
    run_world_size_one_native_parity,
    run_world_size_two_native_ep_moe_layer,
    run_world_size_two_route_partition_only,
)


def _resolve_run_id(prefix: str, explicit_run_id: str | None, world_size: int) -> str:
    if explicit_run_id:
        return explicit_run_id
    if int(world_size) > 1:
        return str(os.environ.get("TORCHELASTIC_RUN_ID", f"{prefix}-shared"))
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _result_exit_code(payload: dict[str, object]) -> int:
    execution_mode = str(payload.get("execution_mode", ""))
    correctness_status = str(payload.get("correctness_status", ""))
    if execution_mode == "world_size_1_local_moe_reconstruction_parity":
        return 0 if payload.get("numerical_correctness_pass") is True else 2
    if execution_mode in {"online_ws2_route_partition_only", "online_ws2_hidden_dispatch_only"}:
        return 0 if correctness_status == "metadata_passed" else 2
    if execution_mode in {
        "online_ws2_native_ep_moe_layer",
        "online_ws2_native_ep_moe_layer_harness",
    }:
        return 0 if correctness_status in {"passed", "skipped_no_remote_route"} else 2
    return 2


def _write_merged_summary(output_dir: Path, run_id: str, world_size: int) -> None:
    payloads: list[dict[str, object]] = []
    for rank in range(int(world_size)):
        rank_path = output_dir / f"{run_id}-rank{rank}.json"
        if not rank_path.exists():
            raise RuntimeError(f"missing rank summary for merged summary: {rank_path}")
        payloads.append(json.loads(rank_path.read_text(encoding="utf-8")))
    merged = {
        "run_id": run_id,
        "world_size": int(world_size),
        "execution_mode": payloads[0].get("execution_mode"),
        "claim_scope": payloads[0].get("claim_scope"),
        "backend": payloads[0].get("backend"),
        "device_mapping": [item.get("device_info", {}) for item in payloads],
        "rank_gpu_mapping": [
            {
                "rank": item.get("rank"),
                "device": item.get("device_info", {}).get("device"),
                "cuda_device_index": item.get("device_info", {}).get("cuda_device_index"),
                "cuda_device_name": item.get("device_info", {}).get("cuda_device_name"),
            }
            for item in payloads
        ],
        "remote_route_count": sum(int(item.get("remote_route_count", 0) or 0) for item in payloads),
        "dispatch_rows": sum(int(item.get("dispatch_rows", 0) or 0) for item in payloads),
        "combine_rows": sum(int(item.get("combine_rows", 0) or 0) for item in payloads),
        "rank_summaries": payloads,
    }
    (output_dir / f"{run_id}-merged.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")


def _failure_payload(
    *,
    run_id: str,
    world_size: int,
    backend: str,
    execution_mode: str,
    rank: int | None,
    rank_device: torch.device,
    stage: str | None,
    error_text: str,
    failures: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "pipeline": "online",
        "execution_mode": execution_mode,
        "trace_origin": "not_collected",
        "future_information_mode": "none",
        "claim_scope": "ws2_distributed_moe_layer_correctness_only",
        "backend": backend,
        "verified_backend": "nccl_gpu" if backend == "nccl" else "gloo_cpu_test_only",
        "rank": rank,
        "world_size": int(world_size),
        "device": str(rank_device),
        "correctness_status": "failed",
        "numerical_correctness_pass": False,
        "failure_stage": stage,
        "failure_error": error_text,
        "failure_details": failures or [],
        "failed_at_utc": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
    }


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
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--route-partition-only", action="store_true")
    parser.add_argument("--hidden-dispatch-only", action="store_true")
    parser.add_argument("--validate-metadata", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--require-remote-route", action="store_true")
    parser.add_argument("--allow-identical-prompts", action="store_true")
    parser.add_argument("--backend", type=str, choices=("nccl", "gloo"), default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/online/bench_native_ep")
    args = parser.parse_args(argv)
    run_id = _resolve_run_id("online-native-bench", args.run_id, args.world_size)
    backend = resolve_backend(world_size=args.world_size, requested_backend=args.backend)
    require_ws2_native_ep_backend(
        world_size=args.world_size,
        backend=backend,
        route_partition_only=bool(args.route_partition_only),
        hidden_dispatch_only=bool(args.hidden_dispatch_only),
    )
    rank_device = resolve_distributed_device(
        requested_device_index=args.device_index,
        world_size=args.world_size,
        backend=backend,
    )
    if int(args.world_size) == 1:
        parity = run_world_size_one_native_parity(
            model_id=args.model,
            model_path=args.model_path,
            prompt_text=args.prompt,
            layer_index=args.layer_index,
            precision=args.precision,
            device_index=(rank_device.index if rank_device.type == "cuda" else 0),
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
        if backend == "nccl":
            torch.cuda.set_device(rank_device)
        init_process_group_checked(backend=backend)
        try:
            distinct_device_indices = assert_distinct_cuda_device_mapping(
                backend=backend,
                rank_device=rank_device,
                world_size=args.world_size,
            )
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
                rank_device=rank_device,
                validate_metadata=bool(args.validate_metadata),
                backend=backend,
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
                "is_complete_ep_dispatch": False,
                "is_transport_calibration_trace": False,
                "correctness_status": observed.agreement.validation.correctness_status,
                "numerical_correctness_pass": None,
                "performance_claim_eligible": False,
                "world_size": 2,
                "backend": backend,
                "verified_backend": "nccl_gpu" if backend == "nccl" else "gloo_cpu_test_only",
                "gloo_test_mode": backend == "gloo",
                "is_gpu_transport_verified": backend == "nccl",
                "device_info": observed.device_info.to_dict(),
                "distinct_cuda_device_indices": distinct_device_indices,
                "placement_hash": observed.placement.placement_hash,
                "manifest_hash": observed.manifest.manifest_hash,
                "request_protocol_hash": observed.manifest.request_protocol_hash,
                "request_table_hash": observed.request_table_hash,
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
                    request_id_table=observed.request_id_table,
                    microbatch_id_table=observed.microbatch_id_table,
                )
                payload.update(
                    {
                        "execution_mode": "online_ws2_hidden_dispatch_only",
                        "trace_origin": "observed_online_ws2_hidden_dispatch",
                        "claim_scope": "distributed_route_partition_count_agreement_and_dispatch_only",
                        "is_real_ep_transport": True,
                        "is_complete_ep_dispatch": False,
                        "correctness_status": hidden_dispatch.validation.correctness_status,
                        "hidden_dispatch_transport": hidden_dispatch.transport_record.to_dict(),
                        "received_remote_route_count": len(hidden_dispatch.received_routes),
                    }
                )
        finally:
            pass
    elif int(args.world_size) == 2:
        if backend == "nccl":
            torch.cuda.set_device(rank_device)
        init_process_group_checked(backend=backend)
        try:
            payload_result = run_world_size_two_native_ep_moe_layer(
                run_id=run_id,
                model_id=args.model,
                model_path=args.model_path,
                prompts_by_rank=[
                    str(args.prompt_rank0 if args.prompt_rank0 is not None else args.prompt),
                    str(args.prompt_rank1 if args.prompt_rank1 is not None else args.prompt),
                ],
                layer_index=args.layer_index,
                precision=args.precision,
                rank_device=rank_device,
                backend=backend,
                require_remote_route=bool(args.require_remote_route),
                validate=bool(args.validate),
                output_dir=args.output_dir,
            )
            payload = {
                "pipeline": "online",
                "execution_mode": "online_ws2_native_ep_moe_layer_harness",
                "trace_origin": "observed_online_ws2_moe_layer_harness",
                "future_information_mode": "none",
                "claim_scope": "ws2_distributed_moe_layer_correctness_only",
                "backend": backend,
                "verified_backend": "nccl_gpu" if backend == "nccl" else "gloo_cpu_test_only",
                "gloo_test_mode": backend == "gloo",
                "is_gpu_transport_verified": backend == "nccl",
                "is_real_ep_transport": bool(payload_result.transport_exercised),
                "is_complete_ep_dispatch": bool(payload_result.transport_exercised),
                "is_real_ep_runtime": False,
                "claim_scope_detail": "ws2_distributed_moe_layer_correctness_only",
                "execution_harness": "observed_online_ws2_moe_layer_harness",
                "prediction_used": False,
                "expert_residency_mode": "full_checkpoint_then_local_extract",
                "checkpoint_loading_is_memory_efficient": False,
                "correctness_status": payload_result.validation.correctness_status,
                "numerical_correctness_pass": payload_result.validation.numerical_correctness_pass,
                "max_abs_error": payload_result.validation.max_abs_error,
                "mean_abs_error": payload_result.validation.mean_abs_error,
                "relative_error": payload_result.validation.relative_error,
                "cosine_similarity": payload_result.validation.cosine_similarity,
                "performance_claim_eligible": False,
                "validation_reference": "hf_single_rank_layer_output",
                "transport_exercised": payload_result.transport_exercised,
                "rank": payload_result.rank,
                "device_info": payload_result.device_info.to_dict(),
                "distinct_cuda_device_indices": payload_result.distinct_cuda_device_indices,
                "placement_hash": payload_result.placement.placement_hash,
                "manifest_hash": payload_result.manifest.manifest_hash,
                "request_protocol_hash": payload_result.manifest.request_protocol_hash,
                "request_table_hash": payload_result.manifest.request_table_hash,
                "local_route_count": payload_result.local_route_count,
                "remote_route_count": payload_result.remote_route_count,
                "dispatch_rows": payload_result.dispatch_rows,
                "combine_rows": payload_result.combine_rows,
                "transport_operations": [record.to_dict() for record in payload_result.trace.transport_operations],
                "validation_details": payload_result.validation.details,
                "trace_metadata": payload_result.trace.metadata,
            }
        except DistributedStageError as exc:
            payload = _failure_payload(
                run_id=run_id,
                world_size=args.world_size,
                backend=backend,
                execution_mode="online_ws2_native_ep_moe_layer_harness",
                rank=(dist.get_rank() if dist.is_initialized() else None),
                rank_device=rank_device,
                stage=exc.stage,
                error_text=str(exc),
                failures=exc.failures,
            )
        except Exception as exc:
            payload = _failure_payload(
                run_id=run_id,
                world_size=args.world_size,
                backend=backend,
                execution_mode="online_ws2_native_ep_moe_layer_harness",
                rank=(dist.get_rank() if dist.is_initialized() else None),
                rank_device=rank_device,
                stage=None,
                error_text=f"{type(exc).__name__}: {exc}",
            )
        finally:
            pass
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
    if int(args.world_size) == 2:
        rank_suffix = payload.get("rank")
        if rank_suffix is None:
            rank_suffix = payload.get("metadata_details", {}).get("rank")
        if rank_suffix is not None:
            output_name = f"{run_id}-rank{rank_suffix}.json"
    output_path = output_dir / output_name
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if int(args.world_size) == 2 and dist.is_initialized():
        if payload.get("correctness_status") != "failed":
            dist.barrier()
            if dist.get_rank() == 0:
                _write_merged_summary(output_dir, run_id, args.world_size)
            dist.barrier()
        destroy_process_group_checked()
    print(json.dumps(payload, indent=2))
    return _result_exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
