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
    collect_world_size_one_local_moe_observed_trace,
    execute_ws2_hidden_dispatch_only,
    export_ws2_hidden_dispatch_trace_artifacts,
    export_single_rank_local_moe_trace_artifacts,
    export_ws2_native_ep_moe_layer_harness_trace_artifacts,
    export_ws2_route_partition_trace_artifacts,
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
    if execution_mode == "world_size_1_local_moe_reconstruction_observation":
        return 0 if payload.get("numerical_correctness_pass") is True else 2
    if execution_mode in {"online_ws2_route_partition_only", "online_ws2_hidden_dispatch_only"}:
        return 0 if correctness_status == "metadata_passed" else 2
    if execution_mode == "online_ws2_native_ep_moe_layer_harness":
        return 0 if correctness_status in {"passed", "skipped_no_remote_route"} else 2
    return 2


def _write_merged_summary(output_dir: Path, run_id: str, world_size: int) -> None:
    payloads: list[dict[str, object]] = []
    for rank in range(int(world_size)):
        rank_path = output_dir / f"{run_id}-rank{rank}_summary.json"
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
        "artifact_paths": [
            {
                "rank": item.get("rank"),
                "jsonl_path": item.get("jsonl_path"),
                "metadata_path": item.get("metadata_path"),
                "summary_path": str(output_dir / f"{run_id}-rank{item.get('rank')}_summary.json"),
            }
            for item in payloads
        ],
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
    parser = argparse.ArgumentParser(description="Collect online native EP trace.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Explain mixture-of-experts routing in one paragraph.")
    parser.add_argument("--prompt-rank0", type=str, default=None)
    parser.add_argument("--prompt-rank1", type=str, default=None)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--route-partition-only", action="store_true")
    parser.add_argument("--hidden-dispatch-only", action="store_true")
    parser.add_argument("--validate-metadata", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--require-remote-route", action="store_true")
    parser.add_argument("--allow-identical-prompts", action="store_true")
    parser.add_argument("--backend", type=str, choices=("nccl", "gloo"), default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/online/native_ep_trace")
    args = parser.parse_args(argv)
    run_id = _resolve_run_id("online-native-trace", args.run_id, args.world_size)
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
        observed = collect_world_size_one_local_moe_observed_trace(
            model_id=args.model,
            model_path=args.model_path,
            prompt_text=args.prompt,
            layer_index=args.layer_index,
            precision=args.precision,
            device_index=(rank_device.index if rank_device.type == "cuda" else 0),
        )
        jsonl_path, metadata_path = export_single_rank_local_moe_trace_artifacts(
            output_dir=args.output_dir,
            run_id=run_id,
            trace=observed.execution_trace,
            extra_metadata={
                **observed.metadata,
                "entrypoint": "collect_native_ep_trace",
                "implemented": True,
                "implemented_scope": "world_size_1_local_moe_reconstruction_observation",
                "expert_residency_mode": "full_model_local_weight_extract_for_parity",
                "correctness_status": (
                    "passed" if observed.parity.numerical_correctness_pass else "failed"
                ),
                "is_real_ep_runtime": False,
                "is_real_ep_transport": False,
                "is_transport_calibration_trace": False,
            },
        )
        payload = {
            "pipeline": "online",
            "execution_mode": "world_size_1_local_moe_reconstruction_observation",
            "trace_origin": "observed_single_rank_local_moe",
            "future_information_mode": "none",
            "correctness_status": "passed" if observed.parity.numerical_correctness_pass else "failed",
            "numerical_correctness_pass": observed.parity.numerical_correctness_pass,
            "performance_claim_eligible": False,
            "is_real_ep_runtime": False,
            "is_real_ep_transport": False,
            "is_transport_calibration_trace": False,
            "jsonl_path": str(jsonl_path),
            "metadata_path": str(metadata_path),
            "parity": observed.parity.to_dict(),
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
            rank_suffix = f"rank{observed.rank}"
            trace = observed.trace
            export_fn = export_ws2_route_partition_trace_artifacts
            trace_origin = "observed_online_ws2_route_partition"
            execution_mode = "online_ws2_route_partition_only"
            claim_scope = "distributed_route_partition_and_count_agreement_only"
            transport_payload = observed.agreement.transport_record.to_dict()
            correctness_status = observed.agreement.validation.correctness_status
            is_real_ep_transport = False
            implemented_scope = "world_size_2_route_partition_and_count_agreement_only"
            extra_metadata = {
                **observed.metadata,
                "entrypoint": "collect_native_ep_trace",
                "implemented": True,
                "implemented_scope": implemented_scope,
                "rank": observed.rank,
                "backend": backend,
                "verified_backend": "nccl_gpu" if backend == "nccl" else "gloo_cpu_test_only",
                "gloo_test_mode": backend == "gloo",
                "is_gpu_transport_verified": backend == "nccl",
                "is_complete_ep_dispatch": False,
                "device_info": observed.device_info.to_dict(),
                "distinct_cuda_device_indices": distinct_device_indices,
                "placement_hash": observed.placement.placement_hash,
                "manifest_hash": observed.manifest.manifest_hash,
                "request_protocol_hash": observed.manifest.request_protocol_hash,
                "request_table_hash": observed.request_table_hash,
                "gathered_send_count_matrix": observed.agreement.gathered_send_count_matrix,
                "gathered_manifest_hashes": observed.agreement.gathered_manifest_hashes,
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
                trace = build_ws2_hidden_dispatch_trace(
                    partition=observed.partition,
                    placement=observed.placement,
                    manifest=observed.manifest,
                    hidden_dispatch=hidden_dispatch,
                )
                export_fn = export_ws2_hidden_dispatch_trace_artifacts
                trace_origin = "observed_online_ws2_hidden_dispatch"
                execution_mode = "online_ws2_hidden_dispatch_only"
                claim_scope = "distributed_route_partition_count_agreement_and_dispatch_only"
                transport_payload = hidden_dispatch.transport_record.to_dict()
                correctness_status = hidden_dispatch.validation.correctness_status
                is_real_ep_transport = True
                implemented_scope = "world_size_2_hidden_dispatch_only"
                extra_metadata.update(
                    {
                        "implemented_scope": implemented_scope,
                        "dispatch_transport_operation": hidden_dispatch.transport_record.to_dict(),
                        "received_remote_route_count": len(hidden_dispatch.received_routes),
                    }
                )
            jsonl_path, metadata_path = export_fn(
                output_dir=args.output_dir,
                run_id=f"{run_id}-{rank_suffix}",
                trace=trace,
                extra_metadata=extra_metadata,
            )
            payload = {
                "pipeline": "online",
                "execution_mode": execution_mode,
                "trace_origin": trace_origin,
                "future_information_mode": "none",
                "claim_scope": claim_scope,
                "correctness_status": correctness_status,
                "numerical_correctness_pass": None,
                "performance_claim_eligible": False,
                "is_real_ep_runtime": False,
                "is_real_ep_transport": is_real_ep_transport,
                "is_complete_ep_dispatch": False,
                "is_transport_calibration_trace": False,
                "rank": observed.rank,
                "backend": backend,
                "verified_backend": "nccl_gpu" if backend == "nccl" else "gloo_cpu_test_only",
                "gloo_test_mode": backend == "gloo",
                "is_gpu_transport_verified": backend == "nccl",
                "device_info": observed.device_info.to_dict(),
                "distinct_cuda_device_indices": distinct_device_indices,
                "jsonl_path": str(jsonl_path),
                "metadata_path": str(metadata_path),
                "placement_hash": observed.placement.placement_hash,
                "manifest_hash": observed.manifest.manifest_hash,
                "request_protocol_hash": observed.manifest.request_protocol_hash,
                "request_table_hash": observed.request_table_hash,
                "local_route_count": len(observed.partition.local_routes),
                "remote_route_count": len(observed.partition.remote_send_routes),
                "per_peer_send_rows": observed.partition.per_peer_send_rows,
                "transport_operation": transport_payload,
            }
        finally:
            pass
    elif int(args.world_size) == 2:
        if backend == "nccl":
            torch.cuda.set_device(rank_device)
        init_process_group_checked(backend=backend)
        try:
            prompts_by_rank = [
                str(args.prompt_rank0 if args.prompt_rank0 is not None else args.prompt),
                str(args.prompt_rank1 if args.prompt_rank1 is not None else args.prompt),
            ]
            result = run_world_size_two_native_ep_moe_layer(
                run_id=run_id,
                model_id=args.model,
                model_path=args.model_path,
                prompts_by_rank=prompts_by_rank,
                layer_index=args.layer_index,
                precision=args.precision,
                rank_device=rank_device,
                backend=backend,
                require_remote_route=bool(args.require_remote_route),
                validate=bool(args.validate),
                output_dir=args.output_dir,
            )
            rank_jsonl, rank_metadata = export_ws2_native_ep_moe_layer_harness_trace_artifacts(
                output_dir=args.output_dir,
                run_id=f"{run_id}-rank{result.rank}",
                trace=result.trace,
                extra_metadata={
                    **result.trace.metadata,
                    "entrypoint": "collect_native_ep_trace",
                    "backend": backend,
                    "verified_backend": "nccl_gpu" if backend == "nccl" else "gloo_cpu_test_only",
                    "rank": result.rank,
                    "device_info": result.device_info.to_dict(),
                    "distinct_cuda_device_indices": result.distinct_cuda_device_indices,
                    "placement_hash": result.placement.placement_hash,
                    "manifest_hash": result.manifest.manifest_hash,
                    "request_protocol_hash": result.manifest.request_protocol_hash,
                    "request_table_hash": result.manifest.request_table_hash,
                    "local_route_count": result.local_route_count,
                    "remote_route_count": result.remote_route_count,
                    "dispatch_rows": result.dispatch_rows,
                    "combine_rows": result.combine_rows,
                    "correctness_status": result.validation.correctness_status,
                    "numerical_correctness_pass": result.validation.numerical_correctness_pass,
                    "trace_origin": "observed_online_ws2_moe_layer_harness",
                },
            )
            payload = {
                "pipeline": "online",
                "execution_mode": "online_ws2_native_ep_moe_layer_harness",
                "trace_origin": "observed_online_ws2_moe_layer_harness",
                "future_information_mode": "none",
                "claim_scope": "ws2_distributed_moe_layer_correctness_only",
                "is_real_ep_runtime": False,
                "is_complete_ep_dispatch": bool(result.transport_exercised),
                "is_real_ep_transport": bool(result.transport_exercised),
                "transport_backend": "online_native_a2a_ep",
                "prediction_used": False,
                "prediction_confidence": 0.0,
                "backend": backend,
                "verified_backend": "nccl_gpu" if backend == "nccl" else "gloo_cpu_test_only",
                "rank": result.rank,
                "device_info": result.device_info.to_dict(),
                "distinct_cuda_device_indices": result.distinct_cuda_device_indices,
                "placement_hash": result.placement.placement_hash,
                "manifest_hash": result.manifest.manifest_hash,
                "request_protocol_hash": result.manifest.request_protocol_hash,
                "request_table_hash": result.manifest.request_table_hash,
                "correctness_status": result.validation.correctness_status,
                "numerical_correctness_pass": result.validation.numerical_correctness_pass,
                "max_abs_error": result.validation.max_abs_error,
                "mean_abs_error": result.validation.mean_abs_error,
                "relative_error": result.validation.relative_error,
                "cosine_similarity": result.validation.cosine_similarity,
                "local_route_count": result.local_route_count,
                "remote_route_count": result.remote_route_count,
                "dispatch_rows": result.dispatch_rows,
                "combine_rows": result.combine_rows,
                "jsonl_path": str(rank_jsonl),
                "metadata_path": str(rank_metadata),
                "transport_operations": [record.to_dict() for record in result.trace.transport_operations],
                "validation_details": result.validation.details,
                "trace_metadata": result.trace.metadata,
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
                "entrypoint": "collect_native_ep_trace",
                "output_dir": args.output_dir,
                "implemented_reason": "online observer is not implemented yet for world_size>1",
            },
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{run_id}_summary.json"
    if int(args.world_size) == 2:
        rank_suffix = payload.get("rank")
        if rank_suffix is not None:
            output_name = f"{run_id}-rank{rank_suffix}_summary.json"
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
