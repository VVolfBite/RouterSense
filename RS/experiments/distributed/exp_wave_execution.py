#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

import torch  # type: ignore

from rs.runtime import load_model_and_tokenizer
from rs.runtime.distributed_ep.adapter.runner import (
    LEGACY_TRACE_REPLAY_MODE,
    REAL_EP_MODE,
    SCHEDULED_COLLECTIVE_PARTITION_REPLAY,
    TRACE_REPLAY_MODE,
    UNSCHEDULED_COLLECTIVE_REPLAY,
    WAVE_COLLECTIVE_REPLAY,
    DistributedRunnerConfig,
    build_distributed_runner_plan,
    execute_scheduled_inference,
)
from rs.topology import load_inventory, resolve_preferred_model_path
from rs.trace import collect_full_sequence_trace


def _load_prompt_samples(
    *,
    prompt: str,
    prompt_file: str | None,
    sample_limit: int,
    text_key: str,
) -> list[dict[str, Any]]:
    if prompt_file is None:
        return [{"sample_id": "wave-sample-0", "request_id": "wave-exec-0", "text": prompt}]

    path = Path(prompt_file)
    if not path.exists():
        raise RuntimeError(f"prompt file not found: {path}")

    samples: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                text = str(payload.get(text_key, "")).strip()
                if not text:
                    continue
                samples.append(
                    {
                        "sample_id": str(payload.get("sample_id", payload.get("document_id", f"sample-{index}"))),
                        "request_id": str(payload.get("request_id", payload.get("message_id", f"req-{index}"))),
                        "text": text,
                        "metadata": payload,
                    }
                )
                if len(samples) >= sample_limit:
                    break
    else:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index, text in enumerate(lines[:sample_limit]):
            samples.append({"sample_id": f"sample-{index}", "request_id": f"req-{index}", "text": text})

    if not samples:
        raise RuntimeError(f"no prompt samples loaded from {path}")
    return samples


def _summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(float(value) for value in values)

    def _pct(p: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return ordered[index]

    return {
        "count": float(len(ordered)),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p50": _pct(0.50),
        "p95": _pct(0.95),
    }


def _summarize_correctness(samples: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: list[str] = []
    token_conservation_values: list[bool] = []
    gate_weight_values: list[bool] = []
    max_abs_errors: list[float] = []
    mean_abs_errors: list[float] = []
    cosine_similarities: list[float] = []

    for sample in samples:
        for rank_payload in sample.get("ranks", []):
            correctness = rank_payload["execution"].get("correctness", {})
            statuses.append(str(correctness.get("correctness_status", "unsupported")))

            token_conservation = correctness.get("token_conservation_pass")
            if token_conservation is not None:
                token_conservation_values.append(bool(token_conservation))

            gate_weight = correctness.get("gate_weight_conservation_pass")
            if gate_weight is not None:
                gate_weight_values.append(bool(gate_weight))

            max_abs_error = correctness.get("max_abs_error")
            if max_abs_error is not None:
                max_abs_errors.append(float(max_abs_error))

            mean_abs_error = correctness.get("mean_abs_error")
            if mean_abs_error is not None:
                mean_abs_errors.append(float(mean_abs_error))

            cosine_similarity = correctness.get("cosine_similarity")
            if cosine_similarity is not None:
                cosine_similarities.append(float(cosine_similarity))

    status_counts: dict[str, int] = {}
    for status in statuses:
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "status_counts": status_counts,
        "token_conservation_pass": all(token_conservation_values) if token_conservation_values else None,
        "gate_weight_conservation_pass": all(gate_weight_values) if gate_weight_values else None,
        "max_abs_error": max(max_abs_errors) if max_abs_errors else None,
        "mean_abs_error": (sum(mean_abs_errors) / len(mean_abs_errors)) if mean_abs_errors else None,
        "cosine_similarity_p50": _summarize_numeric(cosine_similarities)["p50"] if cosine_similarities else None,
    }


def _summarize_batch(result: dict[str, Any]) -> dict[str, Any]:
    samples = list(result.get("samples", []))
    if not samples:
        return {"sample_count": 0}

    effective_sample_ms: list[float] = []
    planner_ms: list[float] = []
    control_plane_ms: list[float] = []
    scheduled_comm_ms: list[float] = []
    native_comm_ms: list[float] = []
    communication_saved_ms: list[float] = []
    communication_ratio: list[float] = []
    token_counts: list[float] = []
    trace_ms: list[float] = []
    validation_ms: list[float] = []
    plan_build_ms: list[float] = []

    for sample in samples:
        rank_payloads = list(sample.get("ranks", []))
        if not rank_payloads:
            continue
        effective_sample_ms.append(max(float(rank_payload.get("sample_wall_ms", 0.0)) for rank_payload in rank_payloads))
        token_counts.append(float(sample.get("trace_summary", {}).get("token_count", 0)))
        trace_ms.append(float(sample.get("trace_ms", 0.0)))
        plan_build_ms.append(float(sample.get("plan_build_ms", 0.0)))
        validation_ms.append(max(float(rank_payload.get("validation_ms", 0.0)) for rank_payload in rank_payloads))
        for rank_payload in rank_payloads:
            execution = rank_payload["execution"]
            control = execution.get("control_plane_ms", {})
            wave_execution = execution.get("wave_execution", {})
            native = execution.get("unscheduled_collective_replay", {})
            scheduled_total = float(wave_execution.get("dispatch_comm_ms", 0.0)) + float(wave_execution.get("combine_comm_ms", 0.0))
            native_total = float(native.get("dispatch_comm_ms", 0.0)) + float(native.get("combine_comm_ms", 0.0))
            planner_ms.append(float(control.get("planner_ms", 0.0)))
            control_plane_ms.append(
                float(control.get("matrix_build_ms", 0.0))
                + float(control.get("all_reduce_ms", 0.0))
                + float(control.get("planner_ms", 0.0))
                + float(control.get("wave_convert_ms", 0.0))
            )
            scheduled_comm_ms.append(scheduled_total)
            native_comm_ms.append(native_total)
            communication_saved_ms.append(native_total - scheduled_total)
            if native_total > 0:
                communication_ratio.append(scheduled_total / native_total)

    batch_wall_ms = float(result.get("run", {}).get("batch_wall_ms", 0.0))
    sample_count = len(samples)
    total_tokens = sum(token_counts)
    return {
        "sample_count": sample_count,
        "batch_wall_ms": batch_wall_ms,
        "samples_per_second": (sample_count * 1000.0 / batch_wall_ms) if batch_wall_ms > 0 else 0.0,
        "tokens_per_second": (total_tokens * 1000.0 / batch_wall_ms) if batch_wall_ms > 0 else 0.0,
        "total_trace_tokens": total_tokens,
        "trace_ms": _summarize_numeric(trace_ms),
        "plan_build_ms": _summarize_numeric(plan_build_ms),
        "effective_sample_ms": _summarize_numeric(effective_sample_ms),
        "validation_ms": _summarize_numeric(validation_ms),
        "scheduler_planner_ms": _summarize_numeric(planner_ms),
        "control_plane_ms": _summarize_numeric(control_plane_ms),
        "scheduled_comm_ms": _summarize_numeric(scheduled_comm_ms),
        "native_comm_ms": _summarize_numeric(native_comm_ms),
        "communication_saved_ms": _summarize_numeric(communication_saved_ms),
        "communication_ratio_vs_native": _summarize_numeric(communication_ratio),
        "correctness": _summarize_correctness(samples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated compatibility shim for legacy distributed trace replay. "
            "This is not the formal online EP runtime."
        )
    )
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--inventory", type=str, default=None)
    parser.add_argument("--node-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="U_gated_maxweight_matching")
    parser.add_argument("--prompt", type=str, default="Explain mixture-of-experts routing in one paragraph.")
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--runtime-mode", choices=[TRACE_REPLAY_MODE, REAL_EP_MODE], default=TRACE_REPLAY_MODE)
    parser.add_argument(
        "--execution-mode",
        choices=[UNSCHEDULED_COLLECTIVE_REPLAY, WAVE_COLLECTIVE_REPLAY, SCHEDULED_COLLECTIVE_PARTITION_REPLAY],
        default=UNSCHEDULED_COLLECTIVE_REPLAY,
    )
    parser.add_argument(
        "--transport-granularity",
        choices=["wave", "atomic"],
        default="wave",
        help="Transport granularity: 'wave' = one all_to_all per wave, 'atomic' = one all_to_all per transfer op. Independent of strategy.",
    )
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--max-memory-gb", type=str, default=None)
    parser.add_argument("--distributed-control-plane", action="store_true", default=False)
    parser.add_argument("--compute-mode", choices=["actual_olmoe_expert", "simulated_delay"], default="actual_olmoe_expert")
    parser.add_argument("--expert-compute-delay", type=float, default=0.0)
    parser.add_argument("--layer-index", type=int, default=0, help="Index into MoE layer ids, not raw transformer layer id.")
    parser.add_argument("--max-waves", type=int, default=0, help="Cap the number of execution waves; 0 means uncapped.")
    parser.add_argument(
        "--validation",
        choices=["off", "sampled", "always"],
        default="off",
        help="Whether to run native reference replay and correctness comparison inside the benchmark harness.",
    )
    parser.add_argument("--validation-every", type=int, default=64, help="When --validation=sampled, validate every Nth sample.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "wave_execution"))
    args = parser.parse_args(argv)
    if args.runtime_mode == REAL_EP_MODE:
        raise RuntimeError(
            "runtime_mode=real_ep is not implemented in the current RS mainline; "
            "only legacy_trace_replay is currently supported by this deprecated compatibility path"
        )

    import torch.distributed as dist  # type: ignore

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    model_path = args.model_path
    if model_path is None and args.inventory:
        inventory = load_inventory(Path(args.inventory))
        candidate = resolve_preferred_model_path(inventory, args.model, preferred_node_name=args.node_name)
        if candidate is not None:
            model_path = str(candidate)

    model, tokenizer, _, _, _ = load_model_and_tokenizer(
        model_id=args.model,
        model_path=model_path,
        precision=args.precision,
        device_index=local_rank,
        device_map=args.device_map,
        max_memory_gb=args.max_memory_gb,
    )
    samples = _load_prompt_samples(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        sample_limit=max(1, args.sample_limit),
        text_key=args.text_key,
    )
    batch_execution_ms = 0.0
    gathered_samples: list[dict[str, Any]] = []
    layer_id: int | None = None
    for sample_index, sample in enumerate(samples):
        trace_started = time.perf_counter()
        trace = collect_full_sequence_trace(
            model,
            tokenizer,
            str(sample["text"]),
            request_id=str(sample["request_id"]),
            sample_id=str(sample["sample_id"]),
        )
        trace_ms = (time.perf_counter() - trace_started) * 1000.0
        moe_layer_ids = list(trace["summary"].get("moe_layer_ids", []))
        if not moe_layer_ids:
            raise RuntimeError("trace returned no MoE layer ids")
        if args.layer_index < 0 or args.layer_index >= len(moe_layer_ids):
            raise RuntimeError(f"layer_index {args.layer_index} out of range for {len(moe_layer_ids)} MoE layers")
        layer_id = int(moe_layer_ids[args.layer_index])

        plan_started = time.perf_counter()
        runner_plan = build_distributed_runner_plan(
            model=model,
            trace=trace,
            config=DistributedRunnerConfig(
                world_size=world_size,
                node_rank=int(os.environ.get("NODE_RANK", 0)),
                model_id=args.model,
                origin_rank=0,
            ),
            rank=rank,
            host=socket.gethostname(),
            gpu_name=f"cuda:{local_rank}",
        )
        plan_build_ms = (time.perf_counter() - plan_started) * 1000.0
        hidden_state_rows = trace["hidden_states"][layer_id][0].to(
            dtype=torch.float16,
            device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
        )
        plan_index = runner_plan.dispatch_plans.index(next(plan for plan in runner_plan.dispatch_plans if int(plan.layer_id) == layer_id))

        should_validate = args.validation == "always" or (
            args.validation == "sampled" and args.validation_every > 0 and sample_index % args.validation_every == 0
        )
        execution_started = time.perf_counter()
        execution = execute_scheduled_inference(
            dispatch_plans=runner_plan.dispatch_plans,
            rank=rank,
            world_size=world_size,
            strategy_name=args.strategy,
            hidden_size=int(runner_plan.adapter.get("hidden_size", hidden_state_rows.shape[-1])),
            expert_compute_delay=args.expert_compute_delay if args.compute_mode == "simulated_delay" else 0.0,
            use_distributed=args.distributed_control_plane,
            execution_mode=args.execution_mode,
            local_expert_weights=runner_plan.local_expert_weight_bundle,
            hidden_state_rows=hidden_state_rows,
            plan_index=plan_index,
            max_waves=(args.max_waves if args.max_waves > 0 else None),
            transport_granularity=args.transport_granularity,
            verify_correctness=should_validate,
            runtime_mode=args.runtime_mode,
        )
        execution_wall_ms = (time.perf_counter() - execution_started) * 1000.0
        validation_ms = float(execution.get("control_plane_ms", {}).get("conservation_check_ms", 0.0))
        sample_wall_ms = plan_build_ms + execution_wall_ms - validation_ms
        payload = {
            "rank": rank,
            "sample_index": sample_index,
            "sample_id": sample["sample_id"],
            "request_id": sample["request_id"],
            "world_size": world_size,
            "model": args.model,
            "model_path": model_path,
            "pipeline": "legacy",
            "execution_mode": LEGACY_TRACE_REPLAY_MODE,
            "transport_execution_mode": args.execution_mode,
            "compute_mode": args.compute_mode,
            "strategy": args.strategy,
            "layer_id": layer_id,
            "prompt": sample["text"],
            "trace_ms": trace_ms,
            "plan_build_ms": plan_build_ms,
            "execution_wall_ms": execution_wall_ms,
            "validation_ms": validation_ms,
            "sample_wall_ms": sample_wall_ms,
            "trace_summary": trace["summary"],
            "execution": execution,
        }
        gathered: list[dict] | None = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
        if rank == 0:
            batch_execution_ms += max(float(item.get("sample_wall_ms", 0.0)) for item in gathered if item is not None)
            gathered_samples.append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample["sample_id"],
                    "request_id": sample["request_id"],
                    "prompt": sample["text"],
                    "prompt_metadata": sample.get("metadata"),
                    "trace_ms": trace_ms,
                    "plan_build_ms": plan_build_ms,
                    "trace_summary": trace["summary"],
                    "ranks": gathered,
                }
            )
    if rank == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = {
            "pipeline": "legacy",
            "execution_mode": LEGACY_TRACE_REPLAY_MODE,
            "claim_scope": "transport_replay_only",
            "trace_origin": "legacy_trace_replay",
            "future_information_mode": "oracle_full_trace",
            "is_real_ep_runtime": False,
            "uses_oracle_future_trace": True,
            "baseline_semantics": (
                "unscheduled_collective_replay"
                if args.execution_mode == UNSCHEDULED_COLLECTIVE_REPLAY
                else "scheduled_collective_replay"
            ),
            "source_ownership_mode": "synthetic_token_position_modulo_partition",
            "expert_residency_mode": "rank_local_expert_weight_cache_from_full_model",
            "performance_claim_eligible": False,
            "deprecated_entrypoint": True,
            "run": {
                "model": args.model,
                "strategy": args.strategy,
                "pipeline": "legacy",
                "execution_mode": LEGACY_TRACE_REPLAY_MODE,
                "transport_execution_mode": args.execution_mode,
                "claim_scope": "transport_replay_only",
                "trace_origin": "legacy_trace_replay",
                "future_information_mode": "oracle_full_trace",
                "is_real_ep_runtime": False,
                "uses_oracle_future_trace": True,
                "replay_sample_count": len(samples),
                "compute_mode": args.compute_mode,
                "distributed_control_plane": args.distributed_control_plane,
                "transport_granularity": args.transport_granularity,
                "validation": args.validation,
                "validation_every": args.validation_every,
                "layer_id": layer_id,
                "world_size": world_size,
                "max_waves": args.max_waves,
                "sample_limit": len(samples),
                "prompt_file": args.prompt_file,
                "batch_wall_ms": batch_execution_ms,
            },
            "samples": gathered_samples,
        }
        result["summary"] = _summarize_batch(result)
        result["correctness_status"] = (
            "not_checked"
            if result["summary"]["correctness"]["status_counts"].get("not_checked")
            and len(result["summary"]["correctness"]["status_counts"]) == 1
            else "failed"
            if result["summary"]["correctness"]["status_counts"].get("failed")
            else "passed"
            if result["summary"]["correctness"]["status_counts"].get("passed")
            else "unsupported"
        )
        suffix = f"batch{len(samples)}" if len(samples) > 1 else f"layer{layer_id}"
        (out / f"{args.execution_mode}_{args.strategy}_{suffix}.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
