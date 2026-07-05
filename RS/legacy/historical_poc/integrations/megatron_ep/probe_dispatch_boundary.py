#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.megatron_ep.native_runtime import (  # noqa: E402
    build_position_ids,
    destroy_distributed,
    dtype_from_name,
    gather_rank_payloads,
    get_process_group_ranks_safe,
    get_process_group_root_safe,
    init_distributed,
    load_prompts,
    stage_barrier,
    summarize_rank_environment,
)
from integrations.megatron_ep.routersense.trace_writer import write_json, write_jsonl  # noqa: E402
from integrations.megatron_ep.verify_env import main as verify_env_main  # noqa: E402


def _sha256_file(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _timeline_record(rows: list[dict[str, object]], **payload: object) -> None:
    rows.append({"ts_us": int(time.time() * 1e6), **payload})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--dispatcher", type=str, default="alltoall")
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-id", type=str, default="dispatcher-probe")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status = verify_env_main(["--model", args.model])
    if status != 0:
        return status

    rank = 0
    local_rank = 0
    world_size = 1
    timeline: list[dict[str, object]] = []
    try:
        ids = init_distributed(backend=args.backend, timeout_seconds=300)
        rank = ids["rank"]
        local_rank = ids["local_rank"]
        world_size = ids["world_size"]
        torch.cuda.set_device(local_rank)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        prompts = load_prompts(args.prompt_file)
        dtype = dtype_from_name(args.precision)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=Path(args.model).exists(),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"].to(device=f"cuda:{local_rank}")
        position_ids = build_position_ids(tokens)
        stage_barrier("probe_tokenizer", ok=True, detail=f"batch={tokens.size(0)} seqlen={tokens.size(1)}")

        from megatron.bridge import AutoBridge
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher_mod

        bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.expert_model_parallel_size = args.ep_size
        provider.moe_token_dispatcher_type = args.dispatcher
        provider.parallel_output = False
        provider.pipeline_dtype = dtype
        provider.params_dtype = dtype
        provider.fp16 = dtype == torch.float16
        provider.bf16 = dtype == torch.bfloat16
        provider.gradient_accumulation_fusion = False
        provider.masked_softmax_fusion = False
        provider.bias_activation_fusion = False
        provider.tp_comm_overlap = False
        provider.finalize()
        models = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
        model = models[0].cuda(local_rank).eval()

        dispatcher_capabilities: dict[str, object] = {}
        original_all_to_all = token_dispatcher_mod.all_to_all

        transport_phase = {"value": "unknown"}

        def traced_all_to_all(group, input_tensor, output_split_sizes, input_split_sizes, **kwargs):
            group_ranks = list(get_process_group_ranks_safe(group))
            phase_name = str(transport_phase["value"])
            _timeline_record(
                timeline,
                rank=rank,
                local_rank=local_rank,
                event="actual_transport_enter" if phase_name == "dispatch" else "actual_combine_transport_enter",
                group_ranks=group_ranks,
                transport_phase=phase_name,
                input_shape=list(input_tensor.shape),
                input_dtype=str(input_tensor.dtype),
                output_split_sizes=list(output_split_sizes.tolist() if hasattr(output_split_sizes, "tolist") else output_split_sizes),
                input_split_sizes=list(input_split_sizes.tolist() if hasattr(input_split_sizes, "tolist") else input_split_sizes),
            )
            out = original_all_to_all(group, input_tensor, output_split_sizes, input_split_sizes, **kwargs)
            _timeline_record(
                timeline,
                rank=rank,
                local_rank=local_rank,
                event="actual_transport_exit" if phase_name == "dispatch" else "actual_combine_transport_exit",
                group_ranks=group_ranks,
                transport_phase=phase_name,
                output_shape=list(out.shape),
                output_dtype=str(out.dtype),
            )
            return out

        token_dispatcher_mod.all_to_all = traced_all_to_all

        for name, module in model.named_modules():
            dispatcher = getattr(module, "token_dispatcher", None)
            if dispatcher is None or getattr(dispatcher, "_routersense_probe_wrapped", False):
                continue

            orig_pre = getattr(dispatcher, "dispatch_preprocess", None)
            orig_dispatch = dispatcher.token_dispatch
            orig_combine_pre = getattr(dispatcher, "combine_preprocess", None)
            orig_combine = dispatcher.token_combine
            orig_combine_post = getattr(dispatcher, "combine_postprocess", None)
            ep_group = getattr(dispatcher, "ep_group", None)
            dispatcher_capabilities[name] = {
                "dispatcher_class": type(dispatcher).__name__,
                "module_path": inspect.getsourcefile(type(dispatcher)),
                "module_sha256": _sha256_file(inspect.getsourcefile(type(dispatcher))),
                "dispatch_preprocess_signature": str(inspect.signature(orig_pre)) if orig_pre is not None else None,
                "token_dispatch_signature": str(inspect.signature(orig_dispatch)),
                "combine_preprocess_signature": str(inspect.signature(orig_combine_pre)) if orig_combine_pre is not None else None,
                "token_combine_signature": str(inspect.signature(orig_combine)),
                "combine_postprocess_signature": str(inspect.signature(orig_combine_post)) if orig_combine_post is not None else None,
                "ep_group_ranks": list(get_process_group_ranks_safe(ep_group)),
                "ep_group_root_global_rank": get_process_group_root_safe(ep_group),
            }

            if orig_pre is not None:
                def wrapped_pre(hidden_states, routing_map, probs, _orig=orig_pre, _dispatcher=dispatcher, _name=name):
                    _timeline_record(
                        timeline,
                        rank=rank,
                        local_rank=local_rank,
                        layer=_name,
                        event="dispatch_preprocess_before",
                        hidden_shape=list(hidden_states.shape),
                        routing_map_shape=list(routing_map.shape),
                        probs_shape=list(probs.shape),
                    )
                    out = _orig(hidden_states, routing_map, probs)
                    packed_hidden = out[0] if isinstance(out, tuple) and len(out) >= 1 else None
                    packed_probs = out[1] if isinstance(out, tuple) and len(out) >= 2 else None
                    _timeline_record(
                        timeline,
                        rank=rank,
                        local_rank=local_rank,
                        layer=_name,
                        event="dispatch_preprocess_after",
                        input_splits=list(getattr(_dispatcher, "input_splits", []).tolist() if hasattr(getattr(_dispatcher, "input_splits", None), "tolist") else (getattr(_dispatcher, "input_splits", None) or [])),
                        output_splits=list(getattr(_dispatcher, "output_splits", []).tolist() if hasattr(getattr(_dispatcher, "output_splits", None), "tolist") else (getattr(_dispatcher, "output_splits", None) or [])),
                        packed_hidden_shape=list(packed_hidden.shape) if isinstance(packed_hidden, torch.Tensor) else None,
                        packed_probs_shape=list(packed_probs.shape) if isinstance(packed_probs, torch.Tensor) else None,
                        has_safe_pre_transport_boundary=bool(
                            isinstance(packed_hidden, torch.Tensor)
                            and getattr(_dispatcher, "input_splits", None) is not None
                            and getattr(_dispatcher, "output_splits", None) is not None
                        ),
                    )
                    return out
                dispatcher.dispatch_preprocess = wrapped_pre

            def wrapped_dispatch(hidden_states, probs, _orig=orig_dispatch, _dispatcher=dispatcher, _name=name):
                _timeline_record(
                    timeline,
                    rank=rank,
                    local_rank=local_rank,
                    layer=_name,
                    event="token_dispatch_enter",
                    packed_hidden_shape=list(hidden_states.shape),
                    packed_probs_shape=list(probs.shape),
                    input_splits=list(getattr(_dispatcher, "input_splits", []).tolist() if hasattr(getattr(_dispatcher, "input_splits", None), "tolist") else (getattr(_dispatcher, "input_splits", None) or [])),
                    output_splits=list(getattr(_dispatcher, "output_splits", []).tolist() if hasattr(getattr(_dispatcher, "output_splits", None), "tolist") else (getattr(_dispatcher, "output_splits", None) or [])),
                )
                transport_phase["value"] = "dispatch"
                out = _orig(hidden_states, probs)
                transport_phase["value"] = "unknown"
                _timeline_record(
                    timeline,
                    rank=rank,
                    local_rank=local_rank,
                    layer=_name,
                    event="token_dispatch_exit",
                    out0_shape=list(out[0].shape) if isinstance(out, tuple) and isinstance(out[0], torch.Tensor) else None,
                    out1_shape=list(out[1].shape) if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], torch.Tensor) else None,
                )
                return out

            if orig_combine_pre is not None:
                def wrapped_combine_pre(hidden_states, _orig=orig_combine_pre, _name=name):
                    _timeline_record(timeline, rank=rank, local_rank=local_rank, layer=_name, event="expert_compute_boundary", hidden_shape=list(hidden_states.shape))
                    return _orig(hidden_states)
                dispatcher.combine_preprocess = wrapped_combine_pre

            def wrapped_combine(hidden_states, _orig=orig_combine, _dispatcher=dispatcher, _name=name):
                _timeline_record(
                    timeline,
                    rank=rank,
                    local_rank=local_rank,
                    layer=_name,
                    event="token_combine_enter",
                    hidden_shape=list(hidden_states.shape),
                    input_splits=list(getattr(_dispatcher, "input_splits", []).tolist() if hasattr(getattr(_dispatcher, "input_splits", None), "tolist") else (getattr(_dispatcher, "input_splits", None) or [])),
                    output_splits=list(getattr(_dispatcher, "output_splits", []).tolist() if hasattr(getattr(_dispatcher, "output_splits", None), "tolist") else (getattr(_dispatcher, "output_splits", None) or [])),
                )
                transport_phase["value"] = "combine"
                out = _orig(hidden_states)
                transport_phase["value"] = "unknown"
                _timeline_record(
                    timeline,
                    rank=rank,
                    local_rank=local_rank,
                    layer=_name,
                    event="token_combine_exit",
                    output_shape=list(out.shape) if isinstance(out, torch.Tensor) else None,
                )
                return out
            dispatcher.token_dispatch = wrapped_dispatch
            dispatcher.token_combine = wrapped_combine
            dispatcher._routersense_probe_wrapped = True

        with torch.inference_mode():
            _ = model(tokens, position_ids, None)
        stage_barrier("probe_forward", ok=True, detail="completed")

        token_dispatcher_mod.all_to_all = original_all_to_all
        env_payload = summarize_rank_environment(rank, local_rank)
        env_payload.update(
            {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "model": args.model,
                "precision": args.precision,
                "dispatcher": args.dispatcher,
                "timeline_rows": len(timeline),
                "dispatcher_capabilities": dispatcher_capabilities,
            }
        )
        gathered = gather_rank_payloads(env_payload)
        write_jsonl(run_dir / f"{args.run_id}-rank{rank}-timeline.jsonl", timeline)
        if rank == 0:
            has_safe_boundary = any(row.get("event") == "dispatch_preprocess_after" and row.get("has_safe_pre_transport_boundary") for row in timeline)
            status_payload = {
                "status": "ready" if has_safe_boundary else "blocked_host_api",
                "reason": None if has_safe_boundary else "no_safe_pre_transport_boundary",
                "backend": args.backend,
                "ep_size": args.ep_size,
                "dispatcher": args.dispatcher,
                "model": args.model,
                "rank_summaries": gathered,
                "has_safe_pre_transport_boundary": has_safe_boundary,
            }
            source_fp = {
                "token_dispatcher_module": dispatcher_capabilities,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_available": dist.is_nccl_available(),
            }
            write_json(run_dir / "dispatcher_capabilities.json", status_payload)
            write_jsonl(run_dir / "dispatcher_call_timeline.jsonl", timeline)
            write_json(run_dir / "source_fingerprint.json", source_fp)
        return 0
    except Exception as exc:
        write_json(
            run_dir / f"{args.run_id}-rank{rank}-error.json",
            {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    finally:
        destroy_distributed()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    raise SystemExit(main())
