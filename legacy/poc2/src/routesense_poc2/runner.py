from __future__ import annotations

import json
import queue
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
POC1_SRC = ROOT / "src"
if str(POC1_SRC) not in sys.path:
    sys.path.insert(0, str(POC1_SRC))

from routesense_poc1.model_inspector import inspect_model, select_middle_moe_layer
from routesense_poc1.serialization import save_json

from .eval import build_eval_summary
from .scheduler import BucketRecord, RuntimeState, SchedulerInput, create_scheduler


SUPPORTED_MODELS = {
    "olmoe": {
        "hf_id": "allenai/OLMoE-1B-7B-0924",
        "local_dir": str((ROOT.parent / "models" / "OLMoE-1B-7B-0924").resolve()),
    },
    "qwen_moe": {
        "hf_id": "Qwen/Qwen1.5-MoE-A2.7B",
        "local_dir": str((ROOT.parent / "models" / "Qwen1.5-MoE-A2.7B").resolve()),
    },
}

SERVICE_BACKENDS = {
    "mock-sleep",
    "synthetic-matmul",
    "synthetic-token-linear",
    "transfer-aware",
}


@dataclass
class RunnerConfig:
    model_key: str = "olmoe"
    model_path: str | None = None
    output_dir: str = str(ROOT / "outputs" / "poc2_single_node_eval")
    prompts_path: str | None = None
    layer: str = "auto"
    microbatch_size: int = 4
    microbatch_count: int = 3
    max_length: int = 96
    precision: str = "bf16"
    world_size: int = 4
    top_k: int | None = None
    seed: int = 42
    mock_routing: bool = False
    service_backend: str = "synthetic-matmul"
    service_scale: int = 64
    per_token_compute_us: int = 600
    strategies: list[str] = field(default_factory=lambda: ["fifo", "state-only", "dependency-only", "full"])

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["routing_mode"] = _routing_mode(self)
        payload["routing_is_real"] = not self.mock_routing
        payload["service_mode"] = _service_mode(self)
        payload["service_is_synthetic"] = True
        payload["backend_realism_level"] = _backend_realism_level(self.service_backend)
        payload["service_backend_description"] = _service_backend_description(self.service_backend)
        return payload


def run_single_node_experiment(config: RunnerConfig) -> dict[str, Any]:
    if config.service_backend not in SERVICE_BACKENDS:
        raise ValueError(f"unsupported service backend: {config.service_backend}")

    random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_goal(output_dir, config)
    save_json(config.to_dict(), output_dir / "config.json")

    prompts = _load_prompts(config.prompts_path, config.microbatch_size * config.microbatch_count)
    microbatches = [
        prompts[index : index + config.microbatch_size]
        for index in range(0, config.microbatch_size * config.microbatch_count, config.microbatch_size)
    ]

    model_bundle = None
    if not config.mock_routing:
        model_bundle = _load_model_bundle(config)
        save_json(model_bundle["model_info"], output_dir / "model_info.json")

    strategy_runs: list[dict[str, Any]] = []
    baseline_release_orders: dict[int, list[str]] = {}
    run_metadata = _run_metadata(config)
    for strategy in config.strategies:
        runtime_state = RuntimeState()
        scheduler = create_scheduler(strategy)
        run_payload = {
            "strategy": strategy,
            "routing_mode": run_metadata["routing_mode"],
            "routing_is_real": run_metadata["routing_is_real"],
            "service_mode": run_metadata["service_mode"],
            "service_is_synthetic": run_metadata["service_is_synthetic"],
            "backend_realism_level": run_metadata["backend_realism_level"],
            "service_backend": config.service_backend,
            "service_backend_description": run_metadata["service_backend_description"],
            "microbatches": [],
        }
        for microbatch_id, batch_prompts in enumerate(microbatches):
            routing = _collect_microbatch_routing(config, batch_prompts, microbatch_id, model_bundle)
            scheduler_input = SchedulerInput(
                strategy=strategy,
                microbatch_id=microbatch_id,
                layer_id=routing["layer_id"],
                layer_path=routing["layer_path"],
                bucket_records=routing["bucket_records"],
                runtime_state=runtime_state,
                metadata={
                    "prompts": batch_prompts,
                    "sample_count": len(batch_prompts),
                    "active_destinations": routing["active_destinations"],
                },
            )
            decision = scheduler.build_decision(scheduler_input)
            execution = _execute_microbatch(config, decision, routing["bucket_map"])
            _update_runtime_state(runtime_state, execution)
            execution["scheduler_decision"] = decision.to_dict()
            execution["routing_summary"] = routing["routing_summary"]
            execution["routing_mode"] = run_metadata["routing_mode"]
            execution["routing_is_real"] = run_metadata["routing_is_real"]
            execution["service_mode"] = run_metadata["service_mode"]
            execution["service_is_synthetic"] = run_metadata["service_is_synthetic"]
            execution["backend_realism_level"] = run_metadata["backend_realism_level"]
            execution["service_backend"] = config.service_backend
            execution["service_backend_description"] = run_metadata["service_backend_description"]
            if strategy == "fifo":
                baseline_release_orders[microbatch_id] = list(decision.release_order)
                execution["release_order_delta_vs_fifo"] = 0.0
            else:
                execution["release_order_delta_vs_fifo"] = _release_order_delta(
                    baseline_release_orders.get(microbatch_id, []), decision.release_order
                )
            run_payload["microbatches"].append(execution)
        strategy_runs.append(run_payload)
        save_json(run_payload, output_dir / f"{strategy.replace('-', '_')}_run.json")

    eval_summary = build_eval_summary(strategy_runs)
    payload = {
        "routing_mode": run_metadata["routing_mode"],
        "routing_is_real": run_metadata["routing_is_real"],
        "service_mode": run_metadata["service_mode"],
        "service_is_synthetic": run_metadata["service_is_synthetic"],
        "backend_realism_level": run_metadata["backend_realism_level"],
        "service_backend": config.service_backend,
        "service_backend_description": run_metadata["service_backend_description"],
        "summary_note": (
            "Routing is real only when routing_mode=real_router_logits. "
            "Service remains synthetic in this POC2 stage and should not be interpreted as final EP runtime latency."
        ),
        "runs": strategy_runs,
        "summary": eval_summary,
    }
    save_json(payload, output_dir / "summary.json")
    return payload


def _load_prompts(prompts_path: str | None, required: int) -> list[str]:
    if prompts_path:
        path = Path(prompts_path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                prompts = [str(item) for item in payload]
            else:
                prompts = [str(item) for item in payload.get("prompts", [])]
        else:
            prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(prompts) >= required:
            return prompts[:required]
    seed_prompts = [
        "Explain how sparse expert routing changes the compute path of a token.",
        "Summarize why dependency-aware scheduling may differ from state-only scheduling.",
        "Describe a bottleneck that appears when many experts map to one destination rank.",
        "What makes a bucket critical if it is not the largest bucket in the batch?",
        "How can co-active experts expose future join pressure before computation starts?",
        "Write a short note on why queue size is not the same as dependency structure.",
        "List two reasons a hot bucket might still be a poor scheduling priority target.",
        "Discuss how expert parallel imbalance can emerge in a single-node setup.",
        "Why can bridge destinations delay global completion even with modest token counts?",
        "Give a concise example of routing-derived structure changing service order.",
        "How should a lightweight MoE runtime collect just enough state for scheduling?",
        "Describe how per-rank idle time can reveal a bad bucket release order.",
    ]
    while len(seed_prompts) < required:
        seed_prompts.extend(seed_prompts)
    return seed_prompts[:required]


def _load_model_bundle(config: RunnerConfig) -> dict[str, Any]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("POC2 real routing requires torch and transformers") from exc

    model_spec = SUPPORTED_MODELS[config.model_key]
    model_path = config.model_path or model_spec["local_dir"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("real POC2 execution requires CUDA; use --mock-routing during no-GPU development")

    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    world_size = min(config.world_size, torch.cuda.device_count())
    max_memory = {idx: "20GiB" for idx in range(world_size)}
    offload_folder = Path(config.output_dir) / ".hf_offload"
    offload_folder.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map.get(config.precision, torch.bfloat16),
        device_map="balanced",
        max_memory=max_memory,
        offload_folder=str(offload_folder),
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "config"):
        model.config.output_router_logits = True
    layers = inspect_model(model)
    selected = select_middle_moe_layer(layers)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_info": {
            "model_key": config.model_key,
            "model_path": model_path,
            "hf_id": model_spec["hf_id"],
            "cuda_device_count": torch.cuda.device_count(),
            "selected_layer_path": selected.layer_path if selected else None,
            "selected_layer_class": selected.module_class if selected else None,
            "candidate_layer_count": len(layers),
            "routing_mode": _routing_mode(config),
            "service_mode": _service_mode(config),
            "backend_realism_level": _backend_realism_level(config.service_backend),
            "service_backend": config.service_backend,
        },
    }


def _collect_microbatch_routing(
    config: RunnerConfig,
    prompts: list[str],
    microbatch_id: int,
    model_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    if config.mock_routing:
        return _mock_microbatch_routing(config, prompts, microbatch_id)
    return _real_microbatch_routing(config, prompts, microbatch_id, model_bundle)


def _mock_microbatch_routing(config: RunnerConfig, prompts: list[str], microbatch_id: int) -> dict[str, Any]:
    rng = random.Random(config.seed + microbatch_id)
    num_experts = 64 if config.model_key == "olmoe" else 60
    top_k = config.top_k or (8 if config.model_key == "olmoe" else 4)
    token_routes: list[list[int]] = []
    for prompt_index, prompt in enumerate(prompts):
        token_budget = max(6, min(18, len(prompt.split()) + 4))
        for token_pos in range(token_budget):
            width = max(2, min(top_k, 4 + (token_pos + prompt_index) % max(2, top_k)))
            base = rng.randint(0, num_experts - 1)
            experts = sorted({(base + rng.randint(0, 9) + shift * 7) % num_experts for shift in range(width)})
            token_routes.append(experts[:top_k])
    return _build_routing_artifacts(
        model_key=config.model_key,
        microbatch_id=microbatch_id,
        layer_id=0,
        layer_path="mock.layers.0.moe",
        token_routes=token_routes,
        token_ids=list(range(len(token_routes))),
        world_size=config.world_size,
    )


def _real_microbatch_routing(
    config: RunnerConfig,
    prompts: list[str],
    microbatch_id: int,
    model_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    if model_bundle is None:
        raise RuntimeError("model bundle is missing")
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("real routing requires torch") from exc

    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_length,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True, return_dict=True, use_cache=False)
    router_logits_by_layer = getattr(outputs, "router_logits", None)
    if not router_logits_by_layer:
        raise RuntimeError("model forward did not return router_logits")
    layer_id = _resolve_layer_id(config.layer, len(router_logits_by_layer))
    layer_router_logits = router_logits_by_layer[layer_id].detach().float().cpu()
    input_ids = inputs["input_ids"].detach().cpu()
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.detach().cpu()
    batch_size, sequence_length = input_ids.shape
    expert_count = layer_router_logits.shape[-1]
    if layer_router_logits.shape[0] != batch_size * sequence_length:
        raise RuntimeError(
            f"unexpected router logits shape {tuple(layer_router_logits.shape)} for batch={batch_size} seq={sequence_length}"
        )
    layer_router_logits = layer_router_logits.view(batch_size, sequence_length, expert_count)
    top_k = int(config.top_k or getattr(model.config, "num_experts_per_tok", 0) or 2)
    top_k = max(1, min(top_k, expert_count))
    token_routes: list[list[int]] = []
    token_ids: list[int] = []
    token_values: list[int] = []
    token_owner_batch_positions: list[tuple[int, int]] = []
    for batch_index in range(batch_size):
        valid_tokens = int(attention_mask[batch_index].sum().item())
        for token_pos in range(max(1, valid_tokens - 1)):
            probabilities = torch.softmax(layer_router_logits[batch_index, token_pos], dim=-1)
            experts = torch.topk(probabilities, top_k).indices.tolist()
            token_routes.append([int(item) for item in experts])
            token_ids.append(len(token_ids))
            token_values.append(int(input_ids[batch_index, token_pos].item()))
            token_owner_batch_positions.append((batch_index, token_pos))
    return _build_routing_artifacts(
        model_key=config.model_key,
        microbatch_id=microbatch_id,
        layer_id=layer_id,
        layer_path=_default_layer_path(layer_id),
        token_routes=token_routes,
        token_ids=token_ids,
        token_values=token_values,
        token_owner_batch_positions=token_owner_batch_positions,
        world_size=config.world_size,
    )


def _build_routing_artifacts(
    model_key: str,
    microbatch_id: int,
    layer_id: int,
    layer_path: str,
    token_routes: list[list[int]],
    world_size: int,
    token_ids: list[int] | None = None,
    token_values: list[int] | None = None,
    token_owner_batch_positions: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    normalized_token_ids = token_ids or list(range(len(token_routes)))
    if len(normalized_token_ids) != len(token_routes):
        raise RuntimeError("token_ids length does not match token_routes length")
    normalized_token_values = token_values or normalized_token_ids
    if len(normalized_token_values) != len(token_routes):
        raise RuntimeError("token_values length does not match token_routes length")
    normalized_positions = token_owner_batch_positions or [(0, index) for index in range(len(token_routes))]
    if len(normalized_positions) != len(token_routes):
        raise RuntimeError("token_owner_batch_positions length does not match token_routes length")

    bucket_positions: dict[int, list[int]] = {}
    coactive_destinations: dict[int, set[int]] = {}
    coactive_event_counts: dict[int, int] = {}
    route_items: list[dict[str, Any]] = []
    active_destinations = sorted({int(dest) for row in token_routes for dest in row})
    max_width = max((len(row) for row in token_routes), default=1)
    total_routes = sum(len(row) for row in token_routes)
    for token_pos, row in enumerate(token_routes):
        peers = {int(dest) for dest in row}
        for destination_id in peers:
            bucket_positions.setdefault(destination_id, []).append(token_pos)
            coactive_destinations.setdefault(destination_id, set()).update(peers - {destination_id})
            coactive_event_counts[destination_id] = coactive_event_counts.get(destination_id, 0) + len(peers - {destination_id})
        token_id = int(normalized_token_ids[token_pos])
        token_value = int(normalized_token_values[token_pos])
        batch_index, prompt_token_pos = normalized_positions[token_pos]
        for route_rank, destination_id in enumerate(row):
            route_items.append(
                {
                    "route_id": f"mb{microbatch_id}_tok{token_id}_r{route_rank}_e{int(destination_id)}",
                    "token_id": token_id,
                    "token_value": token_value,
                    "token_pos": token_pos,
                    "batch_index": int(batch_index),
                    "prompt_token_pos": int(prompt_token_pos),
                    "route_rank": int(route_rank),
                    "expert_id": int(destination_id),
                    "destination_id": int(destination_id),
                    "layer_id": int(layer_id),
                    "layer_path": layer_path,
                    "microbatch_id": int(microbatch_id),
                }
            )

    bucket_sizes = {destination_id: len(positions) for destination_id, positions in bucket_positions.items()}
    max_bucket_size = max(bucket_sizes.values(), default=1)
    avg_bucket_size = total_routes / max(1, len(bucket_positions))
    ranked_destinations = sorted(bucket_sizes, key=lambda destination_id: (-bucket_sizes[destination_id], destination_id))
    bucket_records: list[BucketRecord] = []
    bucket_map: dict[str, dict[str, Any]] = {}
    for arrival_index, destination_id in enumerate(ranked_destinations):
        token_positions = bucket_positions[destination_id]
        source_coverage = len(token_positions) / max(1, len(token_routes))
        peer_degree = len(coactive_destinations.get(destination_id, set())) / max(1, len(active_destinations) - 1)
        coactive_density = coactive_event_counts.get(destination_id, 0) / max(1, len(token_routes) * max(0, max_width - 1))
        spread = (
            (max(token_positions) - min(token_positions)) / max(1, len(token_routes) - 1)
            if len(token_positions) >= 2
            else 0.0
        )
        bridge = source_coverage * peer_degree
        token_count = bucket_sizes[destination_id]
        route_share = token_count / max(1, total_routes)
        density_over_mean = min(1.0, token_count / max(1.0, avg_bucket_size) / 3.0)
        size_norm = token_count / max(1, max_bucket_size)
        inverse_rank = 1.0 - (arrival_index / max(1, len(ranked_destinations) - 1)) if ranked_destinations else 0.0
        destination_rank = int(destination_id) % max(1, world_size)
        estimated_service_units = max(1.0, token_count * (1.0 + 0.35 * bridge + 0.15 * coactive_density))
        bucket_id = f"mb{microbatch_id}_dest{destination_id}"
        record = BucketRecord(
            bucket_id=bucket_id,
            destination_id=int(destination_id),
            destination_rank=destination_rank,
            arrival_index=arrival_index,
            token_count=token_count,
            token_positions=token_positions,
            source_count=len(token_routes),
            active_destinations=active_destinations,
            coactive_destinations=sorted(coactive_destinations.get(destination_id, set())),
            source_coverage=source_coverage,
            coactive_peer_degree=peer_degree,
            coactive_event_density=coactive_density,
            position_spread=spread,
            bridge_score=bridge,
            route_share=route_share,
            density_over_mean=density_over_mean,
            size_norm=size_norm,
            inverse_size_rank_norm=inverse_rank,
            is_hot_bucket=token_count > 1,
            estimated_service_units=estimated_service_units,
        )
        bucket_records.append(record)
        bucket_map[bucket_id] = record.to_dict()

    routing_summary = {
        "model_key": model_key,
        "microbatch_id": microbatch_id,
        "layer_id": layer_id,
        "layer_path": layer_path,
        "source_count": len(token_routes),
        "active_destinations": active_destinations,
        "total_selected_routes": total_routes,
        "source_destination_matrix": token_routes,
        "token_ids": normalized_token_ids,
        "token_values": normalized_token_values,
        "token_owner_batch_positions": [[int(a), int(b)] for a, b in normalized_positions],
        "route_items": route_items,
        "bucket_count": len(bucket_records),
        "buckets": [record.to_dict() for record in bucket_records],
    }
    return {
        "layer_id": layer_id,
        "layer_path": layer_path,
        "active_destinations": active_destinations,
        "bucket_records": bucket_records,
        "bucket_map": bucket_map,
        "route_items": route_items,
        "routing_summary": routing_summary,
    }


def _execute_microbatch(config: RunnerConfig, decision: Any, bucket_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    rank_queues = {rank: queue.Queue() for rank in range(config.world_size)}
    workers = []
    for rank in range(config.world_size):
        worker = threading.Thread(
            target=_rank_worker,
            args=(config, rank, rank_queues[rank], result_queue),
            daemon=True,
        )
        worker.start()
        workers.append(worker)

    start_time = time.perf_counter()
    release_events = []
    for bucket_id in decision.release_order:
        bucket = bucket_map[bucket_id]
        rank = int(bucket["destination_rank"])
        rank_queues[rank].put((bucket_id, bucket))
        release_events.append(
            {
                "bucket_id": bucket_id,
                "destination_id": bucket["destination_id"],
                "destination_rank": rank,
                "release_ts": time.perf_counter() - start_time,
            }
        )
    for rank in range(config.world_size):
        rank_queues[rank].put(None)

    bucket_results = []
    for _ in decision.release_order:
        bucket_results.append(result_queue.get())
    for worker in workers:
        worker.join()

    bucket_results.sort(key=lambda item: float(item["start_offset"]))
    per_rank = _summarize_per_rank(bucket_results)
    batch_completion = max((float(item["end_offset"]) for item in bucket_results), default=0.0)
    earliest_start = min((float(item["start_offset"]) for item in bucket_results), default=0.0)
    barrier_proxy = batch_completion - earliest_start
    critical = max(bucket_results, key=lambda item: float(item.get("estimated_service_units", 0.0)), default=None)
    return {
        "microbatch_id": decision.microbatch_id,
        "strategy": decision.strategy,
        "release_events": release_events,
        "bucket_results": bucket_results,
        "per_rank_summary": per_rank,
        "batch_completion_time": batch_completion,
        "barrier_proxy": barrier_proxy,
        "per_rank_idle_proxy_sum": sum(float(item["idle_time"]) for item in per_rank.values()),
        "critical_bucket_id": critical["bucket_id"] if critical else None,
        "critical_bucket_completion_time": float(critical["end_offset"]) if critical else 0.0,
        "critical_bucket_position": int(next((index for index, item in enumerate(bucket_results) if item["bucket_id"] == critical["bucket_id"]), -1))
        if critical
        else -1,
    }


def _rank_worker(
    config: RunnerConfig,
    rank: int,
    task_queue: queue.Queue[Any],
    result_queue: queue.Queue[dict[str, Any]],
) -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None
    device = None
    stream = None
    if torch is not None and torch.cuda.is_available() and rank < torch.cuda.device_count():
        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.Stream(device=device)
    worker_start = time.perf_counter()
    last_end = worker_start
    while True:
        item = task_queue.get()
        if item is None:
            break
        bucket_id, bucket = item
        start = time.perf_counter()
        idle_time = max(0.0, start - last_end)
        duration = _service_bucket(config, rank, bucket, device, stream)
        end = time.perf_counter()
        last_end = end
        result_queue.put(
            {
                "bucket_id": bucket_id,
                "destination_id": int(bucket["destination_id"]),
                "destination_rank": int(bucket["destination_rank"]),
                "start_offset": start - worker_start,
                "end_offset": end - worker_start,
                "service_time": max(duration, end - start),
                "idle_before_start": idle_time,
                "estimated_service_units": float(bucket["estimated_service_units"]),
                "token_count": int(bucket["token_count"]),
                "service_backend": config.service_backend,
                "backend_cost_hint": _backend_cost_hint(bucket),
            }
        )


def _service_bucket(config: RunnerConfig, rank: int, bucket: dict[str, Any], device: Any, stream: Any) -> float:
    if config.service_backend == "transfer-aware":
        return _service_backend_transfer_aware(config, bucket, device, stream)
    if config.service_backend == "mock-sleep":
        return _service_backend_mock_sleep(config, bucket)
    if config.service_backend == "synthetic-token-linear":
        return _service_backend_synthetic_token_linear(config, bucket, device, stream)
    if config.service_backend == "synthetic-matmul":
        return _service_backend_synthetic_matmul(config, bucket, device, stream)
    raise ValueError(f"unsupported service backend: {config.service_backend}")


def _service_backend_mock_sleep(config: RunnerConfig, bucket: dict[str, Any]) -> float:
    token_count = int(bucket["token_count"])
    units = float(bucket["estimated_service_units"])
    size_norm = float(bucket.get("size_norm", 0.0))
    sleep_seconds = max(
        0.0002,
        (
            config.per_token_compute_us * max(1.0, token_count + 0.5 * units + 2.0 * size_norm)
        )
        / 1_000_000.0,
    )
    start = time.perf_counter()
    time.sleep(sleep_seconds)
    return time.perf_counter() - start


def _service_backend_synthetic_matmul(config: RunnerConfig, bucket: dict[str, Any], device: Any, stream: Any) -> float:
    token_count = int(bucket["token_count"])
    units = float(bucket["estimated_service_units"])
    if device is None:
        return _service_backend_mock_sleep(config, bucket)

    import torch  # type: ignore

    hidden = int(config.service_scale)
    rounds = max(1, int(round(units + token_count * 0.25)))
    rows = max(1, token_count)
    start = time.perf_counter()
    with torch.cuda.device(device):
        left = torch.randn((rows, hidden), device=device, dtype=torch.float16)
        right = torch.randn((hidden, hidden), device=device, dtype=torch.float16)
        with torch.cuda.stream(stream):
            acc = left
            for _ in range(rounds):
                acc = torch.matmul(acc, right)
                acc = torch.relu(acc)
            torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        _ = float(acc.sum().item())
    return time.perf_counter() - start


def _service_backend_synthetic_token_linear(config: RunnerConfig, bucket: dict[str, Any], device: Any, stream: Any) -> float:
    token_count = int(bucket["token_count"])
    units = float(bucket["estimated_service_units"])
    route_share = float(bucket.get("route_share", 0.0))
    if device is None:
        return _service_backend_mock_sleep(config, bucket)

    import torch  # type: ignore

    width = int(config.service_scale)
    rows = max(1, token_count * max(1, int(round(1.0 + units * 0.2))))
    layers = max(1, int(round(1.0 + units * 0.35 + route_share * 8.0)))
    start = time.perf_counter()
    with torch.cuda.device(device):
        activations = torch.randn((rows, width), device=device, dtype=torch.float16)
        weights = [torch.randn((width, width), device=device, dtype=torch.float16) for _ in range(min(layers, 8))]
        with torch.cuda.stream(stream):
            acc = activations
            for weight in weights:
                acc = torch.nn.functional.linear(acc, weight)
                acc = torch.nn.functional.gelu(acc)
            torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        _ = float(acc.sum().item())
    return time.perf_counter() - start


def _service_backend_transfer_aware(config: RunnerConfig, bucket: dict[str, Any], device: Any, stream: Any) -> float:
    token_count = int(bucket["token_count"])
    units = float(bucket["estimated_service_units"])
    size_norm = float(bucket.get("size_norm", 0.0))
    if device is None:
        return _service_backend_mock_sleep(config, bucket)

    import torch  # type: ignore

    width = int(config.service_scale)
    rows = max(1, token_count * max(1, int(round(1.0 + size_norm * 4.0))))
    payload = torch.randn((rows, width), device="cpu", dtype=torch.float32, pin_memory=True)
    rounds = max(1, int(round(1.0 + units * 0.5)))
    start = time.perf_counter()
    with torch.cuda.device(device):
        with torch.cuda.stream(stream):
            device_payload = payload.to(device=device, dtype=torch.float16, non_blocking=True)
            weight = torch.randn((width, width), device=device, dtype=torch.float16)
            acc = device_payload
            for _ in range(min(rounds, 8)):
                acc = torch.matmul(acc, weight)
                acc = torch.relu(acc)
            joined = acc.mean(dim=0, keepdim=True).to("cpu", dtype=torch.float32, non_blocking=True)
            torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        _ = float(joined.sum().item())
    return time.perf_counter() - start


def _backend_cost_hint(bucket: dict[str, Any]) -> dict[str, float]:
    return {
        "token_count": float(bucket["token_count"]),
        "estimated_service_units": float(bucket["estimated_service_units"]),
        "size_norm": float(bucket.get("size_norm", 0.0)),
        "route_share": float(bucket.get("route_share", 0.0)),
        "bridge_score": float(bucket.get("bridge_score", 0.0)),
    }


def _summarize_per_rank(bucket_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_rank: dict[str, dict[str, Any]] = {}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in bucket_results:
        grouped.setdefault(int(item["destination_rank"]), []).append(item)
    for rank, items in grouped.items():
        items.sort(key=lambda entry: float(entry["start_offset"]))
        first_start = float(items[0]["start_offset"]) if items else 0.0
        last_end = float(items[-1]["end_offset"]) if items else 0.0
        busy = sum(float(entry["service_time"]) for entry in items)
        idle = max(0.0, last_end - first_start - busy)
        per_rank[str(rank)] = {
            "bucket_count": len(items),
            "service_order": [entry["bucket_id"] for entry in items],
            "busy_time": busy,
            "idle_time": idle,
            "completion_time": last_end,
        }
    return per_rank


def _update_runtime_state(state: RuntimeState, execution: dict[str, Any]) -> None:
    for bucket in execution.get("bucket_results", []):
        destination_id = int(bucket["destination_id"])
        rank = int(bucket["destination_rank"])
        service_time = float(bucket["service_time"])
        token_count = int(bucket["token_count"])
        state.destination_pending_work[destination_id] = max(
            0.0, float(state.destination_pending_work.get(destination_id, 0.0)) * 0.15
        )
        state.destination_queue_depth[destination_id] = 0
        state.rank_backlog_work[rank] = max(0.0, float(state.rank_backlog_work.get(rank, 0.0)) * 0.15)
        state.rank_queue_depth[rank] = 0
        state.destination_hotness[destination_id] = min(10.0, float(state.destination_hotness.get(destination_id, 0.0)) * 0.5 + token_count)
        state.destination_latency_history.setdefault(destination_id, []).append(service_time)
        state.destination_latency_history[destination_id] = state.destination_latency_history[destination_id][-6:]

    pending_work: dict[int, float] = {}
    destination_queue_depth: dict[int, int] = {}
    rank_backlog_work: dict[int, float] = {}
    rank_queue_depth: dict[int, int] = {}
    for release in execution.get("release_events", []):
        rank = int(release["destination_rank"])
        destination_id = int(release["destination_id"])
        bucket = next(item for item in execution["bucket_results"] if item["bucket_id"] == release["bucket_id"])
        pending_work[destination_id] = pending_work.get(destination_id, 0.0) + float(bucket["estimated_service_units"])
        destination_queue_depth[destination_id] = destination_queue_depth.get(destination_id, 0) + 1
        rank_backlog_work[rank] = rank_backlog_work.get(rank, 0.0) + float(bucket["estimated_service_units"])
        rank_queue_depth[rank] = rank_queue_depth.get(rank, 0) + 1
    state.destination_pending_work = pending_work
    state.destination_queue_depth = destination_queue_depth
    state.rank_backlog_work = rank_backlog_work
    state.rank_queue_depth = rank_queue_depth
    state.batch_index += 1


def _release_order_delta(baseline: list[str], other: list[str]) -> float:
    if not baseline or len(baseline) != len(other):
        return 0.0
    baseline_pos = {bucket_id: index for index, bucket_id in enumerate(baseline)}
    return float(sum(abs(index - baseline_pos.get(bucket_id, index)) for index, bucket_id in enumerate(other)))


def _resolve_layer_id(layer_spec: str, layer_count: int) -> int:
    if layer_spec in ("", "auto", None):
        return layer_count // 2
    if str(layer_spec).isdigit():
        layer_id = int(layer_spec)
    else:
        digits = "".join(char for char in str(layer_spec) if char.isdigit())
        layer_id = int(digits) if digits else layer_count // 2
    if layer_id < 0 or layer_id >= layer_count:
        raise ValueError(f"layer id out of range: {layer_id}")
    return layer_id


def _default_layer_path(layer_id: int) -> str:
    return f"model.layers.{layer_id}.mlp"


def _routing_mode(config: RunnerConfig) -> str:
    return "mock_routing" if config.mock_routing else "real_router_logits"


def _service_mode(config: RunnerConfig) -> str:
    return f"synthetic::{config.service_backend}"


def _service_backend_description(service_backend: str) -> str:
    if service_backend == "mock-sleep":
        return "CPU sleep backend; synthetic timing derived from bucket features."
    if service_backend == "synthetic-token-linear":
        return "GPU linear-stack backend; synthetic compute scales with token_count and estimated_service_units."
    if service_backend == "transfer-aware":
        return "GPU transfer-aware backend; bucket tensors are materialized on CPU, moved to destination GPU, processed, then joined back to CPU."
    return "GPU matmul backend; synthetic compute scales with token_count and estimated_service_units."


def _backend_realism_level(service_backend: str) -> str:
    if service_backend == "transfer-aware":
        return "transfer-aware"
    return "synthetic"


def _run_metadata(config: RunnerConfig) -> dict[str, Any]:
    return {
        "routing_mode": _routing_mode(config),
        "routing_is_real": not config.mock_routing,
        "service_mode": _service_mode(config),
        "service_is_synthetic": True,
        "backend_realism_level": _backend_realism_level(config.service_backend),
        "service_backend_description": _service_backend_description(config.service_backend),
    }


def _write_goal(output_dir: Path, config: RunnerConfig) -> None:
    text = (
        "# POC2 Single Node Scheduler Prototype\n\n"
        "This stage upgrades proxy analysis into a runnable single-node 4-GPU scheduler prototype.\n\n"
        f"Routing mode: `{_routing_mode(config)}`.\n"
        f"Service mode: `{_service_mode(config)}`.\n"
        "Current boundary: routing can be real, but service execution is still synthetic. "
        "This is not a final EP runtime and the reported latency-like metrics remain proxy metrics.\n"
    )
    (output_dir / "goal.md").write_text(text, encoding="utf-8")
