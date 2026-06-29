from __future__ import annotations

import json
import time
from pathlib import Path

from .intervention import RouteAblationContext
from .metrics import next_token_nll
from .schemas import AblationRecord, MoELayerSpec, Window
from .trace import collect_routing_context


def _load_completed_pairs(checkpoint_path: Path) -> set[tuple[int, int, int]]:
    if not checkpoint_path.exists():
        return set()
    completed: set[tuple[int, int, int]] = set()
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add((row["window_id"], row["layer_id"], row["topk_rank"]))
    return completed


def _append_checkpoint(checkpoint_path: Path, record: AblationRecord) -> None:
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


def run_ablation(
    config,
    model,
    tokenizer,
    windows: list[Window],
    moe_layers: list[MoELayerSpec],
    output_dir: Path,
    resume: bool = False,
) -> list[AblationRecord]:
    checkpoint_path = output_dir / "ablation_checkpoint.jsonl"
    completed = _load_completed_pairs(checkpoint_path) if resume else set()
    records: list[AblationRecord] = []
    total_windows = len(windows)
    started = time.perf_counter()

    import torch  # type: ignore

    for window_index, window in enumerate(windows, start=1):
        for moe_layer in moe_layers:
            contexts = collect_routing_context(model, tokenizer, window, moe_layer, config.run_id)
            input_ids = torch.tensor([window.input_ids], device=model.device)
            with torch.inference_mode():
                baseline_outputs = model(input_ids=input_ids, use_cache=False, output_router_logits=True)
            baseline_nll = next_token_nll(baseline_outputs.logits, input_ids)
            for context in contexts:
                key = (context.window_id, context.layer_id, context.topk_rank)
                if key in completed:
                    continue
                with RouteAblationContext(
                    model=model,
                    layer_path=moe_layer.module_path,
                    token_pos=window.target_pos,
                    expert_rank=context.topk_rank,
                    expected_expert_id=context.expert_id,
                ):
                    with torch.inference_mode():
                        ablated_outputs = model(input_ids=input_ids, use_cache=False, output_router_logits=True)
                ablated_nll = next_token_nll(ablated_outputs.logits, input_ids)
                record = AblationRecord(
                    run_id=config.run_id,
                    document_id=context.document_id,
                    window_id=context.window_id,
                    layer_id=context.layer_id,
                    layer_path=context.layer_path,
                    token_pos=context.token_pos,
                    expert_id=context.expert_id,
                    topk_rank=context.topk_rank,
                    router_logit=context.router_logit,
                    router_probability=context.router_probability,
                    effective_gate_weight=context.effective_gate_weight,
                    top1_top2_gap=context.top1_top2_gap,
                    routing_entropy=context.routing_entropy,
                    baseline_nll=baseline_nll,
                    ablated_nll=ablated_nll,
                    delta_nll=ablated_nll - baseline_nll,
                )
                records.append(record)
                _append_checkpoint(checkpoint_path, record)
        elapsed = time.perf_counter() - started
        avg = elapsed / max(window_index, 1)
        eta = avg * max(total_windows - window_index, 0)
        print(f"[ablation] {window_index}/{total_windows} avg={avg:.2f}s eta={eta:.2f}s", flush=True)
    return records

