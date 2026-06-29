from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .schemas import RunConfig, Window


def _build_window_records(tokenized_documents: list[tuple[int, list[int]]], config: RunConfig) -> list[Window]:
    rng = random.Random(config.seed)
    windows: list[Window] = []
    window_length = config.context_length + 1
    candidates = list(tokenized_documents)
    rng.shuffle(candidates)
    for window_id, (document_id, input_ids) in enumerate(candidates[: config.num_windows]):
        start = 0 if len(input_ids) == window_length else rng.randint(0, len(input_ids) - window_length)
        window_tokens = input_ids[start : start + window_length]
        windows.append(
            Window(
                document_id=document_id,
                window_id=window_id,
                input_ids=window_tokens,
                target_pos=config.context_length - 1,
                ground_truth_token_id=window_tokens[-1],
            )
        )
    return windows


def _load_prompt_documents(prompt_file: str | Path) -> list[tuple[int, str]]:
    path = Path(prompt_file)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[tuple[int, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                text = str(payload.get("text", "")).strip()
                if text:
                    rows.append((int(payload.get("document_id", index)), text))
        return rows
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(index, text) for index, text in enumerate(texts)]


def load_and_prepare_data(config: RunConfig, tokenizer: Any) -> list[Window]:
    tokenized_documents: list[tuple[int, list[int]]] = []
    window_length = config.context_length + 1
    if config.prompt_file:
        rows = _load_prompt_documents(config.prompt_file)
        for document_id, text in rows:
            input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(input_ids) < window_length:
                continue
            tokenized_documents.append((document_id, input_ids))
    else:
        from datasets import load_dataset  # type: ignore

        dataset = load_dataset(config.dataset_id, config.dataset_config, split=config.dataset_split)
        for document_id, record in enumerate(dataset):
            text = (record.get("text") or "").strip()
            if not text:
                continue
            input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(input_ids) < window_length:
                continue
            tokenized_documents.append((document_id, input_ids))
    return _build_window_records(tokenized_documents, config)
