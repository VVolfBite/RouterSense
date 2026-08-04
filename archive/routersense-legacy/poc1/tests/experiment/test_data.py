from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from routesense_poc1.experiment.data import _build_window_records, _load_prompt_documents


def test_window_length() -> None:
    config = SimpleNamespace(context_length=3, num_windows=2, seed=42)
    windows = _build_window_records([(0, [1, 2, 3, 4, 5]), (1, [6, 7, 8, 9, 10])], config)
    assert all(len(window.input_ids) == 4 for window in windows)


def test_target_position() -> None:
    config = SimpleNamespace(context_length=3, num_windows=1, seed=42)
    windows = _build_window_records([(0, [1, 2, 3, 4, 5])], config)
    assert windows[0].target_pos == 2


def test_document_id_preserved() -> None:
    config = SimpleNamespace(context_length=3, num_windows=1, seed=42)
    windows = _build_window_records([(17, [1, 2, 3, 4, 5])], config)
    assert windows[0].document_id == 17


def test_seed_reproducibility() -> None:
    config = SimpleNamespace(context_length=3, num_windows=1, seed=42)
    left = _build_window_records([(0, [1, 2, 3, 4, 5, 6])], config)
    right = _build_window_records([(0, [1, 2, 3, 4, 5, 6])], config)
    assert [window.input_ids for window in left] == [window.input_ids for window in right]


def test_different_seeds_differ() -> None:
    left_config = SimpleNamespace(context_length=3, num_windows=1, seed=1)
    right_config = SimpleNamespace(context_length=3, num_windows=1, seed=2)
    left = _build_window_records([(0, [1, 2, 3, 4, 5, 6, 7])], left_config)
    right = _build_window_records([(0, [1, 2, 3, 4, 5, 6, 7])], right_config)
    assert [window.input_ids for window in left] != [window.input_ids for window in right]


def test_load_prompt_documents_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"document_id": 10, "text": "alpha beta gamma"},
        {"document_id": 11, "text": "delta epsilon zeta"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    loaded = _load_prompt_documents(path)
    assert loaded == [(10, "alpha beta gamma"), (11, "delta epsilon zeta")]
