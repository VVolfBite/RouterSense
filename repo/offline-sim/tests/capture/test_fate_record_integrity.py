from __future__ import annotations

import json
from pathlib import Path

import pytest

from rs_sim.trace.collection.fate_artifacts import canonical_fate_record_digest
from rs_sim.trace.collection.fixture_builder import _captured_fate_predictions
from rs_sim.trace.schema.model import TraceValidationError


def _row(source_rank: int, *, request_id: str = "request-0") -> dict:
    row = {
        "schema_version": "RS_SIM_CAPTURE_FATE_P2_ROW",
        "capture_id": "capture-0",
        "request_id": request_id,
        "sample_id": "sample-0",
        "decode_step": 0,
        "model_id": "model-0",
        "source_rank": source_rank,
        "world_size": 2,
        "source_layer_id": 0,
        "target_layer_id": 1,
        "sample_token_count": 2,
        "original_token_count": 4,
        "top_k": 1,
        "num_experts": 2,
        "routing_rows_by_destination": [3, 1] if source_rank == 0 else [1, 3],
        "sampling_method": "DETERMINISTIC_EVEN_MIDPOINT_V1",
        "estimator_kind": "SAMPLED_HARD_TOPK_RANK_COUNTS_V1",
        "predictor_id": "fate-test",
        "confidence_ppm": 900000,
    }
    row["record_digest"] = canonical_fate_record_digest(row)
    return row


def _write(raw_dir: Path, rows: list[dict]) -> None:
    raw_dir.mkdir(parents=True)
    path = raw_dir / "rank_fate_p2_rows.jsonl"
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_captured_fate_rows_require_valid_digest_and_identity(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write(raw, [_row(0), _row(1)])
    predictions, digest = _captured_fate_predictions(raw)
    assert ("sample-0", 0, 1) in predictions
    assert digest.startswith("sha256:")


def test_captured_fate_bad_digest_fails_closed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    rows = [_row(0), _row(1)]
    rows[1]["record_digest"] = "sha256:" + "0" * 64
    _write(raw, rows)
    with pytest.raises(TraceValidationError, match="record_digest mismatch"):
        _captured_fate_predictions(raw)


def test_captured_fate_mixed_request_identity_fails_closed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write(raw, [_row(0), _row(1, request_id="request-other")])
    with pytest.raises(TraceValidationError, match="differs across ranks"):
        _captured_fate_predictions(raw)
