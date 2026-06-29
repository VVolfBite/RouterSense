from __future__ import annotations

import math

import numpy as np
import pytest

from routesense_poc1.metrics import next_token_nll


def test_nll_basic() -> None:
    logits = np.array([[[0.0, 0.0], [2.0, 0.0]]], dtype=np.float32)
    input_ids = np.array([[0, 0]], dtype=np.int64)
    assert next_token_nll(logits, input_ids) > 0.0


def test_nll_perfect_prediction() -> None:
    logits = np.array([[[0.0, 0.0, 20.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
    input_ids = np.array([[1, 2]], dtype=np.int64)
    assert next_token_nll(logits, input_ids) < 1e-6


def test_nll_uniform_prediction() -> None:
    vocab_size = 5
    logits = np.zeros((1, 2, vocab_size), dtype=np.float32)
    input_ids = np.array([[1, 3]], dtype=np.int64)
    assert math.isclose(next_token_nll(logits, input_ids), math.log(vocab_size), rel_tol=1e-5)


def test_nll_batch_mismatch() -> None:
    with pytest.raises(ValueError):
        next_token_nll(np.zeros((1, 2, 4)), np.zeros((2, 2), dtype=np.int64))


def test_nll_short_sequence() -> None:
    with pytest.raises(ValueError):
        next_token_nll(np.zeros((1, 1, 4)), np.zeros((1, 1), dtype=np.int64))
