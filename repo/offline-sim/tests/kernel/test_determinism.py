from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tests.support.kernel_fixture import run_kernel_determinism_fixture


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "kernel_determinism_fixture.json"
)


def normalized_result():
    return asdict(run_kernel_determinism_fixture())


def test_fixture_is_identical_across_100_runs() -> None:
    results = [normalized_result() for _ in range(100)]
    encoded = [json.dumps(result, sort_keys=True) for result in results]
    assert len(set(encoded)) == 1


def test_fixture_matches_frozen_digest_file() -> None:
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    actual = json.loads(json.dumps(normalized_result()))
    assert actual == expected
