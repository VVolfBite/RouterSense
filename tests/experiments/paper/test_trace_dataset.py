from __future__ import annotations

import json

from experiments.paper.contracts import RecordMetadata
from experiments.paper.trace_dataset import load_replay_fixture, replay_fixture_to_trace_sample


def test_replay_fixture_to_trace_sample_is_deterministic(tmp_path) -> None:
    path = tmp_path / "replay_layer_1.json"
    path.write_text(
        json.dumps(
            {
                "num_gpus": 2,
                "p0_dispatch_matrix": [[0, 3], [2, 0]],
                "p1_return_matrix": [[0, 2], [3, 0]],
                "p2_next_dispatch_matrix": [[0, 4], [1, 0]],
                "metadata": {"layer_id": 1},
            }
        ),
        encoding="utf-8",
    )
    fixture = load_replay_fixture(path)
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    a = replay_fixture_to_trace_sample(fixture, model_id="m", model_revision="rev", metadata=metadata)
    b = replay_fixture_to_trace_sample(fixture, model_id="m", model_revision="rev", metadata=metadata)
    assert a.to_dict() == b.to_dict()
