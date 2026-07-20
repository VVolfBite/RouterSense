from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_no_object_collective_in_prediction_and_prepared_plan_paths() -> None:
    paths = [
        ROOT / "src/rs/runtime/online/megatron_ep/control/p2_matrix.py",
        ROOT / "src/rs/runtime/online/megatron_ep/prediction/contracts.py",
        ROOT / "src/rs/runtime/online/megatron_ep/prediction/simple_predictors.py",
        ROOT / "src/rs/runtime/online/megatron_ep/prediction/audit.py",
        ROOT / "src/rs/runtime/online/megatron_ep/control/p2_provider.py",
        ROOT / "src/rs/runtime/online/megatron_ep/lifecycle.py",
        *sorted((ROOT / "src/rs/runtime/online/megatron_ep/lifecycle_parts").glob("*.py")),
    ]
    forbidden = ("all_gather_object", "gather_object", "broadcast_object_list")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{needle} should not appear in {path}"
