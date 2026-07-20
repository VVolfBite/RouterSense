from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _read_lifecycle() -> str:
    root = ROOT / "src/rs/runtime/online/megatron_ep"
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in [root / "lifecycle.py", *sorted((root / "lifecycle_parts").glob("*.py"))]
    )


def test_object_collectives_are_absent_from_runtime_hot_paths() -> None:
    hot_paths = [
        "src/rs/runtime/online/megatron_ep/lifecycle.py",
        *[str(path.relative_to(ROOT)) for path in sorted((ROOT / "src/rs/runtime/online/megatron_ep/lifecycle_parts").glob("*.py"))],
        "src/rs/runtime/online/megatron_ep/async_release/agreement.py",
        "src/rs/runtime/online/megatron_ep/async_release/executor.py",
        "src/rs/runtime/online/megatron_ep/prediction/expert_trace_capture.py",
        "src/rs/runtime/online/megatron_ep/control/p2_matrix.py",
    ]
    forbidden = ("all_gather_object", "broadcast_object_list", "gather_object", "pickle")
    for relpath in hot_paths:
        text = _read(relpath)
        for needle in forbidden:
            assert needle not in text, f"{needle} must not appear in hot path {relpath}"


def test_known_object_collective_exceptions_are_confined_to_host_preflight() -> None:
    host_text = _read("src/rs/runtime/online/megatron_ep/host.py")
    plan_agreement_text = _read("src/rs/runtime/online/megatron_ep/control/plan_agreement.py")
    assert "all_gather_object" in host_text
    assert "stage_barrier" in host_text
    assert "gather_rank_payloads" in host_text
    assert "broadcast_object_list" in plan_agreement_text


def test_expert_trace_capture_is_default_off_and_compact() -> None:
    contracts_text = _read("src/rs/runtime/online/megatron_ep/contracts.py")
    capture_text = _read("src/rs/runtime/online/megatron_ep/prediction/expert_trace_capture.py")
    lifecycle_text = _read_lifecycle()
    assert "capture_expert_trace: bool = False" in contracts_text
    assert "if not enabled or recorder is None:\n        return" in capture_text
    assert '"heavy_debug_trace": False' in capture_text
    assert '"expert_ids"' not in capture_text
    assert '"routing_weights"' not in capture_text
    assert 'bool(getattr(self.config, "capture_expert_trace", False))' in lifecycle_text


def test_async_release_real_collectives_require_explicit_flags() -> None:
    text = _read("src/rs/runtime/online/megatron_ep/async_release/executor.py")
    assert "allow_real_collectives" in text
    assert "dry_run" in text
    assert "global_order_agreement_passed" in text
    assert "fallback_to_phase_sync" in text

