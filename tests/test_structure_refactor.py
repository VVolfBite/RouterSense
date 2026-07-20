from pathlib import Path


def test_formal_mainline_contains_converged_components() -> None:
    root = Path(__file__).resolve().parents[1]
    required = (
        root / "src/rs/core/contracts/performance.py",
        root / "src/rs/evaluation/window_metrics.py",
        root / "src/rs/planning/registry.py",
        root / "src/rs/scheduling/catalog.py",
        root / "src/rs/scheduling/p012_future/_kernel/families.py",
        root / "src/rs/scheduling/runtime_bridge/prepared_priority.py",
        root / "src/rs/reference/baselines/fast_bvn_fixed.py",
        root / "src/rs/reference/baselines/aurora_fixed.py",
        root / "src/rs/reference/baselines/islip_round_robin.py",
    )
    for path in required:
        assert path.exists(), path


def test_retired_algorithm_sources_are_outside_installable_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src/rs/runtime/online/megatron_ep/pending_window").exists()
    assert not (root / "src/rs/planning/legacy_aliases.py").exists()
    archive_policy = (root / "archive/README.md").read_text(encoding="utf-8")
    assert "Round 1 removed-source snapshot is intentionally excluded" in archive_policy


def test_reference_baselines_are_not_in_deployable_phase_local_directory() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("aurora_fixed.py", "fast_bvn_fixed.py", "islip_round_robin.py"):
        assert not (root / "src/rs/scheduling/phase_local" / name).exists()
        assert (root / "src/rs/reference/baselines" / name).exists()
