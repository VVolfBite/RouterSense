from __future__ import annotations

import ast
from pathlib import Path


def _collect_imports(root: Path) -> list[tuple[Path, int, str]]:
    rows: list[tuple[Path, int, str]] = []
    for path in root.rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    rows.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                rows.append((path, node.lineno, module))
    return rows


def test_core_does_not_import_upper_layers() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/core")):
        if module.startswith(("rs.scheduling", "rs.runtime", "experiments", "integrations", "legacy")):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_runtime_does_not_import_experiments() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/runtime")):
        if module.startswith("experiments"):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_formal_src_does_not_import_integrations() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs")):
        if module.startswith("integrations.megatron_ep"):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_formal_src_does_not_import_old_rs_namespaces() -> None:
    bad = []
    forbidden = (
        "rs.scheduler",
        "rs.evaluation",
        "rs.trace",
        "rs.online",
        "rs.runtime.distributed_ep",
    )
    for path, lineno, module in _collect_imports(Path("src/rs")):
        if module.startswith(forbidden):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_scheduling_does_not_import_torch_or_runtime() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/scheduling")):
        if module.startswith(("torch", "megatron", "experiments", "integrations", "legacy", "rs.runtime")):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_prediction_package_does_not_import_runtime_or_experiments() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/prediction")):
        if module.startswith(("rs.runtime", "experiments")):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_planning_package_does_not_import_runtime_or_offline() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/planning")):
        if module.startswith(("rs.runtime", "rs.runtime.offline", "experiments")):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_planning_api_does_not_import_legacy_unified_interface() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/planning")):
        if path.name != "api.py":
            continue
        if module.startswith("rs.scheduling.unified_interface"):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_runtime_formal_modules_do_not_import_private_planning_bridge() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/runtime/online/megatron_ep")):
        if module.startswith("rs.planning.runtime_bridge"):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_runtime_does_not_import_legacy_prediction_implementations() -> None:
    bad = []
    forbidden = (
        "rs.runtime.online.megatron_ep.prediction.simple_predictors",
        "rs.runtime.online.megatron_ep.prediction.gate_replay_predictor",
    )
    for path, lineno, module in _collect_imports(Path("src/rs/runtime")):
        if module.startswith(forbidden):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_runtime_does_not_import_legacy_registry_catalog_modules() -> None:
    bad = []
    forbidden = (
        "rs.scheduling.unified_interface",
        "rs.scheduling.registry",
        "rs.scheduling.catalog",
        "rs.scheduling.algorithm_catalog",
        "rs.scheduling.public_catalog",
    )
    for path, lineno, module in _collect_imports(Path("src/rs/runtime")):
        if module.startswith(forbidden):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_formal_materialization_modules_do_not_import_legacy_phase_execution() -> None:
    bad = []
    forbidden = (
        "rs.scheduling.phase_execution",
        "rs.scheduling.unified_interface",
    )
    for path, lineno, module in _collect_imports(Path("src/rs/runtime/online/megatron_ep/materialization")):
        if module.startswith(forbidden):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def _lifecycle_sources() -> str:
    root = Path(__file__).resolve().parents[1] / "src/rs/runtime/online/megatron_ep"
    paths = [root / "lifecycle.py", *sorted((root / "lifecycle_parts").glob("*.py"))]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_lifecycle_does_not_use_legacy_enqueue_or_main_thread_prediction() -> None:
    source = _lifecycle_sources()
    assert ".enqueue(" not in source
    assert source.count("_predict_dispatch_matrix(") == 1


def test_lifecycle_facade_remains_small_after_split() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/rs/runtime/online/megatron_ep/lifecycle.py").stat().st_size < 40_000
    assert (root / "src/rs/scheduling/p012_future/_kernel/event_core.py").stat().st_size < 20_000


def test_lifecycle_and_target_planning_do_not_create_process_groups() -> None:
    root = Path(__file__).resolve().parents[1] / "src/rs/runtime/online/megatron_ep"
    bad = []
    lifecycle_relatives = [Path("lifecycle.py"), *sorted(Path("lifecycle_parts").glob("*.py"))]
    for relative in (
        *lifecycle_relatives,
        Path("target_planning/planner_service.py"),
        Path("target_planning/store.py"),
        Path("target_planning/contracts.py"),
        Path("target_planning/reconcile.py"),
    ):
        source = (root / relative).read_text(encoding="utf-8")
        if "dist.new_group(" in source:
            bad.append(str(relative))
    assert not bad, bad


def test_lifecycle_does_not_import_runtime_module() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/runtime/online/megatron_ep")):
        if path.name != "lifecycle.py":
            continue
        if module == "rs.runtime.online.megatron_ep.runtime":
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_formal_planning_package_does_not_export_internal_runtime_builders() -> None:
    import rs.planning as planning

    assert not hasattr(planning, "build_runtime_policy")
    assert not hasattr(planning, "build_runtime_request_from_problem")


def test_formal_experiments_do_not_import_integrations_or_poc_line1() -> None:
    bad = []
    forbidden = (
        "integrations.megatron_ep",
        "experiments.poc_line1",
        "rs.scheduler",
        "rs.evaluation",
        "rs.trace",
        "rs.online",
        "rs.offline",
        "rs.runtime.distributed_ep",
    )
    for path, lineno, module in _collect_imports(Path("experiments")):
        if "legacy" in path.parts or "distributed" in path.parts or "poc_line1" in path.parts:
            continue
        if module.startswith(forbidden):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_formal_experiments_do_not_import_private_runtime_modules() -> None:
    bad = []
    forbidden_tokens = (
        "._facade",
        "._host_impl",
        "._lifecycle",
        "._observation",
        "._layout_join_impl",
        "._plan_agreement_impl",
        "global_ready_set_impl",
    )
    for path, lineno, module in _collect_imports(Path("experiments")):
        if "legacy" in path.parts:
            continue
        if any(token in module for token in forbidden_tokens):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_official_online_wrappers_do_not_import_strategy_comparison_name() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        Path("experiments/run_online_phase_sync.py"),
        Path("experiments/run_online_async_release.py"),
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "run_strategy_comparison" not in source
        assert "run_online_evaluation" in source
        assert "online_evaluation_runner" in source
        assert "strategy_comparison_runner" not in source


def test_formal_runtime_async_path_uses_canonical_backend_module() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        Path("src/rs/runtime/online/megatron_ep/execution/executor_facade.py"),
        Path("src/rs/runtime/online/megatron_ep/execution/transport_adapter.py"),
        Path("experiments/distributed/run_stage1_gloo_e2e_gate.py"),
        Path("experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py"),
        Path("experiments/distributed/run_stage3_runtime_integrated_gloo_gate_lowmem.py"),
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "async_release_backend" in source
        assert "async_p2p_executor" not in source


def test_all_relative_import_targets_exist_in_formal_src() -> None:
    src_root = Path("src")
    missing: list[str] = []
    for path in (src_root / "rs").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        package_parts = list(path.relative_to(src_root).with_suffix("").parts[:-1])
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or int(node.level) <= 0:
                continue
            keep = len(package_parts) - int(node.level) + 1
            target_parts = package_parts[: max(0, keep)]
            if node.module:
                target_parts.extend(str(node.module).split("."))
            target = src_root.joinpath(*target_parts)
            if not (target.with_suffix(".py").exists() or (target / "__init__.py").exists()):
                missing.append(f"{path}:{node.lineno}:{'.'.join(target_parts)}")
    assert not missing, missing
