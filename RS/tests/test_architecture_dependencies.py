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


def test_runtime_formal_modules_do_not_import_private_legacy_planning_runtime() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/runtime/online/megatron_ep")):
        if module.startswith("rs.planning._legacy_runtime"):
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


def test_lifecycle_does_not_use_legacy_enqueue_or_main_thread_prediction() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/rs/runtime/online/megatron_ep/lifecycle.py").read_text(encoding="utf-8")
    assert ".enqueue(" not in source
    assert source.count("_predict_dispatch_matrix(") == 1


def test_lifecycle_does_not_import_runtime_module() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("src/rs/runtime/online/megatron_ep")):
        if path.name != "lifecycle.py":
            continue
        if module == "rs.runtime.online.megatron_ep.runtime":
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad


def test_formal_planning_package_does_not_export_legacy_runtime_builders() -> None:
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
