from __future__ import annotations

import ast
from pathlib import Path


def _collect_imports(root: Path) -> list[tuple[Path, int, str]]:
    rows: list[tuple[Path, int, str]] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
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


def test_formal_experiments_do_not_import_integrations_or_poc_line1() -> None:
    bad = []
    for path, lineno, module in _collect_imports(Path("experiments")):
        if "legacy" in path.parts or "distributed" in path.parts or "poc_line1" in path.parts:
            continue
        if module.startswith(("integrations.megatron_ep", "experiments.poc_line1")):
            bad.append(f"{path}:{lineno}:{module}")
    assert not bad, bad
