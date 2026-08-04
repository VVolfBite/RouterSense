from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import yaml

from rs_sim.scheduler.core.birkhoff_core import order_birkhoff
from rs_sim.scheduler.core.literature_cores import (
    order_aurora,
    order_fast,
    order_islip,
    order_residual_mwm,
)
from rs_sim.scheduler.core.oracle import solve_exact_wire
from rs_sim.scheduler.core.rscf_core import order_rscf
from rs_sim.scheduler.decorators.composition import REGISTERED_CORE_IDS


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
RS_SIM_ROOT = SOURCE_ROOT / "rs_sim"

FORBIDDEN_TEXT = re.compile(
    r"releasefrontier_window_joint|rscf_scoring_profile|"
    r"enable_global_safe_selector|BIRKHOFF_LOCAL_FALLBACK|P12_ONLINE_V|"
    r"\bFUTURE\b|\bP01\b|\bP012\b|\bP0123\b|"
    r"\bM[0-5][_-]|(?:^|_)R\d+(?:_\d+)*|(?:^|_)V\d+(?:_\d+)*"
)


def test_source_tree_contains_one_public_package() -> None:
    packages = sorted(
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    )
    assert packages == ["rs_sim"]


def test_production_source_has_no_historical_route_identifiers() -> None:
    violations: list[str] = []
    for path in RS_SIM_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN_TEXT.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}:{line.strip()}")
    assert violations == []


def test_registered_cores_have_one_scope_free_entry_function() -> None:
    entries = {
        "rscf": order_rscf,
        "birkhoff": order_birkhoff,
        "islip": order_islip,
        "residual_mwm": order_residual_mwm,
        "fast": order_fast,
        "aurora": order_aurora,
        "oracle": solve_exact_wire,
    }
    assert set(entries).issubset(REGISTERED_CORE_IDS)
    forbidden_parameters = {
        "scope",
        "planning",
        "planning_mode",
        "joint_scope",
        "joint_release",
        "profile",
        "profile_id",
        "safe",
        "fallback",
        "version",
    }
    for core_id, entry in entries.items():
        parameters = set(inspect.signature(entry).parameters)
        assert parameters.isdisjoint(forbidden_parameters), (core_id, parameters)


def test_controller_cannot_select_an_algorithm_by_name() -> None:
    path = RS_SIM_ROOT / "scheduler" / "execution" / "controller.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "create_plan" not in methods
    assert "activate_plan" in methods


def test_experiment_configs_use_only_composed_algorithm_expression() -> None:
    for path in (ROOT / "configs" / "experiment").rglob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        treatments = payload["experiments"]["treatments"]
        assert treatments
        for treatment in treatments:
            assert isinstance(treatment.get("algorithm"), str)
            assert not ({"core", "scope", "planning", "safe"} & set(treatment))


def test_layered_packages_have_no_flat_compatibility_modules() -> None:
    forbidden = (
        "runtime/runtime.py",
        "runtime/backend_adapter.py",
        "runtime/scheduler_adapter.py",
        "runtime/trace_bridge.py",
        "runtime/communication_metrics.py",
        "transport/control_plane.py",
        "transport/data_plane.py",
        "transport/profiles.py",
        "transport/builders.py",
        "transport/driver.py",
        "trace/model.py",
        "trace/fixtures.py",
        "trace/serialization.py",
        "trace/profiles.py",
        "trace/collector.py",
        "trace/realization.py",
    )
    assert [item for item in forbidden if (RS_SIM_ROOT / item).exists()] == []


def test_simulation_semantics_do_not_read_environment_variables() -> None:
    roots = (
        RS_SIM_ROOT / "backend",
        RS_SIM_ROOT / "scheduler",
        RS_SIM_ROOT / "runtime",
        RS_SIM_ROOT / "transport",
    )
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "os.getenv" in text or "os.environ" in text or "getenv(" in text:
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_removed_streaming_truth_bypass_is_absent_everywhere() -> None:
    violations = []
    for path in RS_SIM_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "RS_SIM_STREAMING_P2_FRACTION" in text or "streaming_p2_fraction" in text:
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_repository_assets_have_no_retired_stage_schema_identifiers() -> None:
    retired_markers = (
        "RS_SIM_PAYLOAD_V4_1_1_R2_2",
        "RS_SIM_DESCRIPTOR_METADATA_V4_1_1_R2_2",
        "RS_SIM_THREE_LINE_PROFILE_V4_1_1_R2_5",
        "RS_SIM_RECEIVER_CALIBRATION_V4_1_1",
        "RS_SIM_FORMAL_RUN_RESULT_V4_1_1_R2_5",
        "RS_SIM_FORMAL_RUN_RECOVERY_V4_1_1_R2_5",
        "routesense_source_expert_counts_adapter_r2_2",
        "rs_sim_v4_1_1_contract_ep4",
        "rs_sim_v4_1_1_r2_3_experiment_matrix",
        "rs-sim-trace-provider-0.2.0-r2.1.1",
        "rs-sim-trace-provider-0.3.0-r2.2",
        "rs-sim-trace-provider-0.4.0-r2.3",
    )
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "fixtures", ROOT / "configs")
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in retired_markers:
                if marker in text:
                    violations.append(f"{path.relative_to(ROOT)}:{marker}")
    assert violations == []
