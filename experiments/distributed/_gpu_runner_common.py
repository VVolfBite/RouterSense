from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rs.experiments_support.gpu_runner_common import (
    available_cuda_count,
    build_policy_correctness_config,
    build_strategy_comparison_config,
    child_env,
    copy_config,
    dump_yaml,
    load_official_config,
    load_yaml,
    python_module_command,
    read_json,
    repo_relative,
    run_subprocess,
    torchrun_policy_command,
    write_json,
    write_runner_result_bundle,
)

__all__ = [
    "available_cuda_count",
    "build_policy_correctness_config",
    "build_strategy_comparison_config",
    "child_env",
    "copy_config",
    "dump_yaml",
    "load_official_config",
    "load_yaml",
    "python_module_command",
    "read_json",
    "repo_relative",
    "run_subprocess",
    "torchrun_policy_command",
    "write_json",
    "write_runner_result_bundle",
]
