from pathlib import Path


def test_shell_scripts_exist_and_use_strict_mode():
    for script in [
        Path("scripts/bootstrap_and_inspect.sh"),
        Path("scripts/setup_env.sh"),
        Path("scripts/run_poc1.sh"),
    ]:
        assert script.exists(), f"missing {script}"
        text = script.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
