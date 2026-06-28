from __future__ import annotations

from routesense_poc1.cli import main


def test_cli_main_creates_output_dir(tmp_path):
    output_dir = tmp_path / "outputs"
    exit_code = main(["--output-dir", str(output_dir)])
    assert exit_code == 0
    assert output_dir.exists()
