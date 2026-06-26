from __future__ import annotations

import subprocess
from pathlib import Path


def test_package_source_only_excludes_runtime_dirs(tmp_path):
    root = Path("/root/autodl-tmp/RouterSense")
    script = root / "scripts" / "package_source_only.sh"
    archive = tmp_path / "src.tar.gz"
    subprocess.run([str(script), str(archive)], cwd=root, check=True)
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True)
    assert "outputs/" not in listing
    assert "artifacts/" not in listing
    assert ".pytest_cache/" not in listing
    assert "__pycache__/" not in listing
