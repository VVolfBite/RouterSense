from __future__ import annotations

import subprocess
from pathlib import Path


def test_source_package_includes_deployment_files_and_excludes_runtime(tmp_path):
    root = Path("/root/autodl-tmp/RouterSense")
    archive = tmp_path / "RouterSense.TAR.GZ"
    subprocess.run([str(root / "scripts" / "package_source_only.sh"), str(archive)], check=True)
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True)
    assert "src/routesense/runtime/single_gpu.py" in listing
    assert "deploy/scripts/check_cluster_access.sh" in listing
    assert "experiments/deployment/single_gpu_olmoe_smoke.py" in listing
    assert "outputs/" not in listing
    assert "artifacts/" not in listing
    assert "deploy/logs/" not in listing
