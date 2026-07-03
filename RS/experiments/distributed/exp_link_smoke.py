#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense.topology import load_inventory


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise RuntimeError("inventory file must parse to a mapping")
        return payload


def _resolve_auth(raw_inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    auth_by_name: dict[str, dict[str, Any]] = {}
    for item in raw_inventory.get("nodes", []):
        if not isinstance(item, dict):
            continue
        auth = item.get("auth", {}) or {}
        auth_by_name[str(item.get("name"))] = dict(auth)
    return auth_by_name


def _ensure_keypair(path: Path) -> tuple[Path, Path]:
    private_key = path
    public_key = Path(f"{path}.pub")
    private_key.parent.mkdir(parents=True, exist_ok=True)
    if private_key.exists() and public_key.exists():
        return private_key, public_key
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return private_key, public_key


def _password_for(node_name: str, auth: dict[str, dict[str, Any]]) -> str:
    node_auth = auth.get(node_name, {})
    direct = str(node_auth.get("ssh_password") or "").strip()
    if direct:
        return direct
    env_name = str(node_auth.get("ssh_password_env") or "").strip()
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise RuntimeError(f"missing ssh password for node {node_name}")


def _run(cmd: list[str], *, capture: bool = True) -> str:
    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def _password_ssh_base(node: Any, password: str) -> list[str]:
    return [
        "sshpass",
        "-p",
        password,
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=20",
        "-p",
        str(node.port),
        f"{node.ssh_user}@{node.host}",
    ]


def _key_ssh_base(node: Any, private_key: Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(private_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=20",
        "-p",
        str(node.port),
        f"{node.ssh_user}@{node.host}",
    ]


def _install_public_key(node: Any, password: str, public_key: Path) -> None:
    pub = public_key.read_text(encoding="utf-8").strip()
    remote_cmd = (
        "set -euo pipefail; "
        "mkdir -p ~/.ssh; chmod 700 ~/.ssh; touch ~/.ssh/authorized_keys; "
        f"grep -qxF {shlex.quote(pub)} ~/.ssh/authorized_keys || echo {shlex.quote(pub)} >> ~/.ssh/authorized_keys; "
        "chmod 600 ~/.ssh/authorized_keys"
    )
    _run(_password_ssh_base(node, password) + [remote_cmd], capture=True)


def _prepare_remote(node: Any, private_key: Path, remote_root: str, artifact_root: str) -> dict[str, Any]:
    remote_log = f"{artifact_root}/link_smoke/{node.name}_remote.json"
    remote_cmd = f"""
set -euo pipefail
mkdir -p {shlex.quote(remote_root)} {shlex.quote(artifact_root)} {shlex.quote(artifact_root)}/link_smoke
if command -v apt-get >/dev/null 2>&1; then
  missing=""
  command -v git >/dev/null 2>&1 || missing="$missing git"
  command -v rsync >/dev/null 2>&1 || missing="$missing rsync"
  command -v python3 >/dev/null 2>&1 || missing="$missing python3 python3-venv python3-pip"
  if [ -n "$missing" ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y $missing >/dev/null 2>&1 || true
  fi
fi
python3 - <<'PY'
import json, os, socket, subprocess, sys
payload = {{
    "hostname": socket.gethostname(),
    "fqdn": socket.getfqdn(),
    "host": {node.host!r},
    "port": {node.port},
    "ip_candidates": subprocess.check_output("hostname -I", shell=True, text=True).strip().split(),
    "python_version": sys.version.split()[0],
    "cwd": os.getcwd(),
}}
print(json.dumps(payload, indent=2))
open({remote_log!r}, "w", encoding="utf-8").write(json.dumps(payload, indent=2))
PY
    """.strip()
    output = _run(_key_ssh_base(node, private_key) + [remote_cmd], capture=True)
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"failed to parse remote JSON payload for {node.name}: {output}")
    return json.loads(output[start : end + 1])


def _ensure_remote_root(node: Any, private_key: Path, remote_root: str, artifact_root: str) -> None:
    remote_cmd = (
        f"set -euo pipefail; mkdir -p {shlex.quote(str(Path(remote_root).parent))} "
        f"{shlex.quote(remote_root)} {shlex.quote(artifact_root)}"
    )
    _run(_key_ssh_base(node, private_key) + [remote_cmd], capture=True)


def _rsync_repo(node: Any, private_key: Path, remote_root: str) -> None:
    ssh_cmd = (
        f"ssh -i {shlex.quote(str(private_key))} -p {node.port} "
        "-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    )
    cmd = [
        "rsync",
        "-az",
        "--delete",
        "--exclude",
        ".git",
        "--exclude",
        "__pycache__",
        "--exclude",
        ".pytest_cache",
        "--exclude",
        "artifacts",
        "--exclude",
        "archive",
        "--exclude",
        "*.pyc",
        "-e",
        ssh_cmd,
        f"{ROOT}/",
        f"{node.ssh_user}@{node.host}:{remote_root}/",
    ]
    _run(cmd, capture=False)


def _fetch_remote_log(node: Any, private_key: Path, remote_file: str, local_file: Path) -> None:
    local_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp",
        "-i",
        str(private_key),
        "-P",
        str(node.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"{node.ssh_user}@{node.host}:{remote_file}",
        str(local_file),
    ]
    _run(cmd, capture=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare remote deployment links and validate repo sync.")
    parser.add_argument(
        "--inventory",
        type=str,
        default=str(ROOT / "deploy" / "inventory" / "hosts.linktest.local.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "deployment" / "link_smoke"),
    )
    parser.add_argument(
        "--key-path",
        type=str,
        default=str(Path.home() / ".ssh" / "routesense_link_ed25519"),
    )
    args = parser.parse_args(argv)

    inventory_path = Path(args.inventory)
    raw_inventory = _load_mapping(inventory_path)
    auth_by_name = _resolve_auth(raw_inventory)
    inventory = load_inventory(inventory_path)
    private_key, public_key = _ensure_keypair(Path(args.key_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results: list[dict[str, Any]] = []
    for node in inventory.nodes:
        password = _password_for(node.name, auth_by_name)
        remote_root = str(node.paths.get("remote_rs_root") or "")
        artifact_root = str(node.paths.get("artifact_root") or f"{remote_root}/artifacts")
        if not remote_root:
            raise RuntimeError(f"missing remote_rs_root for node {node.name}")

        _install_public_key(node, password, public_key)
        key_check = _run(_key_ssh_base(node, private_key) + ["echo passwordless_ok"])
        _ensure_remote_root(node, private_key, remote_root, artifact_root)
        _rsync_repo(node, private_key, remote_root)
        remote_info = _prepare_remote(node, private_key, remote_root, artifact_root)
        remote_log = f"{artifact_root}/link_smoke/{node.name}_remote.json"
        local_log = output_dir / f"{node.name}_remote.json"
        _fetch_remote_log(node, private_key, remote_log, local_log)
        results.append(
            {
                "node_name": node.name,
                "host": node.host,
                "port": node.port,
                "remote_rs_root": remote_root,
                "artifact_root": artifact_root,
                "passwordless_check": key_check.strip(),
                "remote_info": remote_info,
                "retrieved_log": str(local_log),
            }
        )

    payload = {
        "mode": "link_smoke",
        "inventory": str(inventory_path),
        "local_public_key": str(public_key),
        "started_at_epoch_s": started,
        "finished_at_epoch_s": time.time(),
        "node_count": len(results),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
