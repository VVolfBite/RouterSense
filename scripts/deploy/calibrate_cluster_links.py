#!/usr/bin/env python3
from __future__ import annotations

"""Launch NCCL link calibration, retrieve the profile, and distribute it."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.topology import DEFAULT_DEPLOYMENT_MODEL_ID, inventory_cli_summary, load_inventory, load_link_cost_profile
from scripts.deploy.launch_remote import _local_addresses, _run_node
from scripts.deploy.sync_repo import _auth_prefix, _copy_to_node


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-id", default=DEFAULT_DEPLOYMENT_MODEL_ID)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--nccl-socket-ifname", default=os.environ.get("NCCL_SOCKET_IFNAME", ""))
    parser.add_argument("--rows", default="1,16,64,256")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args(argv)


def _copy_from_node(node: Any, source: str, target: Path, *, local_addresses: set[str]) -> None:
    host = str(node.ssh_host or node.host)
    target.parent.mkdir(parents=True, exist_ok=True)
    if str(node.host) in local_addresses or host in local_addresses:
        shutil.copy2(Path(source), target)
        return
    command = [
        *_auth_prefix(),
        "scp",
        "-P",
        str(node.port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"{node.ssh_user}@{host}:{source}",
        str(target),
    ]
    subprocess.run(command, check=True)


def _command(
    *,
    node: Any,
    summary: dict[str, Any],
    inventory_name: str,
    run_id: str,
    model_id: str,
    python_bin: str,
    socket_ifname: str,
    rows: str,
    warmup: int,
    repeats: int,
) -> dict[str, str]:
    resolved = summary["resolved_paths"]
    remote_root = str(resolved[f"{node.name}_remote_rs_root"])
    artifact_root = str(resolved[f"{node.name}_artifact_root"])
    remote_inventory = f"{remote_root}/deploy/inventory/{inventory_name}"
    run_root = f"{artifact_root.rstrip('/')}/{run_id}/link_calibration"
    canonical_profile = f"{remote_root}/outputs/deployment_profiles/{run_id}/link_cost_profile.json"
    log_path = f"{run_root}/{node.name}.log"
    exit_path = f"{run_root}/{node.name}.exit"
    pid_path = f"{run_root}/{node.name}.pid"
    master_name = str(summary["rendezvous"]["master_node"])
    master = next(item for item in summary["nodes"] if str(item["name"]) == master_name)
    master_addr = str(master.get("host") or master.get("ssh_host"))
    master_port = int(summary["rendezvous"]["master_port"]) + 1
    nnodes = len(summary["nodes"])
    model_resolver = (
        f"{shlex.quote(python_bin)} scripts/verify/verify_model_cache.py "
        f"{shlex.quote(remote_inventory)} {shlex.quote(str(node.name))} "
        f"--model-id {shlex.quote(model_id)} --print-model-path"
    )
    env = [
        f"PYTHONPATH={shlex.quote(remote_root + '/src:' + remote_root)}",
        "NCCL_ASYNC_ERROR_HANDLING=1",
        "TORCHDISTRIBUTED_DEBUG=DETAIL",
    ]
    if socket_ifname:
        env.append(f"NCCL_SOCKET_IFNAME={shlex.quote(socket_ifname)}")
    torchrun = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={int(node.target_gpu_count)}",
        f"--node_rank={int(node.node_rank)}",
        "--rdzv-backend=c10d",
        f"--rdzv-id={run_id}-link-calibration",
        f"--rdzv-endpoint={master_addr}:{master_port}",
        "-m",
        "experiments.distributed.calibrate_link_costs",
        "--output",
        canonical_profile,
        "--model-path",
        '"$MODEL_PATH"',
        "--rows",
        rows,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    strict = (
        f"set -euo pipefail; cd {shlex.quote(remote_root)}; "
        f"mkdir -p {shlex.quote(run_root)} {shlex.quote(str(Path(canonical_profile).parent))}; "
        f"MODEL_PATH=$({model_resolver}); export {' '.join(env)}; "
        + " ".join(shlex.quote(part) if part != '"$MODEL_PATH"' else part for part in torchrun)
    )
    wrapper = (
        "set +e; "
        f"bash -lc {shlex.quote(strict)}; rc=$?; "
        f"printf '%s\\n' \"$rc\" > {shlex.quote(exit_path)}; exit \"$rc\""
    )
    detached = (
        f"rm -f {shlex.quote(exit_path)}; "
        f"nohup setsid bash -lc {shlex.quote(wrapper)} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"pid=$!; printf '%s\\n' \"$pid\" > {shlex.quote(pid_path)}; printf '%s\\n' \"$pid\""
    )
    return {
        "node_name": str(node.name),
        "remote_root": remote_root,
        "canonical_profile": canonical_profile,
        "log_path": log_path,
        "exit_path": exit_path,
        "pid_path": pid_path,
        "command": strict,
        "detached_command": detached,
    }


def _wait(*, inventory: Any, rows: list[dict[str, Any]], local_addresses: set[str], timeout: int) -> list[dict[str, Any]]:
    nodes = {str(node.name): node for node in inventory.nodes}
    pending = {str(row["node_name"]): row for row in rows}
    results: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(int(timeout), 1)
    while pending:
        if time.monotonic() >= deadline:
            results.extend({**row, "status": "TIMEOUT", "exit_code": None} for row in pending.values())
            break
        for name, row in list(pending.items()):
            probe = f"if [ -f {shlex.quote(row['exit_path'])} ]; then cat {shlex.quote(row['exit_path'])}; else echo RUNNING; fi"
            output = _run_node(nodes[name], probe, local_addresses=local_addresses).strip()
            if output == "RUNNING":
                continue
            try:
                code = int(output.splitlines()[-1])
            except ValueError:
                code = 255
            results.append({**row, "status": "PASS" if code == 0 else "FAIL", "exit_code": code})
            pending.pop(name, None)
        if pending:
            time.sleep(3)
    return results



def _stop_calibration_jobs(
    *,
    inventory: Any,
    rows: list[dict[str, Any]],
    local_addresses: set[str],
) -> list[dict[str, Any]]:
    nodes = {str(node.name): node for node in inventory.nodes}
    stopped: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["node_name"])
        pid_path = str(row["pid_path"])
        command = (
            f"set -euo pipefail; if [ ! -f {shlex.quote(pid_path)} ]; then echo MISSING; exit 0; fi; "
            f"pid=$(cat {shlex.quote(pid_path)}); "
            "if ! printf '%s' \"$pid\" | grep -Eq '^[0-9]+$'; then echo INVALID; exit 2; fi; "
            "if kill -0 \"$pid\" 2>/dev/null; then "
            "kill -TERM -- \"-$pid\" 2>/dev/null || kill -TERM \"$pid\"; echo STOPPED; "
            "else echo NOT_RUNNING; fi"
        )
        try:
            result = _run_node(nodes[name], command, local_addresses=local_addresses).strip()
            stopped.append({"node_name": name, "result": result, "status": "PASS" if result in {"STOPPED", "NOT_RUNNING", "MISSING"} else "FAIL"})
        except Exception as exc:
            stopped.append({"node_name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    return stopped

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_path = Path(args.inventory)
    inventory = load_inventory(inventory_path)
    summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("rs-%Y%m%dT%H%M%SZ")
    commands = [
        _command(
            node=node,
            summary=summary,
            inventory_name=inventory_path.name,
            run_id=run_id,
            model_id=str(args.model_id),
            python_bin=str(args.python_bin),
            socket_ifname=str(args.nccl_socket_ifname),
            rows=str(args.rows),
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        for node in inventory.nodes
    ]
    local_profile = ROOT / "outputs" / "deployment_profiles" / run_id / "link_cost_profile.json"
    payload: dict[str, Any] = {
        "schema_version": "routersense.deploy.link_calibration.v1",
        "apply_mode": bool(args.apply),
        "run_id": run_id,
        "world_size": sum(int(node.target_gpu_count) for node in inventory.nodes),
        "inventory": summary,
        "local_profile": str(local_profile),
        "commands": commands,
    }
    if not args.apply:
        payload.update({"status": "DRY_RUN", "profile_ready": False})
        print(json.dumps(payload, indent=2))
        return 0

    local_addresses = _local_addresses()
    ordered = sorted(inventory.nodes, key=lambda node: (node.name == inventory.rendezvous.master_node, node.node_rank))
    by_name = {str(row["node_name"]): row for row in commands}
    launches: list[dict[str, Any]] = []
    try:
        for node in ordered:
            row = by_name[str(node.name)]
            pid = _run_node(node, str(row["detached_command"]), local_addresses=local_addresses).splitlines()[-1]
            launches.append({**row, "pid": pid})
    except Exception:
        payload["cleanup"] = _stop_calibration_jobs(
            inventory=inventory, rows=launches, local_addresses=local_addresses
        )
        raise
    completed = _wait(
        inventory=inventory,
        rows=launches,
        local_addresses=local_addresses,
        timeout=int(args.timeout_seconds),
    )
    payload["completed"] = completed
    if not completed or any(row["status"] != "PASS" for row in completed):
        payload["cleanup"] = _stop_calibration_jobs(
            inventory=inventory, rows=launches, local_addresses=local_addresses
        )
        payload.update({"status": "FAIL", "profile_ready": False})
        print(json.dumps(payload, indent=2))
        return 2

    master_node = next(node for node in inventory.nodes if node.name == inventory.rendezvous.master_node)
    master_profile = by_name[str(master_node.name)]["canonical_profile"]
    _copy_from_node(master_node, master_profile, local_profile, local_addresses=local_addresses)
    profile = load_link_cost_profile(local_profile)
    if int(profile.world_size) != int(payload["world_size"]):
        raise RuntimeError("calibrated profile world size mismatch")
    local_sha256 = hashlib.sha256(local_profile.read_bytes()).hexdigest()
    distributed: list[dict[str, Any]] = []
    for node in inventory.nodes:
        target = by_name[str(node.name)]["canonical_profile"]
        _run_node(node, f"mkdir -p {shlex.quote(str(Path(target).parent))}", local_addresses=local_addresses)
        _copy_to_node(node, local_profile, target, local_addresses=local_addresses)
        remote_hash = _run_node(
            node,
            f"sha256sum {shlex.quote(target)} | awk '{{print $1}}'",
            local_addresses=local_addresses,
        ).splitlines()[-1]
        distributed.append({
            "node_name": str(node.name),
            "target": target,
            "sha256": remote_hash,
            "status": "PASS" if remote_hash == local_sha256 else "FAIL",
        })
    payload.update(
        {
            "status": "PASS",
            "profile_ready": True,
            "profile_id": str(profile.profile_id),
            "profile": profile.to_dict(),
            "profile_sha256": local_sha256,
            "distributed": distributed,
        }
    )
    if any(row["status"] != "PASS" for row in distributed):
        payload["status"] = "FAIL"
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
