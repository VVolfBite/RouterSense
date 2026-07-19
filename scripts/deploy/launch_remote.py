#!/usr/bin/env python3
from __future__ import annotations

"""Inventory-driven multi-node launcher for one formal RouterSense strategy."""

import argparse
import json
import os
import shlex
import shutil
import socket
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

from rs.topology import DEFAULT_DEPLOYMENT_MODEL_ID, inventory_cli_summary, load_inventory


DEFAULT_CONFIG = "configs/official/online_p012_deploy_smoke.yaml"
DEFAULT_STRATEGY = "routersense_future_p012_joint_global_rscf_async"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", nargs="?", default="deploy/inventory/hosts.local.yaml")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--comparison-config", default=DEFAULT_CONFIG)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-id", default=DEFAULT_DEPLOYMENT_MODEL_ID)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--nccl-socket-ifname", default=os.environ.get("NCCL_SOCKET_IFNAME", ""))
    return parser.parse_args(argv)


def _local_addresses() -> set[str]:
    values = {"127.0.0.1", "localhost", "::1"}
    try:
        values.update(subprocess.check_output(["hostname", "-I"], text=True).split())
    except Exception:
        pass
    try:
        values.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        pass
    return values


def _ssh_base(node: Any) -> list[str]:
    password = os.environ.get("RSSH_PASSWORD") or os.environ.get("SSHPASS")
    base: list[str] = []
    if password:
        if shutil.which("sshpass") is None:
            raise RuntimeError("RSSH_PASSWORD/SSHPASS is set but sshpass is not installed")
        base.extend(["sshpass", "-p", password])
    base.extend(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=20",
            "-p",
            str(node.port),
            f"{node.ssh_user}@{node.ssh_host or node.host}",
        ]
    )
    return base


def _run_node(node: Any, command: str, *, local_addresses: set[str]) -> str:
    """Run one node command while keeping machine-readable stdout isolated.

    Deployment helpers return JSON or compact probes on stdout.  Package
    managers, Hugging Face, SSH, and NCCL may write diagnostics to stderr; those
    diagnostics must not corrupt the JSON contract consumed by the caller.
    """

    host = str(node.ssh_host or node.host)
    argv = ["bash", "-lc", command] if str(node.host) in local_addresses or host in local_addresses else _ssh_base(node) + [command]
    completed = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        detail = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        raise RuntimeError(f"node command failed rc={completed.returncode}: {detail[-4000:]}")
    return (completed.stdout or "").strip()


def _node_paths(summary: dict[str, Any], node_name: str) -> tuple[str, str, str]:
    resolved = summary["resolved_paths"]
    remote_root = str(resolved.get(f"{node_name}_remote_rs_root") or "")
    model_cache = str(resolved.get(f"{node_name}_model_cache") or "")
    artifact_root = str(resolved.get(f"{node_name}_artifact_root") or "")
    if not remote_root or not model_cache or not artifact_root:
        raise RuntimeError(f"inventory paths incomplete for {node_name}")
    return remote_root, model_cache, artifact_root


def _strategy_command(
    *,
    node: Any,
    inventory_name: str,
    summary: dict[str, Any],
    comparison_config: str,
    strategy: str,
    run_id: str,
    model_id: str,
    python_bin: str,
    nccl_socket_ifname: str,
) -> dict[str, str]:
    remote_root, _model_cache, artifact_root = _node_paths(summary, node.name)
    remote_inventory = f"{remote_root}/deploy/inventory/{inventory_name}"
    remote_config = comparison_config if comparison_config.startswith("/") else f"{remote_root}/{comparison_config}"
    run_root = f"{artifact_root.rstrip('/')}/{run_id}"
    log_path = f"{run_root}/logs/{node.name}.log"
    pid_path = f"{run_root}/logs/{node.name}.pid"
    exit_path = f"{run_root}/logs/{node.name}.exit"
    master_name = str(summary["rendezvous"]["master_node"])
    master = next(item for item in summary["nodes"] if str(item["name"]) == master_name)
    master_addr = str(master.get("host") or master.get("ssh_host"))
    master_port = int(summary["rendezvous"]["master_port"])
    backend = str(summary["rendezvous"].get("backend", "c10d"))
    nnodes = len(summary["nodes"])
    env_parts = [
        f"PYTHONPATH={shlex.quote(remote_root + '/src:' + remote_root)}",
        "TORCHDISTRIBUTED_DEBUG=DETAIL",
        "NCCL_ASYNC_ERROR_HANDLING=1",
    ]
    if nccl_socket_ifname:
        env_parts.append(f"NCCL_SOCKET_IFNAME={shlex.quote(nccl_socket_ifname)}")
    model_resolver = (
        f"{shlex.quote(python_bin)} scripts/verify/verify_model_cache.py "
        f"{shlex.quote(remote_inventory)} {shlex.quote(node.name)} "
        f"--model-id {shlex.quote(model_id)} --print-model-path"
    )
    torchrun = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={int(node.target_gpu_count)}",
        f"--node_rank={int(node.node_rank)}",
        f"--rdzv-backend={backend}",
        f"--rdzv-id={run_id}",
        f"--rdzv-endpoint={master_addr}:{master_port}",
        "-m",
        "experiments.online.run_deployed_strategy",
        "--comparison-config",
        remote_config,
        "--strategy",
        strategy,
        "--output-dir",
        run_root,
        "--run-id",
        run_id,
        "--model-path",
        '"$MODEL_PATH"',
    ]
    run_command = (
        f"set -euo pipefail; cd {shlex.quote(remote_root)}; "
        f"mkdir -p {shlex.quote(run_root + '/logs')}; "
        f"MODEL_PATH=$({model_resolver}); export {' '.join(env_parts)}; "
        + " ".join(shlex.quote(part) if part != '"$MODEL_PATH"' else part for part in torchrun)
    )
    # Keep an outer non-errexit shell alive so the exit marker is written even
    # when the strict torchrun command fails.  Without this, wait-mode can hang
    # until its timeout because ``set -e`` exits before persisting ``$?``.
    completion_wrapper = (
        "set +e; "
        f"bash -lc {shlex.quote(run_command)}; "
        "rc=$?; "
        f"printf '%s\\n' \"$rc\" > {shlex.quote(exit_path)}; "
        "exit \"$rc\""
    )
    detached = (
        f"rm -f {shlex.quote(exit_path)}; "
        f"nohup setsid bash -lc {shlex.quote(completion_wrapper)} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"pid=$!; printf '%s\\n' \"$pid\" > {shlex.quote(pid_path)}; printf '%s\\n' \"$pid\""
    )
    return {
        "node_name": str(node.name),
        "run_root": run_root,
        "log_path": log_path,
        "pid_path": pid_path,
        "exit_path": exit_path,
        "command": run_command,
        "detached_command": detached,
    }


def _wait_for_jobs(
    *,
    inventory: Any,
    launches: list[dict[str, Any]],
    local_addresses: set[str],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + max(int(timeout_seconds), 1)
    pending = {row["node_name"]: row for row in launches}
    completed: list[dict[str, Any]] = []
    nodes = {str(node.name): node for node in inventory.nodes}
    while pending:
        if time.monotonic() >= deadline:
            for row in pending.values():
                completed.append({**row, "status": "TIMEOUT", "exit_code": None})
            break
        for name, row in list(pending.items()):
            probe = f"if [ -f {shlex.quote(row['exit_path'])} ]; then cat {shlex.quote(row['exit_path'])}; else echo RUNNING; fi"
            result = _run_node(nodes[name], probe, local_addresses=local_addresses).strip()
            if result != "RUNNING":
                try:
                    exit_code = int(result.splitlines()[-1])
                except ValueError:
                    exit_code = 255
                completed.append({**row, "status": "PASS" if exit_code == 0 else "FAIL", "exit_code": exit_code})
                pending.pop(name, None)
        if pending:
            time.sleep(5)
    return completed


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_path = Path(args.inventory)
    inventory = load_inventory(inventory_path)
    summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("rs-%Y%m%dT%H%M%SZ")
    commands = [
        _strategy_command(
            node=node,
            inventory_name=inventory_path.name,
            summary=summary,
            comparison_config=str(args.comparison_config),
            strategy=str(args.strategy),
            run_id=str(run_id),
            model_id=str(args.model_id),
            python_bin=str(args.python_bin),
            nccl_socket_ifname=str(args.nccl_socket_ifname),
        )
        for node in inventory.nodes
    ]
    payload: dict[str, Any] = {
        "schema_version": "routersense.deploy.launch.v2",
        "apply_mode": bool(args.apply),
        "dry_run": not bool(args.apply),
        "wait": bool(args.wait),
        "nnodes": len(inventory.nodes),
        "nproc_per_node": (int(inventory.nodes[0].target_gpu_count) if inventory.nodes and len({int(node.target_gpu_count) for node in inventory.nodes}) == 1 else None),
        "gpu_capacity_sufficient": all(int(node.current_gpu_count) >= int(node.target_gpu_count) for node in inventory.nodes),
        "launch_block_status": ("READY" if all(int(node.current_gpu_count) >= int(node.target_gpu_count) for node in inventory.nodes) else "MULTINODE_EP_BLOCKED_BY_GPU_CAPACITY"),
        "inventory": summary,
        "comparison_config": str(args.comparison_config),
        "strategy": str(args.strategy),
        "run_id": str(run_id),
        "world_size": sum(int(node.target_gpu_count) for node in inventory.nodes),
        "commands": commands,
    }
    if not args.apply:
        payload["status"] = "DRY_RUN"
        print(json.dumps(payload, indent=2))
        return 0

    if not payload["gpu_capacity_sufficient"]:
        payload["status"] = "FAIL"
        payload["reason"] = "inventory current_gpu_count is below target_gpu_count"
        print(json.dumps(payload, indent=2))
        return 2

    local_addresses = _local_addresses()
    launches: list[dict[str, Any]] = []
    # Start workers first, rendezvous master last.
    ordered = sorted(inventory.nodes, key=lambda node: (node.name == inventory.rendezvous.master_node, node.node_rank))
    command_by_name = {str(row["node_name"]): row for row in commands}
    for node in ordered:
        row = command_by_name[str(node.name)]
        pid = _run_node(node, str(row["detached_command"]), local_addresses=local_addresses).splitlines()[-1]
        launches.append({**row, "pid": pid, "launch_status": "STARTED"})
    payload["launches"] = launches
    if args.wait:
        completed = _wait_for_jobs(
            inventory=inventory,
            launches=launches,
            local_addresses=local_addresses,
            timeout_seconds=int(args.timeout_seconds),
        )
        payload["completed"] = completed
        payload["status"] = "PASS" if completed and all(row["status"] == "PASS" for row in completed) else "FAIL"
    else:
        payload["status"] = "STARTED"
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] in {"STARTED", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
