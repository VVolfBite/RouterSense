from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from rs_sim.trace.collection.fixture_builder import build_fixtures_from_capture
from rs_sim.trace.collection.pipeline import doctor, inspect_capture_artifacts, launch_collection
from rs_sim.trace.runners.preflight import run_megatron_model_preflight, write_preflight

from .artifact_identity import checkpoint_identity, sha256_file, stable_json_digest
from .collect import _bundle_trace, _write_trace_manifest, normalize_collect_config
from .config_io import ConfigError


PHASES = {"prepare", "worker", "finalize", "all"}


def _replace_torchrun_option(command: list[str], name: str, value: str) -> list[str]:
    prefix = f"--{name}="
    output = [item for item in command if not item.startswith(prefix)]
    insert_at = 1 if output and Path(output[0]).name == "torchrun" else 0
    output.insert(insert_at, f"--{name}={value}")
    return output


def _semantic_config(internal: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy({key: value for key, value in internal.items() if not str(key).startswith("__")})
    payload.pop("output_dir", None)
    launcher = payload.get("launcher", {})
    launcher.pop("clean_output_before_collect", None)
    command = [item for item in launcher.get("command", []) if not str(item).startswith("--node_rank=")]
    launcher["command"] = command
    capture = payload.get("capture", {})
    capture.pop("accepted_source_ranks", None)
    return payload


def collection_contract_digest(internal: dict[str, Any]) -> str:
    return stable_json_digest(_semantic_config(internal))


def _topology(internal: dict[str, Any]) -> tuple[int, int, int]:
    command = list(internal["launcher"]["command"])
    def option(name: str, default: int) -> int:
        prefix = f"--{name}="
        raw = next((item.split("=", 1)[1] for item in command if item.startswith(prefix)), None)
        return int(raw) if raw is not None else int(default)
    parallel = internal["model_runner"]["parallel"]
    nnodes = option("nnodes", 1)
    nproc = option("nproc_per_node", parallel["ep"])
    world = nnodes * nproc
    return nnodes, nproc, world


def _node_ranks(node_rank: int, nproc_per_node: int) -> list[int]:
    start = int(node_rank) * int(nproc_per_node)
    return list(range(start, start + int(nproc_per_node)))


def _node_output(base: Path, node_rank: int) -> Path:
    return base / "node_staging" / f"node{int(node_rank):04d}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_multinode(config: dict[str, Any]) -> dict[str, Any]:
    internal = normalize_collect_config(config)
    nnodes, nproc, world = _topology(internal)
    if nnodes < 2:
        raise ConfigError("multinode prepare requires launch.nnodes >= 2")
    base = Path(internal["output_dir"])
    base.mkdir(parents=True, exist_ok=True)
    identity = checkpoint_identity(
        Path(internal["capture"]["model_path"]),
        explicit_digest=config.get("model", {}).get("checkpoint_digest"),
    )
    digest = collection_contract_digest(internal)
    plan = {
        "schema_version": "RS_SIM_MULTINODE_COLLECTION_PLAN",
        "status": "PASS",
        "capture_id": internal["capture"]["capture_id"],
        "contract_digest": digest,
        "artifact_owner_node_rank": int(config.get("multinode", {}).get("artifact_owner_node_rank", 0)),
        "nnodes": nnodes,
        "nproc_per_node": nproc,
        "world_size": world,
        "rank_to_node": internal["capture"]["rank_to_node"],
        "node_workers": [
            {
                "node_rank": node,
                "expected_source_ranks": _node_ranks(node, nproc),
                "command": f"rs-sim collect --config {config['__config_path']} --phase worker --node-rank {node}",
                "artifact_name": f"{internal['capture']['capture_id']}_node{node:04d}_capture.zip",
            }
            for node in range(nnodes)
        ],
        "finalize_command_template": (
            f"rs-sim collect --config {config['__config_path']} --phase finalize "
            + " ".join(f"--node-artifact <NODE{node}_ZIP>" for node in range(nnodes))
        ),
        "checkpoint_identity": identity,
    }
    _write_json(base / "multinode_collection_plan.json", plan)
    _write_json(base / "checkpoint_identity.json", identity)
    _write_json(base / "resolved_collection_contract.json", _semantic_config(internal))
    return plan


def _run_preflight(internal: dict[str, Any], output_dir: Path) -> dict[str, Any] | None:
    if not internal.get("model_runner", {}).get("self_contained"):
        return None
    parallel = internal["model_runner"]["parallel"]
    command = internal["launcher"]["command"]
    nproc_arg = next(item for item in command if item.startswith("--nproc_per_node="))
    nproc_per_node = int(nproc_arg.split("=", 1)[1])
    report = run_megatron_model_preflight(
        model_path=internal["capture"]["model_path"],
        hf_config_path=next(
            (command[index + 1] for index, item in enumerate(command[:-1]) if item == "--hf-config-path"),
            None,
        ),
        model_format=internal["model_runner"]["model_format"],
        ep=parallel["ep"],
        tp=parallel["tp"],
        pp=parallel["pp"],
        dp=parallel["dp"],
        cp=parallel["cp"],
        etp=parallel["etp"],
        nproc_per_node=nproc_per_node,
        trust_remote_code="--trust-remote-code" in command,
        require_cuda=True,
        require_fate_route=internal["prediction"]["mode"] == "FATE_P2",
        require_compute_hooks=bool(internal["capture"].get("capture_compute", True)),
    )
    write_preflight(output_dir / "model_preflight.json", report)
    return report


def _bundle_node_artifact(node_output: Path, capture_id: str, node_rank: int) -> tuple[Path, Path]:
    files: list[Path] = []
    lines: list[str] = []
    for path in sorted(node_output.rglob("*")):
        if not path.is_file() or "node_artifacts" in path.relative_to(node_output).parts:
            continue
        if path.name == "NODE_ARTIFACT_MANIFEST.sha256":
            continue
        files.append(path)
        lines.append(f"{sha256_file(path)}  {path.relative_to(node_output).as_posix()}\n")
    manifest = node_output / "NODE_ARTIFACT_MANIFEST.sha256"
    manifest.write_text("".join(lines), encoding="utf-8")
    files.append(manifest)
    artifact_dir = node_output / "node_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in capture_id)
    archive = artifact_dir / f"{safe}_node{node_rank:04d}_capture.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in files:
            handle.write(path, path.relative_to(node_output).as_posix())
    sha_path = archive.with_suffix(archive.suffix + ".sha256")
    sha_path.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    return archive, sha_path


def run_multinode_worker(config: dict[str, Any], *, node_rank: int) -> dict[str, Any]:
    internal = normalize_collect_config(config)
    nnodes, nproc, world = _topology(internal)
    if nnodes < 2:
        raise ConfigError("worker phase requires launch.nnodes >= 2")
    if not 0 <= int(node_rank) < nnodes:
        raise ConfigError(f"node_rank must be in [0,{nnodes - 1}]")
    base = Path(internal["output_dir"])
    node_output = _node_output(base, int(node_rank))
    if node_output.exists():
        shutil.rmtree(node_output)
    node_output.mkdir(parents=True, exist_ok=True)
    internal["output_dir"] = str(node_output)
    internal["capture"]["accepted_source_ranks"] = _node_ranks(int(node_rank), nproc)
    internal["launcher"]["command"] = _replace_torchrun_option(
        list(internal["launcher"]["command"]), "node_rank", str(int(node_rank))
    )
    internal["launcher"]["clean_output_before_collect"] = True
    runtime_config = node_output / "capture_runtime_config.json"
    _write_json(runtime_config, {key: value for key, value in internal.items() if not str(key).startswith("__")})
    internal["__config_path"] = str(runtime_config)
    internal["__config_dir"] = str(node_output)

    identity = checkpoint_identity(
        Path(internal["capture"]["model_path"]),
        explicit_digest=config.get("model", {}).get("checkpoint_digest"),
    )
    _write_json(node_output / "checkpoint_identity.json", identity)
    preflight = _run_preflight(internal, node_output)
    environment = doctor(internal, require_megatron=internal["capture"]["backend"] == "MEGATRON_CORE_AUTO")
    launch = launch_collection(internal)
    artifacts = inspect_capture_artifacts(internal)
    if artifacts.get("status") != "PASS":
        raise RuntimeError(f"node-local capture acceptance failed: {artifacts}")
    summary = {
        "schema_version": "RS_SIM_MULTINODE_NODE_RESULT",
        "status": "PASS",
        "capture_id": internal["capture"]["capture_id"],
        "contract_digest": collection_contract_digest(internal),
        "checkpoint_digest": identity.get("checkpoint_digest"),
        "node_rank": int(node_rank),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "nnodes": nnodes,
        "nproc_per_node": nproc,
        "world_size": world,
        "expected_source_ranks": _node_ranks(int(node_rank), nproc),
        "environment": environment,
        "model_preflight": preflight,
        "launch": launch,
        "capture_artifacts": artifacts,
    }
    _write_json(node_output / "node_collection_summary.json", summary)
    archive, sha_path = _bundle_node_artifact(node_output, internal["capture"]["capture_id"], int(node_rank))
    result = dict(summary)
    result["node_artifact"] = str(archive)
    result["node_artifact_sha256_file"] = str(sha_path)
    return result


def _safe_extract_zip(archive: Path, target: Path) -> None:
    target = Path(target).resolve()
    with zipfile.ZipFile(archive, "r") as handle:
        for member in handle.infolist():
            destination = (target / member.filename).resolve()
            if target != destination and target not in destination.parents:
                raise RuntimeError(f"unsafe path in node artifact: {member.filename}")
        handle.extractall(target)


def _verify_extracted_artifact(root: Path) -> None:
    manifest = root / "NODE_ARTIFACT_MANIFEST.sha256"
    if not manifest.is_file():
        raise RuntimeError(f"node artifact lacks NODE_ARTIFACT_MANIFEST.sha256: {root}")
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(f"invalid node manifest line {line_no}: {line}") from exc
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"node artifact missing file: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"node artifact digest mismatch: {relative}")


def _copy_tree_files(source: Path, target: Path) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != sha256_file(path):
                raise RuntimeError(f"conflicting merged artifact: {destination}")
            continue
        shutil.copy2(path, destination)


def finalize_multinode(config: dict[str, Any], *, node_artifacts: Iterable[Path]) -> dict[str, Any]:
    internal = normalize_collect_config(config)
    nnodes, nproc, world = _topology(internal)
    if nnodes < 2:
        raise ConfigError("finalize phase requires launch.nnodes >= 2")
    artifacts = [Path(path).expanduser().resolve() for path in node_artifacts]
    if len(artifacts) != nnodes:
        raise ConfigError(f"expected exactly {nnodes} node artifacts, got {len(artifacts)}")
    base = Path(internal["output_dir"])
    imported = base / "imported_nodes"
    if imported.exists():
        shutil.rmtree(imported)
    for name in ("raw", "runner", "inputs", "fixtures", "bundles"):
        path = base / name
        if path.exists():
            shutil.rmtree(path)
    for name in (
        "trace_manifest.json",
        "finalize_summary.json",
        "multinode_collection_summary.json",
        "ARTIFACT_MANIFEST.sha256",
        "capture_runtime_config.json",
    ):
        path = base / name
        if path.exists():
            path.unlink()
    imported.mkdir(parents=True, exist_ok=True)

    expected_contract = collection_contract_digest(internal)
    checkpoint_digests: set[str] = set()
    seen_nodes: set[int] = set()
    node_summaries: list[dict[str, Any]] = []
    for archive in artifacts:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        with tempfile.TemporaryDirectory(prefix="rs-sim-node-artifact-") as tmp:
            root = Path(tmp)
            _safe_extract_zip(archive, root)
            _verify_extracted_artifact(root)
            summary_path = root / "node_collection_summary.json"
            if not summary_path.is_file():
                raise RuntimeError(f"node artifact lacks node_collection_summary.json: {archive}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            node_rank = int(summary["node_rank"])
            if node_rank in seen_nodes:
                raise RuntimeError(f"duplicate node_rank artifact: {node_rank}")
            seen_nodes.add(node_rank)
            if summary.get("contract_digest") != expected_contract:
                raise RuntimeError(
                    f"node {node_rank} contract digest differs: {summary.get('contract_digest')} != {expected_contract}"
                )
            if summary.get("checkpoint_digest"):
                checkpoint_digests.add(str(summary["checkpoint_digest"]))
            node_target = imported / f"node{node_rank:04d}"
            shutil.copytree(root, node_target)
            _copy_tree_files(root / "raw", base / "raw")
            _copy_tree_files(root / "runner", base / "runner")
            _copy_tree_files(root / "inputs", base / "inputs")
            node_summaries.append(summary)
    expected_nodes = set(range(nnodes))
    if seen_nodes != expected_nodes:
        raise RuntimeError(f"node artifacts cover {sorted(seen_nodes)}, expected {sorted(expected_nodes)}")
    if len(checkpoint_digests) > 1:
        raise RuntimeError(f"checkpoint identity mismatch across nodes: {sorted(checkpoint_digests)}")

    internal["capture"].pop("accepted_source_ranks", None)
    internal["output_dir"] = str(base)
    runtime_config = base / "capture_runtime_config.json"
    _write_json(runtime_config, {key: value for key, value in internal.items() if not str(key).startswith("__")})
    internal["__config_path"] = str(runtime_config)
    internal["__config_dir"] = str(base)
    acceptance = inspect_capture_artifacts(internal)
    if acceptance.get("status") != "PASS":
        raise RuntimeError(f"global capture acceptance failed: {acceptance}")
    qualification = _aggregate_qualification(base, world)
    internal["capture"]["performance_eligible"] = bool(qualification.get("performance_eligible", False))
    internal["capture"]["performance_qualification"] = qualification
    fixtures = build_fixtures_from_capture(internal)
    trace_manifest = _write_trace_manifest(internal, tuple(fixtures))
    trace_payload = json.loads(trace_manifest.read_text(encoding="utf-8"))
    trace_payload["performance_qualification"] = qualification
    trace_payload["performance_eligible"] = bool(qualification.get("performance_eligible", False))
    trace_payload["multinode"] = {
        "nnodes": nnodes,
        "nproc_per_node": nproc,
        "world_size": world,
        "artifact_owner_node_rank": int(config.get("multinode", {}).get("artifact_owner_node_rank", 0)),
        "node_ranks": sorted(seen_nodes),
    }
    _write_json(trace_manifest, trace_payload)
    bundle, bundle_sha = _bundle_trace(base, internal["capture"]["capture_id"])
    summary = {
        "schema_version": "RS_SIM_MULTINODE_FINALIZE_RESULT",
        "status": "PASS",
        "capture_id": internal["capture"]["capture_id"],
        "contract_digest": expected_contract,
        "checkpoint_digest": next(iter(checkpoint_digests), None),
        "nnodes": nnodes,
        "nproc_per_node": nproc,
        "world_size": world,
        "node_summaries": node_summaries,
        "capture_artifacts": acceptance,
        "performance_qualification": qualification,
        "trace_manifest": str(trace_manifest),
        "fixture_paths": [str(path) for path in fixtures],
        "artifact_bundle": str(bundle),
        "artifact_bundle_sha256_file": str(bundle_sha),
    }
    _write_json(base / "multinode_collection_summary.json", summary)
    return summary


def _aggregate_qualification(output_dir: Path, world_size: int) -> dict[str, Any]:
    reports = []
    for path in sorted((Path(output_dir) / "runner").glob("rank*_megatron_runner.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        qualification = payload.get("capture_qualification")
        if qualification:
            reports.append({"rank": payload.get("rank"), **qualification})
    expected = int(world_size)
    if len(reports) != expected:
        return {
            "status": "NOT_QUALIFIED",
            "performance_eligible": False,
            "reason": f"qualification reports={len(reports)}, expected={expected}",
            "rank_reports": reports,
        }
    max_overhead = max(float(item.get("global_max_overhead_ratio", 1.0)) for item in reports)
    threshold = min(float(item.get("threshold_ratio", 0.05)) for item in reports)
    return {
        "status": "PASS" if max_overhead <= threshold else "FAILED",
        "performance_eligible": bool(max_overhead <= threshold),
        "global_max_overhead_ratio": max_overhead,
        "threshold_ratio": threshold,
        "rank_reports": reports,
    }
