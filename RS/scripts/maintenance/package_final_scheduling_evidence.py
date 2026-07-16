from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from experiments.paper.result_bundle import build_result_bundle, sha256_file, write_json
from rs.core.contracts.provenance import compute_source_tree_digest

WINDOWS_ABSOLUTE = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]")
WINDOWS_TEMP = re.compile(r"(?i)[A-Z]:[\\/].*?(appdata[\\/]+local[\\/]+temp|temp)[\\/]")
POSIX_TEMP = re.compile(r"(?<![A-Za-z0-9])/(tmp|var/tmp)/")


@dataclass(frozen=True)
class GitIdentity:
    branch: str
    head: str
    remote: str
    status_short: str
    remote_commit: str


@dataclass(frozen=True)
class EvidenceValidation:
    expected_commit: str
    commit_identity_valid: bool
    forbidden_ephemeral_path_count: int
    external_resource_path_count: int
    scanned_files: int


@dataclass(frozen=True)
class PackageVerification:
    checksums_valid: bool
    artifact_index_valid: bool
    commit_identity_valid: bool
    source_digest_valid: bool
    portable_zip_paths: bool
    text_encoding_valid: bool
    oracle_control_valid: bool
    runtime_matrix_valid: bool
    audit_result_bundle_consistent: bool
    forbidden_ephemeral_path_count: int
    external_resource_path_count: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _repo_dir(repo_root: Path) -> Path:
    if (repo_root / ".git").exists():
        return repo_root
    if (repo_root / "RS" / ".git").exists():
        return repo_root / "RS"
    raise FileNotFoundError(f"could not resolve git repo under {repo_root}")


def _source_repo_dir(repo_root: Path) -> Path:
    repo = _repo_dir(repo_root)
    candidate = repo / "RS"
    if (candidate / "experiments" / "paper" / "cli.py").exists():
        return candidate
    return repo


def _run_git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo_root), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def resolve_git_identity(repo_root: Path) -> GitIdentity:
    repo = _repo_dir(repo_root)
    return GitIdentity(
        branch=_run_git(repo, "branch", "--show-current"),
        head=_run_git(repo, "rev-parse", "HEAD"),
        remote=_run_git(repo, "remote", "get-url", "origin"),
        status_short=_run_git(repo, "status", "--short"),
        remote_commit="",
    )


def assert_remote_synced(repo_root: Path, branch: str, expected_commit: str) -> str:
    repo = _repo_dir(repo_root)
    remote_commit = _run_git(repo, "rev-parse", f"origin/{branch}")
    if remote_commit != expected_commit:
        raise RuntimeError(f"remote commit mismatch: {remote_commit} != {expected_commit}")
    return remote_commit


def _iter_text_files(root: Path) -> list[Path]:
    suffixes = {".json", ".jsonl", ".txt", ".md", ".sha256", ".yaml", ".yml"}
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes],
        key=lambda item: item.relative_to(root).as_posix(),
    )


def _scan_for_commit_values(value: Any, expected_commit: str, *, hits: list[str], path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"commit", "final_commit", "commit_sha"} and child not in {None, ""} and str(child) != str(expected_commit):
                hits.append(f"{path}:{key}={child}")
            _scan_for_commit_values(child, expected_commit, hits=hits, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_for_commit_values(child, expected_commit, hits=hits, path=f"{path}[{index}]")


def _scan_text_for_paths(text: str) -> tuple[int, int]:
    forbidden = len(WINDOWS_TEMP.findall(text)) + len(POSIX_TEMP.findall(text)) + text.count("%TEMP%")
    external = 0
    for match in WINDOWS_ABSOLUTE.finditer(text):
        token = match.group(0)
        if WINDOWS_TEMP.search(token):
            continue
        external += 1
    return int(forbidden), int(external)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_evidence_root(evidence_root: Path, expected_commit: str) -> EvidenceValidation:
    hits: list[str] = []
    forbidden = 0
    external = 0
    scanned = 0
    for path in _iter_text_files(evidence_root):
        scanned += 1
        text = path.read_text(encoding="utf-8-sig")
        local_forbidden, local_external = _scan_text_for_paths(text)
        forbidden += local_forbidden
        external += local_external
        if path.suffix.lower() == ".json":
            _scan_for_commit_values(_load_json(path), expected_commit, hits=hits, path=path.relative_to(evidence_root).as_posix())
        elif path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate([line for line in text.splitlines() if line.strip()], start=1):
                _scan_for_commit_values(json.loads(line), expected_commit, hits=hits, path=f"{path.relative_to(evidence_root).as_posix()}:{line_number}")
    if hits:
        raise RuntimeError("evidence commit mismatch: " + "; ".join(hits[:20]))
    return EvidenceValidation(
        expected_commit=str(expected_commit),
        commit_identity_valid=True,
        forbidden_ephemeral_path_count=int(forbidden),
        external_resource_path_count=int(external),
        scanned_files=int(scanned),
    )


def copy_evidence_tree(evidence_root: Path, staging_root: Path) -> None:
    if staging_root.exists():
        shutil.rmtree(staging_root)
    shutil.copytree(evidence_root, staging_root)


def build_relative_artifact_index(staging_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted((p for p in staging_root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(staging_root).as_posix()):
        relative = path.relative_to(staging_root).as_posix()
        if relative == "checksums.sha256":
            continue
        result[relative] = relative
    return result


def write_relative_checksums(staging_root: Path) -> Path:
    lines: list[str] = []
    files = sorted(
        (path for path in staging_root.rglob("*") if path.is_file() and path.name != "checksums.sha256"),
        key=lambda path: path.relative_to(staging_root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(staging_root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}\n")
    output = staging_root / "checksums.sha256"
    output.write_text("".join(lines), encoding="utf-8", newline="\n")
    return output


def build_portable_zip(staging_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((p for p in staging_root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(staging_root).as_posix()):
            archive.write(path, arcname=path.relative_to(staging_root).as_posix())


def _verify_checksums(root: Path) -> bool:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.exists():
        return False
    if checksum_path.read_bytes().startswith(b"\xef\xbb\xbf"):
        return False
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if "  " not in line:
            return False
        digest, relative = line.split("  ", 1)
        if not relative or "\\" in relative or Path(relative).is_absolute():
            return False
        target = root / relative
        if not target.exists() or sha256_file(target) != digest:
            return False
    return True


def _portable_zip_paths(zip_path: Path) -> bool:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return all("\\" not in name and not name.startswith("/") for name in zf.namelist())


def _source_digest_valid(root: Path) -> bool:
    manifest_path = root / "source" / "source_manifest.json"
    archive_path = root / "source" / "canonical_source.zip"
    if not manifest_path.exists() or not archive_path.exists():
        return False
    manifest = _load_json(manifest_path)
    with tempfile.TemporaryDirectory(prefix="rs_pkg_source_") as tmp:
        unpack = Path(tmp)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(unpack)
        rs_root = unpack / "RS"
        if any(path.name == "__pycache__" for path in rs_root.rglob("*")):
            return False
        if any(path.suffix.lower() == ".pyc" for path in rs_root.rglob("*")):
            return False
        if (rs_root / ".git").exists() or (rs_root / ".pytest_cache").exists():
            return False
        return str(manifest.get("source_tree_digest")) == compute_source_tree_digest(rs_root)


def _sanitize_source_manifest(source_manifest_path: Path) -> None:
    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _sanitize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [_sanitize(child) for child in value]
        if isinstance(value, str) and (":\\" in value or ":/" in value or value.startswith("/tmp/")):
            return Path(value).name
        return value

    write_json(source_manifest_path, _sanitize(_load_json(source_manifest_path)))


def _derive_verification_status(fields: dict[str, bool], *, forbidden_ephemeral_path_count: int) -> str:
    return "PASS" if all(fields.values()) and int(forbidden_ephemeral_path_count) == 0 else "FAILED"


def _oracle_control_valid(root: Path) -> bool:
    path = root / "oracle" / "oracle_control_summary.json"
    if not path.exists():
        return False
    summary = _load_json(path)
    cases = dict(summary.get("cases", {}) or {})
    if summary.get("status") != "READY":
        return False
    if int(summary.get("dominance_violation_count", 1)) != 0:
        return False
    if not all(dict(cases.get(name, {})).get("status") == "PASS" for name in ("joint_advantage", "tie", "unsupported")):
        return False
    for record in list(summary.get("records", []) or []):
        if bool(record.get("comparable")) and not bool(record.get("coverage_valid")):
            return False
    return True


def _runtime_group_valid(root: Path, family: str, backend: str) -> bool:
    summary_path = root / "runtime" / family / backend / "formal_runner_summary.json"
    parity_path = root / "runtime" / family / backend / "parity.json"
    if not summary_path.exists() or not parity_path.exists():
        return False
    summary = _load_json(summary_path)
    parity = _load_json(parity_path)
    expected_policy = {
        "B": "B_barrier_criticality_core_independent",
        "U": "U_barrier_criticality_global_matching",
    }[family]
    return all(
        (
            str(summary.get("status")) == "passed",
            str(summary.get("requested_policy_id")) == expected_policy,
            bool(summary.get("policy_identity_match", False)),
            int(sum(int(row.get("submitted_task_count", 0) or 0) for row in summary.get("ranks", []))) == int(sum(int(row.get("completed_task_count", 0) or 0) for row in summary.get("ranks", []))),
            int(sum(int(row.get("unresolved_task_count", 0) or 0) for row in summary.get("ranks", []))) == 0,
            int(summary.get("fallback_count", -1)) == 0,
            summary.get("fallback_reasons") == [],
            bool(summary.get("native_fallback_invoked", True)) is False,
            str(summary.get("materialized_task_manifest_digest", "")) != "",
            str(summary.get("executed_task_manifest_digest", "")) != "",
            str(summary.get("materialized_task_manifest_digest")) == str(summary.get("executed_task_manifest_digest")),
            bool(summary.get("tensor_parity_pass", False)),
            bool(summary.get("full_reconstruction_parity_pass", False)),
            bool(parity.get("allclose", False)),
            bool(dict(parity.get("full_reconstruction_parity", {}) or {}).get("allclose", False)),
        )
    )


def _runtime_matrix_valid(root: Path) -> bool:
    return all(
        _runtime_group_valid(root, family, backend)
        for family in ("B", "U")
        for backend in ("phase_sync", "async_release")
    )


def _audit_result_bundle_consistent(root: Path) -> bool:
    capability_path = root / "audit" / "capability_matrix.json"
    bundle_path = root / "results" / "result_bundle.json"
    scheduling_path = root / "results" / "scheduling_summary.json"
    if not capability_path.exists() or not bundle_path.exists() or not scheduling_path.exists():
        return False
    capability = _load_json(capability_path)
    bundle = _load_json(bundle_path)
    scheduling = _load_json(scheduling_path)
    strict_ready = bool(dict(scheduling.get("same_core_pair_summary", {}) or {}).get("comparable"))
    runtime_ready = str(dict(capability.get("gloo_execution_wrapper", {}) or {}).get("status", "")) == "READY"
    oracle_ready = str(dict(capability.get("O_local", {}) or {}).get("status", "")).startswith("READY")
    return all(
        (
            bool(bundle.get("runtime_correctness_eligible", False)) == runtime_ready,
            bool(bundle.get("oracle_claim_eligible", False)) == oracle_ready,
            bool(bundle.get("strict_pair_claim_eligible", False)) == strict_ready,
        )
    )


def fresh_unpack_verify(zip_path: Path, expected_commit: str) -> PackageVerification:
    with tempfile.TemporaryDirectory(prefix="rs_pkg_verify_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        artifact_index = dict(_load_json(root / "results" / "result_bundle.json").get("artifact_index", {}) or {})
        artifact_index_valid = all((root / relative).exists() for relative in artifact_index.values())
        commit_validation = validate_evidence_root(root, expected_commit)
        checksums_valid = _verify_checksums(root)
        source_digest_valid = _source_digest_valid(root)
        portable_zip_paths = _portable_zip_paths(zip_path)
        text_encoding_valid = not (root / "checksums.sha256").read_bytes().startswith(b"\xef\xbb\xbf")
        oracle_control_valid = _oracle_control_valid(root)
        runtime_matrix_valid = _runtime_matrix_valid(root)
        audit_result_bundle_consistent = _audit_result_bundle_consistent(root)
        fields = {
            "checksums_valid": checksums_valid,
            "artifact_index_valid": artifact_index_valid,
            "commit_identity_valid": commit_validation.commit_identity_valid,
            "source_digest_valid": source_digest_valid,
            "portable_zip_paths": portable_zip_paths,
            "text_encoding_valid": text_encoding_valid,
            "oracle_control_valid": oracle_control_valid,
            "runtime_matrix_valid": runtime_matrix_valid,
            "audit_result_bundle_consistent": audit_result_bundle_consistent,
        }
        return PackageVerification(
            checksums_valid=checksums_valid,
            artifact_index_valid=artifact_index_valid,
            commit_identity_valid=commit_validation.commit_identity_valid,
            source_digest_valid=source_digest_valid,
            portable_zip_paths=portable_zip_paths,
            text_encoding_valid=text_encoding_valid,
            oracle_control_valid=oracle_control_valid,
            runtime_matrix_valid=runtime_matrix_valid,
            audit_result_bundle_consistent=audit_result_bundle_consistent,
            forbidden_ephemeral_path_count=commit_validation.forbidden_ephemeral_path_count,
            external_resource_path_count=commit_validation.external_resource_path_count,
            status=_derive_verification_status(fields, forbidden_ephemeral_path_count=commit_validation.forbidden_ephemeral_path_count),
        )


def _copy_source_archive(repo_root: Path, staging_root: Path) -> None:
    source_repo = _source_repo_dir(repo_root)
    packager = source_repo / "scripts" / "maintenance" / "package_source_archive.py"
    if not packager.exists():
        raise FileNotFoundError(f"could not locate source archive packager under {source_repo}")
    source_root = staging_root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    archive = source_root / "canonical_source.zip"
    subprocess.run(
        [sys.executable, str(packager), "--scope", "mainline", str(archive)],
        cwd=str(source_repo),
        text=True,
        capture_output=True,
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="rs_source_manifest_") as tmp:
        unpack = Path(tmp)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(unpack)
        shutil.copy2(unpack / "source_manifest.json", source_root / "source_manifest.json")
    _sanitize_source_manifest(source_root / "source_manifest.json")


def _write_git_files(staging_root: Path, identity: GitIdentity, *, commit: str, remote_commit: str) -> None:
    git_root = staging_root / "git"
    git_root.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "branch.txt": identity.branch,
        "starting_commit.txt": commit,
        "final_commit.txt": commit,
        "remote_commit.txt": remote_commit,
        "status.txt": identity.status_short,
        "remote.txt": identity.remote,
        "commits.txt": f"{commit}\n",
    }.items():
        (git_root / name).write_text(str(value) + ("" if str(value).endswith("\n") else "\n"), encoding="utf-8", newline="\n")


def _run_audit(source_repo: Path, staging_root: Path) -> None:
    audit_output = staging_root / "audit"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.paper.cli",
            "audit",
            "--config",
            "configs/official/paper/capability_audit.yaml",
            "--evidence-dir",
            str(staging_root),
            "--output-dir",
            str(audit_output),
        ],
        cwd=str(source_repo),
        text=True,
        capture_output=True,
        check=True,
    )
    results_root = staging_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    for name in ("scheduling_summary.json", "prediction_summary.json", "hiding_summary.json", "runtime_summary.json", "result_bundle.json"):
        src = audit_output / name
        if src.exists():
            shutil.copy2(src, results_root / name)
    consumed_config = audit_output / "consumed_config.json"
    if consumed_config.exists():
        consumed_config.unlink()


def _rebuild_bundle_and_manifest(
    *,
    staging_root: Path,
    branch: str,
    commit: str,
    remote_commit: str,
    package_verification: dict[str, Any] | None,
    remote_synced: bool,
    git_clean: bool,
) -> None:
    results_root = staging_root / "results"
    existing_bundle = _load_json(results_root / "result_bundle.json")
    scheduling_summary = _load_json(results_root / "scheduling_summary.json") if (results_root / "scheduling_summary.json").exists() else None
    prediction_summary = _load_json(results_root / "prediction_summary.json") if (results_root / "prediction_summary.json").exists() else None
    hiding_summary = _load_json(results_root / "hiding_summary.json") if (results_root / "hiding_summary.json").exists() else None
    runtime_summary = _load_json(results_root / "runtime_summary.json") if (results_root / "runtime_summary.json").exists() else None
    oracle_control_summary = _load_json(staging_root / "oracle" / "oracle_control_summary.json") if (staging_root / "oracle" / "oracle_control_summary.json").exists() else None
    artifact_index = build_relative_artifact_index(staging_root)
    rebuilt = build_result_bundle(
        branch=branch,
        commit=commit,
        config_digest=str(existing_bundle["run_identity"]["config_digest"]),
        claim_scope=str(existing_bundle["run_identity"]["claim_scope"]),
        status=str(existing_bundle["status"]),
        scheduling_summary=scheduling_summary,
        prediction_summary=prediction_summary,
        hiding_summary=hiding_summary,
        runtime_summary=runtime_summary,
        oracle_control_summary=oracle_control_summary,
        package_verification=package_verification,
        remote_synced=remote_synced,
        git_clean=git_clean,
        artifact_index=artifact_index,
    )
    write_json(results_root / "result_bundle.json", rebuilt)
    runtime_status = None if runtime_summary is None else runtime_summary.get("status")
    strict_pair_status = None if scheduling_summary is None else dict(scheduling_summary.get("same_core_pair_summary", {}) or {}).get("status")
    oracle_status = None if oracle_control_summary is None else oracle_control_summary.get("status")
    package_status = None if package_verification is None else package_verification.get("status")
    write_json(
        staging_root / "run_manifest.json",
        {
            "branch": branch,
            "starting_commit": commit,
            "final_commit": commit,
            "remote_commit": remote_commit,
            "git_clean": bool(git_clean),
            "remote_synced": bool(remote_synced),
            "oracle_control_status": oracle_status,
            "strict_pair_status": strict_pair_status,
            "runtime_matrix_status": runtime_status,
            "package_verification_status": package_status,
            "final_status": rebuilt["status"],
        },
    )


def _prepare_staging(
    *,
    repo_root: Path,
    source_repo: Path,
    evidence_root: Path,
    staging_root: Path,
    identity: GitIdentity,
    commit: str,
    remote_commit: str,
) -> None:
    copy_evidence_tree(evidence_root, staging_root)
    _copy_source_archive(repo_root, staging_root)
    _write_git_files(staging_root, identity, commit=commit, remote_commit=remote_commit)
    _run_audit(source_repo, staging_root)
    _rebuild_bundle_and_manifest(
        staging_root=staging_root,
        branch=identity.branch,
        commit=commit,
        remote_commit=remote_commit,
        package_verification={"status": "FAILED"},
        remote_synced=True,
        git_clean=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    evidence_root = Path(args.evidence_root)
    output_dir = Path(args.output_dir)
    repo = _repo_dir(repo_root)
    source_repo = _source_repo_dir(repo_root)
    identity = resolve_git_identity(repo_root)
    if identity.branch != args.expected_branch or identity.head != args.expected_commit or identity.status_short.strip():
        raise SystemExit("git identity mismatch or dirty repo")
    remote_commit = assert_remote_synced(repo_root, args.expected_branch, args.expected_commit)
    validate_evidence_root(evidence_root, args.expected_commit)

    with tempfile.TemporaryDirectory(prefix="rs_final_stage_") as tmp:
        staging_root = Path(tmp) / "pkg"
        _prepare_staging(
            repo_root=repo_root,
            source_repo=source_repo,
            evidence_root=evidence_root,
            staging_root=staging_root,
            identity=identity,
            commit=args.expected_commit,
            remote_commit=remote_commit,
        )
        write_json(staging_root / "package_verification.json", {"status": "FAILED"})
        write_relative_checksums(staging_root)
        candidate_zip = output_dir / f"RouterSense_final_scheduling_evidence_{args.expected_commit[:8]}_candidate.zip"
        output_dir.mkdir(parents=True, exist_ok=True)
        build_portable_zip(staging_root, candidate_zip)
        candidate_verification = fresh_unpack_verify(candidate_zip, args.expected_commit)
        if candidate_verification.status != "PASS":
            if candidate_zip.exists():
                candidate_zip.unlink()
            raise SystemExit("candidate package verification failed")

        write_json(staging_root / "package_verification.json", candidate_verification.to_dict())
        _rebuild_bundle_and_manifest(
            staging_root=staging_root,
            branch=identity.branch,
            commit=args.expected_commit,
            remote_commit=remote_commit,
            package_verification=candidate_verification.to_dict(),
            remote_synced=True,
            git_clean=True,
        )
        write_relative_checksums(staging_root)

        zip_name = f"RouterSense_final_scheduling_evidence_{args.expected_commit[:8]}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        final_zip = output_dir / zip_name
        for old in output_dir.iterdir():
            if old == final_zip:
                continue
            if old.is_file():
                old.unlink()
            else:
                shutil.rmtree(old)
        build_portable_zip(staging_root, final_zip)
        final_verification = fresh_unpack_verify(final_zip, args.expected_commit)
        internal_verification = _load_json(staging_root / "package_verification.json")
        if final_verification.to_dict() != internal_verification or final_verification.status != "PASS":
            if final_zip.exists():
                final_zip.unlink()
            raise SystemExit("final package verification failed")
        print(
            json.dumps(
                {
                    "zip_path": str(final_zip),
                    "zip_sha256": sha256_file(final_zip),
                    "verification": final_verification.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
