#!/usr/bin/env python3
from __future__ import annotations

"""Validate locally collected deployment artifacts and emit one summary."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.topology import load_inventory


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _explicit_failure(payload: dict[str, Any]) -> str | None:
    failure_markers = ("fail", "error", "invalid", "reject", "blocked", "timeout", "abort")
    for key in ("status", "correctness_status", "validation_status", "execution_status"):
        value = str(payload.get(key, "")).strip().lower()
        if value and any(marker in value for marker in failure_markers):
            return f"{key}={value}"
    if int(payload.get("fallback_count", 0) or 0) > 0 and bool(payload.get("formal_execution_expected", False)):
        return "unexpected fallback_count"
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory = load_inventory(Path(args.inventory))
    root = Path(args.input_dir or (ROOT / "outputs" / "deployment" / args.run_id)).resolve()
    payload: dict[str, Any] = {
        "schema_version": "routersense.deploy.collected_summary.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "apply_mode": bool(args.apply),
        "run_id": str(args.run_id),
        "input_dir": str(root),
        "nodes": [],
    }
    if not args.apply:
        payload.update({"status": "DRY_RUN", "result_ready": False})
        print(json.dumps(payload, indent=2))
        return 0

    failures: list[str] = []
    master_has_result = False
    for node in inventory.nodes:
        node_root = root / str(node.name)
        exit_files = sorted(node_root.rglob("*.exit")) if node_root.is_dir() else []
        exit_codes: list[int] = []
        for path in exit_files:
            try:
                exit_codes.append(int(path.read_text(encoding="utf-8").strip().splitlines()[-1]))
            except Exception:
                exit_codes.append(255)
        json_files = sorted(node_root.rglob("*.json")) if node_root.is_dir() else []
        result_files = [
            path for path in json_files
            if path.name in {"summary.json", "result_bundle.json", "comparison_report.json"}
        ]
        explicit_failures: list[dict[str, str]] = []
        for path in json_files:
            row = _read_json(path)
            if row is None:
                continue
            reason = _explicit_failure(row)
            if reason:
                explicit_failures.append({"path": str(path.relative_to(node_root)), "reason": reason})
        if str(node.name) == str(inventory.rendezvous.master_node) and result_files:
            master_has_result = True
        status = "PASS"
        if not node_root.is_dir():
            status = "MISSING"
            failures.append(f"{node.name}: collected directory missing")
        elif not exit_files:
            status = "FAIL"
            failures.append(f"{node.name}: remote exit marker missing")
        elif any(code != 0 for code in exit_codes):
            status = "FAIL"
            failures.append(f"{node.name}: non-zero remote exit code {exit_codes}")
        elif explicit_failures:
            status = "FAIL"
            failures.append(f"{node.name}: explicit failure artifacts")
        payload["nodes"].append(
            {
                "node_name": str(node.name),
                "status": status,
                "exit_files": [str(path.relative_to(node_root)) for path in exit_files],
                "exit_codes": exit_codes,
                "json_artifact_count": len(json_files),
                "result_files": [str(path.relative_to(node_root)) for path in result_files],
                "explicit_failures": explicit_failures,
            }
        )
    if not master_has_result:
        failures.append("rendezvous master result_bundle/summary/comparison_report missing")
    payload["failures"] = failures
    payload["result_ready"] = not failures
    payload["status"] = "PASS" if not failures else "FAIL"
    root.mkdir(parents=True, exist_ok=True)
    (root / "deployment_result_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
