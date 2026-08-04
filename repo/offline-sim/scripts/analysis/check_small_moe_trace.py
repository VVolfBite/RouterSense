#!/usr/bin/env python3
"""Validate and summarize one rs-sim trace manifest/bundle for server smoke runs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import zipfile
from pathlib import Path

from rs_sim.trace.io.serialization import load_fixture
from rs_sim.trace.schema.validation import validate_fixture


def _resolve_manifest(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.suffix.lower() == ".zip":
        temp = tempfile.TemporaryDirectory(prefix="rs-trace-check-")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(temp.name)
        candidates = list(Path(temp.name).rglob("trace_manifest.json"))
        if len(candidates) != 1:
            temp.cleanup()
            raise RuntimeError(f"bundle must contain exactly one trace_manifest.json, found {len(candidates)}")
        return candidates[0], temp
    if path.is_dir():
        path = path / "trace_manifest.json"
    return path, None


def _transpose(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(matrix[src][dst]) for src in range(len(matrix))) for dst in range(len(matrix)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="trace_manifest.json, trace directory, or trace bundle zip")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    manifest_path, temp = _resolve_manifest(args.trace.expanduser().resolve())
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "RS_SIM_TRACE_MANIFEST":
            raise RuntimeError(f"not an RS_SIM_TRACE_MANIFEST: {manifest_path}")
        rows = []
        failures = []
        for entry in manifest.get("fixtures", []):
            fixture_path = (manifest_path.parent / str(entry["path"])).resolve()
            fixture = load_fixture(fixture_path)
            validation = validate_fixture(fixture)
            layer_ids = [int(window.layer_id) for window in fixture.windows]
            consecutive = all(layer_ids[i + 1] == layer_ids[i] + 1 for i in range(len(layer_ids) - 1))
            compute_vectors = [
                list(window.local_compute.dispatch_release_to_combine_source_ready_ns)
                for window in fixture.windows
            ]
            compute_flat = [int(value) for vector in compute_vectors for value in vector]
            transpose_ok = all(window.combine_rows == _transpose(window.dispatch_rows) for window in fixture.windows)
            mass_ok = all(
                sum(map(sum, window.dispatch_rows)) == sum(map(sum, window.combine_rows))
                for window in fixture.windows
            )
            row = {
                "fixture": fixture.fixture_id,
                "validation_status": validation.get("status"),
                "world_size": fixture.world_size,
                "window_count": len(fixture.windows),
                "layer_ids": layer_ids,
                "consecutive_layers": consecutive,
                "dispatch_combine_transpose": transpose_ok,
                "routing_mass_closed": mass_ok,
                "compute_samples": len(compute_flat),
                "compute_nonzero_fraction": (
                    sum(value > 0 for value in compute_flat) / len(compute_flat) if compute_flat else 0.0
                ),
                "compute_p50_ns": statistics.median(compute_flat) if compute_flat else 0,
                "compute_max_ns": max(compute_flat) if compute_flat else 0,
                "truth_digest": fixture.truth_digest(),
            }
            rows.append(row)
            if validation.get("status") != "PASS":
                failures.append(f"{fixture.fixture_id}: validation={validation.get('status')}")
            if len(fixture.windows) < 2 or not consecutive:
                failures.append(f"{fixture.fixture_id}: Current-P12 needs >=2 consecutive layers")
            if not transpose_ok or not mass_ok:
                failures.append(f"{fixture.fixture_id}: dispatch/combine routing closure failed")
            if not compute_flat or not any(value > 0 for value in compute_flat):
                failures.append(f"{fixture.fixture_id}: local compute capture is entirely zero")

        report = {
            "schema_version": "RS_SIM_SMALL_MOE_TRACE_CHECK",
            "status": "PASS" if rows and not failures else "FAIL",
            "manifest": str(manifest_path),
            "fixture_count": len(rows),
            "fixtures": rows,
            "failures": failures,
        }
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        print(text, end="")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(text, encoding="utf-8")
        return 0 if report["status"] == "PASS" else 2
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
