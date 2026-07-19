#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze one RouterSense server-suite stage")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--started-at", type=float, required=True)
    parser.add_argument("--returncode", type=int, required=True)
    parser.add_argument("--command-b64", required=True)
    args = parser.parse_args()

    stage = args.output_dir / "stages" / args.name
    artifacts = stage / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, object]] = []
    roots = (Path("outputs"), Path("RS/outputs"), Path("results"))
    threshold = args.started_at - 2.0
    for root in roots:
        if not root.is_dir():
            continue
        for source in sorted(p for p in root.rglob("*") if p.is_file()):
            try:
                modified = source.stat().st_mtime
            except OSError:
                continue
            if modified < threshold:
                continue
            target = artifacts / root.as_posix().replace("/", "__") / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append({
                "source": source.as_posix(),
                "path": target.relative_to(stage).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            })

    command = base64.b64decode(args.command_b64.encode("ascii")).decode("utf-8")
    payload = {
        "schema_version": "routersense.server.stage.v1",
        "name": args.name,
        "status": "passed" if args.returncode == 0 else "failed",
        "returncode": args.returncode,
        "command": command,
        "started_at_epoch": args.started_at,
        "finished_at_epoch": time.time(),
        "artifact_count": len(copied),
        "artifacts": copied,
    }
    (stage / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
