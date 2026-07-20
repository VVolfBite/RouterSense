from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


FORBIDDEN = [
    "rs.runtime.distributed_ep",
    "rs.online.olmoe_ep.ws2_native_ep",
    "RouteItem",
    "RankManifest",
]


def test_no_legacy_runtime_imports() -> None:
    roots = [
        REPO_ROOT / "src/rs/runtime/online/megatron_ep",
        REPO_ROOT / "experiments/online",
    ]
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name == "test_no_legacy_runtime_imports.py":
                continue
            text = path.read_text(encoding="utf-8")
            for needle in FORBIDDEN:
                assert needle not in text, f"{needle} found in {path}"
