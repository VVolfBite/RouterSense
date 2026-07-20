from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("contract", "offline", "unit")
PATTERN = re.compile(r"(^|\n)\s*(from\s+experiments\b|import\s+experiments\b)")


def test_formal_tests_do_not_import_experiments_modules() -> None:
    matches: list[str] = []
    for directory in SCAN_DIRS:
        for path in (ROOT / directory).rglob("test_*.py"):
            content = path.read_text(encoding="utf-8")
            if PATTERN.search(content):
                matches.append(str(path.relative_to(ROOT.parent)))
    assert matches == []
