from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / "RS" / "src", ROOT / "RS", ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)
