from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / "RS" / "src", ROOT / "RS", ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)
