"""Test setup: this folder's ``src`` and ``scripts`` come first on ``sys.path``.

So every test imports this copy's code, whatever ran before it and whatever else the
env has installed, and a single test file runs the same as the whole suite.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for folder in ("scripts", "src"):
    path = str(ROOT / folder)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
