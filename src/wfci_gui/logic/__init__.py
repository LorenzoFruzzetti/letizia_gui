"""Pure logic of the GUI: no Qt imports anywhere in this package (tested with plain pytest).

New study scripts (scan_cohort.py, qc_cohort.py, ...) import from here too, which is
allowed: the dependency goes from the scripts to this package, never the other way.
"""

from pathlib import Path

# src/wfci_gui/logic/__init__.py -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
