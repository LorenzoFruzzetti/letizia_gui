"""Environment checks for the Setup page, run INSIDE the wfci_gui env.

    conda run --no-capture-output -n wfci_gui python -u -m wfci_gui.setup_check --out <json> [--tests]

Writes one JSON file the GUI reads for the step-0 status:
    {"checked": "...", "checks": [{"name": ..., "ok": ..., "detail": ...}, ...]}

Each package is imported in a child process, so one broken package (a DLL that fails
to load) is reported as a failed check instead of stopping the rest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Import name -> what it is for. statsmodels and psutil are pip-installed.
PACKAGES = {
    "numpy": "arrays", "scipy": "statistics", "pandas": "tables", "tifffile": "raw TIFFs",
    "matplotlib": "figures", "yaml": "cohort and ROI files", "pytest": "tests",
    "pyqtgraph": "GUI plots", "PySide6": "GUI", "pytestqt": "GUI tests",
    "statsmodels": "mixed models", "psutil": "memory checks",
}
# A BLAS call: an env whose MKL DLLs are not on PATH dies here, not at import.
BLAS_PROBE = "import numpy as np; a = np.random.rand(22, 300); a @ a.T; np.corrcoef(a)"


def run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=REPO_ROOT)


def check_packages() -> list[dict]:
    checks = []
    for name, purpose in PACKAGES.items():
        result = run_python(f"import {name}; print(getattr({name}, '__version__', 'ok'))")
        detail = result.stdout.strip() if result.returncode == 0 else (
            (result.stderr.strip().splitlines() or [f"exit code {result.returncode}"])[-1])
        checks.append(dict(name=f"package {name} ({purpose})", ok=result.returncode == 0, detail=detail))
    blas = run_python(BLAS_PROBE)
    checks.append(dict(name="numpy BLAS works", ok=blas.returncode == 0,
                       detail="ok" if blas.returncode == 0 else
                       f"exit code {blas.returncode:#x}: run through conda run"))
    return checks


def check_wfci_checkout() -> dict:
    result = run_python("import wfci; print(wfci.__file__)")
    location = result.stdout.strip()
    expected = REPO_ROOT / "src" / "wfci"
    ok = result.returncode == 0 and Path(location).resolve().parent == expected.resolve()
    detail = location if ok else (
        f"wfci is loaded from {location or '(import failed)'}, not {expected} "
        "(an editable install may point at another checkout): run setup_env.ps1 or set PYTHONPATH")
    return dict(name="wfci from this checkout", ok=ok, detail=detail)


def check_tests() -> dict:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    last = (result.stdout.strip().splitlines() or ["no output"])[-1]
    return dict(name="tests pass", ok=result.returncode == 0, detail=last)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the wfci_gui env for the GUI's Setup page.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tests", action="store_true", help="Also run the test suite (minutes).")
    args = parser.parse_args()
    checks = [dict(name="python", ok=sys.version_info[:2] >= (3, 10), detail=sys.version.split()[0])]
    checks += check_packages()
    checks.append(check_wfci_checkout())
    if args.tests:
        print("running the test suite ...", flush=True)
        checks.append(check_tests())
    for check in checks:
        print(f"{'PASS' if check['ok'] else 'FAIL'}  {check['name']}: {check['detail']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checked=datetime.now().isoformat(timespec="seconds"), checks=checks)
    # Keep an earlier test result when this run did not include the tests.
    if not args.tests and args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        payload["checks"] += [c for c in previous.get("checks", []) if c["name"] == "tests pass"]
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [c for c in checks if not c["ok"]]
    print(f"{len(checks) - len(failed)} of {len(checks)} checks pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
