"""Entry point: ``python -m wfci_gui [<workspace>/<cohort>/cohort.yaml]``.

Run it through conda, from the repository root:

    $env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
    & "$env:USERPROFILE\\anaconda3\\condabin\\conda.bat" run --no-capture-output -n wfci_gui python -m wfci_gui

Errors are not caught inside the GUI: ``sys.excepthook`` shows the traceback in a dialog
and appends it to ``<settings folder>/wfci_gui.log`` (GUI_DESIGN_GUIDELINES.md 6.4).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Edit this section to run from the editor without CLI arguments.
RUN_CONFIG = {
    "cohort": None,          # path to a cohort file; None = start with no cohort open
    "config_dir": None,      # None = %APPDATA%\wfci_gui (layout, shortcuts, setup checks, error log)
}


def default_config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "wfci_gui"


def install_excepthook(config_dir: Path, window_getter) -> None:
    from qtkit.qt import QtWidgets
    log_path = config_dir / "wfci_gui.log"

    def hook(kind, value, tb):
        text = "".join(traceback.format_exception(kind, value, tb))
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n--- {datetime.now().isoformat(timespec='seconds')}\n{text}")
        window = window_getter()
        if window is not None:
            window.controller.log_line.emit(f"[gui] ERROR: {value}")
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "Error", str(value) or kind.__name__)
        box.setDetailedText(text)
        box.setInformativeText(f"The full traceback is in {log_path}.")
        box.exec()

    sys.excepthook = hook


def main() -> None:
    parser = argparse.ArgumentParser(description="WFCI cohort pipeline GUI")
    parser.add_argument("cohort", nargs="?", default=RUN_CONFIG["cohort"])
    parser.add_argument("--config-dir", type=Path, default=RUN_CONFIG["config_dir"])
    args = parser.parse_args()
    config_dir = Path(args.config_dir) if args.config_dir else default_config_dir()

    from qtkit.qt import QtCore, QtWidgets
    from .window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("wfci_gui")
    app.setApplicationName("wfci_gui")
    holder: list = []
    install_excepthook(config_dir, lambda: holder[0] if holder else None)
    window = MainWindow(QtCore.QSettings("wfci_gui", "wfci_gui"), config_dir)
    holder.append(window)
    # First launch: the user chooses where the cohorts are kept (cancel quits).
    if not window.has_workspace() and not window.choose_workspace(first=True):
        sys.exit(0)
    if args.cohort:
        window.open_cohort(Path(args.cohort))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
