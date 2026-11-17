"""ROI editor: ``python -m wfci_gui.roi_editor [--cohort cohorts/<name>.yaml]``.

Run from the repository root (the VS Code launch configs "ROI editor (wfci_gui.roi_editor)"
and "ROI editor (empty)" do). With no argument the window opens EMPTY and data is loaded
from File > Open cohort… / Open manifest… / Recent (roi_editor_spec.md D7, D8). With
``--cohort`` the cohort's pages open at once, as the main GUI's step 3 passes it; with
``--manifest`` a manifest's (its settings from the cohort that names it, else from the
flags below); with ``--folder`` every recording folder under those roots. Any flag below
overrides the cohort's value.

Reads: the recording folders (the first ``frames_per_parity`` odd and even frames, when a
page is first opened), ``<roi_set_dir>/alignment_images`` (cached odd / even images), the
ROI sets in the save and seed folders, ``outputs/channel_rectangle/params.json``.
Writes: ``<roi_set_dir>/<key>.yaml`` on save (with ``channel_rectangle`` for a page whose
rectangle was moved), ``<roi_set_dir>/alignment_images/<key>_odd.tif`` / ``_even.tif`` /
``_odd_even.json``, the rebuilt sets on "Rebuild saved sets", and Qt preferences (window
layout, shortcuts, Recent).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Edit this section to run the editor without passing CLI flags. None = nothing / the
# cohort file's value / the editor default.
RUN_CONFIG: dict[str, Any] = {
    "cohort": None,            # e.g. r"cohorts\botox_2023.yaml": open its pages at start
    "manifest": None,          # e.g. r"manifests\botox_2023_manifest_rectangle.csv"
    "folder": None,            # list of root folders; every subfolder with images is a page
    "pattern": None,           # image glob, e.g. "*.tif"
    "frame_order": None,       # "name" | "number"
    "rot90_k": None,           # quarter turns per frame (-1 = clockwise)
    "roi_set_dir": None,       # where ROI sets are saved
    "roi_seed_dir": None,      # read-only starting layouts
    "rebuilt_dir": None,       # where "Rebuild saved sets" writes (default: the seed folder)
    "start_from": None,        # one ROI set every page without its own set starts from
    "frames_per_parity": None,  # odd / even frames averaged into the panels and the decision (10)
    "config_dir": None,        # window layout / shortcuts (default: %APPDATA%\wfci_gui\roi_editor)
    "prefer_cli_args": True,
}
SETTING_KEYS = ("folder", "pattern", "frame_order", "rot90_k", "roi_set_dir", "roi_seed_dir",
                "rebuilt_dir", "start_from", "frames_per_parity")


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROI editor (odd | even panels).")
    p.add_argument("--cohort", default=defaults["cohort"], help="Cohort file to open at start.")
    p.add_argument("--manifest", default=defaults["manifest"], help="Manifest to open at start.")
    p.add_argument("--folder", nargs="+", default=defaults["folder"])
    p.add_argument("--pattern", default=defaults["pattern"])
    p.add_argument("--frame-order", choices=["name", "number"], default=defaults["frame_order"])
    p.add_argument("--rot90-k", dest="rot90_k", type=int, default=defaults["rot90_k"])
    p.add_argument("--roi-set-dir", default=defaults["roi_set_dir"])
    p.add_argument("--roi-seed-dir", default=defaults["roi_seed_dir"])
    p.add_argument("--rebuilt-dir", default=defaults["rebuilt_dir"])
    p.add_argument("--start-from", default=defaults["start_from"])
    p.add_argument("--frames-per-parity", type=int, default=defaults["frames_per_parity"])
    p.add_argument("--config-dir", default=defaults["config_dir"])
    return p.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{k: v for k, v in config.items() if k != "prefer_cli_args"})


def default_config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "wfci_gui" / "roi_editor"


def install_excepthook(config_dir: Path) -> None:
    """Errors are not caught in the editor: show the traceback, log it, keep running."""
    from qtkit.qt import QtWidgets
    log_path = config_dir / "roi_editor.log"

    def hook(kind, value, tb):
        text = "".join(traceback.format_exception(kind, value, tb))
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n--- {datetime.now().isoformat(timespec='seconds')}\n{text}")
        print(text, file=sys.stderr, flush=True)
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "Error", str(value) or kind.__name__)
        box.setDetailedText(text)
        box.setInformativeText(f"The full traceback is in {log_path}.")
        box.exec()

    sys.excepthook = hook


def open_at_start(window, args: argparse.Namespace) -> None:
    """--cohort / --manifest / --folder: load before the window is shown (no dialog)."""
    from . import sources
    from .logic import resolve_settings
    flags = {key: getattr(args, key) for key in SETTING_KEYS}
    if args.cohort:
        window.controller.open_cohort(args.cohort, flags)
    elif args.manifest:
        settings, cohort = sources.manifest_settings(args.manifest, flags)
        if settings is None:
            # No cohort names it: the defaults plus the flags, stated on the terminal.
            settings = resolve_settings(None, flags, need_folder=False)
            print(f"No cohort names {args.manifest}: reading it with {settings}", flush=True)
        window.controller.load(sources.load_manifest(args.manifest, settings, cohort))
    elif args.folder:
        window.controller.load(sources.load_folders(resolve_settings(None, flags)))


def main() -> None:
    args = build_runtime_args()
    from .logic import REPO_ROOT

    # The ROI-set folders and the scripts' own defaults are relative to the repository.
    os.chdir(REPO_ROOT)
    from qtkit.qt import QtCore, QtWidgets
    from .window import EditorWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setOrganizationName("wfci_gui")
    app.setApplicationName("wfci_roi_editor")
    config_dir = Path(args.config_dir) if args.config_dir else default_config_dir()
    install_excepthook(config_dir)
    window = EditorWindow(QtCore.QSettings("wfci_gui", "wfci_roi_editor"), config_dir)
    open_at_start(window, args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
