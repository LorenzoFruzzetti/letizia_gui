"""Interactive midline-rectangle viewer: which interleaved group (odd / even) is GCaMP?

Spec: channel_rectangle_viewer_spec.md (repo root). Lab script only; `wfci` is not touched.

The odd and even mean images of one recording are shown side by side ("odd" | "even",
no channel label, to avoid bias), with a rectangle drawn along the Bregma-Lambda axis
(superior sagittal sinus). Hypothesis to check by eye first: the reflectance channel
shows more vessel contrast in that strip than GCaMP. The top bar shows which parity
each system calls GCaMP:

    library    the pixel cache's `channel_order` (pixels_meta_full.npz, ROI-union median)
    manifest   manifests/channel_assignment_2024.csv (whole-frame median)
    rectangle  live, from the current rectangle and statistic on the two mean images

Geometry (256 grid, continuous coordinates, pixel (r, c) covers [r, r+1) x [c, c+1)):
B = Bregma, L = Lambda = B + 2 * lambda_row_offset along the axis, d = |L - B|,
u = unit vector from L toward B. The rectangle is centred on the axis, starts at
L + start_offset_factor * d * u, is length_factor * d long and width_factor * d wide.
Rotation is locked to the axis. Parameters are factors of d, so one saved set applies
to every recording at its own scale.

Inputs (defaults in RUN_CONFIG, all relative to the repo root):
    pixel_data_2024/<day>_<unit>/<t#>/pixels_meta_full.npz   mean_f, mean_r, channel_order, ...
    roi_sets/rebuilt/<day>_<unit>_<t#>.yaml                   Bregma, Lambda offset, angle, flip
    manifests/channel_assignment_2024.csv                     recording_key -> channel_order
                                                              ('none': no table, column unassigned)
Rule and default parameters: channel_rectangle.py (defaults = the params.json below).
Output: outputs/channel_rectangle/params.json (Save) =
    {length_factor, width_factor, start_offset_factor, statistic}

Run (wfci_gui env; CLAUDE.md 9.17: always through conda run):
    & "$env:USERPROFILE\\anaconda3\\condabin\\conda.bat" run -n wfci_gui python scripts/channel_rectangle_viewer.py
or press Run in the editor (RUN_CONFIG below). CLI flags override RUN_CONFIG.

Keys: Up/Down in the table or PgUp/PgDn = previous/next recording; Ctrl+, parameters;
Ctrl+S / Ctrl+O save / load params.json; Ctrl+L shared contrast scale; Ctrl+0 fit view.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# qtkit lives in src/qtkit; the editable install puts src/ on sys.path, but a plain
# editor run from a fresh checkout should work too.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyqtgraph as pg  # noqa: E402
from qtkit import ActionSpec, DataFrameModel, FieldHint, ParameterDialog, ShellWindow  # noqa: E402
from qtkit.qt import QtCore, QtWidgets  # noqa: E402

# Edit this section to run the viewer without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "day": "240118",  # 2024 cohort day 3 (build_design_2024.py: 240115 + 3)
    "units": ["SP20_000", "SP20_R00", "SP20_RR0", "SP20_RRR",
              "SP21_000", "SP21_R00", "SP21_RR0", "SP21_RRR"],
    "recordings": ["t1", "t2", "t3", "t4", "t5"],
    "pixel_root": "pixel_data_2024",
    "roi_set_dir": "roi_sets/rebuilt",
    "assignment": "manifests/channel_assignment_2024.csv",
    "params": "outputs/channel_rectangle/params.json",
    # Start from params.json when it exists (otherwise the defaults below).
    "load_saved_params": True,
    "prefer_cli_args": True,
}

# Rule, parameters and geometry live in channel_rectangle.py (repo root), shared with
# roi_editor.py, which draws the same rectangle and states its decision on every page.
from channel_rectangle import (  # noqa: E402,F401  (re-exported for the tests)
    PARITIES, STATISTICS, UNASSIGNED, RectangleParams, agreement_marks, agreement_summary,
    axis_frame, bregma_point, gcamp_parity_from_order, gcamp_parity_from_statistic, lambda_point,
    load_params, make_params, rectangle_corners, rectangle_fractions, rectangle_mask,
    rectangle_statistic, save_params, value_ratio,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Recording:
    key: str            # <day>_<unit>/<t#>, the manifest's recording_key
    unit: str
    recording: str
    odd: np.ndarray     # (256, 256) mean image of the odd frames
    even: np.ndarray    # (256, 256) mean image of the even frames
    bregma: np.ndarray  # (row, col), continuous 256 grid
    lambda_: np.ndarray
    library_order: str  # cache meta channel_order
    manifest_order: str  # gcamp_first | emo_first | "" (no label); derived for 2023, see below
    verdict: str        # cache meta channel_verdict
    level_region: str   # cache meta channel_level_region


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def manifest_order_2023(row: pd.Series, odd: np.ndarray, even: np.ndarray) -> str:
    """channel_order from a CLAUDE.md 9.67 row (manifests/channel_assignment_2023.csv).

    That file has no channel_order: `verdict` (correct / swapped / inseparable) is relative
    to the OCTOBER dumps' labelling, which later reruns changed for some recordings (9.71).
    Its `f_over_r_full_field` is median(old mean_f) / median(old mean_r) over the full
    256 image, so comparing it with median(odd) / median(even) of today's images tells
    which parity the old "f" was (checked: exactly one of the two matches on all 96).
    """
    if row["verdict"] == "inseparable":
        return ""
    old_ratio = float(row["f_over_r_full_field"])
    ratio = float(np.median(odd) / np.median(even))
    # The CSV value is rounded to 3 decimals.
    old_f_is_odd, old_f_is_even = abs(ratio - old_ratio) < 1e-3, abs(1 / ratio - old_ratio) < 1e-3
    if old_f_is_odd == old_f_is_even:
        raise ValueError(f"{row.name}: f_over_r_full_field {old_ratio} matches neither or both of "
                         f"odd/even {ratio:.4f} and its inverse")
    old_f = "odd" if old_f_is_odd else "even"
    if row["verdict"] == "correct":
        gcamp = old_f
    elif row["verdict"] == "swapped":
        gcamp = "even" if old_f == "odd" else "odd"
    else:
        raise ValueError(f"{row.name}: unknown verdict {row['verdict']!r}")
    return "gcamp_first" if gcamp == "odd" else "emo_first"


def load_recordings(args: argparse.Namespace) -> list[Recording]:
    """Read every <day>_<unit>/<t#>: cache meta, ROI set, manifest label. Missing = error.

    The manifest is either a `channel_order` table (2024) or the 2023 9.67 verdict table.
    """
    # "none": a cohort without an assignment table; the manifest column stays unassigned.
    no_manifest = args.assignment is None or str(args.assignment).lower() == "none"
    manifest = (pd.DataFrame(columns=["recording_key", "channel_order"]) if no_manifest else
                pd.read_csv(resolve(args.assignment), dtype=str, keep_default_na=False))
    manifest = manifest.set_index("recording_key", verify_integrity=True)
    is_2023_schema = "channel_order" not in manifest.columns
    if is_2023_schema and not {"verdict", "f_over_r_full_field"} <= set(manifest.columns):
        raise ValueError(f"{args.assignment}: needs channel_order, or verdict + f_over_r_full_field")
    recordings = []
    for unit in args.units:
        for name in args.recordings:
            key = f"{args.day}_{unit}/{name}"
            folder = resolve(args.pixel_root) / f"{args.day}_{unit}" / name
            # A labelled dump has pixels_meta_full.npz; a not-interleaved one only the
            # odd/even sidecar (CLAUDE.md 9.72). Exactly one of the two must exist.
            labelled_path = folder / "pixels_meta_full.npz"
            odd_even_path = folder / "pixels_meta_odd_even_full.npz"
            if labelled_path.is_file() == odd_even_path.is_file():
                raise FileNotFoundError(f"{folder}: expected exactly one of {labelled_path.name} "
                                        f"and {odd_even_path.name}")
            meta = np.load(labelled_path if labelled_path.is_file() else odd_even_path)
            roi_set_path = resolve(args.roi_set_dir) / f"{args.day}_{unit}_{name}.yaml"
            roi_set = yaml.safe_load(roi_set_path.read_text(encoding="utf-8"))
            # The ROI set must be the one the cache was cropped around.
            for item in ("bregma_row", "bregma_col"):
                if int(roi_set[item]) != int(meta[item]):
                    raise ValueError(f"{roi_set_path}: {item} {roi_set[item]} != cache {int(meta[item])}")
            if key not in manifest.index and not no_manifest:
                raise KeyError(f"{key} missing from {args.assignment}")
            library_order = str(meta["channel_order"])
            library_parity = gcamp_parity_from_order(library_order)
            if library_parity == UNASSIGNED:
                odd = np.asarray(meta["mean_odd"], dtype=np.float64)    # (256, 256)
                even = np.asarray(meta["mean_even"], dtype=np.float64)  # (256, 256)
            else:
                # mean_f / mean_r are already labelled; channel_order says which parity each is.
                mean_f = np.asarray(meta["mean_f"], dtype=np.float64)  # (256, 256) GCaMP per the cache
                mean_r = np.asarray(meta["mean_r"], dtype=np.float64)  # (256, 256) reflectance per the cache
                odd, even = (mean_f, mean_r) if library_parity == "odd" else (mean_r, mean_f)
            bregma = bregma_point(int(roi_set["bregma_row"]), int(roi_set["bregma_col"]))
            recordings.append(Recording(
                key=key, unit=unit, recording=name, odd=odd, even=even, bregma=bregma,
                lambda_=lambda_point(bregma, float(roi_set["lambda_row_offset"]),
                                     float(roi_set.get("lambda_angle_deg", 0.0)),
                                     bool(roi_set.get("lambda_above_bregma", False))),
                library_order=library_order,
                manifest_order=("" if no_manifest else
                                manifest_order_2023(manifest.loc[key], odd, even) if is_2023_schema
                                else str(manifest.loc[key, "channel_order"])),
                verdict=str(meta["channel_verdict"]),
                # Footer text only; caches written before the region was recorded lack it.
                level_region=(str(meta["channel_level_region"]) if "channel_level_region" in meta.files
                              else "region not recorded"),
            ))
    return recordings


def predict(recording: Recording, params: RectangleParams) -> dict[str, Any]:
    """Every system's GCaMP parity plus the rectangle's raw statistic per parity."""
    corners = rectangle_corners(recording.bregma, recording.lambda_, params.length_factor,
                                params.width_factor, params.start_offset_factor)
    odd_value = rectangle_statistic(recording.odd, corners, params.statistic)
    even_value = rectangle_statistic(recording.even, corners, params.statistic)
    labels = {
        "library": gcamp_parity_from_order(recording.library_order),
        "manifest": gcamp_parity_from_order(recording.manifest_order),
        "rectangle": gcamp_parity_from_statistic(odd_value, even_value, params.statistic),
    }
    return {"labels": labels, "marks": agreement_marks(labels), "odd": odd_value, "even": even_value,
            "ratio": value_ratio(odd_value, even_value)}


def prediction_table(recordings: list[Recording], params: RectangleParams) -> pd.DataFrame:
    rows = []
    for recording in recordings:
        result = predict(recording, params)
        rows.append({"unit": recording.unit, "t#": recording.recording,
                     "library": result["labels"]["library"], "manifest": result["labels"]["manifest"],
                     "rectangle": result["labels"]["rectangle"], "ratio": round(result["ratio"], 3),
                     "agree": agreement_summary(result["labels"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class Controller:
    """Every ActionSpec method; state lives here, the window only draws it."""

    def __init__(self, window: "ViewerWindow"):
        self.window = window

    def previous_recording(self):
        self.window.select_row(self.window.current - 1)

    def next_recording(self):
        self.window.select_row(self.window.current + 1)

    def parameters(self):
        dialog = ParameterDialog(
            self.window, RectangleParams, asdict(self.window.params), make_params,
            {"statistic": FieldHint("choice", choices=STATISTICS),
             **{name: FieldHint(decimals=3) for name in ("length_factor", "width_factor", "start_offset_factor")}},
            title="Rectangle parameters",
            note="Lengths are factors of d = |Lambda - Bregma| of each recording. "
                 "median_level: lower median = GCaMP. centre_flank: |median(middle third) / "
                 "median(outer thirds) - 1|, larger = reflectance. spatial_cv: std / mean, "
                 "larger = reflectance.",
        )
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.window.set_params(dialog.value, refresh_table=True)

    def reset_rectangle(self):
        self.window.set_params(RectangleParams(statistic=self.window.params.statistic), refresh_table=True)

    def save(self):
        save_params(self.window.params_path, self.window.params)
        self.window.notify(f"Saved {self.window.params_path}")

    def load(self):
        self.window.set_params(load_params(self.window.params_path), refresh_table=True)
        self.window.notify(f"Loaded {self.window.params_path}")

    def shared_scale(self):
        self.window.update_levels()

    def fit_view(self):
        self.window.views["odd"].autoRange()

    def shortcuts(self):
        self.window.edit_shortcuts()

    def reset_commands(self):
        self.window.reset_commands()

    def quit(self):
        self.window.close()


ACTIONS = (
    ActionSpec("rect.previous", "Previous recording", "PgUp", "previous_recording", "SP_ArrowUp"),
    ActionSpec("rect.next", "Next recording", "PgDown", "next_recording", "SP_ArrowDown"),
    ActionSpec("rect.parameters", "Rectangle parameters…", "Ctrl+,", "parameters", "SP_FileDialogDetailedView"),
    ActionSpec("rect.reset", "Reset rectangle", "", "reset_rectangle", "SP_DialogResetButton"),
    ActionSpec("rect.shared_scale", "Shared contrast", "Ctrl+L", "shared_scale", "SP_DesktopIcon"),
    ActionSpec("rect.fit", "Fit view", "Ctrl+0", "fit_view", "SP_TitleBarMaxButton"),
    ActionSpec("rect.save", "Save parameters", "Ctrl+S", "save", "SP_DialogSaveButton"),
    ActionSpec("rect.load", "Load parameters", "Ctrl+O", "load", "SP_DialogOpenButton"),
    ActionSpec("settings.shortcuts", "Keyboard shortcuts…", "", "shortcuts", "SP_ComputerIcon"),
    ActionSpec("settings.reset", "Reset commands to defaults", "", "reset_commands", "SP_DialogResetButton"),
    ActionSpec("app.quit", "Quit", "Ctrl+Q", "quit", "SP_DialogCloseButton"),
)
LAYOUT = {
    "menus": {
        "File": ["rect.save", "rect.load", "-", "app.quit"],
        "Rectangle": ["rect.parameters", "rect.reset", "-", "rect.previous", "rect.next"],
        "Display": ["rect.shared_scale", "rect.fit"],
        "Settings": ["settings.shortcuts", "settings.reset"],
    },
    "toolbars": {"main": ["rect.previous", "rect.next", "-", "rect.parameters", "rect.reset",
                          "-", "rect.shared_scale", "rect.fit", "-", "rect.save", "rect.load"]},
    "shortcuts": {},
}


class ViewerWindow(ShellWindow):
    def __init__(self, recordings: list[Recording], params: RectangleParams, params_path: Path,
                 settings: QtCore.QSettings, config_dir: Path):
        super().__init__(settings, config_dir)
        self.setWindowTitle("Channel rectangle viewer")
        self.resize(1500, 850)
        self.recordings = recordings
        self.params = params
        self.params_path = params_path
        self.current = 0
        # True while the code (not the mouse) moves the ROIs, so their signals are ignored.
        self.syncing = False
        self.controller = Controller(self)
        self.install_commands(ACTIONS, LAYOUT, self.controller)
        self.registry.actions["rect.shared_scale"].setCheckable(True)

        # Central: prediction bar above two linked image panels.
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.prediction_label = QtWidgets.QLabel()
        self.prediction_label.setTextFormat(QtCore.Qt.RichText)
        self.prediction_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.prediction_label)
        panels = QtWidgets.QHBoxLayout()
        layout.addLayout(panels, stretch=1)
        self.views, self.images, self.rois = {}, {}, {}
        for parity in PARITIES:
            plot = pg.PlotWidget(title=parity)
            plot.invertY(True)
            plot.setAspectLocked(True)
            image = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image)
            markers = pg.ScatterPlotItem(size=10, pen=pg.mkPen("k"), brush=None)
            plot.addItem(markers)
            # Rotation is locked to the axis: no rotate handle; translation and the two
            # scale handles are projected back onto the axis in roi_changed.
            roi = pg.ROI((0, 0), (1, 1), pen=pg.mkPen("y", width=2), rotatable=False)
            roi.addScaleHandle([1, 0.5], [0, 0.5])    # length: end edge, start edge fixed
            roi.addScaleHandle([0.5, 1], [0.5, 0.5])  # width: symmetric about the axis
            roi.addScaleHandle([0.5, 0], [0.5, 0.5])
            roi.sigRegionChanged.connect(lambda item, parity=parity: self.roi_changed(parity, finished=False))
            roi.sigRegionChangeFinished.connect(lambda item, parity=parity: self.roi_changed(parity, finished=True))
            plot.addItem(roi)
            panels.addWidget(plot)
            self.views[parity], self.images[parity], self.rois[parity] = plot, image, roi
            setattr(self, f"markers_{parity}", markers)
        self.views["even"].setXLink(self.views["odd"])
        self.views["even"].setYLink(self.views["odd"])
        self.setCentralWidget(central)

        # Dock: one row per recording.
        self.model = DataFrameModel(self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.add_dock("recordings", "Recordings", self.table, QtCore.Qt.LeftDockWidgetArea, 420)

        # Footer: cache vs manifest for the current recording.
        self.footer = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.footer)

        self.build_commands()
        self.restore_window()
        self.refresh_table()
        # The selection model survives set_frame's model reset, so connect it once.
        self.table.selectionModel().currentRowChanged.connect(self.row_changed)
        self.show_recording(0)
        # Fit once on the first image; later recordings keep the user's zoom (same grid).
        self.views["odd"].autoRange()

    # ----- table -------------------------------------------------------------------
    def refresh_table(self):
        self.model.set_frame(prediction_table(self.recordings, self.params))
        self.table.resizeColumnsToContents()
        # set_frame resets the model and with it the selection: restore it silently.
        selection = self.table.selectionModel()
        selection.blockSignals(True)
        self.table.selectRow(self.current)
        selection.blockSignals(False)

    def row_changed(self, current, previous):
        # Table Up/Down, a click and PgUp/PgDn (select_row) all arrive here.
        if current.isValid() and current.row() != self.current:
            self.show_recording(current.row())

    def select_row(self, row: int):
        self.table.selectRow(min(max(row, 0), len(self.recordings) - 1))

    # ----- drawing -----------------------------------------------------------------
    def show_recording(self, row: int):
        self.current = row
        recording = self.recordings[row]
        for parity in PARITIES:
            # Image before any transform (CLAUDE.md 9.33); row-major, 1 px = 1 unit.
            self.images[parity].setImage(getattr(recording, parity), autoLevels=False)
            markers = getattr(self, f"markers_{parity}")
            markers.setData(
                [recording.bregma[1], recording.lambda_[1]], [recording.bregma[0], recording.lambda_[0]],
                symbol=["+", "x"], brush=["c", "m"], pen=[pg.mkPen("c", width=2), pg.mkPen("m", width=2)],
                size=14, data=["Bregma", "Lambda"])
        self.update_levels()
        self.draw_rectangle()
        self.update_labels()
        self.setWindowTitle(f"Channel rectangle viewer — {recording.key} ({row + 1}/{len(self.recordings)})")

    def update_levels(self):
        """1-99 percentile per panel, or one scale over both when "Shared contrast" is on."""
        recording = self.recordings[self.current]
        shared = self.registry.actions["rect.shared_scale"].isChecked()
        both = np.concatenate([recording.odd.ravel(), recording.even.ravel()])
        for parity in PARITIES:
            source = both if shared else getattr(recording, parity)
            self.images[parity].setLevels(np.percentile(source, [1, 99]))

    def draw_rectangle(self):
        """Place both ROIs from the parameters (local x = along the axis, y = across)."""
        recording = self.recordings[self.current]
        corners = rectangle_corners(recording.bregma, recording.lambda_, self.params.length_factor,
                                    self.params.width_factor, self.params.start_offset_factor)
        d, u, _ = axis_frame(recording.bregma, recording.lambda_)
        state = {
            "pos": pg.Point(corners[0][1], corners[0][0]),  # view x = col, y = row
            "size": pg.Point(self.params.length_factor * d, self.params.width_factor * d),
            "angle": math.degrees(math.atan2(u[0], u[1])),
        }
        self.syncing = True
        for roi in self.rois.values():
            roi.setState(state)
        self.syncing = False

    def roi_changed(self, parity: str, finished: bool):
        """A drag on one panel: read its ROI back as factors, then redraw both on the axis."""
        if self.syncing:
            return
        recording = self.recordings[self.current]
        d, u, n = axis_frame(recording.bregma, recording.lambda_)
        state = self.rois[parity].getState()
        corner = np.array([state["pos"][1], state["pos"][0]])  # (row, col)
        length, width = abs(float(state["size"][0])), abs(float(state["size"][1]))
        start = corner + 0.5 * width * n
        # Only the component along the axis is kept: the rectangle stays centred on it.
        offset = float((start - recording.lambda_) @ u)
        limits = {item.name: item.metadata for item in fields(RectangleParams)}

        def clip(name: str, value: float) -> float:
            # Keep a drag inside the dialog's range so Save always writes a loadable file.
            return min(max(value, limits[name]["minimum"]), limits[name]["maximum"])

        self.set_params(RectangleParams(
            length_factor=clip("length_factor", length / d),
            width_factor=clip("width_factor", width / d),
            start_offset_factor=clip("start_offset_factor", offset / d),
            statistic=self.params.statistic,
        ), refresh_table=finished)

    def set_params(self, params: RectangleParams, refresh_table: bool):
        self.params = params
        self.draw_rectangle()
        self.update_labels()
        # The whole table is recomputed on release only, so dragging stays smooth.
        if refresh_table:
            self.refresh_table()

    def update_labels(self):
        recording = self.recordings[self.current]
        result = predict(recording, self.params)
        lines = []
        for name in ("library", "manifest", "rectangle"):
            label = result["labels"][name]
            text = (f"<b>{name}</b>: no parity ({recording.verdict}) {result['marks'][name]}"
                    if label == UNASSIGNED else f"<b>{name}</b>: {label} = GCaMP {result['marks'][name]}")
            if name == "rectangle":
                text += (f" &nbsp; <i>{self.params.statistic}</i> odd {result['odd']:.4g}, "
                         f"even {result['even']:.4g}, higher/lower {result['ratio']:.3f}")
            lines.append(text)
        p = self.params
        lines.append(f"<span style='color:gray'>length {p.length_factor:.3f} d, width {p.width_factor:.3f} d, "
                     f"start {p.start_offset_factor:+.3f} d from Lambda</span>")
        self.prediction_label.setText("<br>".join(lines))
        if result["labels"]["library"] == UNASSIGNED:
            comparison = "? cache has no channel label"
        elif result["labels"]["library"] != result["labels"]["manifest"]:
            comparison = "⚠ cache and manifest disagree"
        else:
            comparison = "✓ cache and manifest agree"
        self.footer.setText(
            f"cache: {recording.library_order} ({recording.level_region}) | manifest: {recording.manifest_order}"
            f" | verdict: {recording.verdict} | {comparison}")

    def closeEvent(self, event):
        self.save_window()
        event.accept()


def install_excepthook(window: QtWidgets.QWidget) -> None:
    """Show uncaught errors (e.g. a rejected parameter) instead of losing them in a slot."""
    def hook(kind, value, trace):
        text = "".join(traceback.format_exception(kind, value, trace))
        sys.stderr.write(text)
        QtWidgets.QMessageBox.critical(window, kind.__name__, text)
    sys.excepthook = hook


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", default=defaults["day"])
    parser.add_argument("--units", nargs="+", default=defaults["units"])
    parser.add_argument("--recordings", nargs="+", default=defaults["recordings"])
    parser.add_argument("--pixel-root", default=defaults["pixel_root"])
    parser.add_argument("--roi-set-dir", default=defaults["roi_set_dir"])
    parser.add_argument("--assignment", default=defaults["assignment"],
                        help="channel_assignment CSV, or 'none' for a cohort without one.")
    parser.add_argument("--params", default=defaults["params"])
    parser.add_argument("--no-load-saved-params", dest="load_saved_params", action="store_false",
                        default=defaults["load_saved_params"])
    return parser.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{key: value for key, value in config.items() if key != "prefer_cli_args"})


def initial_params(args: argparse.Namespace) -> RectangleParams:
    path = resolve(args.params)
    return load_params(path) if args.load_saved_params and path.is_file() else RectangleParams()


def main() -> int:
    args = build_runtime_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setOrganizationName("wfci_gui")
    app.setApplicationName("ChannelRectangleViewer")
    recordings = load_recordings(args)
    print(f"{len(recordings)} recordings of {args.day} from {resolve(args.pixel_root)}")
    window = ViewerWindow(
        recordings, initial_params(args), resolve(args.params),
        QtCore.QSettings("wfci_gui", "ChannelRectangleViewer"),
        Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppConfigLocation)),
    )
    install_excepthook(window)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
