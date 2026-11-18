"""The ROI editor window: verdict bar + odd | even panels + docks.

Layout, after scripts/channel_rectangle_viewer.py:

* **verdict bar** (top): which parity each system calls GCaMP, with ✓ / ⚠ / – marks
  (``channel_rectangle.agreement_marks``): the manifest's ``channel_order``, the channel
  rectangle (live while it is dragged), the pixel cache's order when there is one, and a
  grey line with the rectangle's factors;
* the user's own call of which parity is GCaMP (Channels menu, Properties), which then
  names the two panels and is saved in the ROI set as ``channel_order``;
* **two ParityPanels**, "odd" and "even", pan and zoom linked, the same boxes, landmarks
  and rectangle drawn on both; a drag in either moves both;
* docks: **Recordings** (sortable, filterable table: key, group, manifest, rectangle,
  agree, status, bregma, lambda), **Properties**, **Display**, **Activity**, **Controls**.

The window draws; ``controller.Controller`` changes the state (``model.EditorModel``).
With no data the central area says how to open some and every page command is disabled.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from qtkit import DataFrameModel, FieldHint, ParameterDialog, ShellWindow
from qtkit.qt import QtCore, QtGui, QtWidgets

from . import sources
from .controller import (
    ACTIONS, CHECKABLE, DEFAULT_LAYOUT, EXTRA_SHORTCUTS, HELP, NEW_SHORTCUTS, WITHOUT_DATA, Controller,
)
from .logic import (
    REPO_ROOT, STATUSES, EditorSettings, file_times, make_settings, page_status, preflight_page, rebuild_sources,
)
from .model import bregma_128, lambda_128
from .view import ParityPanel

from channel_rectangle import (  # noqa: E402  (repo root, via logic)
    UNASSIGNED, agreement_marks, agreement_summary, factors_of, gcamp_parity_from_order, roi_state,
)
from roi_editor import MIN_LAMBDA_OFFSET  # noqa: E402

Qt = QtCore.Qt
EMPTY_TEXT = "No data loaded — File › Open cohort… / Open manifest…"
COLUMNS = ("key", "group", "manifest", "rectangle", "choice", "agree", "status", "bregma", "lambda")
CHOICES = ("not chosen", "odd", "even")       # Properties: which parity is GCaMP
STATUS_COLOURS = {"not opened": "#888888", "opened": "#444444", "edited": "#d62728", "saved": "#1f77b4",
                  "rebuilt": "#2ca02c", "failed pre-flight": "#ff7f0e"}
SETTINGS_HINTS = {
    "folder": FieldHint("list"),
    "frame_order": FieldHint("choice", choices=("name", "number")),
    "roi_set_dir": FieldHint("dir"),
    "roi_seed_dir": FieldHint("dir", optional=True),
    "rebuilt_dir": FieldHint("dir", optional=True, placeholder="empty = the seed folder"),
    "start_from": FieldHint("file", optional=True),
    "rectangle_params": FieldHint("file", optional=True, placeholder="empty = the default parameters"),
}
CONTROLS_TEXT = (HELP + "\n\nTWO PANELS: 'odd' and 'even' are the two interleaved frame groups (no channel "
                 "label on purpose). Everything is drawn on both; a drag in either moves both.\n"
                 "CHANNEL RECTANGLE (orange): drag it along the Bregma-Lambda axis, or its handles for "
                 "length and width. A moved rectangle belongs to this recording and is saved in its ROI "
                 "set; Rectangle > Reset goes back to the cohort's channel_rectangle_params.json.\n"
                 "Every command is also in the menus; shortcuts can be changed in Settings > Keyboard "
                 "shortcuts. Ctrl+Z / Ctrl+Y undo and redo any edit, PgUp / PgDn change page.")
RECENT_LIMIT = 10


def drop_conflicting_overrides(config_dir: Path) -> list[str]:
    """User shortcut overrides that take a key the new commands use are dropped.

    Returns one line per dropped binding, for the changelog shown in the Activity panel.
    """
    path = Path(config_dir) / "layout.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    taken = {QtGui.QKeySequence(key).toString(QtGui.QKeySequence.PortableText): key for key in NEW_SHORTCUTS}
    owners = {spec.shortcut: spec.id for spec in ACTIONS if spec.shortcut in NEW_SHORTCUTS}
    dropped = []
    for action_id, value in list(data.get("shortcuts", {}).items()):
        sequence = QtGui.QKeySequence(value).toString(QtGui.QKeySequence.PortableText)
        if sequence in taken and owners[taken[sequence]] != action_id:
            del data["shortcuts"][action_id]
            dropped.append(f"Your shortcut {value} for '{action_id}' was removed: {value} is now "
                           f"'{owners[taken[sequence]]}'.")
    if dropped:
        from qtkit.actions import write_json_atomic
        write_json_atomic(path, data)
    return dropped


class PagesTable(DataFrameModel):
    """The Recordings table: rows updated in place, status in its colour."""

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.ForegroundRole and index.isValid() and self.frame.columns[index.column()] == "status":
            return QtGui.QBrush(QtGui.QColor(STATUS_COLOURS[self.frame.iat[index.row(), index.column()]]))
        return super().data(index, role)

    def update_rows(self, rows: dict[int, dict]) -> None:
        for row, values in rows.items():
            for column, value in values.items():
                self.frame.iat[row, self.frame.columns.get_loc(column)] = value
        if rows:
            self.dataChanged.emit(self.index(min(rows), 0), self.index(max(rows), len(self.frame.columns) - 1))


class PageFilter(QtCore.QSortFilterProxyModel):
    """Key text filter + status filter over the Recordings table."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.text, self.status = "", ""
        self.setSortRole(Qt.UserRole)

    def set_filters(self, text: str, status: str) -> None:
        self.text, self.status = text.strip().lower(), status
        self.invalidate()

    def filterAcceptsRow(self, row, parent) -> bool:
        frame = self.sourceModel().frame
        if frame is None:
            return True
        key = str(frame.iat[row, frame.columns.get_loc("key")]).lower()
        status = frame.iat[row, frame.columns.get_loc("status")]
        return self.text in key and self.status in ("", status)


class EditorWindow(ShellWindow):
    LAYOUT_VERSION = 2          # docks changed: saved layouts of the old window are ignored

    def __init__(self, settings: QtCore.QSettings, config_dir) -> None:
        super().__init__(settings, config_dir)
        self.setWindowTitle("ROI editor")
        self.resize(1600, 950)
        self.controller = Controller(self)
        self.confirm_close = True
        self.process: QtCore.QProcess | None = None
        self.copy_dir: Path | None = None
        self.preflight: dict[str, list[str]] = {}
        self.file_state: dict[str, tuple] = {}
        self.library: dict[str, str | None] = {}
        self.shown_index: int | None = None
        self.shown_labels: tuple | None = None
        self.shown_shape: tuple | None = None
        self._filling = False          # True while widgets are filled from the state
        self._closed = False

        dropped = drop_conflicting_overrides(Path(config_dir))
        self.install_commands(ACTIONS, DEFAULT_LAYOUT, self.controller)
        for action_id in CHECKABLE:
            self.registry.actions[action_id].setCheckable(True)
        self.registry.actions["rect.show"].setChecked(True)
        self._build_central()
        self._build_docks()
        self.build_commands()
        self.restore_window()
        for line in dropped:
            self.notify(line)
        self.show_empty()

    @property
    def model(self):
        return self.controller.model

    # ----- construction ----------------------------------------------------------------
    def _build_central(self) -> None:
        self.stack = QtWidgets.QStackedWidget()
        self.empty_label = QtWidgets.QLabel(EMPTY_TEXT)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #888; font-size: 16px;")
        self.stack.addWidget(self.empty_label)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        self.verdict = QtWidgets.QLabel()
        self.verdict.setTextFormat(Qt.RichText)
        self.verdict.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.verdict)
        row = QtWidgets.QHBoxLayout()
        layout.addLayout(row, stretch=1)
        self.panels = {parity: ParityPanel(parity, self.controller) for parity in ("odd", "even")}
        for panel in self.panels.values():
            row.addWidget(panel)
        self.panels["even"].setXLink(self.panels["odd"])
        self.panels["even"].setYLink(self.panels["odd"])
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.stack.addWidget(content)
        self.setCentralWidget(self.stack)
        self.footer = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.footer)

    def _build_docks(self) -> None:
        # Recordings table with its two filters.
        recordings = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(recordings)
        self.page_filter = QtWidgets.QLineEdit()
        self.page_filter.setPlaceholderText("Filter by key…")
        self.status_filter = QtWidgets.QComboBox()
        self.status_filter.addItems(["All statuses", *STATUSES])
        self.table_model = PagesTable(self)
        self.proxy = PageFilter(self)
        self.proxy.setSourceModel(self.table_model)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.page_count = QtWidgets.QLabel()
        for widget in (self.page_filter, self.status_filter, self.table, self.page_count):
            layout.addWidget(widget)
        self.page_filter.textChanged.connect(lambda _text: self.apply_filter())
        self.status_filter.currentIndexChanged.connect(lambda _index: self.apply_filter())
        self.table.selectionModel().currentRowChanged.connect(self._on_row_chosen)
        self.add_dock("recordings", "Recordings", recordings, Qt.LeftDockWidgetArea, 460)

        # Properties: the numbers behind the two landmarks.
        properties = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(properties)
        self.bregma_row, self.bregma_col = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        self.lambda_distance = QtWidgets.QSpinBox()
        self.scale, self.rotation = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for box in (self.bregma_row, self.bregma_col):
            box.setSingleStep(2)       # ROI sets store Bregma in 2x-grid units
        self.lambda_distance.setSuffix(" px")
        self.scale.setDecimals(3)
        self.scale.setSingleStep(0.01)
        self.scale.setRange(0.05, 20.0)
        self.scale.setSuffix(" x")
        self.rotation.setDecimals(1)
        self.rotation.setRange(-180.0, 180.0)
        self.rotation.setSuffix(" deg")
        self.mirror_lock = QtWidgets.QCheckBox("Mirror lock (edit one box, its twin follows)")
        self.rotate_with_lambda = QtWidgets.QCheckBox("Dragging Lambda sideways rotates the layout")
        self.selected_box = QtWidgets.QLabel()
        self.selected_box.setWordWrap(True)
        self.gcamp_choice = QtWidgets.QComboBox()
        self.gcamp_choice.addItems(["not chosen", "odd frames (even = reflectance)",
                                    "even frames (odd = reflectance)"])
        self.gcamp_choice.setToolTip("Which interleaved group is GCaMP for this recording; saved in its "
                                     "ROI set as channel_order")
        self.gcamp_choice.currentIndexChanged.connect(lambda index: self._panel(
            "GCaMP choice", lambda: self.model.choose_gcamp(None if index == 0 else CHOICES[index])))
        form.addRow("GCaMP is", self.gcamp_choice)
        form.addRow("Bregma row", self.bregma_row)
        form.addRow("Bregma column", self.bregma_col)
        form.addRow("Lambda distance", self.lambda_distance)
        form.addRow("Scale", self.scale)
        form.addRow("Rotation", self.rotation)
        form.addRow(self.mirror_lock)
        form.addRow(self.rotate_with_lambda)
        form.addRow("Selected box", self.selected_box)
        for box in (self.bregma_row, self.bregma_col, self.lambda_distance, self.scale, self.rotation):
            box.setKeyboardTracking(False)     # a typed number applies on Enter, not per digit
        self.bregma_row.valueChanged.connect(lambda _v: self._panel("Move Bregma", lambda: self.model.move_bregma(
            self.bregma_row.value() // 2, self.bregma_col.value() // 2)))
        self.bregma_col.valueChanged.connect(lambda _v: self._panel("Move Bregma", lambda: self.model.move_bregma(
            self.bregma_row.value() // 2, self.bregma_col.value() // 2)))
        self.lambda_distance.valueChanged.connect(
            lambda value: self._panel("Lambda distance", lambda: self.model.set_lambda(int(value))))
        self.scale.valueChanged.connect(lambda value: self._panel("Scale", lambda: self.model.set_scale(value)))
        self.rotation.valueChanged.connect(
            lambda value: self._panel("Rotation", lambda: self.model.set_rotation(value)))
        self.mirror_lock.toggled.connect(lambda on: self._toggle("mirror_lock", on))
        self.rotate_with_lambda.toggled.connect(lambda on: self._toggle("rotate_with_lambda", on))
        self.add_dock("properties", "Properties", properties, Qt.RightDockWidgetArea, 320)

        # Display: contrast, frames per parity, shared contrast, rectangle.
        display = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(display)
        self.contrast_low, self.contrast_high = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for box in (self.contrast_low, self.contrast_high):
            box.setRange(0.0, 100.0)
            box.setDecimals(1)
            box.setSuffix(" percentile")
            box.setKeyboardTracking(False)
            box.valueChanged.connect(lambda _v: self._set_contrast())
        self.frames_per_parity = QtWidgets.QSpinBox()
        self.frames_per_parity.setRange(1, 200)
        self.frames_per_parity.setKeyboardTracking(False)
        self.frames_per_parity.setToolTip("Frames of each parity averaged into both panels and the "
                                          "rectangle decision")
        self.frames_per_parity.valueChanged.connect(
            lambda value: None if self._filling else self.controller.set_frames_per_parity(value))
        self.shared_contrast = QtWidgets.QCheckBox("Shared contrast (one scale for both panels)")
        self.show_rectangle = QtWidgets.QCheckBox("Show the channel rectangle")
        self._bind_checkbox(self.shared_contrast, "display.shared_scale")
        self._bind_checkbox(self.show_rectangle, "rect.show")
        form.addRow("Darkest", self.contrast_low)
        form.addRow("Brightest", self.contrast_high)
        form.addRow("Frames per parity", self.frames_per_parity)
        form.addRow(self.shared_contrast)
        form.addRow(self.show_rectangle)
        self.add_dock("display", "Display", display, Qt.RightDockWidgetArea)

        self.activity = QtWidgets.QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setMaximumBlockCount(5000)
        self.add_dock("activity", "Activity", self.activity, Qt.BottomDockWidgetArea, 120)
        controls = QtWidgets.QPlainTextEdit(CONTROLS_TEXT)
        controls.setReadOnly(True)
        self.add_dock("controls", "Controls", controls, Qt.BottomDockWidgetArea, 120)

    def _bind_checkbox(self, box: QtWidgets.QCheckBox, action_id: str) -> None:
        """A dock checkbox and a checkable action that always agree."""
        action = self.registry.actions[action_id]
        box.setChecked(action.isChecked())
        box.toggled.connect(lambda on: action.isChecked() != on and action.trigger())
        action.toggled.connect(box.setChecked)

    # ----- commands --------------------------------------------------------------------
    def build_commands(self) -> None:
        super().build_commands()
        for action_id, extra in EXTRA_SHORTCUTS.items():
            action = self.registry.actions[action_id]
            action.setShortcuts([action.shortcut(), QtGui.QKeySequence(extra)])

    def extend_menu(self, title, menu) -> None:
        """File > Recent, filled each time the File menu opens."""
        if title != "File":
            return
        self.recent_menu = QtWidgets.QMenu("Recent", menu)
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        actions = menu.actions()
        # Right after Open cohort / Open manifest in the default layout.
        if len(actions) > 2:
            menu.insertMenu(actions[2], self.recent_menu)
        else:
            menu.addMenu(self.recent_menu)

    def recent(self) -> list[tuple[str, str]]:
        return [tuple(item) for item in json.loads(self.settings.value("recent", "[]"))]

    def _store_recent(self, entries) -> None:
        self.settings.setValue("recent", json.dumps([list(item) for item in entries]))

    def add_recent(self, kind: str, path: str) -> None:
        self._store_recent(sources.push_recent(self.recent(), kind, path, RECENT_LIMIT))

    def remove_recent(self, kind: str, path: str) -> None:
        self._store_recent([item for item in self.recent() if item != (kind, path)])

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        entries = self.recent()
        if not entries:
            self.recent_menu.addAction("(nothing opened yet)").setEnabled(False)
        for number, (kind, path) in enumerate(entries):
            action = self.recent_menu.addAction(f"{kind}: {Path(path).name}  ({path})")
            action.setObjectName(f"file.recent.{number}")
            action.triggered.connect(lambda _checked=False, k=kind, p=path: self.controller.open_recent(k, p))

    def update_actions(self) -> None:
        if self.registry is None:
            return
        actions, model = self.registry.actions, self.model
        for action_id, action in actions.items():
            if action_id not in WITHOUT_DATA:
                action.setEnabled(model is not None)
        busy = self.process is not None
        actions["file.settings"].setEnabled(not busy)
        if model is None:
            return
        actions["edit.undo"].setEnabled(model.can_undo())
        actions["edit.redo"].setEnabled(model.can_redo())
        actions["edit.undo"].setText(f"Undo {model.undo_text()}".strip())
        actions["edit.redo"].setText(f"Redo {model.redo_text()}".strip())
        has_rebuilt = bool(self.controller.settings.rebuilt)
        actions["tools.rebuild"].setEnabled(has_rebuilt and not busy)
        actions["tools.preflight"].setEnabled(has_rebuilt and not busy)
        actions["file.cache_all"].setEnabled(self.controller.cache_thread is None)
        actions["rect.reset"].setEnabled(model.page.rectangle is not None)

    # ----- questions (one method each, so tests can answer them) -------------------------
    def ask_unsaved(self, count: int) -> str:
        """'save' | 'discard' | 'cancel' for pages with unsaved edits."""
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "Unsaved pages",
                                    f"{count} page(s) have unsaved changes.", parent=self)
        save = box.addButton("Save all", QtWidgets.QMessageBox.AcceptRole)
        discard = box.addButton("Discard", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        return "save" if box.clickedButton() is save else "discard" if box.clickedButton() is discard else "cancel"

    def ask_settings(self, values: dict, need_folder: bool, note: str) -> EditorSettings | None:
        dialog = ParameterDialog(self, EditorSettings, values,
                                 lambda v: make_settings(v, need_folder=need_folder), SETTINGS_HINTS,
                                 title="ROI editor settings", note=note)
        return dialog.value if dialog.exec() == QtWidgets.QDialog.Accepted else None

    def choose_file(self, title: str, folder: Path, pattern: str) -> str:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, title, str(folder), pattern)
        return path

    # ----- messages ----------------------------------------------------------------------
    def notify(self, message: str) -> None:
        self.statusBar().showMessage(message, 15000)
        self.activity.appendPlainText(message)

    def log(self, message: str) -> None:
        self.activity.appendPlainText(message)

    def show_controls(self) -> None:
        self.docks["controls"].show()
        self.docks["controls"].raise_()

    # ----- data shown --------------------------------------------------------------------
    def show_empty(self) -> None:
        self.stack.setCurrentIndex(0)
        self.setWindowTitle("ROI editor")
        self.footer.setText("")
        self.table_model.set_frame(pd.DataFrame(columns=list(COLUMNS)))
        self.update_actions()

    def show_data(self) -> None:
        """New pages loaded: table, panels, everything."""
        self.stack.setCurrentIndex(1)
        self.preflight = {}
        self.shown_index = self.shown_labels = self.shown_shape = None
        data = self.controller.data
        self.library = {page.key: sources.library_order(data.pixel_root, page.key) for page in self.model.pages}
        self._filling = True
        self.frames_per_parity.setValue(data.settings.frames_per_parity)
        self._filling = False
        self.rescan_files()
        self.table_model.set_frame(pd.DataFrame([self.row_values(i) for i in range(len(self.model.pages))],
                                                columns=list(COLUMNS)))
        self.table.resizeColumnsToContents()
        self.show_page()
        self.apply_filter()

    def show_page(self) -> None:
        """The current page from scratch: images, boxes, view range."""
        model = self.model
        page = model.page
        model.ensure_image(page)
        odd, even = page.odd_even_128
        self.panels["odd"].set_image(odd)
        self.panels["even"].set_image(even, "no cache (old single preview only)")
        labels = tuple(page.boxes)
        if labels != self.shown_labels:
            for panel in self.panels.values():
                panel.build_boxes(labels)
            self.shown_labels = labels
        if page.image.shape != self.shown_shape:
            self.fit_view(page.image.shape)
            self.shown_shape = page.image.shape
        self.shown_index = model.index
        self.update_levels()
        self.sync()

    def redraw(self, page_changed: bool = False, skip=None) -> None:
        if self.model is None:
            return
        if page_changed or self.model.index != self.shown_index:
            self.show_page()
        else:
            self.sync(skip)

    def sync(self, skip=None) -> None:
        """Move every item to the model's geometry; ``skip`` = (panel, box being dragged)."""
        model = self.model
        page = model.page
        bad = set(model.out_of_frame(page))
        rectangle = (roi_state(bregma_128(page), lambda_128(page), factors_of(model.page_params(page)))
                     if self.registry.actions["rect.show"].isChecked() else None)
        self.controller.syncing = True
        try:
            for panel in self.panels.values():
                panel.sync(page, model.selected, bad, rectangle,
                           skip=skip[1] if skip is not None and skip[0] is panel else None)
        finally:
            self.controller.syncing = False
        for parity, panel in self.panels.items():
            # No channel label until the user has made the call (avoids bias).
            role = "" if page.gcamp_choice is None else (" — GCaMP" if parity == page.gcamp_choice
                                                          else " — reflectance")
            panel.setTitle(parity + role)
        self.message.setText(model.message + (f"   |   {len(bad)} box(es) OFF FRAME: {sorted(bad)} -- saving "
                                               f"is refused until they fit" if bad else ""))
        self.message.setStyleSheet(f"color: {'#d62728' if bad else '#1f77b4'}; font-family: monospace;")
        self.update_verdict()
        self.update_properties()
        self.update_footer()
        self.update_actions()

    def update_levels(self) -> None:
        """Darkest / brightest percentile per panel, or over both with shared contrast."""
        if self.model is None:
            return
        images = [image for image in self.model.page.odd_even_128 if image is not None]
        shared = self.registry.actions["display.shared_scale"].isChecked()
        both = np.concatenate([image.ravel() for image in images])
        for panel, image in zip(self.panels.values(), self.model.page.odd_even_128):
            if image is not None:
                panel.set_levels(np.percentile(both if shared else image, self.controller.clip))
        self._filling = True
        self.contrast_low.setValue(self.controller.clip[0])
        self.contrast_high.setValue(self.controller.clip[1])
        self._filling = False

    def _set_contrast(self) -> None:
        if self._filling:
            return
        low, high = self.contrast_low.value(), self.contrast_high.value()
        if high <= low:
            self.notify("The brightest percentile must be above the darkest one.")
            self.update_levels()
            return
        self.controller.clip = [low, high]
        self.update_levels()

    def fit_view(self, shape=None) -> None:
        if self.model is None:
            return
        self.panels["odd"].fit(shape or self.model.page.image.shape)
        self.panels["even"].fit(shape or self.model.page.image.shape)

    # ----- verdict bar, footer, properties -----------------------------------------------
    def labels_of(self, index: int) -> tuple[dict[str, str], object, str]:
        """GCaMP parity per system for one page: (labels, decision, note)."""
        model, data = self.model, self.controller.data
        page = model.pages[index]
        labels = {}
        if data.kind != "folder":
            labels["manifest"] = gcamp_parity_from_order(page.channel_order)
        decision, note = model.decision(page)
        labels["rectangle"] = decision.gcamp_parity if decision is not None else UNASSIGNED
        library = self.library.get(page.key)
        if library is not None:
            labels["library"] = gcamp_parity_from_order(library)
        if page.gcamp_choice is not None:
            labels["choice"] = page.gcamp_choice
        return labels, decision, note

    def update_verdict(self) -> None:
        model = self.model
        page = model.page
        labels, decision, note = self.labels_of(model.index)
        marks = agreement_marks(labels)
        params = model.page_params(page)
        own = " (this recording)" if page.rectangle is not None else ""
        lines = []
        if "choice" in labels:
            other = "even" if page.gcamp_choice == "odd" else "odd"
            lines.append(f"<b>your choice</b>: {page.gcamp_choice} = GCaMP, {other} = reflectance "
                         f"{marks['choice']}")
        else:
            lines.append("<b>your choice</b>: not made (Channels menu: Alt+1 odd, Alt+2 even, "
                         "G = the rectangle's decision)")
        if "manifest" in labels:
            order = page.channel_order or "blank"
            lines.append(f"<b>manifest</b>: no parity ({order}) {marks['manifest']}" if labels["manifest"] == UNASSIGNED
                         else f"<b>manifest</b>: {labels['manifest']} = GCaMP ({order}) {marks['manifest']}")
        if decision is None:
            lines.append(f"<b>rectangle</b>: no decision ({note}){own} {marks['rectangle']}")
        else:
            lines.append(f"<b>rectangle</b>: {decision.gcamp_parity} = GCaMP {marks['rectangle']} &nbsp; "
                         f"<i>{params.statistic}</i> odd {decision.odd_value:.4g}, even {decision.even_value:.4g}, "
                         f"higher/lower {decision.ratio:.3f}{own}")
        if "library" in labels:
            lines.append(f"<b>library</b>: {labels['library'] if labels['library'] != UNASSIGNED else 'no parity'} "
                         f"({self.library[page.key]}) {marks['library']}")
        lines.append(f"<span style='color:gray'>length {params.length_factor:.3f} d, width "
                     f"{params.width_factor:.3f} d, start {params.start_offset_factor:+.3f} d from Lambda "
                     f"({'this recording' if own else 'params.json'})</span>")
        self.verdict.setText("<br>".join(lines))

    def update_footer(self) -> None:
        model, data = self.model, self.controller.data
        page = model.page
        self.footer.setText(f"{page.key} | {page.folder} | {page.seed_source} | {page.image_source}")
        self.setWindowTitle(f"ROI editor — {data.name} — {page.key} ({model.index + 1}/{len(model.pages)})"
                            f"{' *' if page.dirty else ''}")

    def update_properties(self) -> None:
        model = self.model
        page = model.page
        self._filling = True
        rows, cols = page.image.shape
        self.bregma_row.setRange(2, 2 * rows)
        self.bregma_col.setRange(2, 2 * cols)
        self.lambda_distance.setRange(MIN_LAMBDA_OFFSET, max(rows, cols))
        self.bregma_row.setValue(page.bregma_row)
        self.bregma_col.setValue(page.bregma_col)
        self.lambda_distance.setValue(page.lambda_offset)
        self.scale.setValue(page.lambda_scale)
        self.rotation.setValue(page.lambda_angle)
        self.mirror_lock.setChecked(model.mirror_lock)
        self.rotate_with_lambda.setChecked(model.rotate_with_lambda)
        self.gcamp_choice.setCurrentIndex(CHOICES.index(page.gcamp_choice or "not chosen"))
        box = page.boxes.get(model.selected) if model.selected else None
        self.selected_box.setText(
            f"{model.selected}: rows {box.row_start}..{box.row_end}, cols {box.col_start}..{box.col_end} "
            "(offsets from Bregma)" if box else "none (click a box)")
        self._filling = False

    def _panel(self, label: str, edit) -> None:
        if not self._filling and self.model is not None:
            self.controller.panel_edit(label, edit)

    def _toggle(self, name: str, on: bool) -> None:
        if not self._filling and self.model is not None:
            setattr(self.model, name, on)

    # ----- Recordings table --------------------------------------------------------------
    def rescan_files(self) -> None:
        """Which pages have a saved and a rebuilt set (after load, save, rebuild)."""
        settings = self.controller.settings
        self.file_state = {page.key: file_times(page.key, Path(settings.roi_set_dir), settings.rebuilt)
                           for page in self.model.pages}

    def status_of(self, index: int) -> str:
        page = self.model.pages[index]
        reviewed, rebuilt = self.file_state.get(page.key, (None, None))
        return page_status(page.dirty, index in self.model.visited, reviewed, rebuilt, self.preflight.get(page.key))

    def row_values(self, index: int) -> dict:
        page = self.model.pages[index]
        labels, _decision, _note = self.labels_of(index)
        side = " above" if page.lambda_sign < 0 else ""
        angle = f" {page.lambda_angle:+g}°" if page.lambda_angle else ""
        return {"key": page.key, "group": page.group,
                "manifest": "" if labels.get("manifest", UNASSIGNED) == UNASSIGNED else labels["manifest"],
                "rectangle": "" if labels["rectangle"] == UNASSIGNED else labels["rectangle"],
                "choice": page.gcamp_choice or "",
                "agree": agreement_summary(labels), "status": self.status_of(index),
                "bregma": f"{page.bregma_row}, {page.bregma_col}", "lambda": f"{page.lambda_offset}{side}{angle}"}

    def refresh_rows(self) -> None:
        if self.model is None or self.table_model.frame is None or len(self.table_model.frame) != len(self.model.pages):
            return
        self.table_model.update_rows({i: self.row_values(i) for i in range(len(self.model.pages))})
        self.apply_filter()

    def apply_filter(self) -> None:
        text = self.page_filter.text()
        status = self.status_filter.currentText()
        # Re-filtering moves the selection; that must not be read as the user opening a page.
        self._filling = True
        self.proxy.set_filters(text, "" if status == "All statuses" else status)
        self._filling = False
        total = len(self.model.pages) if self.model is not None else 0
        modified = len(self.model.dirty_pages()) if self.model is not None else 0
        self.page_count.setText(f"{self.proxy.rowCount()} of {total} page(s) shown, {modified} with unsaved edits")
        self.select_current_row()

    def select_current_row(self) -> None:
        if self.model is None:
            return
        index = self.proxy.mapFromSource(self.table_model.index(self.model.index, 0))
        self._filling = True
        if index.isValid():
            self.table.selectionModel().setCurrentIndex(
                index, QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Rows)
        self._filling = False

    def _on_row_chosen(self, current, _previous) -> None:
        if self._filling or not current.isValid():
            return
        self.controller.go_to(self.proxy.mapToSource(current).row())

    # ----- rebuild and pre-flight --------------------------------------------------------
    def rebuild(self) -> None:
        """rebuild_roi_sets.py on the sessions that have a new or changed saved set."""
        settings = self.controller.settings
        if not settings.rebuilt:
            self.notify("Set a rebuilt folder in Settings first.")
            return
        if self.process is not None:
            self.notify("A rebuild is already running.")
            return
        if self.model.dirty_pages():
            self.notify("Save the modified pages first: the rebuild reads the saved files.")
            return
        sources_ = rebuild_sources([page.key for page in self.model.pages], Path(settings.roi_set_dir),
                                   Path(settings.rebuilt))
        if not sources_:
            self.notify("No new or changed saved sets: nothing to rebuild.")
            return
        # Only these files are rebuilt, never the whole folder.
        self.copy_dir = Path(tempfile.mkdtemp(prefix="roi_rebuild_"))
        for source in sources_:
            shutil.copy2(source, self.copy_dir / source.name)
        from .logic import rebuild_options
        process = QtCore.QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-u", str(REPO_ROOT / "scripts" / "rebuild_roi_sets.py"), "--roi-set-dir", str(self.copy_dir),
                              "--out-dir", str(settings.rebuilt), *rebuild_options(self.controller.data.cohort)])
        process.setWorkingDirectory(str(self.copy_dir))
        # The same PYTHONPATH and bytecode folder as a GUI job: this copy's code, never
        # another installation of it.
        from wfci_gui.logic.conda import job_environment
        environment = QtCore.QProcessEnvironment()
        for key, value in job_environment(REPO_ROOT).items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_process)
        process.finished.connect(self._rebuild_finished)
        self.process = process
        self.update_actions()
        self.notify(f"Rebuilding {len(sources_)} saved set(s) into {settings.rebuilt} …")
        process.start()

    def _read_process(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text.strip():
            self.activity.appendPlainText(text.rstrip())

    def _rebuild_finished(self, exit_code: int, _status) -> None:
        self._read_process()
        self.process.deleteLater()
        self.process = None
        shutil.rmtree(self.copy_dir)
        self.copy_dir = None
        self.rescan_files()
        self.update_actions()
        if exit_code != 0:
            self.notify(f"Rebuild FAILED (exit code {exit_code}); see the Activity panel.")
            self.refresh_rows()
            return
        self.notify("Rebuild finished.")
        self.run_preflight()

    def run_preflight(self) -> None:
        """roi_preflight.py's check on every page's rebuilt set; results go to the table."""
        import roi_preflight
        settings = self.controller.settings
        if not settings.rebuilt:
            self.notify("Set a rebuilt folder in Settings first.")
            return
        window = roi_preflight.RUN_CONFIG["window"]
        current_grid = list(self.model.page.image.shape)
        self.preflight = {page.key: preflight_page(page.key, Path(settings.rebuilt), window,
                                                   list(page.image.shape) if page.image is not None else current_grid)
                          for page in self.model.pages}
        failed = {key: problems for key, problems in self.preflight.items() if problems}
        self.refresh_rows()
        for key, problems in failed.items():
            self.activity.appendPlainText(f"pre-flight FAIL {key}: {'; '.join(problems)}")
        self.notify(f"Pre-flight: {len(self.preflight) - len(failed)} passed, {len(failed)} failed"
                    + (" (filter the table by 'failed pre-flight')" if failed else ""))

    # ----- closing -----------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        if self.confirm_close and not self.controller.confirm_discard():
            event.ignore()
            return
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(3000)
        if self.controller.cache_thread is not None:
            self.controller.cache_worker.cancelled = True
            self.controller.cache_thread.quit()
            self.controller.cache_thread.wait()
        self.save_window()
        self._closed = True
        event.accept()
