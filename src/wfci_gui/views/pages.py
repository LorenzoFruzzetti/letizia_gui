"""One page per workflow step: Settings | Tables | Results."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from qtkit.qt import QtCore, QtWidgets

from ..logic.cohort import (
    ACQUISITION_KEYS, BOOL_COLUMNS, CHANNEL_ORDERS, EDITABLE_RECORDING_COLUMNS, FRAME_ORDERS,
    used_recordings,
)
from ..logic.conda import display_command
from ..logic.design import check_design
from ..logic.params import JOBS, build_arguments, form_fields, script_parameters, smoke_unit
from ..logic.presets import diff_overrides, hash_warning, list_presets, load_preset, save_preset
from ..logic.steps import (
    CONNECTIVITY_VARIANTS, StepDef, cache_windows, chosen_variants, design_path, free_gb,
    needed_dump_gb, qc_table_path,
)
from .widgets import FrameModel, ParameterForm, ResultsView, TableEditor, text_of


def rectangle_viewer_arguments(cohort, tables, day: str) -> tuple[list[str], str]:
    """CLI of channel_rectangle_viewer.py for one session, and a note ('' when complete).

    The viewer shows every unit x recording it is given and needs a dump for each, so it
    gets the recordings every dumped unit of the day shares; units missing one are left out.
    """
    pixel_root = cohort.layout.pixel_data
    have: dict[str, set[str]] = {}
    for row in used_recordings(tables).itertuples():
        folder = pixel_root / f"{row.day}_{row.animal}" / row.recording
        if row.day == day and ((folder / "pixels_meta_full.npz").exists()
                               or (folder / "pixels_meta_odd_even_full.npz").exists()):
            have.setdefault(row.animal, set()).add(row.recording)
    if not have:
        return [], f"No extracted recordings on {day}: run step 4 first."
    shared = set.intersection(*have.values())
    if not shared:
        return [], f"The units of {day} share no extracted recording."
    units = sorted(have)
    recordings = sorted(shared, key=lambda name: int(name[1:]))
    left_out = sorted(f"{u}/{r}" for u, rs in have.items() for r in rs - shared)
    # The viewer takes the dumps, ROI sets and rectangle parameters from the cohort file. A
    # cohort has no channel-assignment table: the viewer's manifest column stays unassigned.
    arguments = ["--cohort", str(cohort.path), "--day", day, "--units", *units,
                 "--recordings", *recordings, "--assignment", "none"]
    note = f"Rectangle viewer: left out (not shared by every unit): {', '.join(left_out)}" if left_out else ""
    return arguments, note


def colored(text: str, ok: bool | None) -> str:
    color = {True: "#1e8449", False: "#c0392b", None: "#b9770e"}[ok]
    return f"<span style='color:{color}'>{text}</span>"


# ----------------------------------------------------------------------------- job section
class JobSection(QtWidgets.QGroupBox):
    """One job: its parameter form, presets, run buttons and (advanced) command preview."""

    def __init__(self, page: "StepPage", job: str, buttons: list[tuple[str, Any]] | None = None):
        spec = JOBS[job]
        super().__init__(f"{spec.label}  ({spec.script})")
        self.page, self.job = page, job
        self.controller = page.controller
        self.preview_variant = None
        layout = QtWidgets.QVBoxLayout(self)
        self.form = ParameterForm(lambda n, v: self.controller.set_override(job, n, v),
                                  lambda n: self.controller.reset_override(job, n))
        toggle = QtWidgets.QToolButton()
        toggle.setText("Parameters ▸")
        toggle.setCheckable(True)
        toggle.toggled.connect(lambda on: (self.form.setVisible(on), toggle.setText("Parameters ▾" if on else "Parameters ▸")))
        self.form.setVisible(False)
        row = QtWidgets.QHBoxLayout()
        for label, callback in buttons or [("Run", lambda: page.window.run_job(job))]:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, cb=callback: cb())
            row.addWidget(button)
        row.addStretch()
        row.addWidget(toggle)
        save = QtWidgets.QPushButton("Save preset…")
        save.clicked.connect(self.save_preset)
        load = QtWidgets.QPushButton("Load preset…")
        load.clicked.connect(self.load_preset)
        row.addWidget(save)
        row.addWidget(load)
        layout.addLayout(row)
        layout.addWidget(self.form)
        self.preview = QtWidgets.QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(90)
        layout.addWidget(self.preview)

    def refresh(self) -> None:
        state = self.controller.state
        fields, note = form_fields(self.job, state.cohort, state.tables, self.preview_variant)
        self.form.set_fields(fields, note)
        self.preview.setVisible(state.advanced)
        if not state.advanced:
            return
        if note:
            self.preview.setHtml(colored(html.escape(note), False))
            return
        built = build_arguments(self.job, state.cohort, state.tables, self.preview_variant)
        command = display_command(["conda.bat", "run", "--no-capture-output", "-n", "wfci_gui",
                                   "python", "-u", *built.arguments], built.env)
        marked = html.escape(command)
        for name in sorted(built.overridden):
            for token in (f"--{name.replace('_', '-')}", f"--no-{name.replace('_', '-')}", f"{name}="):
                marked = marked.replace(token, f"<b style='color:#b9770e'>{token}</b>")
        self.preview.setHtml(f"<code>{marked}</code><br><i>Bold = changed from the default.</i>")

    def save_preset(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Save preset", f"Preset name for {self.job}:")
        if ok and name.strip():
            spec = JOBS[self.job]
            cohort = self.controller.state.cohort
            path = save_preset(self.job, name.strip(), cohort.overrides(self.job),
                               script_parameters(spec).file_hash, cohort.layout.presets)
            self.page.window.notify(f"Preset saved: {path}")

    def load_preset(self) -> None:
        root = self.controller.state.cohort.layout.presets
        names = list_presets(self.job, root)
        if not names:
            QtWidgets.QMessageBox.information(self, "Load preset", f"No presets saved for {self.job} yet.")
            return
        name, ok = QtWidgets.QInputDialog.getItem(self, "Load preset", "Preset:", names, 0, False)
        if not ok:
            return
        payload = load_preset(self.job, name, root)
        current = self.controller.state.cohort.overrides(self.job)
        changes = diff_overrides(current, payload["overrides"])
        warning = hash_warning(payload, script_parameters(JOBS[self.job]).file_hash)
        text = "\n".join(f"{n}: {o!r} → {v!r}" for n, o, v in changes) or "No change."
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "Load preset",
                                    f"Apply preset '{name}'?\n\n{text}" + (f"\n\n⚠ {warning}" if warning else ""),
                                    QtWidgets.QMessageBox.Apply | QtWidgets.QMessageBox.Cancel, self)
        if box.exec() == QtWidgets.QMessageBox.Apply:
            self.controller.replace_overrides(self.job, payload["overrides"])


# ----------------------------------------------------------------------------- base page
class StepPage(QtWidgets.QWidget):
    jobs: tuple[str, ...] = ()

    def __init__(self, window, step: StepDef):
        super().__init__()
        self.window = window
        self.controller = window.controller
        self.step = step
        self.sections: list[JobSection] = []
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        settings = QtWidgets.QWidget()
        self.settings_layout = QtWidgets.QVBoxLayout(settings)
        help_label = QtWidgets.QLabel(step.help_html)
        help_label.setWordWrap(True)
        help_label.setTextFormat(QtCore.Qt.RichText)
        self.settings_layout.addWidget(help_label)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.settings_layout.addWidget(self.status)
        self.build_settings()
        for section in self.make_sections():
            self.sections.append(section)
            self.settings_layout.addWidget(section)
        self.settings_layout.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings)
        self.tabs.addTab(scroll, "Settings")
        tables = self.build_tables()
        self.tables_index = self.tabs.addTab(tables, "Tables") if tables is not None else -1
        self.results = ResultsView()
        self.tabs.addTab(self.results, "Results")

    # hooks
    def build_settings(self) -> None:
        """Add page-specific widgets above the job sections."""

    def make_sections(self) -> list[JobSection]:
        return [JobSection(self, job) for job in self.jobs]

    def build_tables(self) -> QtWidgets.QWidget | None:
        return None

    def results_folders(self) -> list[Path]:
        cohort = self.controller.state.cohort
        return [cohort.layout.outputs] if cohort else []

    def primary(self) -> None:
        """What Run step (F5) does on this page."""
        if self.jobs:
            self.window.run_job(self.jobs[0])

    def refresh(self) -> None:
        state = self.controller.state
        cohort = state.cohort
        status = state.statuses.get(self.step.key)
        lines = []
        if status is not None:
            mark = "✓ done" if status.done else "not done"
            lines.append(f"<b>Check:</b> {colored(mark, status.done)} — {html.escape(status.detail)}")
            lines += [colored("⚠ " + html.escape(w), None) for w in status.warnings]
            blockers = self.controller.blockers(self.step.key)
            if blockers:
                lines.append(colored("Blocked until these steps are done: " + ", ".join(blockers), False))
        note = self.controller.history_notes().get(self.step.key)
        if note:
            lines.append(f"<i>{html.escape(note)}</i>")
        self.status.setText("<br>".join(lines))
        if cohort is not None:
            for section in self.sections:
                section.refresh()
        self.refresh_page()
        self.results.set_folders(self.results_folders())

    def refresh_page(self) -> None:
        """Page-specific redraw."""


# ----------------------------------------------------------------------------- 0 setup
class SetupPage(StepPage):
    def build_settings(self):
        row = QtWidgets.QHBoxLayout()
        for label, tests in (("Run checks", False), ("Run checks + test suite", True)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, t=tests: self.window.enqueue(self.controller.setup_check_job(t)))
            row.addWidget(button)
        repair = QtWidgets.QPushButton("Repair (setup_env.ps1)")
        repair.clicked.connect(lambda: self.window.enqueue(self.controller.repair_job()))
        row.addWidget(repair)
        row.addStretch()
        self.settings_layout.addLayout(row)
        self.checks = QtWidgets.QTableWidget(0, 3)
        self.checks.setHorizontalHeaderLabels(["Check", "Result", "Detail"])
        self.checks.horizontalHeader().setStretchLastSection(True)
        self.checks.setMinimumHeight(320)
        self.settings_layout.addWidget(self.checks)

    def primary(self):
        self.window.enqueue(self.controller.setup_check_job(False))

    def refresh_page(self):
        state = self.controller.state
        checks = list(state.quick_checks) + list((state.setup_result or {}).get("checks", []))
        self.checks.setRowCount(len(checks))
        for i, check in enumerate(checks):
            self.checks.setItem(i, 0, QtWidgets.QTableWidgetItem(check["name"]))
            self.checks.setItem(i, 1, QtWidgets.QTableWidgetItem("pass" if check["ok"] else "FAIL"))
            self.checks.setItem(i, 2, QtWidgets.QTableWidgetItem(str(check["detail"])))
        self.checks.resizeColumnsToContents()

    def results_folders(self):
        return [self.controller.logs_dir()]


# ----------------------------------------------------------------------------- 1 scan
# The only paths of a cohort file; every other location is fixed by the cohort folder.
COHORT_PATH_KEYS = ("raw_root", "pixel_root")


class ScanPage(StepPage):
    jobs = ("scan",)

    def build_settings(self):
        box = QtWidgets.QGroupBox("Cohort file")
        form = QtWidgets.QFormLayout(box)
        self.path_edits = {}
        for key in COHORT_PATH_KEYS:
            edit = QtWidgets.QLineEdit()
            edit.editingFinished.connect(lambda k=key, e=edit: self._set_path(k, e.text()))
            self.path_edits[key] = edit
            form.addRow(key.replace("_", " "), edit)
        self.timepoints = QtWidgets.QLineEdit()
        self.timepoints.setToolTip("Timepoint classes in order, the first is the baseline, e.g. BL, 3D, 15D, 30D")
        self.timepoints.editingFinished.connect(self._set_timepoints)
        form.addRow("timepoints", self.timepoints)
        self.settings_layout.addWidget(box)

    def _set_path(self, key: str, text: str) -> None:
        cohort = self.controller.state.cohort
        value = text.strip() or None
        if cohort is not None and value != cohort.data.get(key):
            self.controller.set_cohort_value(key, value)

    def _set_timepoints(self) -> None:
        values = [t.strip() for t in self.timepoints.text().split(",") if t.strip()]
        if values != self.controller.state.cohort.timepoints:
            self.controller.set_cohort_value("timepoints", values)

    def build_tables(self):
        tabs = QtWidgets.QTabWidget()
        self.editors = {
            "sessions": TableEditor(
                "Sessions: one row per session folder", lambda f: self.controller.set_table("sessions", f),
                lambda p: self.controller.import_table("sessions", p), bool_columns=BOOL_COLUMNS["sessions"],
                note="session_folder = the folder name under raw_root; day = yymmdd; timepoint = one of the "
                     "cohort's timepoints; day_zero = a pre-baseline session; include = scan it."),
            "animals": TableEditor(
                "Animals: one row per animal folder name", lambda f: self.controller.set_table("animals", f),
                lambda p: self.controller.import_table("animals", p), bool_columns=BOOL_COLUMNS["animals"],
                note="raw_name = the folder name exactly; animal = the mouse (two names with the same animal are "
                     "aliases). Never start an animal with digits only (000 -> A1_000)."),
            "recordings": TableEditor(
                "Recordings: written by the scan", lambda f: self.controller.set_table("recordings", f),
                lambda p: self.controller.import_table("recordings", p),
                editable=EDITABLE_RECORDING_COLUMNS, bool_columns=BOOL_COLUMNS["recordings"],
                choices={"channel_order": CHANNEL_ORDERS}, can_add_rows=False,
                note="Untick 'use' to leave a recording out; set channel_order only when the automatic "
                     "decision is known to be wrong."),
        }
        for kind, editor in self.editors.items():
            tabs.addTab(editor, kind)
        return tabs

    def refresh_page(self):
        state = self.controller.state
        cohort = state.cohort
        if cohort is None:
            return
        for key, edit in self.path_edits.items():
            if not edit.hasFocus():
                edit.setText(str(cohort.data.get(key) or ""))
        if not self.timepoints.hasFocus():
            self.timepoints.setText(", ".join(cohort.timepoints))
        issues = self.controller.table_issues()
        for kind, editor in self.editors.items():
            editor.set_frame(state.tables[kind], issues.get(kind, []))

    def results_folders(self):
        cohort = self.controller.state.cohort
        return [cohort.layout.folder, cohort.layout.tables] if cohort else []


# ----------------------------------------------------------------------------- 2 acquisition
class AcquisitionPage(StepPage):
    jobs = ("facts",)

    def build_settings(self):
        box = QtWidgets.QGroupBox("Acquisition facts of this cohort")
        form = QtWidgets.QFormLayout(box)
        self.fields: dict[str, QtWidgets.QWidget] = {}
        for key in ACQUISITION_KEYS:
            if key == "confirmed":
                continue
            if key == "frame_order":
                widget = QtWidgets.QComboBox()
                widget.addItems(["", *FRAME_ORDERS])
            else:
                widget = QtWidgets.QLineEdit()
            self.fields[key] = widget
            form.addRow(key.replace("_", " "), widget)
        self.fields["sampling_rate_hz"].setToolTip("Per channel, in Hz (for example 10 or 25).")
        self.fields["grid"].setToolTip("rows, cols of the final grid, e.g. 128, 128")
        buttons = QtWidgets.QHBoxLayout()
        apply = QtWidgets.QPushButton("Apply values")
        apply.clicked.connect(lambda: self._apply(False))
        confirm = QtWidgets.QPushButton("Apply and confirm")
        confirm.clicked.connect(lambda: self._apply(True))
        suggest = QtWidgets.QPushButton("Use the values read from the files")
        suggest.clicked.connect(self._use_suggested)
        buttons.addWidget(apply)
        buttons.addWidget(confirm)
        buttons.addWidget(suggest)
        buttons.addStretch()
        form.addRow(buttons)
        self.settings_layout.addWidget(box)
        self.facts = QtWidgets.QLabel()
        self.facts.setWordWrap(True)
        self.facts.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.settings_layout.addWidget(self.facts)

    def _values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, widget in self.fields.items():
            text = (widget.currentText() if isinstance(widget, QtWidgets.QComboBox) else widget.text()).strip()
            if not text:
                values[key] = None
            elif key in ("sampling_rate_hz", "trim_s"):
                values[key] = float(text) if "." in text else int(text)
            elif key in ("rot90_k", "expected_files"):
                values[key] = int(text)
            elif key == "grid":
                values[key] = [int(v) for v in text.replace(";", ",").split(",")]
            else:
                values[key] = text
        return values

    def _apply(self, confirm: bool) -> None:
        self.controller.set_acquisition(self._values(), confirm)

    def facts_path(self) -> Path:
        return self.controller.state.cohort.layout.outputs / "acquisition_facts.json"

    def _use_suggested(self) -> None:
        if not self.facts_path().exists():
            QtWidgets.QMessageBox.information(self, "Acquisition", "Read the facts first (Run).")
            return
        summary = json.loads(self.facts_path().read_text(encoding="utf-8"))["summary"]
        self.fields["sampling_rate_hz"].setText(f"{summary['sampling_rate_hz']:g}")
        self.fields["frame_order"].setCurrentText(summary["frame_order"])
        self.fields["file_glob"].setText(summary["file_glob"])

    def refresh_page(self):
        cohort = self.controller.state.cohort
        if cohort is None:
            return
        for key, widget in self.fields.items():
            if widget.hasFocus():
                continue
            value = cohort.acquisition.get(key)
            if isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(value or "")
            else:
                widget.setText(text_of(value) if not isinstance(value, float) else f"{value:g}")
        if self.facts_path().exists():
            payload = json.loads(self.facts_path().read_text(encoding="utf-8"))
            rows = "".join(
                f"<tr><td>{s['day']}</td><td>{html.escape(s['timepoint'])}</td><td>{s['sampling_rate_hz']}</td>"
                f"<td>{s['suggested_frame_order']}</td><td>{s['image_shape']}</td><td>{s['n_files']}</td>"
                f"<td>{html.escape(' '.join(s['led_tokens']) or '-')}</td></tr>" for s in payload["sessions"])
            summary = payload["summary"]
            warnings = []
            if summary.get("led_settings_change"):
                warnings.append(colored("⚠ LED settings change between sessions: an illumination change "
                                        "creates fake connectivity changes.", None))
            if not summary.get("rate_consistent"):
                warnings.append(colored("⚠ The frame rate differs between sessions.", False))
            self.facts.setText(
                "<b>Read from the files</b><table cellpadding=3><tr><th>day</th><th>timepoint</th><th>Hz</th>"
                f"<th>frame order</th><th>shape</th><th>files</th><th>LED tokens</th></tr>{rows}</table>"
                + "<br>".join(warnings))
        else:
            self.facts.setText("<i>Facts not read yet.</i>")


# ----------------------------------------------------------------------------- 3 ROIs
class RoisPage(StepPage):
    jobs = ("bake", "rebuild", "preflight")

    def make_sections(self):
        run = self.window.run_job
        return [
            JobSection(self, "bake", [("Bake previews", lambda: run("bake"))]),
            JobSection(self, "rebuild", [("Rebuild this cohort's new sets", self._rebuild)]),
            JobSection(self, "preflight", [("Run pre-flight", lambda: run("preflight"))]),
        ]

    def _new_editor(self) -> None:
        self.window.enqueue(self.controller.new_roi_editor_job())

    def _rebuild(self) -> None:
        if not self.controller.has_new_reviewed_sets():
            QtWidgets.QMessageBox.information(self, "Rebuild", "No new or changed reviewed ROI sets for this "
                                              "cohort: nothing to rebuild.")
            return
        self.window.run_job("rebuild")

    def build_settings(self):
        note = QtWidgets.QLabel(
            "The ROI editor opens in its own window; the queue waits until you close it. When it closes, the "
            "GUI rebuilds only this cohort's new sets (never the whole folder) and runs the pre-flight.")
        note.setWordWrap(True)
        self.settings_layout.addWidget(note)
        # The editor takes everything from the cohort file, so it has no parameter form.
        box = QtWidgets.QGroupBox("ROI editor  (python -m wfci_gui.roi_editor)")
        row = QtWidgets.QHBoxLayout(box)
        button = QtWidgets.QPushButton("Open ROI editor")
        button.clicked.connect(lambda checked=False: self._new_editor())
        row.addWidget(button)
        row.addStretch()
        self.settings_layout.addWidget(box)

    def primary(self):
        self.window.run_job("preflight")


# ----------------------------------------------------------------------------- 4 extract
class ExtractPage(StepPage):
    jobs = ("extract",)

    def make_sections(self):
        return [JobSection(self, "extract", [("Smoke test (one unit, debug)", self._smoke),
                                             ("Full run", self.primary)])]

    def build_settings(self):
        self.disk = QtWidgets.QLabel()
        self.disk.setWordWrap(True)
        self.settings_layout.addWidget(self.disk)

    def _smoke(self) -> None:
        unit = smoke_unit(self.controller.state.tables)
        if unit is None:
            QtWidgets.QMessageBox.information(self, "Smoke test", "No recordings to test (scan first).")
            return
        self.window.run_job("extract", extra={"debug": True, "select": [unit]},
                            label=f"Extraction: smoke test {unit}")

    def remaining(self) -> int:
        state = self.controller.state
        used = used_recordings(state.tables)
        cohort = state.cohort
        return sum(not (cohort.layout.outputs / k / "connectivity_full.npz").exists()
                   for k in used["key"]) if not used.empty else 0

    def primary(self):
        cohort = self.controller.state.cohort
        free = free_gb(cohort.layout.pixel_data)
        need = needed_dump_gb(self.remaining(), 0)
        if free is not None and free < need:
            answer = QtWidgets.QMessageBox.warning(
                self, "Disk space", f"About {need:.0f} GB are needed and only {free:.0f} GB are free on the "
                "pixel drive. Run anyway?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self.window.run_job("extract")

    def refresh_page(self):
        cohort = self.controller.state.cohort
        if cohort is None:
            return
        remaining = self.remaining()
        free = free_gb(cohort.layout.pixel_data)
        need = needed_dump_gb(remaining, 0)
        ok = free is not None and free >= need
        free_text = f"{free:.0f} GB free" if free is not None else "drive not found"
        self.disk.setText(colored(f"Disk: {remaining} recordings still to extract need about {need:.0f} GB "
                                  f"in {cohort.layout.pixel_data} ({free_text}).", ok)
                          + f"<br>Time: about {remaining * 1.5 / 60:.1f}–{remaining * 4 / 60:.1f} h "
                            "(1.5–4 min per recording).")


# ----------------------------------------------------------------------------- 5 QC
class QcPage(StepPage):
    jobs = ("qc",)

    def build_settings(self):
        row = QtWidgets.QHBoxLayout()
        for label, callback in (("Channel trace plots", self._traces), ("Traces by animal", self._by_animal),
                                ("Inspect channel labels (viewer)", self._inspect),
                                ("Channel rectangle viewer (selected session)", self._rectangle_viewer)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        row.addStretch()
        self.settings_layout.addLayout(row)
        self.levels = QtWidgets.QLabel()
        self.levels.setWordWrap(True)
        self.settings_layout.addWidget(self.levels)

    def build_tables(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        note = QtWidgets.QLabel(
            "Ticked rows will be excluded. Suggestions (glitch > 1, channels not alternating) are ticked "
            "for you; nothing is excluded until you press <b>Confirm exclusion list</b>, and every "
            "excluded recording needs a reason.")
        note.setWordWrap(True)
        layout.addWidget(note)
        bar = QtWidgets.QHBoxLayout()
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter…")
        self.suggest_filter = QtWidgets.QComboBox()
        self.suggest_filter.addItems(["all", "exclude", "check", "ok", "ticked"])
        bar.addWidget(self.filter, 1)
        bar.addWidget(QtWidgets.QLabel("show:"))
        bar.addWidget(self.suggest_filter)
        confirm = QtWidgets.QPushButton("Confirm exclusion list")
        confirm.clicked.connect(self._confirm)
        reopen = QtWidgets.QPushButton("Reopen list")
        reopen.clicked.connect(lambda: self.controller.set_exclusions(self.controller.state.cohort.exclusions, False))
        bar.addWidget(confirm)
        bar.addWidget(reopen)
        layout.addLayout(bar)
        self.model = FrameModel(editable=("exclude", "exclude_reason"), bool_columns=("exclude",), parent=self)
        self.proxy = QtCore.QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.filter.textChanged.connect(self._filter)
        self.suggest_filter.currentTextChanged.connect(self._filter)
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        layout.addWidget(self.view, 1)
        self.loaded_from: tuple | None = None
        return widget

    def _filter(self) -> None:
        text = self.filter.text()
        choice = self.suggest_filter.currentText()
        pattern = "" if choice == "all" else choice
        # Simple combined filter: free text, plus the suggestion / ticked column.
        if choice == "ticked":
            self.proxy.setFilterKeyColumn(0)
            self.proxy.setFilterFixedString("true")
        elif pattern:
            self.proxy.setFilterKeyColumn(list(self.model.frame.columns).index("suggest"))
            self.proxy.setFilterFixedString(pattern)
        else:
            self.proxy.setFilterKeyColumn(-1)
            self.proxy.setFilterFixedString(text)

    def _load_table(self) -> None:
        cohort = self.controller.state.cohort
        path = qc_table_path(cohort)
        key = (str(path), path.stat().st_mtime if path.exists() else 0, cohort.get("exclusions_confirmed"),
               tuple(e["recording"] for e in cohort.exclusions))
        if key == self.loaded_from:
            return
        self.loaded_from = key
        if not path.exists():
            self.model.set_frame(pd.DataFrame())
            return
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        reasons = {e["recording"]: e["reason"] for e in cohort.exclusions}
        if reasons or cohort.get("exclusions_confirmed"):
            ticked = table["recording"].isin(reasons)
            reason = table["recording"].map(reasons).fillna("")
        else:
            # Pre-tick the suggestions; the reason starts as the QC reason.
            ticked = table["suggest"] == "exclude"
            reason = table["reason"].where(ticked, "")
        table.insert(0, "exclude", ticked.map({True: "true", False: "false"}))
        table.insert(1, "exclude_reason", reason)
        self.model.set_frame(table)
        self.view.resizeColumnsToContents()

    def _confirm(self) -> None:
        frame = self.model.frame
        if frame.empty:
            QtWidgets.QMessageBox.information(self, "QC", "Build the QC table first (Run).")
            return
        ticked = frame[frame["exclude"] == "true"]
        missing = ticked.loc[ticked["exclude_reason"].str.strip() == "", "recording"].tolist()
        if missing:
            QtWidgets.QMessageBox.warning(self, "QC", "Every excluded recording needs a reason. Missing for:\n"
                                          + "\n".join(missing[:20]))
            return
        exclusions = [dict(recording=r.recording, reason=r.exclude_reason.strip()) for r in ticked.itertuples()]
        self.controller.set_exclusions(exclusions, True)
        self.window.save_cohort()

    def _traces(self) -> None:
        cohort = self.controller.state.cohort
        self.window.enqueue(self.controller.viewer_job("scripts/plot_channel_traces.py", [
            "--cohort", str(cohort.path),
            "--channel-assignment-csv", "none", "--old-manifest-csv", "none",
            "--expected-disagreements", "none"], "QC: channel trace plots"))

    def _by_animal(self) -> None:
        cohort = self.controller.state.cohort
        self.window.enqueue(self.controller.viewer_job("scripts/plot_channel_traces_by_animal.py", [
            "--cohort", str(cohort.path), "--recordings-from-design", "--exclude-timepoints"],
            "QC: traces by animal"))

    def _inspect(self) -> None:
        cohort = self.controller.state.cohort
        day = self._selected_day()
        self.controller.launch_detached("scripts/inspect_channel_labels.py", [
            "--cohort", str(cohort.path), "--session", str(day)])

    def _selected_day(self) -> str:
        index = self.view.currentIndex()
        if index.isValid():
            return str(self.model.frame.iloc[self.proxy.mapToSource(index).row()]["day"])
        return str(used_recordings(self.controller.state.tables)["day"].min())

    def _rectangle_viewer(self) -> None:
        """scripts/channel_rectangle_viewer.py on one session: odd | even mean images with
        the midline rectangle; parameters from the cohort's channel_rectangle_params.json."""
        cohort = self.controller.state.cohort
        arguments, note = rectangle_viewer_arguments(cohort, self.controller.state.tables, self._selected_day())
        if not arguments:
            QtWidgets.QMessageBox.information(self, "Channel rectangle viewer", note)
            return
        if note:
            self.window.notify(note)
        self.controller.launch_detached("scripts/channel_rectangle_viewer.py", arguments)

    def refresh_page(self):
        cohort = self.controller.state.cohort
        if cohort is None:
            return
        self._load_table()
        levels = cohort.layout.outputs / "qc_levels_by_session.csv"
        if levels.exists():
            frame = pd.read_csv(levels, dtype=str, keep_default_na=False)
            rows = "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in r) + "</tr>"
                           for r in frame.itertuples(index=False))
            header = "".join(f"<th>{c}</th>" for c in frame.columns)
            self.levels.setText("<b>Illumination level per session</b> (a GCaMP step with flat reflectance is an "
                                f"acquisition change, not biology):<table cellpadding=3><tr>{header}</tr>{rows}</table>")


# ----------------------------------------------------------------------------- 6 caches
class CachesPage(StepPage):
    jobs = ("cache_median", "cache_blpedestal")

    def make_sections(self):
        return [JobSection(self, "cache_median", [("Build median caches", self.primary)]),
                JobSection(self, "cache_blpedestal", [("Build BL-pedestal cache (optional)",
                                                       lambda: self.window.run_job("cache_blpedestal"))])]

    def build_settings(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Running-median windows (s), comma-separated:"))
        self.windows = QtWidgets.QLineEdit()
        self.windows.editingFinished.connect(self._set_windows)
        row.addWidget(self.windows)
        self.settings_layout.addLayout(row)

    def _set_windows(self) -> None:
        values = [float(v) for v in self.windows.text().replace(";", ",").split(",") if v.strip()]
        current = cache_windows(self.controller.state.cohort)
        if values and values != current:
            value: Any = values[0] if len(values) == 1 else values
            if value == 20.0:
                self.controller.reset_override("cache_median", "window_s")
            else:
                self.controller.set_override("cache_median", "window_s", value)

    def primary(self):
        for window in cache_windows(self.controller.state.cohort):
            self.window.run_job("cache_median", variant=window, label=f"Median dF/F cache: {window:g} s")

    def refresh_page(self):
        cohort = self.controller.state.cohort
        if cohort is not None and not self.windows.hasFocus():
            self.windows.setText(", ".join(f"{w:g}" for w in cache_windows(cohort)))

    def results_folders(self):
        cohort = self.controller.state.cohort
        return [cohort.layout.logs] if cohort else []


# ----------------------------------------------------------------------------- 7 design
class DesignPage(StepPage):
    jobs = ("design",)

    def results_folders(self):
        cohort = self.controller.state.cohort
        return [design_path(cohort).parent] if cohort else []

    def build_tables(self):
        self.design_view = TableEditor("design.csv (read only; rebuild it to change it)",
                                       lambda f: None, None, editable=(), can_add_rows=False)
        self.loaded = None
        return self.design_view

    def refresh_page(self):
        cohort = self.controller.state.cohort
        if cohort is None:
            return
        path = design_path(cohort)
        key = (str(path), path.stat().st_mtime if path.exists() else 0)
        if key != self.loaded:
            self.loaded = key
            frame = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()
            self.design_view.set_frame(frame, [])


# ----------------------------------------------------------------------------- A connectivity
class AnalysisAPage(StepPage):
    jobs = ("connectivity", "stats")

    def make_sections(self):
        return [JobSection(self, "connectivity", [("Run connectivity only", self._connectivity_only)]),
                JobSection(self, "stats", [("Run statistics only", self._stats_only)])]

    def build_settings(self):
        box = QtWidgets.QGroupBox("What to analyse")
        grid = QtWidgets.QGridLayout(box)
        self.baseline = QtWidgets.QComboBox()
        self.baseline.textActivated.connect(lambda t: self._set("baseline", t))
        self.focus = QtWidgets.QComboBox()
        self.focus.addItems(["SSp", "motor"])
        self.focus.textActivated.connect(lambda t: self._set("focus", t))
        self.timepoints = QtWidgets.QListWidget()
        self.animals = QtWidgets.QListWidget()
        self.variants = QtWidgets.QListWidget()
        for widget, key in ((self.timepoints, "timepoints"), (self.animals, "animals"), (self.variants, "variants")):
            widget.itemChanged.connect(lambda item, k=key, w=widget: self._set_checked(k, w))
            widget.setMaximumHeight(170)
        grid.addWidget(QtWidgets.QLabel("Baseline timepoint"), 0, 0)
        grid.addWidget(self.baseline, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Region focus"), 0, 2)
        grid.addWidget(self.focus, 0, 3)
        for column, (label, widget) in enumerate((("Timepoints", self.timepoints), ("Animals", self.animals),
                                                  ("Variants", self.variants))):
            grid.addWidget(QtWidgets.QLabel(label), 1, column * 2)
            grid.addWidget(widget, 2, column * 2, 1, 2)
        self.check = QtWidgets.QLabel()
        self.check.setWordWrap(True)
        grid.addWidget(self.check, 3, 0, 1, 6)
        self.run_all = QtWidgets.QPushButton("Queue connectivity for every variant, then the statistics")
        self.run_all.clicked.connect(self.primary)
        grid.addWidget(self.run_all, 4, 0, 1, 6)
        self.settings_layout.addWidget(box)
        self._updating = False

    def _choice(self) -> dict[str, Any]:
        return self.controller.state.cohort.overrides("analysis_a")

    def _set(self, key: str, value: Any) -> None:
        if self._updating or self._choice().get(key) == value:
            return
        if value is None:
            self.controller.reset_override("analysis_a", key)
        else:
            self.controller.set_override("analysis_a", key, value)

    def _set_checked(self, key: str, widget: QtWidgets.QListWidget) -> None:
        if self._updating:
            return
        values = [widget.item(i).text() for i in range(widget.count())
                  if widget.item(i).checkState() == QtCore.Qt.Checked]
        self._set(key, values)

    def design_frame(self) -> pd.DataFrame:
        cohort = self.controller.state.cohort
        path = design_path(cohort)
        if not path.exists():
            return pd.DataFrame(columns=["animal", "timepoint", "day", "recording"])
        design = pd.read_csv(path, dtype=str, keep_default_na=False)
        excluded = cohort.excluded_keys()
        keys = design["day"] + "_" + design["animal"] + "/" + design["recording"]
        return design[~keys.isin(excluded)]

    def available_variants(self) -> list[str]:
        cohort = self.controller.state.cohort
        variants = list(CONNECTIVITY_VARIANTS)
        pixel_root = cohort.layout.pixel_data
        if any(pixel_root.glob("*/*/pixels_median_dff_20s_blpedestal_full.npy")):
            variants += ["pixels_median_dff_20s_blpedestal_full", "pixels_median_dff_20s_blpedestal_full_gsr"]
        return variants

    def current(self) -> tuple[str, list[str], list[str], list[str]]:
        cohort = self.controller.state.cohort
        choice = self._choice()
        design = self.design_frame()
        timepoints = list(choice.get("timepoints") or cohort.timepoints)
        animals = list(choice.get("animals") or sorted(design["animal"].unique()))
        baseline = choice.get("baseline") or timepoints[0]
        return baseline, timepoints, animals, chosen_variants(cohort)

    def design_check(self):
        baseline, timepoints, animals, _ = self.current()
        return check_design(self.design_frame(), baseline, timepoints, animals)

    def refresh_page(self):
        cohort = self.controller.state.cohort
        if cohort is None:
            return
        self._updating = True
        baseline, timepoints, animals, variants = self.current()
        design = self.design_frame()
        self.baseline.clear()
        self.baseline.addItems(cohort.timepoints)
        self.baseline.setCurrentText(baseline)
        self.focus.setCurrentText(self._choice().get("focus", "SSp"))
        for widget, options, chosen in ((self.timepoints, cohort.timepoints, timepoints),
                                        (self.animals, sorted(design["animal"].unique()), animals),
                                        (self.variants, self.available_variants(), variants)):
            widget.clear()
            for option in options:
                item = QtWidgets.QListWidgetItem(option)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked if option in chosen else QtCore.Qt.Unchecked)
                widget.addItem(item)
        self._updating = False
        result = self.design_check()
        self.check.setText(colored(html.escape(result.message), result.complete))
        self.run_all.setEnabled(result.complete and bool(variants))
        for section in self.sections:
            section.preview_variant = variants[0] if variants else None

    def _connectivity_only(self) -> None:
        for variant in self.current()[3]:
            self.window.run_job("connectivity", variant=variant)

    def _stats_only(self) -> None:
        if not self.design_check().complete:
            QtWidgets.QMessageBox.warning(self, "Statistics", "The animal x timepoint design is incomplete.")
            return
        for variant in self.current()[3]:
            self.window.run_job("stats", variant=variant)

    def primary(self):
        if not self.design_check().complete:
            QtWidgets.QMessageBox.warning(self, "Connectivity", self.design_check().message)
            return
        self._connectivity_only()
        self._stats_only()

    def results_folders(self):
        cohort = self.controller.state.cohort
        return [cohort.layout.connectivity, cohort.layout.connectivity_stats] if cohort else []


PAGE_CLASSES = {
    "setup": SetupPage, "scan": ScanPage, "acquisition": AcquisitionPage, "rois": RoisPage,
    "extract": ExtractPage, "qc": QcPage, "caches": CachesPage, "design": DesignPage,
    "analysis_a": AnalysisAPage,
}

