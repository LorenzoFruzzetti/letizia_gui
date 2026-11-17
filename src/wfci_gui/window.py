"""Main window: qtkit ShellWindow with the step list, step pages and the job dock."""

from __future__ import annotations

import html
from pathlib import Path

from qtkit import ActionSpec, ShellWindow, Step
from qtkit.qt import QtCore, QtGui, QtWidgets

from . import __version__
from .controller import Controller
from .logic import REPO_ROOT
from .logic.presets import export_overrides
from .logic.steps import STEPS
from .views.pages import PAGE_CLASSES
from .views.widgets import LogDock

COHORT_DIR = REPO_ROOT / "cohorts"
MAX_RECENT = 10

ACTIONS = (
    ActionSpec("cohort.new", "New cohort…", "Ctrl+N", "new_cohort", "SP_FileIcon"),
    ActionSpec("cohort.open", "Open cohort…", "Ctrl+O", "open_cohort_dialog", "SP_DirOpenIcon"),
    ActionSpec("cohort.save", "Save cohort", "Ctrl+S", "save_cohort", "SP_DialogSaveButton"),
    ActionSpec("cohort.revert", "Revert unsaved changes", "", "revert_cohort", "SP_DialogResetButton"),
    ActionSpec("cohort.export", "Export all parameters…", "", "export_parameters", "SP_ArrowUp"),
    ActionSpec("cohort.import", "Import all parameters…", "", "import_parameters", "SP_ArrowDown"),
    ActionSpec("run.step", "Run step", "F5", "run_current_step", "SP_MediaPlay"),
    ActionSpec("run.stop", "Stop", "Shift+F5", "stop_job", "SP_MediaStop"),
    ActionSpec("run.refresh", "Refresh status", "Ctrl+R", "refresh_status", "SP_BrowserReload"),
    ActionSpec("run.open_folder", "Open folder", "", "open_output_folder", "SP_DirIcon"),
    ActionSpec("view.advanced", "Show commands (advanced)", "", "toggle_advanced", "SP_FileDialogDetailedView"),
    ActionSpec("app.shortcuts", "Keyboard shortcuts…", "", "edit_shortcuts", "SP_FileDialogListView"),
    ActionSpec("app.toolbar", "Customize toolbar…", "", "edit_toolbar", "SP_FileDialogContentsView"),
    ActionSpec("app.reset_commands", "Reset menus and shortcuts", "", "reset_commands", "SP_DialogResetButton"),
    ActionSpec("app.open_config", "Open settings folder", "", "open_config_folder", "SP_DirHomeIcon"),
    ActionSpec("app.about", "About", "", "about", "SP_MessageBoxInformation"),
    ActionSpec("app.quit", "Quit", "Ctrl+Q", "close", "SP_DialogCloseButton"),
)
DEFAULT_LAYOUT = {
    "menus": {
        "File": ["cohort.save", "cohort.revert", "-", "cohort.export", "cohort.import", "-", "app.quit"],
        "Cohort": ["cohort.new", "cohort.open"],
        "Run": ["run.step", "run.stop", "-", "run.refresh", "run.open_folder"],
        "Help": ["app.shortcuts", "app.toolbar", "app.reset_commands", "-", "app.open_config", "app.about"],
    },
    "toolbars": {"main": ["run.step", "run.stop", "run.open_folder", "-", "cohort.save", "run.refresh"]},
    "shortcuts": {},
}


class MainWindow(ShellWindow):
    def __init__(self, settings: QtCore.QSettings, config_dir: Path, environ: dict | None = None):
        super().__init__(settings, config_dir)
        self.setWindowTitle("WFCI cohort pipeline")
        self.resize(1400, 900)
        self.controller = Controller(Path(config_dir), REPO_ROOT, environ)
        self.current_step = "setup"
        self.install_commands(ACTIONS, DEFAULT_LAYOUT, self)
        self.registry.actions["view.advanced"].setCheckable(True)
        self.cohort_combo = QtWidgets.QComboBox()
        self.cohort_combo.setMinimumWidth(240)
        self.cohort_combo.setToolTip("Cohort file")
        self.cohort_combo.textActivated.connect(self._combo_activated)
        # Pages, one per step.
        self.stack = QtWidgets.QStackedWidget()
        self.pages = {}
        for step in STEPS:
            page = PAGE_CLASSES[step.key](self, step)
            self.pages[step.key] = page
            self.stack.addWidget(page)
        self.placeholder = QtWidgets.QLabel("Open a cohort (Cohort ▸ Open, or the box in the toolbar) "
                                            "or create one (Cohort ▸ New).")
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.stack.addWidget(self.placeholder)
        self.setCentralWidget(self.stack)
        self.log_dock = LogDock(self.controller)
        self.add_dock("jobs", "Job queue and log", self.log_dock, QtCore.Qt.BottomDockWidgetArea, 260)
        guide_steps = [Step(s.key, f"{s.number} · {s.title}", None, s.help_html) for s in STEPS]
        self.add_guide(guide_steps, self.progress, QtCore.Qt.LeftDockWidgetArea, 300)
        self.docks["pipeline_guide"].setWindowTitle("Steps")
        self.guide.steps.currentRowChanged.connect(lambda row: self.show_step_view(STEPS[row].key))
        self.build_commands()
        self.restore_window()
        self.controller.changed.connect(self.refresh)
        self.refresh()

    # ----- shell hooks ----------------------------------------------------------------
    def build_commands(self):
        super().build_commands()
        # The View menu is built by ShellWindow; add the advanced toggle to it.
        view_menu = next(m for m in self.menuBar().findChildren(QtWidgets.QMenu) if m.title() == "View")
        view_menu.addSeparator()
        view_menu.addAction(self.registry.actions["view.advanced"])
        if self.toolbars:
            first = self.toolbars[0]
            first.insertWidget(first.actions()[0] if first.actions() else None, self.cohort_combo)
            first.insertSeparator(first.actions()[1] if len(first.actions()) > 1 else None)

    def extend_menu(self, title, menu):
        if title == "Cohort":
            recent = menu.addMenu("Recent")
            for path in self.recent_cohorts():
                recent.addAction(str(path), lambda p=path: self.open_cohort(p))
            recent.addSeparator()
            recent.addAction("Clear list", lambda: (self.settings.setValue("recent", []), self.build_commands()))

    def update_actions(self):
        state = self.controller.state
        has_cohort = state.cohort is not None
        for key in ("cohort.save", "cohort.revert"):
            self.registry.actions[key].setEnabled(has_cohort and (state.cohort_dirty or bool(state.dirty_tables)))
        for key in ("cohort.export", "cohort.import", "run.open_folder"):
            self.registry.actions[key].setEnabled(has_cohort)
        self.registry.actions["run.stop"].setEnabled(self.controller.queue.busy)
        self.registry.actions["run.step"].setEnabled(has_cohort or self.current_step == "setup")

    def show_step_view(self, key):
        self.current_step = key
        if self.controller.state.cohort is None and key != "setup":
            self.stack.setCurrentWidget(self.placeholder)
        else:
            self.stack.setCurrentWidget(self.pages[key])
            self.pages[key].refresh()
        self.update_actions()

    def notify(self, message):
        self.statusBar().showMessage(message, 10000)
        self.controller.log_line.emit(f"[gui] {message}")

    # ----- state -> view --------------------------------------------------------------
    def progress(self, state):
        notes = self.controller.history_notes()
        rows = []
        for step in STEPS:
            status = state.statuses.get(step.key)
            if status is None:
                rows.append((False, "open a cohort" if step.key != "setup" else "not checked"))
                continue
            detail = status.detail
            if notes.get(step.key):
                detail += f" · {notes[step.key]}"
            if status.warnings:
                detail += f" · ⚠ {len(status.warnings)} warning(s)"
            rows.append((status.done, detail))
        return rows

    def refresh(self):
        state = self.controller.state
        cohort = state.cohort
        title = "WFCI cohort pipeline"
        if cohort is not None:
            title = f"{cohort.name}[*] — {title}"
        self.setWindowTitle(title)
        self.setWindowModified(state.cohort_dirty or bool(state.dirty_tables))
        self._fill_combo()
        self.guide.refresh(state)
        self.show_step_view(self.current_step)

    def _fill_combo(self) -> None:
        cohort = self.controller.state.cohort
        files = sorted(COHORT_DIR.glob("*.yaml")) if COHORT_DIR.is_dir() else []
        items = [str(p.stem) for p in files]
        current = cohort.name if cohort else ""
        if [self.cohort_combo.itemText(i) for i in range(self.cohort_combo.count())] != ["", *items]:
            self.cohort_combo.clear()
            self.cohort_combo.addItems(["", *items])
        self.cohort_combo.setCurrentText(current)

    # ----- cohort commands ------------------------------------------------------------
    def _combo_activated(self, name: str) -> None:
        cohort = self.controller.state.cohort
        if name and (cohort is None or cohort.name != name):
            self.open_cohort(COHORT_DIR / f"{name}.yaml")

    def recent_cohorts(self) -> list[Path]:
        values = self.settings.value("recent", []) or []
        values = [values] if isinstance(values, str) else list(values)
        return [Path(v) for v in values if Path(v).exists()]

    def remember(self, path: Path) -> None:
        values = [str(path.resolve())] + [str(p) for p in self.recent_cohorts() if p.resolve() != path.resolve()]
        self.settings.setValue("recent", values[:MAX_RECENT])
        self.build_commands()

    def maybe_save(self) -> bool:
        """Ask about unsaved changes; False means the user cancelled."""
        state = self.controller.state
        if not (state.cohort_dirty or state.dirty_tables):
            return True
        answer = QtWidgets.QMessageBox.question(
            self, "Unsaved changes", "The cohort has unsaved changes. Save them?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if answer == QtWidgets.QMessageBox.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.Save:
            return self.save_cohort()
        return True

    def open_cohort(self, path) -> None:
        if not self.maybe_save():
            return
        self.controller.open_cohort(Path(path))
        self.remember(Path(path))
        if self.current_step == "setup":
            self.current_step = "scan"

    def open_cohort_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open cohort", str(COHORT_DIR), "Cohort files (*.yaml)")
        if path:
            self.open_cohort(path)

    def new_cohort(self) -> None:
        if not self.maybe_save():
            return
        dialog = NewCohortDialog(self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        values, template = dialog.values()
        data = self.controller.new_cohort_data(**values)
        if template is not None:
            from .logic.cohort import load_cohort
            data["overrides"] = load_cohort(template).get("overrides")
            data["overrides"].get("analysis_a", {}).pop("stats_cohort", None)
        self.controller.create_cohort(data, COHORT_DIR)
        self.remember(COHORT_DIR / f"{values['name']}.yaml")
        self.current_step = "scan"
        self.refresh()

    def save_cohort(self) -> bool:
        """Show the diff against the file on disk, then save."""
        state = self.controller.state
        if state.cohort is None:
            return False
        diff = self.controller.pending_diff()
        tables = ", ".join(sorted(state.dirty_tables))
        if not diff and not tables:
            self.notify("Nothing to save.")
            return True
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save cohort")
        dialog.resize(800, 500)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel(f"Changes to {state.cohort.path.name}:" if diff else "No change to the cohort file."))
        text = QtWidgets.QPlainTextEdit(diff)
        text.setReadOnly(True)
        text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        layout.addWidget(text, 1)
        if tables:
            layout.addWidget(QtWidgets.QLabel(f"Tables also saved: {tables}"))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return False
        self.controller.save_cohort()
        self.notify(f"Saved {state.cohort.path}")
        return True

    def revert_cohort(self) -> None:
        self.controller.revert_cohort()

    def export_parameters(self) -> None:
        cohort = self.controller.state.cohort
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export parameters", f"{cohort.name}_parameters.json",
                                                        "JSON (*.json)")
        if path:
            export_overrides(Path(path), cohort.name, cohort.get("overrides"))
            self.notify(f"Parameters exported to {path}")

    def import_parameters(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import parameters", "", "JSON (*.json)")
        if not path:
            return
        from .logic.presets import diff_overrides, import_overrides
        new = import_overrides(Path(path))
        current = self.controller.state.cohort.get("overrides")
        lines = []
        for job in sorted(set(current) | set(new)):
            for name, old, value in diff_overrides(current.get(job, {}), new.get(job, {})):
                lines.append(f"{job}.{name}: {old!r} → {value!r}")
        answer = QtWidgets.QMessageBox.question(
            self, "Import parameters", "Replace every parameter of this cohort?\n\n" + ("\n".join(lines) or "No change."))
        if answer == QtWidgets.QMessageBox.Yes:
            self.controller.import_all_overrides(Path(path))

    # ----- run commands ---------------------------------------------------------------
    def run_job(self, job: str, variant=None, extra=None, label=None) -> None:
        """Queue one job of the current cohort, refusing visibly when the step is blocked."""
        step = self.controller.job_spec(job).step
        blockers = self.controller.blockers(step)
        if blockers:
            names = ", ".join(next(f"{s.number} {s.title}" for s in STEPS if s.key == b) for b in blockers)
            QtWidgets.QMessageBox.information(
                self, "Step blocked", f"This step can run once these steps are done: {names}.")
            return
        self.enqueue(self.controller.make_job(job, variant, extra, label))

    def enqueue(self, job) -> None:
        state = self.controller.state
        if (state.cohort_dirty or state.dirty_tables) and not self.save_cohort():
            return
        self.controller.enqueue(job)
        self.notify(f"Queued: {job.label}")

    def run_current_step(self) -> None:
        self.pages[self.current_step].primary()

    def stop_job(self) -> None:
        self.controller.stop()

    def refresh_status(self) -> None:
        self.controller.refresh_status()

    def open_output_folder(self) -> None:
        folders = [f for f in self.pages[self.current_step].results_folders() if f.is_dir()]
        if folders:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folders[0])))
        else:
            self.notify("This step has no output folder yet.")

    def toggle_advanced(self) -> None:
        self.controller.state.advanced = self.registry.actions["view.advanced"].isChecked()
        self.refresh()

    def open_config_folder(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.config_dir)))

    def about(self) -> None:
        QtWidgets.QMessageBox.about(self, "About", f"<b>WFCI cohort pipeline</b> {__version__}<br>"
                                    f"Repository: {html.escape(str(REPO_ROOT))}")

    def closeEvent(self, event):
        if self.controller.queue.busy:
            answer = QtWidgets.QMessageBox.question(
                self, "Job running", "A job is running. Stop it and quit? (It can be resumed later.)")
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.controller.queue.clear_pending()
            self.controller.stop()
        if not self.maybe_save():
            event.ignore()
            return
        self.save_window()
        event.accept()


class NewCohortDialog(QtWidgets.QDialog):
    """Wizard: name, raw root, junction root, frame rate, frame order, timepoints; optionally
    the parameters of another cohort as a starting point."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("New cohort")
        self.resize(640, 320)
        form = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("e.g. cohort_a (letters, digits, _ and -)")
        self.raw_root = self._path_row(form, "Raw data folder")
        self.link_root = self._path_row(form, "Junction view folder (optional)")
        form.insertRow(0, "Name", self.name)
        self.rate = QtWidgets.QLineEdit()
        self.rate.setPlaceholderText("per channel, e.g. 10 or 25; blank = read it in step 2")
        form.addRow("Frame rate (Hz)", self.rate)
        self.frame_order = QtWidgets.QComboBox()
        self.frame_order.addItems(["", "name", "number"])
        form.addRow("Frame order", self.frame_order)
        self.timepoints = QtWidgets.QLineEdit("BL, 3D, 15D, 30D")
        form.addRow("Timepoints", self.timepoints)
        self.template = QtWidgets.QComboBox()
        self.template.addItems(["(none)", *sorted(p.stem for p in COHORT_DIR.glob("*.yaml"))])
        form.addRow("Start from the parameters of", self.template)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _path_row(self, form, label):
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(lambda: edit.setText(QtWidgets.QFileDialog.getExistingDirectory(self, label) or edit.text()))
        layout.addWidget(edit, 1)
        layout.addWidget(browse)
        form.addRow(label, row)
        return edit

    def values(self):
        rate = self.rate.text().strip()
        values = dict(name=self.name.text().strip(), raw_root=self.raw_root.text().strip(),
                      link_root=self.link_root.text().strip() or None,
                      sampling_rate_hz=float(rate) if rate else None,
                      frame_order=self.frame_order.currentText() or None,
                      timepoints=[t.strip() for t in self.timepoints.text().split(",") if t.strip()])
        template = self.template.currentText()
        return values, (COHORT_DIR / f"{template}.yaml" if template != "(none)" else None)
