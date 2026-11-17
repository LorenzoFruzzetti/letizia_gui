"""Every ROI-editor command and every mouse gesture.

The controller owns the model (``model.EditorModel``) and the loaded data
(``sources.LoadedData``); the window only draws them. A command is: record a pending
mouse gesture, change the model, record one undo step, redraw.

Command ids, labels and default shortcuts are the previous editor's (``KEY_OPS`` plus the
other actions), with the new editor's own commands added.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, NamedTuple

from qtkit import ActionSpec
from qtkit.qt import QtCore

from . import sources
from .logic import REPO_ROOT, EditorSettings
from .model import NUDGE_BIG, EditorModel

from roi_editor import HELP  # noqa: E402,F401  (repo root, via logic; re-exported for the window)


# ----------------------------------------------------------------------------- commands
class KeyOp(NamedTuple):
    """One page command with the previous editor's id, label and single-key shortcut."""
    id: str
    label: str
    shortcut: str
    icon: str
    edit: Callable[[EditorModel], None]   # what it does to the model
    files: bool = False                   # writes ROI sets (the table re-reads the disk)

    @property
    def method(self) -> str:
        return "op_" + self.id.replace(".", "_")


def _moves(prefix: str, what: str, shortcut: str, edit) -> list[KeyOp]:
    ops = []
    for name, icon, (d_row, d_col) in (("Up", "SP_ArrowUp", (-1, 0)), ("Down", "SP_ArrowDown", (1, 0)),
                                       ("Left", "SP_ArrowLeft", (0, -1)), ("Right", "SP_ArrowRight", (0, 1))):
        for step in (1, NUDGE_BIG):
            big = step > 1
            ops.append(KeyOp(f"{prefix}.{name.lower()}{'5' if big else ''}", f"{what} {name.lower()} {step} px",
                             f"{shortcut}{'Shift+' if big else ''}{name}", icon,
                             lambda model, dr=d_row * step, dc=d_col * step: edit(model, dr, dc)))
    return ops


KEY_OPS: tuple[KeyOp, ...] = (
    KeyOp("page.next", "Next page", "N", "SP_ArrowForward", lambda m: m.navigate(+1)),
    KeyOp("page.previous", "Previous page", "P", "SP_ArrowBack", lambda m: m.navigate(-1)),
    KeyOp("page.copy_previous", "Copy the previous page's layout", "C", "SP_FileDialogDetailedView",
          lambda m: m.copy_from_previous()),
    KeyOp("file.save", "Save this page", "S", "SP_DialogSaveButton", lambda m: m.save_current(), True),
    KeyOp("file.save_modified", "Save modified pages", "Ctrl+S", "SP_DialogApplyButton",
          lambda m: m.save_modified(), True),
    KeyOp("file.save_all", "Save all pages", "Shift+S", "SP_DriveHDIcon", lambda m: m.save_all(), True),
    KeyOp("file.write_shared", "Write the shared ROI set", "W", "SP_FileLinkIcon", lambda m: m.write_shared(), True),
    KeyOp("apply.all", "Apply boxes and scale to all pages", "A", "SP_FileDialogListView",
          lambda m: m.apply_to_all(False)),
    KeyOp("apply.all_bregma", "Apply boxes, scale and Bregma to all pages", "Shift+A", "SP_FileDialogListView",
          lambda m: m.apply_to_all(True)),
    KeyOp("reset.page", "Reset this page", "R", "SP_DialogResetButton", lambda m: m.reset_page()),
    KeyOp("reset.all", "Reset all pages", "Shift+R", "SP_DialogResetButton", lambda m: m.reset_all()),
    *_moves("bregma", "Move Bregma", "Ctrl+", lambda m, dr, dc: m.move_bregma_by(dr, dc)),
    *_moves("box", "Move the selected box", "", lambda m, dr, dc: m.nudge(dr, dc)),
    KeyOp("box.grow", "Grow the selected box", "+", "SP_TitleBarMaxButton", lambda m: m.resize_box(+1)),
    KeyOp("box.shrink", "Shrink the selected box", "-", "SP_TitleBarMinButton", lambda m: m.resize_box(-1)),
    KeyOp("lambda.shorter", "Lambda distance -1 px", ",", "SP_ArrowUp", lambda m: m.change_lambda(-1)),
    KeyOp("lambda.longer", "Lambda distance +1 px", ".", "SP_ArrowDown", lambda m: m.change_lambda(+1)),
    KeyOp("lambda.shorter5", "Lambda distance -5 px", "<", "SP_ArrowUp", lambda m: m.change_lambda(-NUDGE_BIG)),
    KeyOp("lambda.longer5", "Lambda distance +5 px", ">", "SP_ArrowDown", lambda m: m.change_lambda(+NUDGE_BIG)),
    KeyOp("lambda.reset_scale", "Back to scale 1.000x", "L", "SP_BrowserReload", lambda m: m.reset_scale()),
    KeyOp("rotate.left", "Rotate left 1 degree", "Z", "SP_ArrowLeft", lambda m: m.rotate(-1)),
    KeyOp("rotate.right", "Rotate right 1 degree", "X", "SP_ArrowRight", lambda m: m.rotate(+1)),
    KeyOp("rotate.left5", "Rotate left 5 degrees", "Shift+Z", "SP_ArrowLeft", lambda m: m.rotate(-NUDGE_BIG)),
    KeyOp("rotate.right5", "Rotate right 5 degrees", "Shift+X", "SP_ArrowRight", lambda m: m.rotate(+NUDGE_BIG)),
    KeyOp("flip.horizontal", "Mirror the layout left-right", "F", "SP_BrowserReload", lambda m: m.flip(True)),
    KeyOp("flip.vertical", "Mirror the layout up-down", "V", "SP_BrowserReload", lambda m: m.flip(False)),
    KeyOp("flip.lambda", "Flip Bregma-Lambda (all pages)", "U", "SP_BrowserReload", lambda m: m.flip_lambda_all()),
    KeyOp("toggle.mirror_lock", "Mirror lock on/off", "M", "SP_DialogYesButton", lambda m: m.toggle_mirror_lock()),
    KeyOp("toggle.rotate_with_lambda", "Rotate with Lambda on/off", "O", "SP_DialogYesButton",
          lambda m: m.toggle_rotate_with_lambda()),
    KeyOp("rect.reset", "Reset rectangle to params.json", "", "SP_DialogResetButton", lambda m: m.reset_rectangle()),
    # Which interleaved group is GCaMP, the user's call for this recording (saved in its
    # ROI set as channel_order; the other group is reflectance).
    KeyOp("channel.odd", "Odd frames are GCaMP (even = reflectance)", "Alt+1", "SP_MediaSeekBackward",
          lambda m: m.choose_gcamp("odd")),
    KeyOp("channel.even", "Even frames are GCaMP (odd = reflectance)", "Alt+2", "SP_MediaSeekForward",
          lambda m: m.choose_gcamp("even")),
    KeyOp("channel.accept_rectangle", "GCaMP = the rectangle's decision", "G", "SP_DialogApplyButton",
          lambda m: m.accept_rectangle()),
    KeyOp("channel.clear", "Clear the GCaMP choice", "Alt+0", "SP_DialogDiscardButton",
          lambda m: m.choose_gcamp(None)),
)

ACTIONS = tuple(ActionSpec(op.id, op.label, op.shortcut, op.method, op.icon) for op in KEY_OPS) + (
    ActionSpec("file.open_cohort", "Open cohort…", "Ctrl+O", "open_cohort_dialog", "SP_DialogOpenButton"),
    ActionSpec("file.open_manifest", "Open manifest…", "Ctrl+Shift+O", "open_manifest_dialog", "SP_FileIcon"),
    ActionSpec("file.cache_all", "Cache all previews", "", "cache_all", "SP_DriveHDIcon"),
    ActionSpec("file.resume", "Reload this page from its saved set", "", "resume", "SP_BrowserReload"),
    ActionSpec("file.settings", "Settings…", "Ctrl+,", "edit_settings", "SP_FileDialogDetailedView"),
    ActionSpec("edit.undo", "Undo", "Ctrl+Z", "undo", "SP_ArrowBack"),
    ActionSpec("edit.redo", "Redo", "Ctrl+Y", "redo", "SP_ArrowForward"),
    ActionSpec("rect.show", "Show the channel rectangle", "", "toggle_rectangle", "SP_DialogYesButton"),
    ActionSpec("display.contrast_down", "Less contrast stretch", "[", "contrast_down", "SP_ArrowDown"),
    ActionSpec("display.contrast_up", "More contrast stretch", "]", "contrast_up", "SP_ArrowUp"),
    ActionSpec("display.shared_scale", "Shared contrast", "Ctrl+L", "toggle_shared_scale", "SP_DesktopIcon"),
    ActionSpec("display.fit", "Fit view", "Ctrl+0", "fit_view", "SP_TitleBarMaxButton"),
    ActionSpec("tools.rebuild", "Rebuild saved sets", "F6", "rebuild", "SP_MediaPlay"),
    ActionSpec("tools.preflight", "Run pre-flight", "F7", "run_preflight", "SP_DialogApplyButton"),
    ActionSpec("help.controls", "Controls", "H", "show_controls", "SP_DialogHelpButton"),
    ActionSpec("app.shortcuts", "Keyboard shortcuts…", "", "edit_shortcuts", "SP_FileDialogListView"),
    ActionSpec("app.toolbar", "Customize toolbar…", "", "edit_toolbar", "SP_FileDialogContentsView"),
    ActionSpec("app.reset_commands", "Reset menus and shortcuts", "", "reset_commands", "SP_DialogResetButton"),
    ActionSpec("app.quit", "Quit", "Q", "quit", "SP_DialogCloseButton"),
)
CHECKABLE = ("rect.show", "display.shared_scale")
# Second bindings (an ActionSpec holds one): PgUp / PgDn page as in the viewer.
EXTRA_SHORTCUTS = {"page.previous": "PgUp", "page.next": "PgDown"}
# Shortcuts the new commands brought; a user binding of the same keys is dropped.
NEW_SHORTCUTS = ("Ctrl+O", "Ctrl+Shift+O", "Ctrl+L", "Ctrl+0")
# Usable with no data loaded; everything else is disabled in the empty window.
WITHOUT_DATA = {"file.open_cohort", "file.open_manifest", "file.settings", "help.controls",
                "app.shortcuts", "app.toolbar", "app.reset_commands", "app.quit",
                "rect.show", "display.shared_scale", "display.contrast_down", "display.contrast_up"}

DEFAULT_LAYOUT = {
    "menus": {
        "File": ["file.open_cohort", "file.open_manifest", "-", "file.save", "file.save_modified",
                 "file.save_all", "file.write_shared", "-", "file.resume", "file.cache_all", "-",
                 "file.settings", "-", "app.quit"],
        "Edit": ["edit.undo", "edit.redo", "-", "reset.page", "reset.all", "-",
                 "page.copy_previous", "apply.all", "apply.all_bregma"],
        "Page": ["page.previous", "page.next"],
        "Layout": [op.id for op in KEY_OPS if op.id.startswith("bregma.")] + ["-"]
                  + [op.id for op in KEY_OPS if op.id.startswith(("lambda.", "rotate."))] + ["-"]
                  + ["flip.horizontal", "flip.vertical", "flip.lambda", "-",
                     "toggle.mirror_lock", "toggle.rotate_with_lambda"],
        "Box": [op.id for op in KEY_OPS if op.id.startswith("box.")],
        "Rectangle": ["rect.show", "rect.reset"],
        "Channels": ["channel.odd", "channel.even", "channel.accept_rectangle", "-", "channel.clear"],
        "Display": ["display.shared_scale", "display.fit", "-", "display.contrast_down", "display.contrast_up"],
        "Tools": ["tools.rebuild", "tools.preflight"],
        "Settings": ["app.shortcuts", "app.toolbar", "app.reset_commands"],
        "Help": ["help.controls"],
    },
    "toolbars": {"main": ["file.open_cohort", "file.open_manifest", "-", "page.previous", "page.next", "-",
                          "file.save", "file.save_modified", "-", "edit.undo", "edit.redo", "-",
                          "page.copy_previous", "file.resume", "-", "rotate.left", "rotate.right", "-",
                          "channel.odd", "channel.even", "channel.accept_rectangle", "-",
                          "rect.reset", "display.fit", "-", "tools.rebuild", "tools.preflight"]},
    "shortcuts": {},
}


# ----------------------------------------------------------------------------- cache job
class CacheWorker(QtCore.QObject):
    """Cache all previews, off the GUI thread: one (key, folder) after the other."""

    progress = QtCore.Signal(str)
    done = QtCore.Signal(int, int)          # cached now, total

    def __init__(self, jobs: list[tuple[str, str]], settings: EditorSettings) -> None:
        super().__init__()
        self.jobs, self.settings = jobs, settings
        self.cancelled = False             # set by the window when it closes

    def run(self) -> None:
        written = 0
        for number, (key, folder) in enumerate(self.jobs, start=1):
            if self.cancelled:
                break
            result = sources.cache_page(key, folder, self.settings)
            written += result == "cached"
            self.progress.emit(f"[{number}/{len(self.jobs)}] {key}: {result}")
        self.done.emit(written, len(self.jobs))


# ----------------------------------------------------------------------------- controller
class Controller:
    """State changes go through here; the window redraws from ``model`` and ``data``."""

    COMMIT_DELAY_MS = 400     # a mouse gesture becomes one undo step once it pauses this long

    def __init__(self, window) -> None:
        self.window = window
        self.data: sources.LoadedData | None = None
        self.model: EditorModel | None = None
        # True while the code (not the mouse) moves items; their signals are then ignored.
        self.syncing = False
        self.clip = [1.0, 99.0]             # display percentiles (darkest, brightest)
        self.commit_timer = QtCore.QTimer()
        self.commit_timer.setSingleShot(True)
        self.commit_timer.setInterval(self.COMMIT_DELAY_MS)
        self.commit_timer.timeout.connect(lambda: self.commit("Drag"))
        self.cache_thread: QtCore.QThread | None = None
        self.cache_worker: CacheWorker | None = None

    # ----- loading ---------------------------------------------------------------------
    @property
    def settings(self) -> EditorSettings | None:
        return self.data.settings if self.data is not None else None

    def load(self, data: sources.LoadedData) -> None:
        """Show freshly loaded pages (the caller has already asked about unsaved pages)."""
        if not data.pages:
            raise ValueError(f"No recordings to edit in {data.name}" +
                             (": " + "; ".join(data.notes[:5]) if data.notes else ""))
        self.commit_timer.stop()
        model = EditorModel(data.pages, data.args, data.profile_name,
                            loader=lambda page: sources.load_images(page, data.settings))
        self.data, self.model = data, model
        model.ensure_image(model.page)
        self.window.show_data()
        for note in data.notes:
            self.window.notify(f"left out: {note}")
        self.window.notify(f"Loaded {len(data.pages)} page(s) from {data.name}"
                           + (" -- raw folders not found: the pages are cached images, the channel "
                              "rectangle gives no decision; editing and saving work as usual."
                              if data.preview_only else ""))
        if data.kind in ("cohort", "manifest"):
            self.window.add_recent(data.kind, data.path)

    def confirm_discard(self) -> bool:
        """Before new data replaces pages with unsaved edits: Save all / Discard / Cancel."""
        self.flush()
        if self.model is None or not self.model.dirty_pages():
            return True
        answer = self.window.ask_unsaved(len(self.model.dirty_pages()))
        if answer == "cancel":
            return False
        if answer == "save":
            refused = [page.key for page in self.model.dirty_pages() if self.model.out_of_frame(page)]
            if refused:
                self.window.notify(f"Not opened: {len(refused)} page(s) cannot be saved, boxes off "
                                   f"the frame: {refused}. Fix or discard them first.")
                return False
            self.model.save_modified()
        return True

    def open_cohort(self, path: str, flags: dict | None = None) -> None:
        if not self.confirm_discard():
            return
        self.load(sources.load_cohort(path, flags))

    def open_manifest(self, path: str, flags: dict | None = None) -> None:
        """Settings from the cohort naming the manifest, else asked for."""
        if not self.confirm_discard():
            return
        settings, cohort = sources.manifest_settings(path, flags)
        if settings is None:
            settings = self.window.ask_settings(
                self._dialog_values(), need_folder=False,
                note=f"No cohort file names {Path(path).name} as its manifest: give the settings "
                     "its recordings are read with (the folder field is not used).")
            if settings is None:
                return
        self.load(sources.load_manifest(path, settings, cohort))

    def open_recent(self, kind: str, path: str) -> None:
        if not Path(path).is_file():
            self.window.remove_recent(kind, path)
            self.window.notify(f"Removed from Recent: {path} no longer exists.")
            return
        (self.open_cohort if kind == "cohort" else self.open_manifest)(path)

    def open_cohort_dialog(self) -> None:
        path = self.window.choose_file("Open cohort", REPO_ROOT / "cohorts", "Cohort files (*.yaml)")
        if path:
            self.open_cohort(path)

    def open_manifest_dialog(self) -> None:
        path = self.window.choose_file("Open manifest", REPO_ROOT / "manifests", "Manifests (*.csv)")
        if path:
            self.open_manifest(path)

    def _dialog_values(self) -> dict:
        from .logic import default_values
        return dataclasses.asdict(self.settings) if self.settings is not None else default_values()

    def edit_settings(self) -> None:
        """Settings dialog. New folders (or nothing loaded, or a folder source) = open those
        folders; otherwise the current cohort / manifest is reloaded with the settings."""
        self.flush()
        settings = self.window.ask_settings(
            self._dialog_values(), need_folder=False,
            note="Folders are scanned for recordings (one page each) when they change or nothing "
                 "is loaded; otherwise OK reloads the current cohort / manifest with these settings. "
                 "Sets are saved to 'Roi set dir'; 'Roi seed dir' holds read-only starting layouts.")
        if settings is None or not self.confirm_discard():
            return
        data = self.data
        if data is None or data.kind == "folder" or settings.folder != data.settings.folder:
            if not settings.folder:
                raise ValueError("Give at least one folder of recordings")
            self.load(sources.load_folders(settings))
        elif data.kind == "manifest":
            self.load(sources.load_manifest(data.path, settings, data.cohort))
        else:
            # A cohort: its manifest (or folder scan) again, read with the new settings.
            from wfci_gui.logic.cohort import load_cohort
            manifest = load_cohort(data.cohort).path_of("manifest")
            pages, args, profile, preview_only, notes = (
                sources.pages_from_manifest(manifest, settings) if manifest is not None and manifest.is_file()
                else sources.pages_from_folders(settings, data.cohort))
            self.load(dataclasses.replace(data, pages=pages, args=args, profile_name=profile, settings=settings,
                                          preview_only=preview_only, notes=notes))

    def set_frames_per_parity(self, value: int) -> None:
        """Display dock: every page's images are read again with the new count."""
        if self.data is None or value == self.settings.frames_per_parity:
            return
        self.data.settings = dataclasses.replace(self.settings, frames_per_parity=int(value))
        self.data.args.frames_per_parity = int(value)
        for page in self.model.pages:
            page.image, page.odd_even, page.odd_even_128, page.image_source = None, None, None, "none"
        self.model.loader = lambda page: sources.load_images(page, self.data.settings)
        self.model.ensure_image(self.model.page)
        self.window.show_page()
        self.window.notify(f"Frames per parity: {value}; images are read again as pages are opened.")

    # ----- undo ------------------------------------------------------------------------
    def schedule_commit(self) -> None:
        self.commit_timer.start()

    def commit(self, label: str) -> None:
        self.commit_timer.stop()
        if self.model is not None and self.model.commit(label):
            self.window.refresh_rows()
        self.window.update_actions()

    def flush(self) -> None:
        """Record a mouse gesture not recorded yet, before the next command runs."""
        if self.commit_timer.isActive():
            self.commit("Drag")

    def undo(self) -> None:
        self.flush()
        self.model.undo()
        self.window.redraw(page_changed=True)
        self.window.refresh_rows()

    def redo(self) -> None:
        self.flush()
        self.model.redo()
        self.window.redraw(page_changed=True)
        self.window.refresh_rows()

    # ----- page commands -----------------------------------------------------------------
    def run(self, op: KeyOp) -> None:
        """One page command as one undo step. A save that raises still refreshes the table."""
        self.flush()
        index = self.model.index
        try:
            op.edit(self.model)
        finally:
            if op.files:
                self.window.rescan_files()
            self.model.commit(op.label)
            self.window.redraw(page_changed=self.model.index != index or op.id in ("reset.page", "reset.all"))
            self.window.refresh_rows()

    def resume(self) -> None:
        self.flush()
        self.model.resume()
        self.model.commit("Reload from the saved set")
        self.window.redraw()
        self.window.refresh_rows()

    def go_to(self, index: int) -> None:
        """Open a page from the table (nothing carried over)."""
        self.flush()
        if self.model is None or index == self.model.index:
            return
        self.model.go_to(index)
        self.window.redraw(page_changed=True)
        self.window.refresh_rows()

    def panel_edit(self, label: str, edit: Callable[[], None]) -> None:
        """An edit typed into the properties panel, as one undo step."""
        self.flush()
        edit()
        self.model.commit(label)
        self.window.redraw()
        self.window.refresh_rows()

    # ----- gestures (from the panels) ----------------------------------------------------
    def box_dragged(self, panel, label: str, roi) -> None:
        if self.syncing:
            return
        pos, size = roi.pos(), roi.size()
        # Snapping keeps these integral; round anyway so int() cannot truncate -0.5 to 0.
        self.model.move_box(label, int(round(pos.y())), int(round(pos.x())),
                            int(round(size.y())), int(round(size.x())))
        self.window.redraw(skip=(panel, label))
        self.schedule_commit()

    def box_clicked(self, label: str) -> None:
        if self.syncing:
            return
        self.model.select(label)
        self.window.redraw()

    def bregma_dragged(self, panel, pos) -> None:
        if self.syncing:
            return
        # The marker sits at the pixel centre: +0.5 gives the 1-based y_1 / x_2 frame.
        self.model.drag_bregma(pos.y() + 0.5, pos.x() + 0.5)
        self.window.redraw()
        self.schedule_commit()

    def lambda_dragged(self, panel, pos) -> None:
        if self.syncing:
            return
        self.model.drag_lambda(pos.y() + 0.5, pos.x() + 0.5)
        self.window.redraw()
        self.schedule_commit()

    def double_clicked(self, view_row: float, view_col: float) -> None:
        if self.syncing or self.model is None:
            return
        self.flush()
        self.model.place_bregma(view_row, view_col)
        self.model.commit("Place Bregma")
        self.window.redraw()
        self.window.refresh_rows()

    def rectangle_dragged(self, panel, roi, finished: bool) -> None:
        """Live verdict while dragging; the table row and the undo step on release."""
        if self.syncing:
            return
        state = roi.getState()
        corner = (float(state["pos"][1]), float(state["pos"][0]))       # (row, col)
        self.model.move_rectangle(corner, float(state["size"][0]), float(state["size"][1]))
        self.window.redraw()
        if finished:
            self.commit("Move rectangle")
        else:
            self.schedule_commit()

    def gesture_finished(self) -> None:
        if not self.syncing and self.commit_timer.isActive():
            self.commit("Drag")

    # ----- display -----------------------------------------------------------------------
    def contrast_down(self) -> None:
        self.clip[1] = max(self.clip[0] + 1.0, min(100.0, self.clip[1] - 2.0))
        self.window.update_levels()

    def contrast_up(self) -> None:
        self.clip[1] = max(self.clip[0] + 1.0, min(100.0, self.clip[1] + 2.0))
        self.window.update_levels()

    def toggle_shared_scale(self) -> None:
        self.window.update_levels()

    def toggle_rectangle(self) -> None:
        if self.model is not None:
            self.window.redraw()

    def fit_view(self) -> None:
        self.window.fit_view()

    def show_controls(self) -> None:
        self.window.show_controls()

    # ----- tools -------------------------------------------------------------------------
    def rebuild(self) -> None:
        self.flush()
        self.window.rebuild()

    def run_preflight(self) -> None:
        self.flush()
        self.window.run_preflight()

    def cache_all(self) -> None:
        """Fill the odd / even cache of every page from its raw frames, in the background."""
        if self.cache_thread is not None:
            self.window.notify("Caching is already running.")
            return
        jobs = [(page.key, page.folder) for page in self.model.pages]
        self.cache_thread = QtCore.QThread()
        self.cache_worker = CacheWorker(jobs, self.settings)
        self.cache_worker.moveToThread(self.cache_thread)
        self.cache_thread.started.connect(self.cache_worker.run)
        self.cache_worker.progress.connect(self.window.log)
        self.cache_worker.done.connect(self._cache_done)
        self.window.notify(f"Caching {len(jobs)} page(s) in "
                           f"{Path(self.settings.roi_set_dir) / sources.CACHE_FOLDER} …")
        self.cache_thread.start()
        self.window.update_actions()

    def _cache_done(self, written: int, total: int) -> None:
        self.cache_thread.quit()
        self.cache_thread.wait()
        self.cache_thread = self.cache_worker = None
        self.window.notify(f"Cache all: {written} written, {total - written} already cached or without "
                           f"a recording folder.")
        self.window.update_actions()

    # ----- app ---------------------------------------------------------------------------
    def edit_shortcuts(self) -> None:
        self.window.edit_shortcuts()

    def edit_toolbar(self) -> None:
        self.window.edit_toolbar()

    def reset_commands(self) -> None:
        self.window.reset_commands()

    def quit(self) -> None:
        self.window.close()


def _key_method(op: KeyOp):
    def method(self) -> None:
        self.run(op)
    method.__name__ = op.method
    method.__doc__ = f"{op.label} (default shortcut '{op.shortcut}')."
    return method


# One handler method per page command, so qtkit can bind every action by name.
for _op in KEY_OPS:
    setattr(Controller, _op.method, _key_method(_op))
