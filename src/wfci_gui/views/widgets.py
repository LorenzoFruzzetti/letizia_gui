"""Reusable widgets: editable table, parameter form, results viewer, log dock."""

from __future__ import annotations


import html
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from qtkit import DataFrameModel
from qtkit.qt import QtCore, QtGui, QtWidgets

from ..logic.params import FormField


# ----------------------------------------------------------------------------- table
class FrameModel(QtCore.QAbstractTableModel):
    """Editable DataFrame of strings. Boolean columns are check boxes, choice columns combos."""

    edited = QtCore.Signal()

    def __init__(self, editable: tuple[str, ...] | None = None, bool_columns: tuple[str, ...] = (),
                 choices: dict[str, tuple[str, ...]] | None = None, parent=None):
        super().__init__(parent)
        self.frame = pd.DataFrame()
        self.editable = editable            # None = every column
        self.bool_columns = set(bool_columns)
        self.choices = choices or {}

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self.frame = frame.copy().reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.frame.columns)

    def _column(self, index) -> str:
        return self.frame.columns[index.column()]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        value = str(self.frame.iat[index.row(), index.column()])
        if self._column(index) in self.bool_columns:
            if role == QtCore.Qt.CheckStateRole:
                return QtCore.Qt.Checked if value.lower() == "true" else QtCore.Qt.Unchecked
            return None
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.UserRole):
            return value
        return None

    def flags(self, index):
        flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        column = self._column(index)
        if self.editable is None or column in self.editable:
            flags |= QtCore.Qt.ItemIsUserCheckable if column in self.bool_columns else QtCore.Qt.ItemIsEditable
        return flags

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        column = self._column(index)
        if column in self.bool_columns and role == QtCore.Qt.CheckStateRole:
            value = "true" if QtCore.Qt.CheckState(value) == QtCore.Qt.Checked else "false"
        elif role != QtCore.Qt.EditRole:
            return False
        self.frame.iat[index.row(), index.column()] = str(value)
        self.dataChanged.emit(index, index)
        self.edited.emit()
        return True

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        return str(self.frame.columns[section]) if orientation == QtCore.Qt.Horizontal else str(section + 1)

    def add_row(self) -> None:
        row = {c: ("true" if c in self.bool_columns else "") for c in self.frame.columns}
        self.set_frame(pd.concat([self.frame, pd.DataFrame([row])], ignore_index=True))
        self.edited.emit()

    def remove_rows(self, rows: list[int]) -> None:
        self.set_frame(self.frame.drop(index=rows))
        self.edited.emit()


class ChoiceDelegate(QtWidgets.QStyledItemDelegate):
    """Combo box editor for the model's choice columns."""

    def createEditor(self, parent, option, index):
        model = index.model()
        source = model.sourceModel() if hasattr(model, "sourceModel") else model
        column = source.frame.columns[index.column()]
        if column in source.choices:
            editor = QtWidgets.QComboBox(parent)
            editor.addItems(list(source.choices[column]))
            return editor
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if isinstance(editor, QtWidgets.QComboBox):
            editor.setCurrentText(index.data(QtCore.Qt.EditRole))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QtWidgets.QComboBox):
            model.setData(index, editor.currentText(), QtCore.Qt.EditRole)
        else:
            super().setModelData(editor, model, index)


class TableEditor(QtWidgets.QWidget):
    """A cohort table: edit cells, add/delete rows, import a CSV. Edits go to ``on_change``."""

    def __init__(self, title: str, on_change: Callable[[pd.DataFrame], None],
                 on_import: Callable[[Path], None] | None, editable=None, bool_columns=(),
                 choices=None, can_add_rows: bool = True, note: str = ""):
        super().__init__()
        self.on_change = on_change
        self.on_import = on_import
        self.model = FrameModel(editable, bool_columns, choices, self)
        self.model.edited.connect(lambda: self.on_change(self.model.frame))
        self.proxy = QtCore.QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(f"<b>{html.escape(title)}</b>")
        header.addWidget(label)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter rows…")
        self.filter.textChanged.connect(self.proxy.setFilterFixedString)
        header.addWidget(self.filter, 1)
        if can_add_rows:
            add = QtWidgets.QPushButton("Add row")
            add.clicked.connect(self.model.add_row)
            header.addWidget(add)
            remove = QtWidgets.QPushButton("Delete selected rows")
            remove.clicked.connect(self._remove_selected)
            header.addWidget(remove)
        if on_import is not None:
            importer = QtWidgets.QPushButton("Import CSV…")
            importer.clicked.connect(self._import)
            header.addWidget(importer)
        layout.addLayout(header)
        if note:
            note_label = QtWidgets.QLabel(note)
            note_label.setWordWrap(True)
            layout.addWidget(note_label)
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.proxy)
        self.view.setItemDelegate(ChoiceDelegate(self.view))
        self.view.setSortingEnabled(True)
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.view, 1)
        self.issues = QtWidgets.QLabel()
        self.issues.setWordWrap(True)
        self.issues.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.issues)

    def set_frame(self, frame: pd.DataFrame, issues: list) -> None:
        if not frame.equals(self.model.frame):
            self.model.set_frame(frame)
            self.view.resizeColumnsToContents()
        text = "\n".join(str(i) for i in issues[:15])
        more = f"\n… and {len(issues) - 15} more" if len(issues) > 15 else ""
        self.issues.setText(f"<span style='color:#c0392b'>Problems:<br>{html.escape(text + more).replace(chr(10), '<br>')}</span>"
                            if issues else "")

    def _remove_selected(self) -> None:
        rows = sorted({self.proxy.mapToSource(i).row() for i in self.view.selectionModel().selectedRows()})
        if rows:
            self.model.remove_rows(rows)

    def _import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import table", "", "CSV files (*.csv)")
        if path:
            self.on_import(Path(path))


# ----------------------------------------------------------------------------- form
NUMBER_INT = re.compile(r"^[+-]?\d+$")
NUMBER_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(text_of(v) for v in value)
    return str(value)


def parse_scalar(text: str, like: Any) -> Any:
    """Text typed in a field -> value of the same kind as the default ``like``."""
    text = text.strip()
    if isinstance(like, bool):
        return text.lower() in ("true", "1", "yes")
    if isinstance(like, int):
        if not NUMBER_INT.match(text):
            raise ValueError(f"{text!r} is not a whole number")
        return int(text)
    if isinstance(like, float):
        if not NUMBER_FLOAT.match(text):
            raise ValueError(f"{text!r} is not a number")
        return float(text)
    if isinstance(like, str):
        return text
    # Unknown kind (default None or computed): numbers, None and booleans by their look.
    if text == "" or text.lower() == "none":
        return None
    if NUMBER_INT.match(text):
        return int(text)
    if NUMBER_FLOAT.match(text):
        return float(text)
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def text_problem(text: str, default: Any) -> str:
    """Why ``text`` cannot be a value like ``default`` ('' when it can)."""
    parts = [p.strip() for p in text.split(",") if p.strip()] if isinstance(default, (list, tuple)) else [text.strip()]
    like = (default[0] if default else None) if isinstance(default, (list, tuple)) else default
    for part in parts:
        if isinstance(like, int) and not isinstance(like, bool) and not NUMBER_INT.match(part):
            return f"{part!r} is not a whole number"
        if isinstance(like, float) and not NUMBER_FLOAT.match(part):
            return f"{part!r} is not a number"
        if like is None and part[:1] in "[(":
            return f"{part!r}: write a list as comma-separated values"
    return ""


def parse_field(text: str, default: Any) -> Any:
    if isinstance(default, (list, tuple)):
        parts = [p for p in (s.strip() for s in text.split(",")) if p]
        like = default[0] if default else None
        values = [parse_scalar(p, like) for p in parts]
        return type(default)(values)
    return parse_scalar(text, default)


class ParameterForm(QtWidgets.QWidget):
    """Form over a job's parameters. Overridden values are bold; each has Reset to default.

    Commits go to ``on_set(name, value)`` / ``on_reset(name)``; out-of-range or malformed
    values are refused visibly (the message appears under the form) and nothing changes.
    """

    def __init__(self, on_set: Callable[[str, Any], None], on_reset: Callable[[str], None]):
        super().__init__()
        self.on_set = on_set
        self.on_reset = on_reset
        self.rows: dict[str, FormField] = {}
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.form = QtWidgets.QFormLayout()
        outer.addLayout(self.form)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        outer.addWidget(self.message)
        self.signature = None

    def set_fields(self, fields: list[FormField], note: str = "") -> None:
        signature = [(f.name, repr(f.value), f.source, f.editable) for f in fields]
        if signature == self.signature:
            return  # unchanged: keep focus and the user's cursor
        self.signature = signature
        while self.form.rowCount():
            self.form.removeRow(0)
        self.rows = {f.name: f for f in fields}
        self.widgets = {}
        for field in fields:
            self.form.addRow(self._label(field), self._row(field))
        self.message.setText(f"<span style='color:#c0392b'>{html.escape(note)}</span>" if note else "")

    def _label(self, field: FormField) -> QtWidgets.QLabel:
        text = field.name.replace("_", " ")
        label = QtWidgets.QLabel(f"<b>{text}</b>" if field.source == "override" else text)
        tooltip = field.hint.help or field.parameter.help
        label.setToolTip(f"{tooltip}\nScript default: {field.parameter.describe_default()}")
        return label

    def _row(self, field: FormField) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        parameter = field.parameter
        boolean = isinstance(field.default, bool) or parameter.action in (
            "store_true", "store_false", "boolean_optional", "bool_pair")
        if not field.editable:
            shown = "(computed in the script; needs an approved edit to change)" if field.source == "computed" \
                else text_of(field.value)
            widget = QtWidgets.QLineEdit(shown)
            widget.setReadOnly(True)
            widget.setEnabled(False)
        elif boolean:
            widget = QtWidgets.QCheckBox()
            widget.setChecked(bool(field.value))
            widget.toggled.connect(lambda checked, name=field.name: self._commit(name, checked))
        elif parameter.choices:
            widget = QtWidgets.QComboBox()
            widget.addItems([str(c) for c in parameter.choices])
            widget.setCurrentText(str(field.value))
            widget.textActivated.connect(lambda text, name=field.name: self._commit_text(name, text))
        else:
            widget = QtWidgets.QLineEdit(text_of(field.value))
            widget.editingFinished.connect(lambda name=field.name, w=widget: self._commit_text(name, w.text()))
        self.widgets[field.name] = widget
        layout.addWidget(widget, 1)
        if field.hint.unit:
            layout.addWidget(QtWidgets.QLabel(field.hint.unit))
        source = QtWidgets.QLabel(f"<i>{field.source}</i>")
        layout.addWidget(source)
        if field.editable:
            reset = QtWidgets.QToolButton()
            reset.setText("Reset")
            reset.setToolTip("Reset to the default")
            reset.setEnabled(field.source == "override")
            reset.clicked.connect(lambda checked=False, name=field.name: self.on_reset(name))
            layout.addWidget(reset)
        return row

    def _commit_text(self, name: str, text: str) -> None:
        field = self.rows[name]
        like = field.default if field.default is not None else field.value
        if text_of(field.value) == text.strip():
            return
        problem = text_problem(text, like)
        if problem:
            self.message.setText(f"<span style='color:#c0392b'>{html.escape(name)}: {html.escape(problem)}; "
                                 "not changed.</span>")
            return
        value = parse_field(text, like) if text.strip() or like is not None else None
        self._commit(name, value)

    def _commit(self, name: str, value: Any) -> None:
        field = self.rows[name]
        hint = field.hint
        numbers = value if isinstance(value, (list, tuple)) else [value]
        for number in numbers:
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                if (hint.minimum is not None and number < hint.minimum) or \
                        (hint.maximum is not None and number > hint.maximum):
                    self.message.setText(f"<span style='color:#c0392b'>{name}: {number} is outside "
                                         f"{hint.minimum}–{hint.maximum}; not changed.</span>")
                    self.signature = None
                    return
        self.message.setText("")
        if value == field.default and field.source != "override":
            return
        self.on_set(name, value)


# ----------------------------------------------------------------------------- results
RESULT_SUFFIXES = (".png", ".svg", ".pdf", ".csv", ".log", ".txt", ".json")


class ResultsView(QtWidgets.QWidget):
    """Files of a step's output folder, newest first: PNG shown, CSV as a table, others opened."""

    def __init__(self):
        super().__init__()
        self.folders: list[Path] = []
        layout = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.folder_label = QtWidgets.QLabel()
        self.folder_label.setWordWrap(True)
        bar.addWidget(self.folder_label, 1)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        bar.addWidget(refresh)
        open_folder = QtWidgets.QPushButton("Open folder")
        open_folder.clicked.connect(self.open_folder)
        bar.addWidget(open_folder)
        layout.addLayout(bar)
        splitter = QtWidgets.QSplitter()
        self.files = QtWidgets.QListWidget()
        self.files.currentItemChanged.connect(self.show_file)
        self.files.itemDoubleClicked.connect(lambda item: self.open_external(Path(item.data(QtCore.Qt.UserRole))))
        splitter.addWidget(self.files)
        self.stack = QtWidgets.QStackedWidget()
        self.image = QtWidgets.QLabel("Select a file")
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.image)
        scroll.setWidgetResizable(True)
        self.table_model = DataFrameModel(self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.table_model)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.external = QtWidgets.QPushButton("Open in the default application")
        self.external.clicked.connect(lambda: self.open_external(self.current))
        for widget in (scroll, self.table, self.text, self.external):
            self.stack.addWidget(widget)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)
        self.current: Path | None = None

    def set_folders(self, folders: list[Path]) -> None:
        if [str(f) for f in folders] != [str(f) for f in self.folders]:
            self.folders = list(folders)
            self.reload()

    def reload(self) -> None:
        self.files.clear()
        shown = "; ".join(str(f) for f in self.folders) or "(no output folder yet)"
        self.folder_label.setText(f"Output: {html.escape(shown)}")
        found = []
        for folder in self.folders:
            if folder.is_dir():
                found += [p for p in folder.rglob("*") if p.suffix.lower() in RESULT_SUFFIXES]
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in found[:2000]:
            relative = next((path.relative_to(f) for f in self.folders if path.is_relative_to(f)), path)
            item = QtWidgets.QListWidgetItem(str(relative))
            item.setData(QtCore.Qt.UserRole, str(path))
            self.files.addItem(item)

    def show_file(self, item) -> None:
        if item is None:
            return
        path = Path(item.data(QtCore.Qt.UserRole))
        self.current = path
        suffix = path.suffix.lower()
        if suffix == ".png":
            pixmap = QtGui.QPixmap(str(path))
            self.image.setPixmap(pixmap.scaledToWidth(min(pixmap.width(), 1400), QtCore.Qt.SmoothTransformation))
            self.stack.setCurrentIndex(0)
        elif suffix == ".csv":
            self.table_model.set_frame(pd.read_csv(path, nrows=5000, dtype=str, keep_default_na=False))
            self.table.resizeColumnsToContents()
            self.stack.setCurrentIndex(1)
        elif suffix in (".log", ".txt", ".json"):
            self.text.setPlainText(path.read_text(encoding="utf-8", errors="replace")[-400_000:])
            self.stack.setCurrentIndex(2)
        else:
            self.external.setText(f"Open {path.name} in the default application")
            self.stack.setCurrentIndex(3)

    def open_external(self, path: Path | None) -> None:
        if path is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def open_folder(self) -> None:
        existing = [f for f in self.folders if f.is_dir()]
        if existing:
            self.open_external(existing[0])


# ----------------------------------------------------------------------------- log dock
class LogDock(QtWidgets.QWidget):
    """Queue of waiting jobs and the live log of the running one."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.running = QtWidgets.QLabel("Idle")
        self.running.setWordWrap(True)
        left.addWidget(self.running)
        left.addWidget(QtWidgets.QLabel("Waiting:"))
        self.pending = QtWidgets.QListWidget()
        left.addWidget(self.pending, 1)
        remove = QtWidgets.QPushButton("Remove from queue")
        remove.clicked.connect(self._remove)
        left.addWidget(remove)
        box = QtWidgets.QWidget()
        box.setLayout(left)
        box.setMaximumWidth(320)
        layout.addWidget(box)
        right = QtWidgets.QVBoxLayout()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        right.addWidget(self.log, 1)
        clear = QtWidgets.QPushButton("Clear log view")
        clear.clicked.connect(self.log.clear)
        right.addWidget(clear, 0, QtCore.Qt.AlignRight)
        layout.addLayout(right, 1)
        controller.log_line.connect(self.log.appendPlainText)
        controller.queue_changed.connect(self.refresh)

    def refresh(self) -> None:
        queue = self.controller.queue
        self.running.setText(f"<b>Running:</b> {html.escape(queue.running.label)}<br>started {queue.running.start}"
                             if queue.running else "Idle")
        self.pending.clear()
        self.pending.addItems([job.label for job in queue.pending])

    def _remove(self) -> None:
        row = self.pending.currentRow()
        if row >= 0:
            self.controller.remove_pending(row)
