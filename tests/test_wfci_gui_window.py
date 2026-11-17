"""The GUI, headless (pytest-qt): commands are triggered directly (gui_spec.md 12)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from gui_cohort_tree import make_cohort, recording_keys, write_extraction  # noqa: E402
from qtkit.testing import check_commands, qtkit_config_dir, qtkit_settings  # noqa: E402,F401
from wfci_gui.logic.cohort import load_cohort  # noqa: E402
from wfci_gui.logic.steps import STEPS  # noqa: E402
from wfci_gui.window import ACTIONS, DEFAULT_LAYOUT, MainWindow  # noqa: E402


@pytest.fixture
def window(qtbot, qtkit_settings, qtkit_config_dir, monkeypatch):  # noqa: F811
    # A modal dialog never returns offscreen (a failed test left unsaved changes, the
    # window's close asked "Save them?" and the run hung): make any of them an error.
    def refuse(*args, **kwargs):
        raise AssertionError(f"a modal dialog opened in a headless test: {args[1:3]}")
    for name in ("question", "information", "warning", "critical", "about"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, staticmethod(refuse))
    win = MainWindow(qtkit_settings, qtkit_config_dir, environ={"USERPROFILE": str(qtkit_config_dir)})
    qtbot.addWidget(win)
    return win


def _scan(path):
    import sys
    import scan_cohort
    argv, sys.argv = sys.argv, ["scan_cohort.py", "--cohort", str(path), "--no-make-links"]
    scan_cohort.main()
    sys.argv = argv


def test_command_table_is_valid():
    check_commands(ACTIONS, DEFAULT_LAYOUT, MainWindow)


def test_window_opens_with_every_step_page(window):
    assert set(window.pages) == {s.key for s in STEPS}
    assert window.controller.state.cohort is None
    window.show_step_view("scan")
    assert window.stack.currentWidget() is window.placeholder   # no cohort yet


def test_open_cohort_shows_status_and_pages(window, tmp_path):
    path = make_cohort(tmp_path)
    _scan(path)
    window.open_cohort(path)
    state = window.controller.state
    assert state.statuses["scan"].done
    for step in STEPS:
        window.show_step_view(step.key)
        assert window.stack.currentWidget() is window.pages[step.key]
    rows = window.progress(state)
    assert rows[1][0] is True and "8 ok" in rows[1][1]


def test_override_marks_dirty_and_saves_atomically(window, tmp_path):
    path = make_cohort(tmp_path)
    _scan(path)
    window.open_cohort(path)
    controller = window.controller
    controller.set_override("extract", "workers", 2)
    assert controller.state.cohort_dirty and "workers: 2" in controller.pending_diff()
    controller.save_cohort()
    assert not controller.state.cohort_dirty
    assert load_cohort(path).overrides("extract") == {"workers": 2}
    controller.reset_override("extract", "workers")
    controller.save_cohort()
    assert "extract" not in load_cohort(path).get("overrides")


def test_jobs_refused_while_unsaved(window, tmp_path):
    path = make_cohort(tmp_path)
    _scan(path)
    window.open_cohort(path)
    window.controller.set_override("qc", "glitch_threshold", 2.0)
    # The fixture's environment has no conda; building the command only needs a path
    # (the job is refused before anything runs).
    window.controller.state.conda = tmp_path / "anaconda3" / "condabin" / "conda.bat"
    with pytest.raises(RuntimeError, match="unsaved"):
        window.controller.enqueue(window.controller.make_job("qc"))
    # Leave nothing unsaved, or closing the window would open the "Save changes?" dialog.
    window.controller.revert_cohort()


def test_qc_page_preticks_suggestions(window, tmp_path):
    path = make_cohort(tmp_path)
    _scan(path)
    cohort = load_cohort(path)
    keys = recording_keys()
    write_extraction(cohort.path_of("output_root"), cohort.path_of("pixel_root"), keys, needs_decision=(keys[0],))
    pd.DataFrame([dict(recording=k, day=k[:6], suggest="exclude" if k == keys[0] else "ok",
                       reason="channels do not alternate" if k == keys[0] else "") for k in keys]
                 ).to_csv(cohort.path_of("output_root") / "qc_table.csv", index=False)
    window.open_cohort(path)
    window.show_step_view("qc")
    frame = window.pages["qc"].model.frame
    ticked = frame.loc[frame["exclude"] == "true", "recording"].tolist()
    assert ticked == [keys[0]]
    assert load_cohort(path).exclusions == []                   # nothing excluded until confirmed


def test_analysis_a_run_disabled_until_design_complete(window, tmp_path):
    path = make_cohort(tmp_path)
    _scan(path)
    cohort = load_cohort(path)
    design = pd.DataFrame(dict(day=["230101", "230101", "230104"], animal=["M1", "M2", "M1"],
                               recording=["t1", "t1", "t1"], timepoint=["BL", "BL", "3D"],
                               group=["ctrl", "ctrl", "ctrl"]))
    cohort.path_of("design_csv").parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(cohort.path_of("design_csv"), index=False)
    window.open_cohort(path)
    window.controller.set_override("analysis_a", "stats_cohort", "2023")
    page = window.pages["analysis_a"]
    window.show_step_view("analysis_a")
    assert not page.run_all.isEnabled()
    assert "M2" in page.check.text()
    window.controller.set_override("analysis_a", "animals", ["M1"])
    assert "at least two animals" in page.check.text()
    window.controller.revert_cohort()


def test_rois_page_starts_the_new_editor_with_the_cohort(window, tmp_path):
    path = make_cohort(tmp_path)
    _scan(path)
    window.open_cohort(path)
    window.controller.state.conda = tmp_path / "anaconda3" / "condabin" / "conda.bat"
    job = window.controller.new_roi_editor_job()
    assert job.step == "rois" and job.command[-4:] == ["-m", "wfci_gui.roi_editor", "--cohort", str(path)]
    labels = [b.text() for b in window.pages["rois"].findChildren(QtWidgets.QPushButton)]
    assert "Open ROI editor" in labels and "Open old ROI editor" in labels
