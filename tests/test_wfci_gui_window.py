"""The GUI, headless (pytest-qt): commands are triggered directly."""

from __future__ import annotations

import os
from pathlib import Path

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
    write_extraction(cohort.layout.outputs, cohort.layout.pixel_data, keys, needs_decision=(keys[0],))
    pd.DataFrame([dict(recording=k, day=k[:6], suggest="exclude" if k == keys[0] else "ok",
                       reason="channels do not alternate" if k == keys[0] else "") for k in keys]
                 ).to_csv(cohort.layout.outputs / "qc_table.csv", index=False)
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
    design = pd.DataFrame(dict(day=["200101", "200101", "200104"], animal=["M1", "M2", "M1"],
                               recording=["t1", "t1", "t1"], timepoint=["BL", "BL", "3D"],
                               group=["ctrl", "ctrl", "ctrl"]))
    design.to_csv(cohort.layout.design_csv, index=False)
    window.open_cohort(path)
    page = window.pages["analysis_a"]
    window.show_step_view("analysis_a")
    assert not page.run_all.isEnabled()
    assert "M2" in page.check.text()
    window.controller.set_override("analysis_a", "animals", ["M1"])
    assert "at least two animals" in page.check.text()
    # A complete design is enough: no per-cohort entry in the statistics script is needed.
    window.controller.set_override("analysis_a", "timepoints", ["BL"])
    window.controller.set_override("analysis_a", "animals", ["M1", "M2"])
    assert "later timepoint" in page.check.text()
    design.loc[len(design)] = dict(day="200104", animal="M2", recording="t1", timepoint="3D", group="ctrl")
    design.to_csv(cohort.layout.design_csv, index=False)
    window.controller.reset_override("analysis_a", "timepoints")
    assert page.run_all.isEnabled(), page.check.text()
    window.controller.revert_cohort()


def test_rois_page_starts_the_new_editor_with_the_cohort(window, tmp_path):
    path = make_cohort(tmp_path)
    _scan(path)
    window.open_cohort(path)
    window.controller.state.conda = tmp_path / "anaconda3" / "condabin" / "conda.bat"
    job = window.controller.new_roi_editor_job()
    assert job.step == "rois" and job.command[-4:] == ["-m", "wfci_gui.roi_editor", "--cohort", str(path)]
    labels = [b.text() for b in window.pages["rois"].findChildren(QtWidgets.QPushButton)]
    assert "Open ROI editor" in labels and "Open old ROI editor" not in labels


def test_workspace_is_kept_in_the_user_settings(window, tmp_path):
    from wfci_gui.logic.layout import cohort_file_in
    assert not window.has_workspace()
    window.set_workspace(tmp_path / "ws")
    assert window.has_workspace() and window.workspace == (tmp_path / "ws").resolve()
    path = make_cohort(tmp_path)            # tmp_path/workspace/fake/cohort.yaml
    window.set_workspace(path.parent.parent)
    assert [window.cohort_combo.itemText(i) for i in range(window.cohort_combo.count())] == ["", "fake"]
    assert cohort_file_in(window.workspace, "fake") == path.resolve()
    with pytest.raises(ValueError, match="inside the program folder"):
        window.set_workspace(Path(__file__).resolve().parents[1] / "workspace")


def test_create_example_cohort_queues_the_generator_into_the_workspace(window, tmp_path):
    window.set_workspace(tmp_path / "ws")
    window.controller.state.conda = tmp_path / "anaconda3" / "condabin" / "conda.bat"
    (tmp_path / "ws" / "example").mkdir()                      # taken: the next name is used
    job, path = window.controller.example_cohort_job(window.workspace)
    assert path == window.workspace / "example_2" / "cohort.yaml"
    assert job.command[-5:] == [str(window.controller.repo / "scripts" / "make_example_cohort.py"),
                                "--workspace", str(window.workspace), "--name", "example_2"]
    assert "cohort.example" in window.registry.actions


def test_a_cohort_saved_by_a_newer_version_is_refused(window, tmp_path):
    import yaml
    path = make_cohort(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["app_version"] = "999.0.0"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        window.controller.open_cohort(path)
    assert window.controller.state.cohort is None


def test_tables_written_by_a_job_are_reread(window, tmp_path):
    path = make_cohort(tmp_path)
    window.open_cohort(path)                      # not scanned yet
    assert not window.controller.state.statuses["scan"].done
    _scan(path)                                   # what the scan job writes
    window.controller.reload_tables()             # done after every finished job
    window.controller.refresh_status()
    assert window.controller.state.statuses["scan"].done
