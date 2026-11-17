"""The ROI editor window (src/wfci_gui/roi_editor, roi_editor_spec.md), headless.

Equivalence with the old editor through the window's own actions is checked here; the
model-level equivalence of every edit is tests/test_roi_editor_model.py.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tifffile  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

import channel_rectangle as cr  # noqa: E402
from qtkit.testing import check_commands, qtkit_config_dir, qtkit_settings  # noqa: E402,F401
from roi_editor import DEFAULT_LAMBDA_OFFSET, ROIEditor, Session  # noqa: E402
from wfci.atlases import CORTEX_22  # noqa: E402
from wfci_gui.roi_editor import logic, sources  # noqa: E402
from wfci_gui.roi_editor.controller import ACTIONS, DEFAULT_LAYOUT, KEY_OPS, WITHOUT_DATA, Controller  # noqa: E402
from wfci_gui.roi_editor.model import PageModel, bregma_128, lambda_128  # noqa: E402
from wfci_gui.roi_editor.window import EditorWindow, drop_conflicting_overrides  # noqa: E402

GRID = (128, 128)
KEYS = ("240101_M1_t1", "240101_M1_t2", "240104_M1_t1")
HEADER = ("day,animal,group,n_recordings,recordings,total_frames,bregma_row,bregma_col,profile,"
          "channel_order,animal_path,recording_paths\n")


def images(seed: int):
    rng = np.random.default_rng(seed)
    odd_256, even_256 = rng.random((256, 256)) + 5, rng.random((256, 256)) + 5
    even_256[:, 128:136] += 3                       # vessel-like strip: the rule calls even reflectance
    return (odd_256, even_256), (sources.to_128(odd_256), sources.to_128(even_256))


def make_pages():
    """Synthetic pages with their images already read, so nothing is decoded."""
    pages = []
    for i, key in enumerate(KEYS):
        odd_even, odd_even_128 = images(i)
        pages.append(PageModel(key=key, folder="(synthetic)", y_1=60, x_2=67, boxes=dict(CORTEX_22.boxes),
                               lambda_offset=DEFAULT_LAMBDA_OFFSET, image=odd_even_128[0], odd_even=odd_even,
                               odd_even_128=odd_even_128, image_source="raw", group="P",
                               channel_order="gcamp_first", seed_source="seeded from the cortical_gsr atlas"))
    return pages


def make_args(roi_set_dir: Path):
    return types.SimpleNamespace(channel_order="auto", preview_frames=1, roi_set_dir=str(roi_set_dir),
                                 roi_seed_dir=None, shared_name="shared.yaml",
                                 lambda_offset=DEFAULT_LAMBDA_OFFSET, lambda_scales_box_size=False,
                                 channel_rectangle_params=None, mirror_lock=True)


def refuse_dialogs(monkeypatch):
    # A modal dialog never returns offscreen: make any of them an error (CLAUDE.md 9.74).
    def refuse(*args, **kwargs):
        raise AssertionError(f"a modal dialog opened in a headless test: {args[1:3]}")
    for name in ("question", "information", "warning", "critical", "about"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, staticmethod(refuse))
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", refuse)


@pytest.fixture
def empty(qtbot, qtkit_settings, qtkit_config_dir, monkeypatch):  # noqa: F811
    refuse_dialogs(monkeypatch)
    win = EditorWindow(qtkit_settings, qtkit_config_dir)
    win.confirm_close = False        # tests leave edited pages behind on purpose
    qtbot.addWidget(win)
    return win


@pytest.fixture
def window(empty, tmp_path):
    reviewed, rebuilt = tmp_path / "reviewed", tmp_path / "rebuilt"
    reviewed.mkdir()
    rebuilt.mkdir()
    settings = logic.EditorSettings(folder=("(synthetic)",), roi_set_dir=str(reviewed), roi_seed_dir=None,
                                    rebuilt_dir=str(rebuilt))
    empty.controller.load(sources.LoadedData(make_pages(), make_args(reviewed), "cortical_gsr", settings,
                                             "manifest", str(tmp_path / "fake.csv"), None, None))
    return empty


def trigger(win, action_id: str) -> None:
    win.registry.actions[action_id].trigger()


def table_column(win, column: str) -> list:
    return list(win.table_model.frame[column])


# ----------------------------------------------------------------------------- logic
def test_page_status_priorities():
    status = logic.page_status
    assert status(False, False, None, None, None) == "not opened"
    assert status(False, True, None, None, None) == "opened"
    assert status(False, True, 10.0, None, None) == "saved"
    assert status(False, True, 10.0, 20.0, None) == "rebuilt"
    assert status(False, True, 30.0, 20.0, None) == "saved"          # saved again after the rebuild
    assert status(False, True, 10.0, 20.0, ["V1L outside the grid"]) == "failed pre-flight"
    assert status(True, True, 10.0, 20.0, ["V1L outside the grid"]) == "edited"
    assert status(False, False, None, 20.0, []) == "rebuilt"


def test_rebuild_sources_take_the_whole_session(tmp_path):
    reviewed, rebuilt = tmp_path / "reviewed", tmp_path / "rebuilt"
    reviewed.mkdir()
    rebuilt.mkdir()
    for key in KEYS:
        (reviewed / f"{key}.yaml").write_text("x", encoding="utf-8")
    # Only 240104_M1_t1 has an up-to-date rebuilt copy.
    (rebuilt / "240104_M1_t1.yaml").write_text("x", encoding="utf-8")
    os.utime(rebuilt / "240104_M1_t1.yaml", (2e9, 2e9))
    sources_ = logic.rebuild_sources(list(KEYS), reviewed, rebuilt)
    assert [p.name for p in sources_] == ["240101_M1_t1.yaml", "240101_M1_t2.yaml"]
    (rebuilt / "240101_M1_t1.yaml").write_text("x", encoding="utf-8")
    os.utime(rebuilt / "240101_M1_t1.yaml", (2e9, 2e9))
    assert len(logic.rebuild_sources(list(KEYS), reviewed, rebuilt)) == 2


def test_snapshot_round_trip_includes_the_rectangle():
    page = make_pages()[0]
    before = logic.snapshot(page)
    page.y_1 += 3
    page.rescale_to(page.lambda_offset + 4)
    page.rectangle = cr.RectangleFactors(0.5, 0.2, 1.0)
    assert logic.snapshot(page) != before
    logic.restore(page, before)
    assert logic.snapshot(page) == before and page.rectangle is None
    # A plain Session (old editor) still snapshots, with no rectangle.
    session = Session(key="k", folder="f", y_1=60, x_2=67, boxes=dict(CORTEX_22.boxes))
    logic.restore(session, logic.snapshot(session))
    assert not hasattr(session, "rectangle")


def test_settings_precedence_and_validation():
    settings = logic.resolve_settings(None, dict(folder=["raw"], rot90_k=-1, frame_order=None))
    assert settings.folder == ("raw",) and settings.rot90_k == -1
    assert settings.frame_order == "name" and settings.frames_per_parity == 10
    assert settings.rebuilt == settings.roi_seed_dir                # no rebuilt_dir: the seed folder
    defaults = logic.default_values()
    with pytest.raises(ValueError, match="at least one folder"):
        logic.make_settings(dict(defaults, folder=[]))
    assert logic.make_settings(dict(defaults, folder=[]), need_folder=False).folder == ()
    with pytest.raises(ValueError, match="frame_order"):
        logic.make_settings(dict(defaults, folder=["x"], frame_order="date"))
    with pytest.raises(ValueError, match="Frames per parity"):
        logic.make_settings(dict(defaults, folder=["x"], frames_per_parity=0))


# ----------------------------------------------------------------------------- commands
def test_command_table_is_valid_and_keeps_the_previous_ids():
    check_commands(ACTIONS, DEFAULT_LAYOUT, Controller)
    shortcuts = [op.shortcut for op in KEY_OPS if op.shortcut]
    assert len(set(shortcuts)) == len(shortcuts)
    ids = {spec.id for spec in ACTIONS}
    previous = {"page.next", "page.previous", "page.copy_previous", "file.save", "file.save_modified",
                "file.save_all", "file.write_shared", "apply.all", "apply.all_bregma", "reset.page",
                "reset.all", "box.grow", "box.shrink", "lambda.reset_scale", "flip.lambda",
                "toggle.mirror_lock", "display.contrast_up", "edit.undo", "file.resume", "file.settings",
                "tools.rebuild", "tools.preflight", "help.controls", "app.quit"}
    new = {"file.open_cohort", "file.open_manifest", "file.cache_all", "rect.reset", "rect.show",
           "display.shared_scale", "display.fit"}
    assert previous | new <= ids


def test_a_user_binding_of_a_new_shortcut_is_dropped(tmp_path):
    (tmp_path / "layout.json").write_text(json.dumps({"shortcuts": {"file.resume": "Ctrl+O",
                                                                    "page.next": "Ctrl+N"}}), encoding="utf-8")
    dropped = drop_conflicting_overrides(tmp_path)
    assert len(dropped) == 1 and "file.resume" in dropped[0]
    assert json.loads((tmp_path / "layout.json").read_text(encoding="utf-8"))["shortcuts"] == {"page.next": "Ctrl+N"}


# ----------------------------------------------------------------------------- empty start
def test_the_window_starts_empty_with_every_page_action_disabled(empty):
    assert empty.stack.currentIndex() == 0 and "Open cohort" in empty.empty_label.text()
    actions = empty.registry.actions
    assert all(not action.isEnabled() for key, action in actions.items() if key not in WITHOUT_DATA)
    assert actions["file.open_cohort"].isEnabled() and actions["file.open_manifest"].isEnabled()


def write_cached_manifest(tmp_path: Path) -> tuple[Path, logic.EditorSettings]:
    """A manifest whose raw folders are gone, with the old single previews cached."""
    reviewed = tmp_path / "reviewed"
    (reviewed / "alignment_images").mkdir(parents=True)
    rows = []
    for animal, t in (("A", "t0"), ("A", "t1"), ("B", "t0")):
        tifffile.imwrite(reviewed / "alignment_images" / f"230711_{animal}_{t}.tif",
                         np.random.default_rng(len(animal + t)).random(GRID).astype(np.float32))
        rows.append(f"230711,{animal},P,1,{t},0,120,134,cortical_gsr,gcamp_first,x,{tmp_path / 'gone' / t}")
    manifest = tmp_path / "rectangle_manifest.csv"
    manifest.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    settings = logic.make_settings(dict(logic.default_values(), folder=[], roi_set_dir=str(reviewed),
                                        roi_seed_dir=None), need_folder=False)
    return manifest, settings


def test_open_manifest_asks_for_settings_loads_and_goes_to_recent(empty, tmp_path, monkeypatch):
    manifest, settings = write_cached_manifest(tmp_path)
    asked = []
    monkeypatch.setattr(empty, "ask_settings", lambda values, need_folder, note: asked.append(note) or settings)
    empty.controller.open_manifest(str(manifest))
    assert asked and "No cohort file names" in asked[0]           # D10: no cohort names it
    assert empty.stack.currentIndex() == 1
    assert table_column(empty, "key") == ["230711_A_t0", "230711_A_t1", "230711_B_t0"]
    assert table_column(empty, "manifest") == ["odd"] * 3
    assert empty.registry.actions["file.save"].isEnabled()
    assert empty.model.page.image_source == "legacy" and empty.panels["even"].note.isVisible()
    assert empty.recent()[0] == ("manifest", str(manifest.resolve()))
    assert "rectangle_manifest.csv" in empty.windowTitle()
    # Cancelling the settings dialog loads nothing.
    monkeypatch.setattr(empty, "ask_settings", lambda *a, **k: None)
    before = empty.model
    empty.controller.open_manifest(str(manifest))
    assert empty.model is before


def test_a_missing_recent_entry_is_removed_with_a_message(empty, tmp_path):
    empty.add_recent("cohort", str(tmp_path / "gone.yaml"))
    empty.controller.open_recent("cohort", str((tmp_path / "gone.yaml").resolve()))
    assert empty.recent() == [] and "no longer exists" in empty.activity.toPlainText()


@pytest.mark.parametrize("answer", ["save", "discard", "cancel"])
def test_opening_data_over_unsaved_pages_asks_first(window, tmp_path, monkeypatch, answer):
    manifest, settings = write_cached_manifest(tmp_path / "next")
    monkeypatch.setattr(window, "ask_settings", lambda *a, **k: settings)
    monkeypatch.setattr(window, "ask_unsaved", lambda count: answer)
    trigger(window, "bregma.down")
    edited = window.model
    window.controller.open_manifest(str(manifest))
    saved = Path(window.controller.settings.roi_set_dir if answer == "cancel" else edited.args.roi_set_dir)
    if answer == "cancel":
        assert window.model is edited and edited.page.dirty
    else:
        assert window.model is not edited and len(window.model.pages) == 3
    assert (saved / f"{KEYS[0]}.yaml").exists() == (answer == "save")


# ----------------------------------------------------------------------------- equivalence
def test_same_edits_write_byte_identical_yaml(window, tmp_path):
    """The window's actions write what the old editor's keys write."""
    control, shift, none = QtCore.Qt.ControlModifier, QtCore.Qt.ShiftModifier, QtCore.Qt.NoModifier
    edits = [("bregma.down5", "Key_Down", control | shift), ("bregma.left", "Key_Left", control),
             ("lambda.longer5", "Key_Period", shift), ("lambda.shorter", "Key_Comma", none),
             ("rotate.right5", "Key_X", shift), ("rotate.left", "Key_Z", none),
             ("page.next", "Key_N", none), ("lambda.longer", "Key_Period", none),
             ("flip.horizontal", "Key_F", none), ("file.save_all", "Key_S", shift)]
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    sessions = [Session(key=p.key, folder=p.folder, y_1=p.y_1, x_2=p.x_2, boxes=dict(p.boxes),
                        lambda_offset=p.lambda_offset, image=p.image) for p in make_pages()]
    old = ROIEditor(sessions, make_args(old_dir), "cortical_gsr")
    old._rebuild_scene()
    for _action, key, modifiers in edits:
        old._on_key(types.SimpleNamespace(key=lambda k=key: getattr(QtCore.Qt, k), modifiers=lambda m=modifiers: m))
    for action, _key, _modifiers in edits:
        trigger(window, action)
    new_dir = Path(window.controller.settings.roi_set_dir)
    names = sorted(p.name for p in old_dir.glob("*.yaml"))
    assert names == [f"{key}.yaml" for key in sorted(KEYS)]
    for name in names:
        assert (new_dir / name).read_bytes() == (old_dir / name).read_bytes(), name


# ----------------------------------------------------------------------------- undo
def test_undo_and_redo(window):
    page = window.model.page
    start = logic.snapshot(page)
    trigger(window, "bregma.down5")
    moved = logic.snapshot(page)
    trigger(window, "rotate.right5")
    assert window.model.position == 2 and page.lambda_angle == 5
    trigger(window, "edit.undo")
    assert logic.snapshot(page) == moved
    trigger(window, "edit.undo")
    assert logic.snapshot(page) == start
    assert page.dirty                                      # an undone page is not "saved"
    assert not window.registry.actions["edit.undo"].isEnabled()
    trigger(window, "edit.redo")
    assert logic.snapshot(page) == moved
    trigger(window, "lambda.longer")
    assert not window.model.can_redo()


def test_undo_covers_edits_on_several_pages(window):
    before = logic.capture(window.model.pages)
    trigger(window, "apply.all_bregma")                    # no change yet: every page is equal
    assert window.model.position == 0
    trigger(window, "bregma.up5")
    trigger(window, "apply.all_bregma")
    assert all(p.y_1 == window.model.page.y_1 for p in window.model.pages)
    trigger(window, "edit.undo")
    trigger(window, "edit.undo")
    assert logic.capture(window.model.pages) == before


# ----------------------------------------------------------------------------- both panels
def test_a_drag_in_either_panel_moves_both_and_is_one_undo_step(window):
    page = window.model.page
    start_row = page.y_1
    even = window.panels["even"]
    for row in (start_row + 2, start_row + 4, start_row + 6):
        even.bregma.setPos(page.x_2 - 0.5, row - 0.5)       # dragged in the EVEN panel
    assert page.y_1 == start_row + 6 and window.controller.commit_timer.isActive()
    assert window.panels["odd"].bregma.pos().y() == pytest.approx(start_row + 6 - 0.5)
    label = "M2L_alta"
    odd_box, even_box = window.panels["odd"].rois[label], even.rois[label]
    assert odd_box.pos() == even_box.pos()
    window.controller.flush()
    assert window.model.position == 1
    trigger(window, "edit.undo")
    assert page.y_1 == start_row and window.panels["odd"].bregma.pos().y() == pytest.approx(start_row - 0.5)


def test_a_box_dragged_in_the_odd_panel_moves_it_and_its_twin_in_the_even_one(window):
    odd, even = window.panels["odd"], window.panels["even"]
    roi = odd.rois["M2L_alta"]
    roi.setPos((roi.pos().x() - 2, roi.pos().y() + 1))
    assert window.model.selected == "M2L_alta" and window.model.page.dirty
    assert even.rois["M2L_alta"].pos() == roi.pos()
    twin = window.model.page.boxes["M2R_alta"]
    assert twin.col_start == -window.model.page.boxes["M2L_alta"].col_end   # mirror lock
    assert even.rois["M2R_alta"].pos() == odd.rois["M2R_alta"].pos()


def test_the_panels_are_linked_and_titled_by_parity_only(window):
    odd, even = window.panels["odd"], window.panels["even"]
    assert odd.getPlotItem().titleLabel.text == "odd" and even.getPlotItem().titleLabel.text == "even"
    odd.getPlotItem().setXRange(10, 40, padding=0)
    assert even.getPlotItem().viewRange()[0] == pytest.approx([10, 40], abs=1e-6)


def test_moving_the_rectangle_updates_the_verdict_live_and_the_table_on_release(window):
    page = window.model.page
    assert "params.json" in window.verdict.text() and "(this recording)" not in window.verdict.text()
    roi = window.panels["odd"].rectangle
    state = cr.roi_state(bregma_128(page), lambda_128(page), cr.RectangleFactors(0.6, 0.3, 1.05))
    roi.setState({"pos": QtCore.QPointF(*state["pos"]), "size": QtCore.QPointF(*state["size"]),
                  "angle": state["angle"]})
    assert page.rectangle is not None and page.dirty
    assert "(this recording)" in window.verdict.text()
    assert window.panels["even"].rectangle.pos() == roi.pos()
    assert window.model.position == 1                       # setState ends with a "finished" signal
    assert table_column(window, "status")[0] == "edited"
    trigger(window, "rect.reset")
    assert page.rectangle is None and "(this recording)" not in window.verdict.text()
    trigger(window, "edit.undo")
    assert page.rectangle is not None


def test_the_verdict_bar_marks_each_system(window):
    text = window.verdict.text()
    decision, _note = window.model.decision()
    assert decision.gcamp_parity == "odd"                   # the strip was put in the even frames
    assert "<b>manifest</b>: odd = GCaMP (gcamp_first) ✓" in text
    assert "<b>rectangle</b>: odd = GCaMP ✓" in text and "library" not in text
    assert table_column(window, "agree") == ["✓", "✓", "✓"]          # every page has raw 256 means
    assert table_column(window, "rectangle") == ["odd", "odd", "odd"]


def test_shared_contrast_and_show_rectangle_follow_their_actions(window):
    trigger(window, "display.shared_scale")
    assert window.shared_contrast.isChecked()
    levels = [panel.image_item.levels for panel in window.panels.values()]
    assert np.allclose(levels[0], levels[1])
    window.show_rectangle.setChecked(False)
    assert not window.registry.actions["rect.show"].isChecked()
    assert not window.panels["odd"].rectangle.isVisible()


# ----------------------------------------------------------------------------- docks
def test_properties_panel_edits_the_page(window):
    page = window.model.page
    window.bregma_row.setValue(page.bregma_row + 10)
    assert page.y_1 == 65 and window.model.undo_text() == "Move Bregma"
    window.rotation.setValue(7.0)
    assert page.lambda_angle == 7.0
    window.lambda_distance.setValue(page.lambda_offset + 3)
    assert page.lambda_offset == DEFAULT_LAMBDA_OFFSET + 3
    assert window.scale.value() == pytest.approx(page.lambda_scale, abs=1e-3)
    window.scale.setValue(1.0)
    assert page.lambda_offset == page.base_lambda
    trigger(window, "bregma.right5")
    assert window.bregma_col.value() == page.bregma_col
    trigger(window, "toggle.mirror_lock")
    assert window.mirror_lock.isChecked() is window.model.mirror_lock is False


def test_table_status_filters_and_sorting(window):
    assert table_column(window, "status") == ["opened", "not opened", "not opened"]
    trigger(window, "bregma.down")
    assert table_column(window, "status")[0] == "edited"
    trigger(window, "file.save")
    assert table_column(window, "status")[0] == "saved" and not window.model.page.dirty
    window.status_filter.setCurrentText("saved")
    assert window.proxy.rowCount() == 1
    window.status_filter.setCurrentText("All statuses")
    window.page_filter.setText("240104")
    assert window.proxy.rowCount() == 1
    window.page_filter.setText("")
    window.table.sortByColumn(0, QtCore.Qt.DescendingOrder)
    first = window.proxy.mapToSource(window.proxy.index(0, 0)).row()
    assert window.table_model.frame.iat[first, 0] == "240104_M1_t1"


def test_choosing_a_page_in_the_table_carries_nothing_over(window):
    trigger(window, "lambda.longer5")
    untouched = logic.snapshot(window.model.pages[2])
    window.table.selectRow(window.proxy.mapFromSource(window.table_model.index(2, 0)).row())
    assert window.model.index == 2
    assert logic.snapshot(window.model.page) == untouched and not window.model.page.dirty
    assert "(3/3)" in window.windowTitle()
    window.table.selectRow(window.proxy.mapFromSource(window.table_model.index(0, 0)).row())
    trigger(window, "page.next")
    first = window.model.pages[0]
    assert window.model.index == 1 and window.model.page.dirty
    assert window.model.page.lambda_offset == first.lambda_offset and window.model.page.boxes == first.boxes


def test_page_up_and_down_also_change_page(window):
    shortcuts = [s.toString() for s in window.registry.actions["page.next"].shortcuts()]
    assert shortcuts == ["N", "PgDown"]


def test_rebuild_then_preflight_from_the_window(window, qtbot):
    trigger(window, "bregma.down")
    trigger(window, "tools.rebuild")
    assert window.process is None                          # refused: there are unsaved pages
    trigger(window, "file.save_all")
    trigger(window, "tools.rebuild")
    assert window.process is not None
    qtbot.waitUntil(lambda: window.process is None, timeout=120000)
    rebuilt = Path(window.controller.settings.rebuilt)
    assert sorted(p.name for p in rebuilt.glob("*.yaml")) == [f"{key}.yaml" for key in sorted(KEYS)], \
        window.activity.toPlainText()
    assert set(window.preflight) == set(KEYS)
    for index, key in enumerate(KEYS):
        expected = "failed pre-flight" if window.preflight[key] else "rebuilt"
        assert table_column(window, "status")[index] == expected
    trigger(window, "tools.rebuild")
    assert window.process is None                          # nothing new to rebuild


def test_cached_previews_stand_in_for_an_unmounted_raw_drive(tmp_path, monkeypatch):
    """No raw folder + a cohort without a manifest file: the cohort's cached pages."""
    reviewed = tmp_path / "reviewed"
    (reviewed / "alignment_images").mkdir(parents=True)
    rows = [("240104", "M1", "t1"), ("240101", "M1", "t2"), ("240101", "M1", "t1")]
    for day, animal, recording in rows[:2]:                 # the third has no cached image
        tifffile.imwrite(reviewed / "alignment_images" / f"{day}_{animal}_{recording}.tif",
                         np.random.default_rng(0).random(GRID).astype(np.float32))
    monkeypatch.setattr(sources, "cohort_recordings", lambda cohort: rows)
    settings = logic.EditorSettings(folder=(str(tmp_path / "not_mounted"),), roi_set_dir=str(reviewed),
                                    roi_seed_dir=None)
    pages, _args, _profile, preview_only, _notes = sources.pages_from_folders(settings, "cohorts/fake.yaml")
    assert preview_only and [p.key for p in pages] == ["240101_M1_t2", "240104_M1_t1"]
    sources.load_images(pages[0], settings)
    assert pages[0].image.shape == GRID and pages[0].image_source == "legacy"
    with pytest.raises(FileNotFoundError, match="raw drive"):
        sources.pages_from_folders(settings)


def test_choosing_which_parity_is_gcamp_names_the_panels_and_is_saved(window):
    assert window.panels["odd"].getPlotItem().titleLabel.text == "odd"
    assert "<b>your choice</b>: not made" in window.verdict.text()
    trigger(window, "channel.even")
    page = window.model.page
    assert page.gcamp_choice == "even" and page.dirty
    assert window.panels["even"].getPlotItem().titleLabel.text == "even — GCaMP"
    assert window.panels["odd"].getPlotItem().titleLabel.text == "odd — reflectance"
    # The rectangle says odd: the two systems now disagree.
    assert "even = GCaMP, odd = reflectance ⚠" in window.verdict.text()
    assert table_column(window, "choice")[0] == "even" and table_column(window, "agree")[0] == "⚠"
    assert window.gcamp_choice.currentIndex() == 2
    trigger(window, "channel.accept_rectangle")
    assert page.gcamp_choice == "odd" and table_column(window, "agree")[0] == "✓"
    window.gcamp_choice.setCurrentIndex(0)                  # Properties: clear it
    assert page.gcamp_choice is None and window.model.undo_text() == "GCaMP choice"
    trigger(window, "edit.undo")
    assert page.gcamp_choice == "odd"
    trigger(window, "file.save")
    saved = Path(window.controller.settings.roi_set_dir) / f"{KEYS[0]}.yaml"
    assert cr.read_roi_set_channel_order(saved) == "gcamp_first"
