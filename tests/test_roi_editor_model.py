"""roi_editor_spec.md 7.1: the new editor's model against the old editor, headless.

The same edit sequence is applied to ``roi_editor.ROIEditor`` (driven through its key
handler and its items, ``QT_QPA_PLATFORM=offscreen``) and to ``EditorModel``. After every
step each page's geometry must be identical, and at the end both write every page: the
files must be byte-identical. Then undo / redo of every edit must give back the exact
snapshots.

Fixture: two animals, five t#, one page with Lambda above Bregma, one rotated layout.
"""

from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6 import QtCore  # noqa: E402

from roi_editor import (  # noqa: E402
    DEFAULT_LAMBDA_OFFSET, ROIEditor, Session, flip_rows_about_bregma, rotate_boxes,
)
from wfci.atlases import CORTEX_22  # noqa: E402
from wfci_gui.roi_editor import logic  # noqa: E402
from wfci_gui.roi_editor.model import EditorModel, OutOfFrameError, PageModel  # noqa: E402

Qt = QtCore.Qt
GRID = (128, 128)
LABEL = "M2L_alta"            # a box with a bilateral twin
SINGLE = next(label for label in CORTEX_22.boxes if label != LABEL)

# key, y_1, x_2, lambda_offset, lambda_sign, angle
PAGES = [
    ("240101_A_t1", 60, 67, DEFAULT_LAMBDA_OFFSET, 1, 0.0),
    ("240101_A_t2", 61, 66, 33, 1, 0.0),
    ("240101_A_t3", 70, 64, 30, -1, 0.0),          # Lambda above Bregma (head upside down)
    ("240104_B_t1", 58, 68, 31, 1, 7.0),           # rotated layout
    ("240104_B_t2", 59, 67, 30, 1, 0.0),
]


def page_kwargs(key, y_1, x_2, lam, sign, angle, seed):
    boxes = dict(CORTEX_22.boxes)
    if sign < 0:
        boxes = flip_rows_about_bregma(boxes)
    if angle:
        boxes = rotate_boxes(boxes, angle)
    image = np.random.default_rng(seed).random(GRID)
    return dict(key=key, folder="(synthetic)", y_1=y_1, x_2=x_2, boxes=boxes, lambda_offset=lam,
                lambda_sign=sign, lambda_angle=angle, image=image)


def make_args(roi_set_dir: Path):
    return types.SimpleNamespace(channel_order="auto", preview_frames=1, roi_set_dir=str(roi_set_dir),
                                 roi_seed_dir=None, shared_name="shared.yaml",
                                 lambda_offset=DEFAULT_LAMBDA_OFFSET, lambda_scales_box_size=False,
                                 mirror_lock=True)


class KeyPress:
    """The two things ROIEditor._on_key reads from a Qt key event."""

    def __init__(self, key, modifiers=Qt.NoModifier):
        self._key, self._modifiers = key, modifiers

    def key(self):
        return self._key

    def modifiers(self):
        return self._modifiers


def editors(tmp_path):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old = ROIEditor([Session(**page_kwargs(*row, seed=i)) for i, row in enumerate(PAGES)],
                    make_args(old_dir), "cortical_gsr")
    old._rebuild_scene()
    new = EditorModel([PageModel(**page_kwargs(*row, seed=i)) for i, row in enumerate(PAGES)],
                      make_args(new_dir), "cortical_gsr")
    return old, new, old_dir, new_dir


# ----------------------------------------------------------------------------- steps
# A step is (old action, new action). Helpers build the common ones.
def key(name: str, modifiers=Qt.NoModifier, new=None):
    return (lambda old: old._on_key(KeyPress(getattr(Qt, name), modifiers)), new)


SHIFT, CTRL = Qt.ShiftModifier, Qt.ControlModifier


def select(label):
    return (lambda old: old._select(label), lambda new: new.select(label))


def drag_box(label, d_row, d_col):
    """Move the old editor's RectROI; read its state back for the model, as the view does."""
    state = {}

    def old_step(old):
        roi = old._rois[label]
        roi.setPos((roi.pos().x() + d_col, roi.pos().y() + d_row))
        pos, size = roi.pos(), roi.size()
        state.update(row=int(round(pos.y())), col=int(round(pos.x())),
                     height=int(round(size.y())), width=int(round(size.x())))

    return old_step, lambda new: new.move_box(label, state["row"], state["col"], state["height"], state["width"])


def resize_box_by_handle(label, grow_rows, grow_cols):
    state = {}

    def old_step(old):
        roi = old._rois[label]
        roi.setSize((roi.size().x() + grow_cols, roi.size().y() + grow_rows))
        pos, size = roi.pos(), roi.size()
        state.update(row=int(round(pos.y())), col=int(round(pos.x())),
                     height=int(round(size.y())), width=int(round(size.x())))

    return old_step, lambda new: new.move_box(label, state["row"], state["col"], state["height"], state["width"])


def drag_bregma(d_row, d_col):
    state = {}

    def old_step(old):
        session = old.session
        x, y = session.x_2 - 0.5 + d_col, session.y_1 - 0.5 + d_row
        old.target.setPos(x, y)
        state.update(x=x, y=y)

    return old_step, lambda new: new.drag_bregma(state["y"] + 0.5, state["x"] + 0.5)


def drag_lambda(d_row, d_col):
    state = {}

    def old_step(old):
        row, col = old.session.lambda_point
        x, y = col - 0.5 + d_col, row - 0.5 + d_row
        old.lambda_target.setPos(x, y)
        state.update(x=x, y=y)

    return old_step, lambda new: new.drag_lambda(state["y"] + 0.5, state["x"] + 0.5)


def double_click(view_row, view_col):
    # ROIEditor._on_scene_clicked maps the click to a view position, then does exactly this.
    return (lambda old: old._move_bregma(math.floor(view_row) + 1, math.floor(view_col) + 1),
            lambda new: new.place_bregma(view_row, view_col))


def save_page():
    return key("Key_S", new=lambda new: new.save_current())


def resume():
    return (lambda old: old._resume_session(), lambda new: new.resume())


def nudges():
    steps = []
    for name, (dr, dc) in (("Up", (-1, 0)), ("Down", (1, 0)), ("Left", (0, -1)), ("Right", (0, 1))):
        steps.append(key(f"Key_{name}", new=lambda new, dr=dr, dc=dc: new.nudge(dr, dc)))
        steps.append(key(f"Key_{name}", SHIFT, new=lambda new, dr=dr, dc=dc: new.nudge(5 * dr, 5 * dc)))
    return steps


def resizes():
    return [key("Key_Plus", new=lambda new: new.resize_box(1)), key("Key_Plus", new=lambda new: new.resize_box(1)),
            key("Key_Minus", new=lambda new: new.resize_box(-1))]


def toggle_mirror():
    return key("Key_M", new=lambda new: new.toggle_mirror_lock())


def toggle_rotate():
    return key("Key_O", new=lambda new: new.toggle_rotate_with_lambda())


def nav(delta):
    return key("Key_N" if delta > 0 else "Key_P", new=lambda new: new.navigate(delta))


LAMBDA_KEYS = [key("Key_Period", new=lambda new: new.change_lambda(1)),
               key("Key_Period", SHIFT, new=lambda new: new.change_lambda(5)),
               key("Key_Comma", new=lambda new: new.change_lambda(-1)),
               key("Key_Comma", SHIFT, new=lambda new: new.change_lambda(-5)),
               key("Key_Period", SHIFT, new=lambda new: new.change_lambda(5)),
               key("Key_L", new=lambda new: new.reset_scale())]
ROTATIONS = [key("Key_X", new=lambda new: new.rotate(1)), key("Key_X", SHIFT, new=lambda new: new.rotate(5)),
             key("Key_Z", new=lambda new: new.rotate(-1)), key("Key_Z", SHIFT, new=lambda new: new.rotate(-5)),
             key("Key_Z", SHIFT, new=lambda new: new.rotate(-5))]
FLIPS = [key("Key_F", new=lambda new: new.flip(True)), key("Key_V", new=lambda new: new.flip(False)),
         key("Key_U", new=lambda new: new.flip_lambda_all())]
BREGMA = [key("Key_Down", CTRL, new=lambda new: new.move_bregma_by(1, 0)),
          key("Key_Left", CTRL | SHIFT, new=lambda new: new.move_bregma_by(0, -5)),
          drag_bregma(3, -2), double_click(64.3, 70.8)]

SCENARIOS = {
    "box_edits_mirror_lock_on": [select(LABEL), *nudges(), *resizes(), drag_box(LABEL, 2, -3),
                                 resize_box_by_handle(LABEL, 1, 2)],
    "box_edits_mirror_lock_off": [toggle_mirror(), select(LABEL), *nudges(), *resizes(),
                                  drag_box(LABEL, -2, 1), toggle_mirror(), select(SINGLE), *nudges()],
    "box_edits_on_the_rotated_page": [nav(1), nav(1), nav(1), select(LABEL), *nudges(), *resizes(),
                                      drag_box(LABEL, 1, 1)],
    "lambda_rescale_and_drag": [*LAMBDA_KEYS, drag_lambda(4, 3), toggle_rotate(), drag_lambda(2, 5),
                                toggle_rotate(), drag_lambda(-3, -1)],
    "rotate": ROTATIONS + [nav(1), nav(1), nav(1), *ROTATIONS],
    "flips": FLIPS + [nav(1), *FLIPS, nav(1), nav(1), key("Key_F", new=lambda new: new.flip(True))],
    "bregma": BREGMA + [nav(1), nav(1), *BREGMA],
    "navigate_carries_layout_over": [*LAMBDA_KEYS[:2], ROTATIONS[1], nav(1), nav(1), nav(1), nav(1), nav(-1)],
    "copy_from_previous": [BREGMA[0], LAMBDA_KEYS[1], nav(1), key("Key_C", new=lambda new: new.copy_from_previous()),
                           nav(1), nav(1), key("Key_C", new=lambda new: new.copy_from_previous())],
    "apply_to_all": [BREGMA[1], LAMBDA_KEYS[1], ROTATIONS[1], key("Key_A", new=lambda new: new.apply_to_all(False))],
    "apply_to_all_with_bregma": [nav(1), nav(1), BREGMA[0], LAMBDA_KEYS[3],
                                 key("Key_A", SHIFT, new=lambda new: new.apply_to_all(True))],
    "reset": [BREGMA[0], LAMBDA_KEYS[1], key("Key_R", new=lambda new: new.reset_page()), nav(1), FLIPS[0],
              nav(1), ROTATIONS[1], key("Key_R", SHIFT, new=lambda new: new.reset_all()), ROTATIONS[0]],
    "resume": [BREGMA[0], save_page(), LAMBDA_KEYS[1], ROTATIONS[1], resume(), nav(1), BREGMA[1], resume(),
               nav(1), save_page(), FLIPS[2], resume()],
}


def page_state(page):
    """Geometry plus the flags a save depends on (the old Session has no rectangle and no
    GCaMP choice: the page-only fields are left out)."""
    values, boxes, base, seed = logic.snapshot(page)
    return values[:-len(logic.PAGE_ONLY_FIELDS)], boxes, base, seed, page.dirty, page.has_own_set


def assert_same(old, new, where):
    assert old.index == new.index, where
    assert old.selected == new.selected, where
    assert old.mirror_lock == new.mirror_lock and old.rotate_with_lambda == new.rotate_with_lambda, where
    for i, (session, page) in enumerate(zip(old.sessions, new.pages)):
        assert page_state(session) == page_state(page), f"{where}, page {i}"
        assert page.rectangle is None and page.gcamp_choice is None


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_same_edits_write_byte_identical_yaml(name, tmp_path):
    old, new, old_dir, new_dir = editors(tmp_path)
    for number, (old_step, new_step) in enumerate(SCENARIOS[name]):
        old_step(old)
        new_step(new)
        assert_same(old, new, f"{name} step {number}")
    # Everything: every page under its own key, and the shared set.
    old._on_key(KeyPress(Qt.Key_S, SHIFT))
    new.save_all()
    old._on_key(KeyPress(Qt.Key_W))
    new.write_shared()
    assert_same(old, new, f"{name} after saving")
    names = sorted(p.name for p in old_dir.glob("*.yaml"))
    assert names == sorted([f"{row[0]}.yaml" for row in PAGES] + ["shared.yaml"])
    assert sorted(p.name for p in new_dir.glob("*.yaml")) == names
    for file_name in names:
        old_bytes, new_bytes = (old_dir / file_name).read_bytes(), (new_dir / file_name).read_bytes()
        # The folder line names the save directory's page: identical by construction.
        assert new_bytes == old_bytes, file_name


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_undo_then_redo_returns_the_identical_snapshot(name, tmp_path):
    _old, new, _old_dir, _new_dir = editors(tmp_path)
    steps = 0
    for _old_step, new_step in SCENARIOS[name]:
        before = logic.capture(new.pages)
        new_step(new)
        index, selected = new.index, new.selected
        if not new.commit("step"):
            continue
        after = logic.capture(new.pages)
        new.undo()
        assert logic.capture(new.pages) == before
        new.redo()
        assert logic.capture(new.pages) == after
        # Undo / redo clear the selection; the scenario's next step expects it back.
        new.index, new.selected = index, selected
        steps += 1
    assert steps > 0


def test_a_new_edit_after_undo_drops_the_redo_branch(tmp_path):
    _old, new, _o, _n = editors(tmp_path)
    new.rotate(5)
    new.commit("rotate")
    new.move_bregma_by(2, 0)
    new.commit("bregma")
    new.undo()
    assert new.can_redo() and new.redo_text() == "bregma"
    new.change_lambda(1)
    new.commit("lambda")
    assert not new.can_redo() and new.undo_text() == "lambda"
    assert not new.commit("nothing changed")


def test_a_box_off_the_frame_refuses_the_save_and_names_it(tmp_path):
    _old, new, _o, new_dir = editors(tmp_path)
    new.move_bregma(1, 1)                      # pushes the layout off the top-left corner
    with pytest.raises(OutOfFrameError, match="off the frame"):
        new.save_current()
    new.navigate(1)
    new.move_bregma_by(1, 0)
    with pytest.raises(OutOfFrameError, match="240101_A_t1"):
        new.save_modified()
    # The page that fits was still written.
    assert [p.name for p in new_dir.glob("*.yaml")] == ["240101_A_t2.yaml"]
