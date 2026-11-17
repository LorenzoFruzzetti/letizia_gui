"""The ROI editor's state and every edit, with no Qt (roi_editor_spec.md D1, section 6.4).

``PageModel`` is one recording: a ``roi_editor.Session`` (boxes as Bregma offsets, Bregma,
Lambda distance / side / angle, the rescale reference and the reset seed) plus the page's
own channel rectangle, its odd / even images and where they came from.

``EditorModel`` holds the pages and does every edit the old editor's ``ROIEditor`` did,
reproducing its arithmetic: the geometry is computed by the same ``roi_editor`` functions
(``scale_boxes`` through ``Session.rescale_to``, ``mirror_box`` / ``mirror_twin``,
``rotate_boxes``, ``flip_rows_about_bregma``, ``unrotated_op``, ``lambda_position``), and
the logic that lived inside ``ROIEditor`` methods (``_mirror_to_twin``, ``_set_lambda``,
``_move_bregma``, ``_navigate``, ``_copy_from_previous``, ``_apply_to_all``, ``_resume_session``,
``_save``) is re-implemented here. tests/test_roi_editor_model.py pins it: the same edits
in both write byte-identical YAML.

Edits change only the model and return nothing; ``message`` holds the line the old editor
printed. Undo is here too (``commit`` / ``undo`` / ``redo``), so it is tested headless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .logic import capture, changes, restore

from channel_rectangle import (  # noqa: E402  (repo root, on sys.path via logic)
    RectangleDecision,
    RectangleFactors,
    append_roi_set_channel_order,
    append_roi_set_rectangle,
    decide,
    decision_problem,
    factors_from_roi,
    params_corners,
    params_or_default,
    order_from_parity,
    params_with,
    parity_of_order,
    read_roi_set_channel_order,
    read_roi_set_rectangle,
)
from roi_editor import (  # noqa: E402
    MIN_LAMBDA_OFFSET,
    Box,
    Session,
    box_from_pixels,
    load_lambda_above,
    load_lambda_angle,
    load_lambda_offset,
    load_roi_set,
    mirror_box,
    mirror_twin,
    normalise_angle,
    pixel_bounds,
    save_roi_set,
    unrotated_op,
)
from wfci import ROIConfig, box_slices_for, get_profile  # noqa: E402

NUDGE_BIG = 5          # shift + arrow / , . / z x step, as in the old editor


class OutOfFrameError(ValueError):
    """A save was refused: boxes lie outside the frame (they are drawn red)."""


@dataclass(eq=False)
class PageModel(Session):
    """One page: the old editor's Session plus the rectangle and the two images.

    ``image`` (inherited) is the odd 128-grid image, or the legacy single preview; it
    gives the grid every geometric check uses. ``odd_even`` (inherited) holds the
    256-grid means the rectangle decision needs, read from the raw frames only.
    """

    group: str = ""
    channel_order: str = ""                 # the manifest's, "" when the source has none
    # The page's own rectangle (None = params.json's). Not None is the "moved" flag: a
    # page moved back onto the params.json values still writes the key (spec 5.1).
    rectangle: RectangleFactors | None = None
    seed_rectangle: RectangleFactors | None = None      # what 'r' (reset) returns to
    # The user's call of which parity is GCaMP ("odd" / "even"; None = not chosen). Saved
    # as `channel_order` in the ROI set. Per recording: never carried to another page.
    gcamp_choice: str | None = None
    seed_gcamp_choice: str | None = None
    odd_even_128: tuple[np.ndarray, np.ndarray | None] | None = None
    image_source: str = "none"              # raw | cache | legacy | none
    seed_source: str = ""                   # footer: "own set" / "seeded from <x>"

    # Pages are compared by identity: Session's generated __eq__ would compare the numpy
    # images element-wise and fail.
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    @property
    def odd_even_256(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self.odd_even

    @property
    def day(self) -> str:
        return self.key.split("_", 1)[0]

    @property
    def recording(self) -> str:
        return self.key.rsplit("_", 1)[1]

    @property
    def animal(self) -> str:
        return self.key.split("_", 1)[1].rsplit("_", 1)[0]


def bregma_128(page: Session) -> np.ndarray:
    """Bregma in continuous 128-grid coordinates: the centre of pixel (y_1 - 1, x_2 - 1)."""
    return np.array([page.y_1 - 0.5, page.x_2 - 0.5])


def lambda_128(page: Session) -> np.ndarray:
    row, col = page.lambda_point
    return np.array([row - 0.5, col - 0.5])


def out_of_frame(page: Session, shape: tuple[int, int]) -> list[str]:
    """Labels whose box does not fit the frame (ROIEditor._out_of_frame)."""
    rows, cols = shape
    bad = []
    for label, box in page.boxes.items():
        r0, r1, c0, c1 = pixel_bounds(box, page.y_1, page.x_2)
        if r0 < 0 or c0 < 0 or r1 >= rows or c1 >= cols:
            bad.append(label)
    return bad


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def lambda_side(page: Session) -> str:
    return "below Bregma" if page.lambda_sign > 0 else "ABOVE Bregma"


class EditorModel:
    """Pages plus the editing state (current page, selected box, the two toggles)."""

    def __init__(self, pages: list[PageModel], args, profile_name: str,
                 loader: Callable[[PageModel], None] | None = None) -> None:
        self.pages = list(pages)
        self.args = args
        self.profile_name = profile_name
        # Fills a page's images the first time it is needed (sources.load_images).
        self.loader = loader
        self.index = 0
        self.selected: str | None = None
        self.mirror_lock = bool(getattr(args, "mirror_lock", True))
        self.rotate_with_lambda = bool(getattr(args, "rotate_with_lambda", True))
        self.message = ""
        self.visited: set[int] = {0} if self.pages else set()
        self.rectangle_params = params_or_default(getattr(args, "channel_rectangle_params", None))
        # Undo: a list of (label, {page index: (before, after)}) and a cursor into it.
        self.history: list[tuple[str, dict]] = []
        self.position = 0
        self.baseline = capture(self.pages)

    # ----- state ---------------------------------------------------------------------
    @property
    def page(self) -> PageModel:
        return self.pages[self.index]

    @property
    def empty(self) -> bool:
        return not self.pages

    def dirty_pages(self) -> list[PageModel]:
        return [page for page in self.pages if page.dirty]

    def say(self, text: str) -> None:
        self.message = text

    def ensure_image(self, page: PageModel) -> None:
        if page.image is None:
            if self.loader is None:
                raise ValueError(f"{page.key}: no image and no loader to read one")
            self.loader(page)

    def shape(self, page: PageModel | None = None) -> tuple[int, int]:
        page = self.page if page is None else page
        self.ensure_image(page)
        return page.image.shape

    def out_of_frame(self, page: PageModel | None = None) -> list[str]:
        page = self.page if page is None else page
        return out_of_frame(page, self.shape(page))

    # ----- undo ----------------------------------------------------------------------
    def commit(self, label: str) -> bool:
        """Record whatever changed since the last step as one undo step (nothing = none)."""
        after = capture(self.pages)
        diff = changes(self.baseline, after)
        if not diff:
            return False
        del self.history[self.position:]          # a new edit drops the redo branch
        self.history.append((label, diff))
        self.position += 1
        self.baseline = after
        return True

    def can_undo(self) -> bool:
        return self.position > 0

    def can_redo(self) -> bool:
        return self.position < len(self.history)

    def undo_text(self) -> str:
        return self.history[self.position - 1][0] if self.can_undo() else ""

    def redo_text(self) -> str:
        return self.history[self.position][0] if self.can_redo() else ""

    def undo(self) -> None:
        label, diff = self.history[self.position - 1]
        self.position -= 1
        self._apply_states({index: pair[0] for index, pair in diff.items()})
        self.say(f"Undone: {label}")

    def redo(self) -> None:
        label, diff = self.history[self.position]
        self.position += 1
        self._apply_states({index: pair[1] for index, pair in diff.items()})
        self.say(f"Redone: {label}")

    def _apply_states(self, states: dict[int, tuple]) -> None:
        for index, state in states.items():
            restore(self.pages[index], state)
            # The page no longer matches its file (or never had one).
            self.pages[index].dirty = True
        if self.index not in states:
            self.index = min(states)
            self.visited.add(self.index)
        self.selected = None
        self.baseline = capture(self.pages)

    # ----- pages ---------------------------------------------------------------------
    def navigate(self, delta: int) -> None:
        """n / p: next / previous page (wraps). A page opened for the first time takes the
        current layout (boxes, Lambda distance, side, angle; not Bregma), unless it has its
        own saved set (ROIEditor._navigate)."""
        source = self.page
        self.index = (self.index + delta) % len(self.pages)
        target = self.page
        note = ""
        if self.index not in self.visited:
            self.visited.add(self.index)
            if target.has_own_set:
                note = (f"{target.key}: kept its own saved ROI set (not carried over from "
                        f"{source.key}); press 'c' to copy the previous page's layout over it.")
            else:
                target.boxes = dict(source.boxes)
                target.lambda_offset = source.lambda_offset
                target.lambda_sign = source.lambda_sign
                target.lambda_angle = source.lambda_angle   # before rebase: it unrotates
                target.rebase()
                target.dirty = True
        self.selected, self.message = None, ""
        self.ensure_image(target)
        if note:
            self.say(note)

    def go_to(self, index: int) -> None:
        """Open a page from the table: nothing is carried over to it."""
        self.index = index
        self.visited.add(index)
        self.selected, self.message = None, ""
        self.ensure_image(self.page)

    def copy_from_previous(self) -> None:
        """c: the previous page's whole layout (boxes, Lambda, Bregma, rectangle) here."""
        if len(self.pages) < 2:
            self.say("Only one session loaded -- there is no previous page to copy from.")
            return
        source = self.pages[(self.index - 1) % len(self.pages)]
        target = self.page
        target.boxes = dict(source.boxes)
        target.lambda_offset = source.lambda_offset
        target.lambda_sign = source.lambda_sign
        target.lambda_angle = source.lambda_angle
        target.y_1, target.x_2 = source.y_1, source.x_2
        target.rectangle = source.rectangle         # D19: the rectangle comes along
        target.rebase()
        target.dirty = True
        self.selected = None
        self.say(f"{target.key}: layout copied from {source.key} ({len(target.boxes)} boxes, "
                 f"Bregma row={target.bregma_row} col={target.bregma_col}, Lambda "
                 f"{target.lambda_offset} px {lambda_side(target)})")

    def apply_to_all(self, with_bregma: bool) -> None:
        """a / A: this page's boxes, scale, side, angle and rectangle (A: and Bregma) on
        every other page (ROIEditor._apply_to_all)."""
        page = self.page
        for other in self.pages:
            if other is page:
                continue
            other.boxes = dict(page.boxes)
            other.lambda_offset = page.lambda_offset
            other.lambda_sign = page.lambda_sign
            other.lambda_angle = page.lambda_angle
            other.rectangle = page.rectangle
            other.rebase()
            if with_bregma:
                other.y_1, other.x_2 = page.y_1, page.x_2
            other.dirty = True
        what = "boxes, scale and Bregma" if with_bregma else "boxes and scale (each keeps its Bregma)"
        self.say(f"Applied {page.key}'s {what} to all {len(self.pages)} sessions -- press S to "
                 f"write them, or w for one shared file.")

    def reset_page(self) -> None:
        """r: back to the layout the page opened with (its seed)."""
        self._reset(self.page)
        self.selected = None
        self.say(f"{self.page.key}: reset to the starting layout.")

    def reset_all(self) -> None:
        """R: every page back to its seed."""
        for page in self.pages:
            self._reset(page)
        self.selected = None
        self.say(f"All {len(self.pages)} sessions reset to starting layouts.")

    @staticmethod
    def _reset(page: PageModel) -> None:
        y_1, x_2, lam, boxes = page.seed
        page.y_1, page.x_2, page.boxes = y_1, x_2, dict(boxes)
        page.lambda_offset = lam
        page.lambda_angle = page.seed_angle
        page.rectangle = page.seed_rectangle
        page.gcamp_choice = page.seed_gcamp_choice
        page.rebase()
        page.dirty = False

    # ----- Bregma, Lambda, rotation, flips -------------------------------------------
    def move_bregma(self, y_1: int, x_2: int) -> None:
        """Bregma to (y_1, x_2), clamped to the frame; every box moves with it."""
        page = self.page
        rows, cols = self.shape(page)
        page.y_1 = _clamp(y_1, 1, rows)
        page.x_2 = _clamp(x_2, 1, cols)
        page.dirty = True
        self.say(f"Bregma -> row={page.bregma_row}, col={page.bregma_col} "
                 f"({len(page.boxes)} boxes moved with it)")

    def move_bregma_by(self, d_row: int, d_col: int) -> None:
        """ctrl + arrows."""
        self.move_bregma(self.page.y_1 + d_row, self.page.x_2 + d_col)

    def drag_bregma(self, row: float, col: float) -> None:
        """The Bregma marker dragged to the 1-based (row, col) = marker + 0.5."""
        self.move_bregma(int(round(row)), int(round(col)))

    def place_bregma(self, view_row: float, view_col: float) -> None:
        """Double-click at a view position: Bregma on the clicked pixel."""
        self.move_bregma(math.floor(view_row) + 1, math.floor(view_col) + 1)

    def _lambda_direction(self, page: PageModel) -> tuple[float, float]:
        radians = math.radians(page.lambda_angle)
        return page.lambda_sign * math.cos(radians), -page.lambda_sign * math.sin(radians)

    def set_lambda(self, lambda_offset: int, rotated: bool = False) -> None:
        """Bregma -> Lambda distance (clamped to the frame along the axis); rescales the
        layout about Bregma (ROIEditor._set_lambda)."""
        page = self.page
        rows, cols = self.shape(page)
        row_dir, col_dir = self._lambda_direction(page)
        limits = []
        for start, step, last in ((page.y_1, row_dir, rows), (page.x_2, col_dir, cols)):
            if step > 1e-9:
                limits.append((last - start) / step)
            elif step < -1e-9:
                limits.append((1 - start) / step)
        room = int(math.floor(min(limits) + 1e-9))
        lambda_offset = _clamp(lambda_offset, MIN_LAMBDA_OFFSET, room)
        page.rescale_to(lambda_offset, self.args.lambda_scales_box_size)
        page.dirty = True
        turned = f", rotated {page.lambda_angle:+.1f} deg" if rotated else ""
        self.say(f"Lambda {lambda_offset} px {lambda_side(page)}{turned} -> scale "
                 f"{page.lambda_scale:.3f}x of the reference layout ({len(page.boxes)} boxes moved)")

    def change_lambda(self, delta: int) -> None:
        """, . < >"""
        self.set_lambda(self.page.lambda_offset + delta)

    def reset_scale(self) -> None:
        """l: back to scale 1.000x (the reference distance)."""
        self.set_lambda(self.page.base_lambda)

    def drag_lambda(self, row: float, col: float) -> None:
        """The Lambda marker dragged to the 1-based (row, col) = marker + 0.5: its distance
        is the scale, its direction the rotation (ROIEditor._on_lambda_moved)."""
        page = self.page
        d_row, d_col = row - page.y_1, col - page.x_2
        if not self.rotate_with_lambda:
            row_dir, col_dir = self._lambda_direction(page)
            self.set_lambda(int(round(d_row * row_dir + d_col * col_dir)))
            return
        sign = page.lambda_sign
        angle = math.degrees(math.atan2(-sign * d_col, sign * d_row))
        page.rotate_to(angle, self.args.lambda_scales_box_size)
        self.set_lambda(int(round(math.hypot(d_row, d_col))), rotated=True)

    def rotate(self, degrees: float) -> None:
        """z / x: Lambda and every box centre further around Bregma."""
        page = self.page
        page.rotate_to(page.lambda_angle + degrees, self.args.lambda_scales_box_size)
        page.dirty = True
        self.say(f"Lambda + constellation rotated to {page.lambda_angle:+.1f} deg around Bregma "
                 f"({len(page.boxes)} axis-aligned boxes moved)")

    def set_rotation(self, degrees: float) -> None:
        """Properties panel: an absolute angle."""
        self.rotate(degrees - self.page.lambda_angle)

    def set_scale(self, scale: float) -> None:
        """Properties panel: scale relative to the reference distance."""
        from roi_editor import round_half_away
        self.set_lambda(round_half_away(scale * self.page.base_lambda))

    def flip(self, horizontal: bool) -> None:
        """f / v: mirror Bregma and every box about the image centre on one axis."""
        page = self.page
        rows, cols = self.shape(page)
        if horizontal:
            page.x_2 = _clamp(cols - (page.x_2 - 1), 1, cols)
            page.boxes = {label: Box(row_start=box.row_start, row_end=box.row_end,
                                     col_start=-box.col_end, col_end=-box.col_start)
                          for label, box in page.boxes.items()}
            what = f"flipped left-right (Bregma col={page.bregma_col})"
        else:
            page.y_1 = _clamp(rows - (page.y_1 - 1), 1, rows)
            page.boxes = {label: Box(row_start=-box.row_end, row_end=-box.row_start,
                                     col_start=box.col_start, col_end=box.col_end)
                          for label, box in page.boxes.items()}
            what = f"flipped up-down (Bregma row={page.bregma_row})"
        # A mirror reverses the sense of rotation, so the rebase unrotates by -angle.
        page.lambda_angle = normalise_angle(-page.lambda_angle)
        page.rebase()
        page.dirty = True
        self.say(f"Constellation {what}, {len(page.boxes)} boxes mirrored")

    def flip_lambda_all(self) -> None:
        """u: Lambda to the other side of Bregma on EVERY page, layouts mirrored up-down."""
        for page in self.pages:
            page.flip_lambda_side()
        self.selected = None
        self.say(f"Bregma-Lambda flipped on all {len(self.pages)} pages: Lambda now "
                 f"{lambda_side(self.page)} on this page, layout mirrored about Bregma's row. "
                 f"Press 'u' again to undo; save to keep it.")

    # ----- one box -------------------------------------------------------------------
    def select(self, label: str | None) -> None:
        self.selected = label

    def _mirror_to_twin(self, label: str) -> str | None:
        """With the mirror lock on, rewrite the bilateral twin as the exact mirror of
        ``label`` (in the unrotated frame on a rotated layout). Returns the twin."""
        if not self.mirror_lock:
            return None
        page = self.page
        twin = mirror_twin(label, page.boxes)
        if twin is None:
            return None
        mirrored = unrotated_op({label: page.boxes[label]}, page.lambda_angle,
                                lambda boxes: {label: mirror_box(boxes[label])})
        page.boxes[twin] = mirrored[label]
        return twin

    def move_box(self, label: str, row: int, col: int, height: int, width: int) -> None:
        """A box dragged or resized with the mouse: its rect's top-left pixel and size."""
        page = self.page
        height, width = max(1, int(height)), max(1, int(width))
        page.boxes[label] = box_from_pixels(row, row + height - 1, col, col + width - 1,
                                            page.y_1, page.x_2)
        twin = self._mirror_to_twin(label)
        page.rebase()
        page.dirty = True
        self.selected = label
        if twin:
            self.say(f"{label} moved, {twin} mirrored with it (m unlocks the pair)")

    def nudge(self, d_row: int, d_col: int) -> None:
        """Arrows: the selected box by (d_row, d_col), kept whole inside the frame."""
        if self.selected is None:
            self.say("No ROI selected -- click one first.")
            return
        page = self.page
        rows, cols = self.shape(page)
        r0, r1, c0, c1 = pixel_bounds(page.boxes[self.selected], page.y_1, page.x_2)
        d_row = _clamp(d_row, -r0, rows - 1 - r1)
        d_col = _clamp(d_col, -c0, cols - 1 - c1)
        page.boxes[self.selected] = box_from_pixels(r0 + d_row, r1 + d_row, c0 + d_col, c1 + d_col,
                                                    page.y_1, page.x_2)
        twin = self._mirror_to_twin(self.selected)
        page.rebase()
        page.dirty = True
        if twin:
            self.say(f"{self.selected} moved, {twin} mirrored with it (m unlocks the pair)")

    def resize_box(self, delta: int) -> None:
        """+ / -: grow / shrink the selected box by ``delta`` px on every side."""
        if self.selected is None:
            self.say("No ROI selected -- click one first.")
            return
        page = self.page
        rows, cols = self.shape(page)
        r0, r1, c0, c1 = pixel_bounds(page.boxes[self.selected], page.y_1, page.x_2)
        r0, r1 = _clamp(r0 - delta, 0, rows - 1), _clamp(r1 + delta, 0, rows - 1)
        c0, c1 = _clamp(c0 - delta, 0, cols - 1), _clamp(c1 + delta, 0, cols - 1)
        if r1 < r0 or c1 < c0:
            self.say("Box is already 1 px -- cannot shrink further.")
            return
        page.boxes[self.selected] = box_from_pixels(r0, r1, c0, c1, page.y_1, page.x_2)
        twin = self._mirror_to_twin(self.selected)
        page.rebase()
        page.dirty = True
        if twin:
            self.say(f"{self.selected} resized, {twin} mirrored with it (m unlocks the pair)")

    def toggle_mirror_lock(self) -> None:
        self.mirror_lock = not self.mirror_lock
        self.say("Mirror lock ON -- editing one box of an L/R pair now mirrors its twin about Bregma."
                 if self.mirror_lock else
                 "Mirror lock OFF -- boxes move independently. Anything you edit one-sided from "
                 "here stays asymmetric, and 'c' carries it forward.")

    def toggle_rotate_with_lambda(self) -> None:
        self.rotate_with_lambda = not self.rotate_with_lambda
        self.say("Rotate with Lambda ON -- dragging Lambda sideways turns it and the whole "
                 "constellation around Bregma." if self.rotate_with_lambda else
                 "Rotate with Lambda OFF -- dragging Lambda only changes its distance "
                 "(z / x still rotate).")

    # ----- channel rectangle ---------------------------------------------------------
    def page_params(self, page: PageModel | None = None):
        """The rectangle parameters of a page: its own factors, else params.json's."""
        page = self.page if page is None else page
        return params_with(self.rectangle_params, page.rectangle)

    def rectangle_corners_128(self, page: PageModel | None = None) -> np.ndarray:
        """(4, 2) corners (row, col) on the 128 grid, the editor's view coordinates."""
        page = self.page if page is None else page
        return params_corners(bregma_128(page), lambda_128(page), self.page_params(page))

    def move_rectangle(self, corner, length: float, width: float) -> None:
        """The rectangle dragged in a panel: ``corner`` = its local origin (row, col) and
        ``length`` / ``width`` its size, on the 128 grid. Projected onto the Bregma-Lambda
        axis and clipped to the parameter limits (D16)."""
        page = self.page
        page.rectangle = factors_from_roi(bregma_128(page), lambda_128(page), corner, length, width)
        page.dirty = True

    def reset_rectangle(self) -> None:
        """Back to params.json's rectangle: the page writes no key any more."""
        page = self.page
        page.rectangle = None
        page.dirty = True
        self.say(f"{page.key}: channel rectangle reset to params.json")

    # ----- which parity is GCaMP (the user's call) --------------------------------------
    def choose_gcamp(self, parity: str | None) -> None:
        """The user says which interleaved group is GCaMP: "odd", "even" or None (clear)."""
        if parity not in ("odd", "even", None):
            raise ValueError(f"parity must be 'odd', 'even' or None, got {parity!r}")
        page = self.page
        page.gcamp_choice = parity
        page.dirty = True
        self.say(f"{page.key}: GCaMP = {parity.upper()} frames, reflectance = "
                 f"{'EVEN' if parity == 'odd' else 'ODD'} frames (channel_order "
                 f"{order_from_parity(parity)})" if parity else f"{page.key}: GCaMP choice cleared")

    def accept_rectangle(self) -> None:
        """Take the rectangle's decision as this page's choice."""
        decision, note = self.decision()
        if decision is None:
            self.say(f"{self.page.key}: the rectangle gives no decision ({note}); choose by hand.")
            return
        self.choose_gcamp(decision.gcamp_parity)

    def decision(self, page: PageModel | None = None) -> tuple[RectangleDecision | None, str]:
        """(decision, '') or (None, why there is none) for a page.

        Only from the 256-grid means read off the raw frames: a decision from the cached
        128 images disagreed with the 256-grid one on 27 of 496 recordings (spec 5.3), so
        a cached page has none.
        """
        page = self.page if page is None else page
        if page.odd_even is None:
            reason = {"cache": "cached 128 images only (raw frames not found)",
                      "legacy": "only the old single preview is cached",
                      "none": "no image read yet"}.get(page.image_source, "no raw frames")
            return None, reason
        params = self.page_params(page)
        odd, even = page.odd_even
        bregma, lambda_ = 2 * bregma_128(page), 2 * lambda_128(page)
        problem = decision_problem(odd.shape, params_corners(bregma, lambda_, params), params.statistic)
        if problem:
            return None, problem
        return decide(odd, even, bregma, lambda_, params), ""

    # ----- saving --------------------------------------------------------------------
    def save_page(self, page: PageModel, path: Path, name: str, own_set: bool = True) -> Path:
        """Write one ROI set (+ its rectangle when moved). Raises when a box is off frame."""
        shape = self.shape(page)
        bad = out_of_frame(page, shape)
        if bad:
            raise OutOfFrameError(f"{page.key} NOT saved -- {len(bad)} box(es) off the frame: {bad}. "
                                  f"Drag them back or move Bregma.")
        # The library's own validation on the geometry as saved.
        box_slices_for(ROIConfig(y_1=page.y_1, x_2=page.x_2, boxes=dict(page.boxes)), shape)
        source = (f"drawn with roi_editor.py on the {self.profile_name} preview grid; "
                  f"folder {page.folder}")
        written = save_roi_set(path, page.boxes, page.bregma_row, page.bregma_col, grid=shape,
                               name=name, source=source, lambda_offset=page.lambda_offset,
                               downsample=get_profile(self.profile_name).downsample,
                               lambda_above_bregma=page.lambda_sign < 0,
                               lambda_angle_deg=page.lambda_angle)
        if page.rectangle is not None:
            append_roi_set_rectangle(written, page.rectangle)
        if page.gcamp_choice is not None:
            append_roi_set_channel_order(written, order_from_parity(page.gcamp_choice))
        page.saved_to = str(written)
        if own_set:
            page.dirty = False
            page.has_own_set = True
        self.say(f"Saved {name} -> {written}")
        return written

    def own_path(self, page: PageModel) -> Path:
        return Path(self.args.roi_set_dir) / f"{page.key}.yaml"

    def save_current(self) -> None:
        """s"""
        page = self.page
        self.save_page(page, self.own_path(page), page.key)

    def save_many(self, pages: list[PageModel], what: str) -> None:
        """Every page that fits is written; the refused ones are named in one error."""
        refused = [page.key for page in pages if self.out_of_frame(page)]
        for page in pages:
            if page.key not in refused:
                self.save_page(page, self.own_path(page), page.key)
        self.say(f"Saved {len(pages) - len(refused)}/{len(pages)} {what} ROI sets to "
                 f"{self.args.roi_set_dir}")
        if refused:
            raise OutOfFrameError(f"{len(refused)} page(s) NOT saved, boxes off the frame: {refused}")

    def save_all(self) -> None:
        """S"""
        self.save_many(self.pages, "session")

    def save_modified(self) -> None:
        """ctrl+s: only the pages with unsaved edits."""
        pending = self.dirty_pages()
        if not pending:
            self.say("Nothing to save -- no session has unsaved changes.")
            return
        self.save_many(pending, "modified")

    def write_shared(self) -> None:
        """w: this page's layout as the one shared ROI set."""
        path = Path(self.args.roi_set_dir) / self.args.shared_name
        self.save_page(self.page, path, Path(self.args.shared_name).stem, own_set=False)

    def _latest_saved_set(self) -> Path | None:
        directory = Path(self.args.roi_set_dir)
        if not directory.is_dir():
            return None
        candidates = [p for p in directory.glob("*.yaml") if p.is_file()]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def resume(self) -> None:
        """Reload this page from its saved set (own, else the seed folder's), else from
        the most recently saved set of any page (ROIEditor._resume_session)."""
        page = self.page
        own_path = self.own_path(page)
        seed_dir = getattr(self.args, "roi_seed_dir", None)
        seed_path = Path(seed_dir) / f"{page.key}.yaml" if seed_dir else None
        path = own_path if own_path.is_file() else None
        if path is None and seed_path is not None and seed_path.is_file():
            path = seed_path
        own = path is not None
        if path is None:
            path = self._latest_saved_set()
        if path is None:
            self.say(f"Nothing to resume from: {page.key} has no saved ROI set, and "
                     f"{self.args.roi_set_dir} does not hold a single one yet.")
            return
        atlas, file_row, file_col = load_roi_set(path)
        lam = load_lambda_offset(path)
        if file_row is not None:
            page.y_1, page.x_2 = file_row // 2, file_col // 2
        page.boxes = dict(atlas)
        if lam is not None:
            page.lambda_offset = lam
        page.lambda_sign = -1 if load_lambda_above(path) else 1
        page.lambda_angle = load_lambda_angle(path)
        page.rectangle = read_roi_set_rectangle(path)
        if own:
            # A choice belongs to its own recording: another page's file brings none.
            order = read_roi_set_channel_order(path)
            page.gcamp_choice = parity_of_order(order) if order else None
        page.rebase()
        if own:
            page.seed = (page.y_1, page.x_2, page.lambda_offset, dict(page.boxes))
            page.seed_angle = page.lambda_angle
            page.seed_rectangle = page.rectangle
            page.seed_gcamp_choice = page.gcamp_choice
            page.dirty = False
            page.has_own_set = True
            self.say(f"{page.key}: resumed from {path}")
        else:
            page.dirty = True
            saved_at = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %H:%M")
            self.say(f"{page.key} has no ROI set of its own -- resumed from the most recently saved "
                     f"one, {path.name} (saved {saved_at}). Check it against THIS anatomy, then "
                     f"press 's' to save it as {page.key}.yaml.")
        self.selected = None
