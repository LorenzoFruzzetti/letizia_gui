"""roi_editor_spec.md 7.2: the per-recording channel rectangle (pure parts, no Qt)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

import channel_rectangle as cr  # noqa: E402
from roi_editor import DEFAULT_LAMBDA_OFFSET, save_roi_set  # noqa: E402
from wfci.atlases import CORTEX_22  # noqa: E402
from wfci_gui.roi_editor import logic  # noqa: E402
from wfci_gui.roi_editor.model import EditorModel, PageModel, bregma_128, lambda_128  # noqa: E402

GRID = (128, 128)
FACTORS = cr.RectangleFactors(0.7444891064863796, 0.1468939228518398, 1.1166427822870573)


def make_model(tmp_path, n_pages=2, raw=True):
    rng = np.random.default_rng(1)
    pages = []
    for i in range(n_pages):
        page = PageModel(key=f"240101_M1_t{i + 1}", folder="(synthetic)", y_1=60, x_2=67,
                         boxes=dict(CORTEX_22.boxes), lambda_offset=DEFAULT_LAMBDA_OFFSET,
                         image=rng.random(GRID))
        if raw:
            # A bright midline strip in the even frames: the rule calls it reflectance.
            odd, even = rng.random((256, 256)) + 5.0, rng.random((256, 256)) + 5.0
            even[:, 128:136] += 3.0
            page.odd_even, page.image_source = (odd, even), "raw"
        pages.append(page)
    args = types.SimpleNamespace(roi_set_dir=str(tmp_path / "reviewed"), roi_seed_dir=None,
                                 shared_name="shared.yaml", lambda_scales_box_size=False,
                                 channel_rectangle_params=None)
    return EditorModel(pages, args, "cortical_gsr")


# ----------------------------------------------------------------------------- the key
def test_factors_are_validated_against_the_parameter_limits():
    assert cr.make_factors({"length_factor": 1, "width_factor": 0.2, "start_offset_factor": 0}) == \
        cr.RectangleFactors(1.0, 0.2, 0.0)
    with pytest.raises(ValueError, match="exactly"):
        cr.make_factors({"length_factor": 1, "width_factor": 0.2})
    with pytest.raises(ValueError, match="outside"):
        cr.make_factors({"length_factor": 9, "width_factor": 0.2, "start_offset_factor": 0})
    with pytest.raises(ValueError, match="number"):
        cr.make_factors({"length_factor": True, "width_factor": 0.2, "start_offset_factor": 0})


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_append_keeps_the_file_and_its_line_endings(tmp_path, newline):
    path = tmp_path / "set.yaml"
    path.write_bytes(b"name: x\nboxes: {}\n".replace(b"\n", newline))
    before = path.read_bytes()
    cr.append_roi_set_rectangle(path, FACTORS)
    after = path.read_bytes()
    assert after.startswith(before) and after.endswith(newline)
    assert after.count(b"\r\n") == (3 if newline == b"\r\n" else 0) and b"\r\r\n" not in after
    assert cr.read_roi_set_rectangle(path) == FACTORS            # full precision round trip
    with pytest.raises(ValueError, match="already has"):
        cr.append_roi_set_rectangle(path, FACTORS)


def test_tiny_and_exponent_values_still_read_back_as_floats(tmp_path):
    path = tmp_path / "set.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    factors = cr.RectangleFactors(0.05, 0.01, 3.4e-17)
    cr.append_roi_set_rectangle(path, factors)
    assert cr.read_roi_set_rectangle(path) == factors


def test_rectangle_params_for_names_its_source(tmp_path):
    global_params = cr.RectangleParams(statistic="median_level")
    path = tmp_path / "set.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    assert cr.rectangle_params_for(path, global_params) == (global_params, "params")
    cr.append_roi_set_rectangle(path, FACTORS)
    params, source = cr.rectangle_params_for(path, global_params)
    assert source == "roi_set" and params.statistic == "median_level"
    assert cr.factors_of(params) == FACTORS
    path.write_text("name: x\nchannel_rectangle: {length_factor: 0.5, width_factor: 9.0, "
                    "start_offset_factor: 0.0}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="width_factor"):
        cr.rectangle_params_for(path, global_params)
    path.write_text("name: x\nchannel_rectangle: {length_factor: 0.5}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        cr.rectangle_params_for(path, global_params)


def test_agreement_helpers_moved_here_keep_their_rules():
    assert cr.gcamp_parity_from_order("auto") == cr.UNASSIGNED
    assert cr.agreement_marks({"manifest": "odd", "rectangle": "even"}) == {"manifest": "⚠", "rectangle": "⚠"}
    assert cr.agreement_summary({"manifest": cr.UNASSIGNED, "rectangle": "odd"}) == "?"


# ----------------------------------------------------------------------------- the model
def test_move_rectangle_projects_onto_the_axis_and_clips(tmp_path):
    model = make_model(tmp_path)
    page = model.page
    bregma, lambda_ = bregma_128(page), lambda_128(page)
    d, u, n = cr.axis_frame(bregma, lambda_)
    state = cr.roi_state(bregma, lambda_, FACTORS)
    corner = np.array([state["pos"][1], state["pos"][0]])
    # Dragged sideways (across the axis) by 4 px: only the along-axis part is kept.
    model.move_rectangle(corner + 4 * n, *state["size"])
    assert [getattr(page.rectangle, name) for name in cr.FACTOR_NAMES] == \
        pytest.approx([getattr(FACTORS, name) for name in cr.FACTOR_NAMES])
    assert page.dirty
    # Along the axis by 0.1 d: only the start moves.
    model.move_rectangle(corner + 0.1 * d * u, state["size"][0], state["size"][1])
    assert page.rectangle.start_offset_factor == pytest.approx(FACTORS.start_offset_factor + 0.1)
    model.move_rectangle(corner, 100 * d, 0.0001 * d)
    assert page.rectangle.length_factor == 5.0 and page.rectangle.width_factor == 0.01


def test_the_decision_is_channel_rectangle_decide_with_the_page_factors(tmp_path):
    model = make_model(tmp_path)
    page = model.page
    model.move_rectangle(*_corner_and_size(page, cr.RectangleFactors(0.6, 0.3, 1.2)))
    decision, note = model.decision()
    expected = cr.decide(*page.odd_even, 2 * bregma_128(page), 2 * lambda_128(page),
                         cr.params_with(model.rectangle_params, page.rectangle))
    assert note == "" and decision == expected
    # Without raw frames there is no decision (spec 5.3).
    page.odd_even, page.image_source = None, "cache"
    assert model.decision() == (None, "cached 128 images only (raw frames not found)")


def _corner_and_size(page, factors):
    state = cr.roi_state(bregma_128(page), lambda_128(page), factors)
    return np.array([state["pos"][1], state["pos"][0]]), *state["size"]


def test_save_round_trip_and_an_unmoved_page_writes_no_key(tmp_path):
    model = make_model(tmp_path)
    model.move_rectangle(*_corner_and_size(model.page, FACTORS))
    moved = model.page.rectangle
    model.save_all()
    first, second = (Path(model.args.roi_set_dir) / f"240101_M1_t{i}.yaml" for i in (1, 2))
    assert cr.read_roi_set_rectangle(first) == moved
    assert "channel_rectangle" not in yaml.safe_load(second.read_text(encoding="utf-8"))
    # The rest of the file is exactly what save_roi_set alone writes.
    reference = tmp_path / "reference.yaml"
    page = model.page
    save_roi_set(reference, page.boxes, page.bregma_row, page.bregma_col, grid=GRID, name=page.key,
                 source=f"drawn with roi_editor.py on the cortical_gsr preview grid; folder {page.folder}",
                 lambda_offset=page.lambda_offset, downsample=0.5)
    assert first.read_bytes().startswith(reference.read_bytes())
    # Resume reads it back, exactly.
    model.reset_rectangle()
    model.resume()
    assert model.page.rectangle == moved and not model.page.dirty


def test_moved_back_onto_params_still_writes_the_key_and_reset_clears_it(tmp_path):
    model = make_model(tmp_path)
    model.move_rectangle(*_corner_and_size(model.page, cr.factors_of(model.rectangle_params)))
    model.save_current()
    path = Path(model.args.roi_set_dir) / "240101_M1_t1.yaml"
    assert cr.read_roi_set_rectangle(path) is not None
    model.reset_rectangle()
    model.save_current()
    assert cr.read_roi_set_rectangle(path) is None


def test_a_rectangle_move_is_an_undoable_edit_carried_by_copy_and_apply(tmp_path):
    model = make_model(tmp_path, n_pages=3)
    before = logic.capture(model.pages)
    model.move_rectangle(*_corner_and_size(model.page, FACTORS))
    assert model.commit("Move rectangle")
    model.undo()
    assert logic.capture(model.pages) == before and model.page.rectangle is None
    model.redo()
    moved = model.page.rectangle
    model.go_to(1)
    model.copy_from_previous()
    assert model.page.rectangle == moved
    model.go_to(0)
    model.apply_to_all(with_bregma=False)
    assert all(page.rectangle == moved for page in model.pages)
    # Reset goes back to what the page opened with (no rectangle of its own).
    model.go_to(2)
    model.reset_page()
    assert model.page.rectangle is None and not model.page.dirty


# ----------------------------------------------------------------------------- GCaMP choice
def test_the_gcamp_choice_is_saved_resumed_reset_and_never_carried(tmp_path):
    model = make_model(tmp_path, n_pages=3)
    model.choose_gcamp("even")
    assert model.page.dirty and model.commit("choice")
    model.save_current()
    path = Path(model.args.roi_set_dir) / "240101_M1_t1.yaml"
    assert cr.read_roi_set_channel_order(path) == "emo_first"
    assert path.read_bytes().endswith(b"channel_order: emo_first" + (b"\r\n" if b"\r\n" in path.read_bytes() else b"\n"))
    # Copy / apply / navigate do not carry a recording's own channel call.
    model.apply_to_all(with_bregma=True)
    model.navigate(1)
    model.copy_from_previous()
    assert [page.gcamp_choice for page in model.pages] == ["even", None, None]
    # Resume reads it back; reset returns to what the page opened with.
    model.go_to(0)
    model.choose_gcamp(None)
    model.resume()
    assert model.page.gcamp_choice == "even" and not model.page.dirty
    model.choose_gcamp("odd")
    model.reset_page()
    assert model.page.gcamp_choice == "even"
    # The rectangle's decision can be taken as the choice.
    model.accept_rectangle()
    assert model.page.gcamp_choice == model.decision()[0].gcamp_parity == "odd"
    with pytest.raises(ValueError, match="parity"):
        model.choose_gcamp("gcamp")


def test_both_appended_keys_together_and_bad_values(tmp_path):
    path = tmp_path / "set.yaml"
    path.write_bytes(b"name: x\r\n")
    cr.append_roi_set_rectangle(path, FACTORS)
    cr.append_roi_set_channel_order(path, "gcamp_first")
    assert cr.read_roi_set_rectangle(path) == FACTORS and cr.read_roi_set_channel_order(path) == "gcamp_first"
    assert b"\r\r\n" not in path.read_bytes() and path.read_bytes().count(b"\n") == 3
    with pytest.raises(ValueError, match="already has"):
        cr.append_roi_set_channel_order(path, "emo_first")
    path.write_text("name: x\nchannel_order: auto\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be one of"):
        cr.read_roi_set_channel_order(path)
