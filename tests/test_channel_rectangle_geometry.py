"""Midline rectangle geometry of scripts/channel_rectangle_viewer.py (no window opened)."""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The viewer imports Qt at module level; never open a display from the tests.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from channel_rectangle_viewer import (  # noqa: E402
    RectangleParams, agreement_marks, agreement_summary, axis_frame, bregma_point,
    gcamp_parity_from_order, gcamp_parity_from_statistic, lambda_point, manifest_order_from_verdict,
    load_params, rectangle_corners, rectangle_mask, rectangle_statistic, save_params, value_ratio,
)

BREGMA = np.array([121.0, 133.0])


def test_bregma_point_is_the_centre_of_the_pipelines_128_pixel():
    # e.g. bregma 122/134 -> y_1 61, x_2 67 -> 256 rows 120-121, cols 132-133
    assert np.array_equal(bregma_point(122, 134), [121.0, 133.0])
    assert np.array_equal(bregma_point(123, 135), [121.0, 133.0])


def test_vertical_axis_rectangle_runs_from_lambda_up_past_bregma():
    lam = lambda_point(BREGMA, 51)                       # 51 rows on the 128 grid = 102 here
    assert np.allclose(lam, [223.0, 133.0])
    d, u, _ = axis_frame(BREGMA, lam)
    assert d == pytest.approx(102.0) and np.allclose(u, [-1, 0])
    corners = rectangle_corners(BREGMA, lam, 1.5, 0.2, 0.0)
    # start edge at Lambda, end edge 0.5 d above Bregma, width 0.2 d centred on the column
    assert np.allclose(corners[[0, 1], 0], 223.0) and np.allclose(corners[[2, 3], 0], 121.0 - 51.0)
    assert sorted(corners[:, 1]) == pytest.approx([133 - 10.2] * 2 + [133 + 10.2] * 2)


def test_lambda_above_bregma_flips_the_direction():
    lam = lambda_point(BREGMA, 51, lambda_above_bregma=True)
    assert np.allclose(lam, [19.0, 133.0])
    _, u, _ = axis_frame(BREGMA, lam)
    assert np.allclose(u, [1, 0])
    corners = rectangle_corners(BREGMA, lam, 1.5, 0.2, 0.0)
    assert np.allclose(corners[[2, 3], 0], 121.0 + 51.0)   # ends 0.5 d below Bregma


def test_rotated_axis_follows_roi_editor_convention():
    # (cos a, -sin a) in (row, col): a = 30 deg puts Lambda down and to the LEFT.
    lam = lambda_point(BREGMA, 50, angle_deg=30.0)
    assert np.allclose(lam - BREGMA, [100 * math.cos(math.radians(30)), -100 * math.sin(math.radians(30))])
    corners = rectangle_corners(BREGMA, lam, 1.0, 0.2, 0.0)
    centre_start = corners[[0, 1]].mean(axis=0)
    centre_end = corners[[2, 3]].mean(axis=0)
    assert np.allclose(centre_start, lam) and np.allclose(centre_end, BREGMA)
    # The width edge is perpendicular to the axis and 0.2 d long.
    width_edge = corners[1] - corners[0]
    assert width_edge @ (BREGMA - lam) == pytest.approx(0.0, abs=1e-9)
    assert np.hypot(*width_edge) == pytest.approx(20.0)


def test_factors_scale_with_d_and_offset_shifts_along_the_axis():
    small = rectangle_corners(BREGMA, lambda_point(BREGMA, 40), 1.5, 0.2, 0.25)
    large = rectangle_corners(BREGMA, lambda_point(BREGMA, 60), 1.5, 0.2, 0.25)
    # Same factors -> same shape relative to (L, d): every side scales by 60 / 40.
    for corners, d in ((small, 80.0), (large, 120.0)):
        assert np.hypot(*(corners[3] - corners[0])) == pytest.approx(1.5 * d)
        assert np.hypot(*(corners[1] - corners[0])) == pytest.approx(0.2 * d)
        assert corners[[0, 1], 0].mean() == pytest.approx(BREGMA[0] + d - 0.25 * d)


@pytest.mark.parametrize("angle", [0.0, 17.0, -40.0])
def test_mask_pixel_count_matches_the_area(angle):
    lam = lambda_point(BREGMA, 51, angle_deg=angle)
    corners = rectangle_corners(BREGMA, lam, 1.5, 0.2, 0.0)
    mask = rectangle_mask((256, 256), corners)
    area = 1.5 * 102 * 0.2 * 102
    assert mask.sum() == pytest.approx(area, rel=0.03)
    # Every masked pixel centre is within the rectangle's bounding box.
    rows, cols = np.nonzero(mask)
    assert rows.min() + 0.5 >= corners[:, 0].min() and rows.max() + 0.5 <= corners[:, 0].max()
    assert cols.min() + 0.5 >= corners[:, 1].min() and cols.max() + 0.5 <= corners[:, 1].max()


def test_statistics_point_the_right_way():
    lam = lambda_point(BREGMA, 51)
    corners = rectangle_corners(BREGMA, lam, 1.5, 0.3, 0.0)
    flat = np.full((256, 256), 100.0)
    vessel = flat.copy()
    vessel[:, 128:139] = 60.0   # dark stripe on the midline = vessel contrast
    assert rectangle_statistic(flat, corners, "centre_flank") == pytest.approx(0.0)
    assert rectangle_statistic(vessel, corners, "centre_flank") > 0.3
    # "vessel" is the reflectance-like image: the flat one is called GCaMP either way.
    for statistic in ("centre_flank", "spatial_cv"):
        odd = rectangle_statistic(flat, corners, statistic)
        even = rectangle_statistic(vessel, corners, statistic)
        assert gcamp_parity_from_statistic(odd, even, statistic) == "odd"
    assert gcamp_parity_from_statistic(5.0, 9.0, "median_level") == "odd"


def test_agreement_marks_skip_an_unlabelled_system():
    assert agreement_marks({"library": "odd", "manifest": "even", "rectangle": "odd"}) == {
        "library": "⚠", "manifest": "⚠", "rectangle": "⚠"}
    # A not-interleaved cache gives no parity: the other two are compared alone.
    labels = {"library": gcamp_parity_from_order("odd_even_unassigned"), "manifest": "odd", "rectangle": "odd"}
    assert agreement_marks(labels) == {"library": "–", "manifest": "✓", "rectangle": "✓"}
    assert agreement_summary(labels) == "?"
    assert agreement_summary({"library": "odd", "manifest": "odd", "rectangle": "odd"}) == "✓"


def test_value_ratio_is_higher_over_lower():
    assert value_ratio(2.0, 8.0) == value_ratio(8.0, 2.0) == pytest.approx(4.0)
    assert value_ratio(0.0, 0.3) == math.inf


def test_verdict_table_is_mapped_through_the_old_full_field_ratio():
    odd, even = np.full((4, 4), 9000.0), np.full((4, 4), 10000.0)
    # Old "f" was the odd group (ratio 0.9 matches odd/even): correct -> odd is GCaMP.
    row = pd.Series({"verdict": "correct", "f_over_r_full_field": "0.900"}, name="k")
    assert manifest_order_from_verdict(row, odd, even) == "gcamp_first"
    assert manifest_order_from_verdict(row.replace("correct", "swapped"), odd, even) == "emo_first"
    # Old "f" was the even group (ratio 1.111 = even/odd).
    row = pd.Series({"verdict": "correct", "f_over_r_full_field": "1.111"}, name="k")
    assert manifest_order_from_verdict(row, odd, even) == "emo_first"
    assert manifest_order_from_verdict(row.replace("correct", "inseparable"), odd, even) == ""
    with pytest.raises(ValueError, match="matches neither"):
        manifest_order_from_verdict(pd.Series({"verdict": "correct", "f_over_r_full_field": "0.5"}, name="k"), odd, even)


def test_params_round_trip(tmp_path):
    params = RectangleParams(length_factor=1.2, width_factor=0.15, start_offset_factor=-0.1,
                             statistic="spatial_cv")
    save_params(tmp_path / "params.json", params)
    assert load_params(tmp_path / "params.json") == params
