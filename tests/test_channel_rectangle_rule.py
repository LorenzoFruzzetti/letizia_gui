"""channel_rectangle.py (shared rule) and its uses in roi_editor.py, qc_cohort.py and the GUI."""

from __future__ import annotations

import json

import numpy as np
import tifffile

import channel_rectangle as rectangle


def test_params_or_default_falls_back_and_reads_relative_paths(tmp_path):
    assert rectangle.params_or_default(tmp_path / "missing.json") == rectangle.RectangleParams()
    path = tmp_path / "p.json"
    path.write_text(json.dumps(dict(length_factor=1.0, width_factor=0.2, start_offset_factor=0.0,
                                    statistic="median_level")), encoding="utf-8")
    assert rectangle.params_or_default(path).statistic == "median_level"


def _sinus_images():
    """odd = smooth (GCaMP-like), even = a dark vertical sinus along col 128 (reflectance)."""
    odd = np.full((256, 256), 1000.0)
    even = np.full((256, 256), 3000.0)
    even[:, 124:132] = 1500.0
    return odd, even


def test_decide_calls_the_vessel_contrast_image_reflectance():
    odd, even = _sinus_images()
    bregma = rectangle.bregma_point(120, 128)                 # col 127 on the 256 grid
    lambda_ = rectangle.lambda_point(bregma, 50)
    params = rectangle.RectangleParams(length_factor=1.0, width_factor=0.3, start_offset_factor=0.0,
                                       statistic="centre_flank")
    decision = rectangle.decide(odd, even, bregma, lambda_, params)
    assert decision.gcamp_parity == "odd" and decision.channel_order == "gcamp_first"
    assert decision.level_parity == "odd" and decision.agrees_with_level
    swapped = rectangle.decide(even, odd, bregma, lambda_, params)
    assert swapped.channel_order == "emo_first"


def test_decision_problem_instead_of_an_exception():
    corners_off = rectangle.rectangle_corners(np.array([-500.0, -500.0]), np.array([-400.0, -500.0]),
                                              1.0, 0.2, 0.0)
    assert "no pixel" in rectangle.decision_problem((256, 256), corners_off, "centre_flank")
    # 1.2 px wide: it holds one column of pixel centres, so a third of the width is empty.
    narrow = rectangle.rectangle_corners(np.array([100.0, 100.0]), np.array([200.0, 100.0]), 1.0, 0.012, 0.0)
    assert "too narrow" in rectangle.decision_problem((256, 256), narrow, "centre_flank")


def test_editor_odd_even_means_are_the_raw_parities(tmp_path):
    from roi_editor import odd_even_means
    for i in range(1, 9):
        level = 100 if i % 2 else 900          # odd files dim, even files bright
        tifffile.imwrite(tmp_path / f"f_{i:03d}.tif", np.full((16, 16), level, np.uint16))
    odd, even = odd_even_means(tmp_path, n_frames=3)
    assert odd.shape == (8, 8) and np.allclose(odd, 100) and np.allclose(even, 900)


def test_qc_rectangle_columns_from_a_labelled_dump(tmp_path):
    import yaml
    from qc_cohort import rectangle_columns
    odd, even = _sinus_images()
    # Labelled emo_first: the even (reflectance-looking) image is called GCaMP -> disagrees.
    np.savez(tmp_path / "pixels_meta_full.npz", channel_order=np.array("emo_first"),
             mean_f=even, mean_r=odd)
    roi_set = tmp_path / "set.yaml"
    roi_set.write_text(yaml.safe_dump(dict(bregma_row=120, bregma_col=128, lambda_row_offset=50)),
                       encoding="utf-8")
    params = rectangle.RectangleParams(length_factor=1.0, width_factor=0.3, start_offset_factor=0.0)
    columns = rectangle_columns(tmp_path, roi_set, params)
    assert columns["rect_channel_order"] == "gcamp_first" and columns["rect_agrees"] is False


def test_gui_viewer_arguments_use_shared_recordings(tmp_path):
    from gui_cohort_tree import make_cohort
    import sys
    import scan_cohort
    from wfci_gui.logic.cohort import load_cohort, load_tables
    from wfci_gui.views.pages import rectangle_viewer_arguments
    path = make_cohort(tmp_path)
    argv, sys.argv = sys.argv, ["scan_cohort.py", "--cohort", str(path), "--no-make-links"]
    scan_cohort.main()
    sys.argv = argv
    cohort = load_cohort(path)
    tables = load_tables(cohort)
    assert rectangle_viewer_arguments(cohort, tables, "200101")[0] == []
    for unit, recordings in (("200101_M1", ("t1", "t2")), ("200101_M2", ("t1",))):
        for recording in recordings:
            folder = cohort.layout.pixel_data / unit / recording
            folder.mkdir(parents=True)
            (folder / "pixels_meta_full.npz").write_bytes(b"x")
    arguments, note = rectangle_viewer_arguments(cohort, tables, "200101")
    assert arguments[arguments.index("--units") + 1:arguments.index("--recordings")] == ["M1", "M2"]
    assert arguments[arguments.index("--recordings") + 1] == "t1"
    assert arguments[arguments.index("--assignment") + 1] == "none"
    assert arguments[arguments.index("--cohort") + 1] == str(path)
    assert "--pixel-root" not in arguments and "--params" not in arguments
    assert "M1/t2" in note


def test_rectangle_parameters_are_the_cohorts(tmp_path):
    """M17: the global rectangle parameters live in the cohort folder; missing = defaults."""
    from gui_cohort_tree import make_cohort
    import qc_cohort
    path = make_cohort(tmp_path)
    params_path = rectangle.cohort_params_path(path)
    assert params_path == path.parent / "channel_rectangle_params.json"
    assert rectangle.params_or_default(params_path) == rectangle.RectangleParams()
    saved = rectangle.RectangleParams(length_factor=1.0, width_factor=0.2, start_offset_factor=0.0,
                                      statistic="median_level")
    rectangle.save_params(params_path, saved)
    assert rectangle.params_or_default(params_path) == saved
    assert "channel_rectangle_params" not in qc_cohort.RUN_CONFIG     # no repository-relative default
