"""rebuild_roi_sets.py on the cases the 2024 cohort brings: underscored animal names
and layouts drawn with Lambda ABOVE Bregma (roi_editor 'u').

The rebuild regenerates every set from the cortex22 baseline, so anything the drawn
file says about orientation is lost unless the rebuild reads it -- these tests pin that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import rebuild_roi_sets  # noqa: E402
from roi_editor import (  # noqa: E402
    flip_rows_about_bregma,
    rotate_boxes,
    save_roi_set,
    scale_boxes,
)
from wfci import load_atlas  # noqa: E402

BASELINE = REPO_ROOT / "atlas" / "cortex22_roi_set.yaml"
REFERENCE_LAMBDA = 55


def _write_session(folder: Path, key: str, above: list[bool], lam: int = 48) -> None:
    """t1..tN of one session, each saved with its own Lambda side."""
    base = dict(load_atlas(BASELINE))
    for i_rec, is_above in enumerate(above, start=1):
        boxes = scale_boxes(base, lam / REFERENCE_LAMBDA)
        if is_above:
            boxes = flip_rows_about_bregma(boxes)
        save_roi_set(folder / f"{key}_t{i_rec}.yaml", boxes, 104, 134, (128, 128),
                     f"{key}_t{i_rec}", "test", lambda_offset=lam,
                     lambda_above_bregma=is_above)


def _run(monkeypatch, roi_dir: Path, out_dir: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["rebuild_roi_sets.py", "--roi-set-dir", str(roi_dir),
                                      "--out-dir", str(out_dir), "--baseline", str(BASELINE)])
    rebuild_roi_sets.main()


def test_key_pattern_groups_underscored_and_digitless_animals():
    for key, animal in [("240108_SP20_000_t1", "240108_SP20_000"),
                        ("261012_Nan_t3", "261012_Nan"),
                        ("260611_R1_t5", "260611_R1")]:
        assert rebuild_roi_sets.KEY_RE.fullmatch(key).group("animal") == animal


def test_rebuild_keeps_lambda_above_bregma_and_outvotes_a_lone_slip(tmp_path, monkeypatch):
    roi_dir, out_dir = tmp_path / "drawn", tmp_path / "rebuilt"
    # t3 was saved the normal way up by mistake; the other four are flipped.
    _write_session(roi_dir, "240108_SP20_000", [True, True, False, True, True])
    _run(monkeypatch, roi_dir, out_dir)

    expected = flip_rows_about_bregma(scale_boxes(dict(load_atlas(BASELINE)),
                                                  48 / REFERENCE_LAMBDA))
    for i_rec in range(1, 6):
        path = out_dir / f"240108_SP20_000_t{i_rec}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["lambda_above_bregma"] is True, f"t{i_rec} was un-flipped"
        assert data["lambda_row_offset"] == 48
        assert dict(load_atlas(path)) == expected


def test_rebuild_of_a_normal_session_writes_no_orientation_key(tmp_path, monkeypatch):
    roi_dir, out_dir = tmp_path / "drawn", tmp_path / "rebuilt"
    _write_session(roi_dir, "240108_SP21_RRR", [False] * 5)
    _run(monkeypatch, roi_dir, out_dir)
    for path in out_dir.glob("*.yaml"):
        assert "lambda_above_bregma" not in yaml.safe_load(path.read_text(encoding="utf-8"))


def test_rebuild_keeps_the_rotation_and_the_drawn_grid(tmp_path, monkeypatch):
    """A layout rotated in the editor (z/x, Lambda drag) on the 2022 cohort's 152x128
    grid must come back rotated and on that grid, not the baseline's 128x128."""
    roi_dir, out_dir = tmp_path / "drawn", tmp_path / "rebuilt"
    lam, angle = 48, -7.5
    scaled = scale_boxes(dict(load_atlas(BASELINE)), lam / REFERENCE_LAMBDA)
    for i_rec in range(1, 4):
        save_roi_set(roi_dir / f"221111_R10_L_t{i_rec}.yaml", rotate_boxes(scaled, angle),
                     150, 128, (152, 128), f"221111_R10_L_t{i_rec}", "test",
                     lambda_offset=lam, lambda_angle_deg=angle)
    _run(monkeypatch, roi_dir, out_dir)

    for i_rec in range(1, 4):
        path = out_dir / f"221111_R10_L_t{i_rec}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["lambda_angle_deg"] == angle
        assert data["grid"] == [152, 128]
        assert dict(load_atlas(path)) == rotate_boxes(scaled, angle)

def test_rebuild_carries_the_editor_keys_per_file_and_changes_nothing_else(tmp_path, monkeypatch):
    """roi_editor_spec.md D17 / 7.2: channel_rectangle (and channel_order) survive the rebuild
    on their own file only; every other file is byte-identical to a rebuild without them."""
    from channel_rectangle import (RectangleFactors, append_roi_set_channel_order, append_roi_set_rectangle,
                                   read_roi_set_channel_order, read_roi_set_rectangle)
    plain, keyed = tmp_path / "plain", tmp_path / "keyed"
    for folder in (plain, keyed):
        _write_session(folder, "240108_SP20_000", [False] * 5)
    factors = RectangleFactors(0.6123456789012345, 0.2, 1.05)
    append_roi_set_rectangle(keyed / "240108_SP20_000_t2.yaml", factors)
    append_roi_set_channel_order(keyed / "240108_SP20_000_t2.yaml", "emo_first")
    _run(monkeypatch, plain, tmp_path / "out_plain")
    _run(monkeypatch, keyed, tmp_path / "out_keyed")
    for i_rec in range(1, 6):
        name = f"240108_SP20_000_t{i_rec}.yaml"
        before, after = (tmp_path / "out_plain" / name).read_bytes(), (tmp_path / "out_keyed" / name).read_bytes()
        if i_rec == 2:
            assert after.startswith(before) and after != before
            assert read_roi_set_rectangle(tmp_path / "out_keyed" / name) == factors
            assert read_roi_set_channel_order(tmp_path / "out_keyed" / name) == "emo_first"
        else:
            assert after == before, name
