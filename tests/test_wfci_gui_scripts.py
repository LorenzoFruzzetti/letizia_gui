"""roi_preflight.py's check of one ROI set against the dump window and the grid."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_preflight_catches_a_box_outside_the_window():
    from roi_preflight import check_set
    template = yaml.safe_load((REPO_ROOT / "atlas" / "cortex22_roi_set.yaml").read_text(encoding="utf-8"))
    data = dict(template, bregma_row=120, bregma_col=134, grid=[128, 128])
    data["boxes"] = dict(data["boxes"], V1L=dict(row_start=45, row_end=50, col_start=-25, col_end=-20))
    problems = check_set(data, [-29, 47, -44, 43], [128, 128])
    assert any("V1L outside the dump window" in p for p in problems)
    near_edge = dict(template, bregma_row=10, bregma_col=134, grid=[128, 128])
    assert any("does not fit" in p for p in check_set(near_edge, [-29, 47, -44, 43], [128, 128]))
