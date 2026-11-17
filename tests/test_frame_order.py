"""wfci.io frame_order: acquisition order for non-zero-padded frame file names.

Some acquisitions name frames ``..._1.tif`` ... ``..._6000.tif``. A name sort gives
1, 10, 100, 1000, 1001, ..., so the odd/even split mixes the two interleaved channels.
``frame_order="number"`` sorts by the trailing integer; ``"name"`` (the default) must
stay exactly the lexicographic sort for zero-padded names.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wfci.io import interleaved_channel_files, sorted_frame_files  # noqa: E402

# Enough frames that a name sort is out of order (10 sorts before 2).
N_FRAMES = 12
GCAMP_LEVEL, EMO_LEVEL = 100, 200


def _write_interleaved(folder: Path, name_of) -> None:
    """Frame i (1-based) is GCaMP when odd, reflectance when even; value encodes i."""
    folder.mkdir()
    for i in range(1, N_FRAMES + 1):
        level = GCAMP_LEVEL if i % 2 else EMO_LEVEL
        tifffile.imwrite(folder / name_of(i), np.full((4, 4), level + i, dtype=np.uint16))


def _frame_index(path: Path) -> int:
    return int(tifffile.imread(str(path))[0, 0]) % 100


def test_number_order_recovers_acquisition_order(tmp_path):
    folder = tmp_path / "t0"
    _write_interleaved(folder, lambda i: f"LEDb100_ExtEDGE16_{i}.tif")
    files = sorted_frame_files(folder, frame_order="number")
    assert [_frame_index(f) for f in files] == list(range(1, N_FRAMES + 1))
    # The name sort is what the option exists to avoid.
    by_name = sorted_frame_files(folder)
    assert [_frame_index(f) for f in by_name] != list(range(1, N_FRAMES + 1))


def test_number_order_splits_channels_cleanly(tmp_path):
    folder = tmp_path / "t0"
    _write_interleaved(folder, lambda i: f"x_{i}.tif")
    gcamp, emo = interleaved_channel_files(folder, frame_order="number")
    assert [_frame_index(f) for f in gcamp] == list(range(1, N_FRAMES + 1, 2))
    assert [_frame_index(f) for f in emo] == list(range(2, N_FRAMES + 1, 2))


def test_zero_padded_names_give_the_same_order_either_way(tmp_path):
    folder = tmp_path / "t1"
    _write_interleaved(folder, lambda i: f"img_{i:05d}.tif")
    assert sorted_frame_files(folder) == sorted_frame_files(folder, frame_order="number")


def test_number_order_rejects_unnumbered_and_repeated_files(tmp_path):
    folder = tmp_path / "t0"
    _write_interleaved(folder, lambda i: f"x_{i}.tif")
    tifffile.imwrite(folder / "x_007.tif", np.zeros((4, 4), dtype=np.uint16))  # repeats 7
    with pytest.raises(ValueError, match="repeated"):
        sorted_frame_files(folder, frame_order="number")
    (folder / "x_007.tif").unlink()
    tifffile.imwrite(folder / "notes.tif", np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match="no trailing frame number"):
        sorted_frame_files(folder, frame_order="number")


def test_unknown_frame_order_raises(tmp_path):
    with pytest.raises(ValueError, match="frame_order"):
        sorted_frame_files(tmp_path, frame_order="mtime")
