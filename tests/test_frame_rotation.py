"""rot90_k / pattern support (.tiff files, camera at 90 degrees)."""
from __future__ import annotations

import numpy as np
import tifffile

from roi_editor import preview_image
from wfci.io import folder_frame_source


def _write_interleaved(folder, n_pairs=3, shape=(12, 20)):
    """Interleaved .tiff folder: dim GCaMP frames with a bright left (anterior) edge."""
    rng = np.random.default_rng(0)
    for i in range(n_pairs):
        gcamp = rng.integers(100, 200, size=shape).astype(np.uint16)
        gcamp[:, :4] += 1000  # "anterior" = left columns of the raw frame
        emo = np.full(shape, 30000, dtype=np.uint16)
        tifffile.imwrite(folder / f"img_{2 * i:04d}.tiff", gcamp)
        tifffile.imwrite(folder / f"img_{2 * i + 1:04d}.tiff", emo)


def test_folder_frame_source_rotates_each_frame(tmp_path):
    _write_interleaved(tmp_path)
    files = sorted(tmp_path.glob("*.tiff"))
    plain = list(folder_frame_source(files).open())
    rotated = list(folder_frame_source(files, rot90_k=-1).open())
    for a, b in zip(plain, rotated):
        assert b.shape == (a.shape[1], a.shape[0])
        assert b.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(b, np.rot90(a, -1))


def test_preview_reads_tiff_and_puts_anterior_on_top(tmp_path):
    _write_interleaved(tmp_path, shape=(16, 24))
    plain = preview_image(tmp_path, pattern="*.tiff")
    rotated = preview_image(tmp_path, pattern="*.tiff", rot90_k=-1)
    assert plain.shape == (4, 6) and rotated.shape == (6, 4)
    np.testing.assert_allclose(rotated, np.rot90(plain, -1))
    # The bright left edge of the raw frame is now the top row.
    assert rotated[0].mean() > rotated[-1].mean()
