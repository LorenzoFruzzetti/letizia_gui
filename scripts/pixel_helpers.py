"""Pixel-dump helpers shared by the cache, connectivity and channel-trace scripts.

Holds only functions, no run configuration and no dataset tables:
    * ``median_window_frames`` -- odd frame count of a running-median window;
    * ``running_median``       -- centred running median down axis 0;
    * ``crop_slices``          -- an atlas box as slices into a saved pixel crop;
    * ``recording_roi_set``    -- the ROI set one pixel dump is analysed with;
    * ``roi_output_suffix``    -- output-folder suffix for overridden ROI sets.

Importing this module is inert: it has no ``__main__`` block.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def median_window_frames(window_s, sampling_rate_hz):
    """Odd frame count spanning `window_s`, the convention the detector also uses."""
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("baseline_median_window_s must be finite and positive")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    return 2 * round(window_s * sampling_rate_hz / 2) + 1


def running_median(traces, n_window):
    """Centred running median down axis 0 of a (n_time, n_pixels) array.

    `min_periods=1` so the partial windows at both ends still give a baseline,
    the same edge convention as the event detector's slow-trend removal.
    """
    return (pd.DataFrame(traces)
            .rolling(window=n_window, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float))


def crop_slices(box, y_1, x_2, region):
    """Atlas offsets are MATLAB 1-based inclusive; region is 0-based exclusive.

    The exploratory test_new_epilectic_pixels.py maps crop pixels using window_rel.
    Use the stored full-grid origin here, and include the atlas's -1 conversion so
    the boxes match wfci.roi.box_slices_for exactly.
    """
    r0, r1, c0, c1 = map(int, region)
    rs = slice(y_1 + box["row_start"] - 1 - r0, y_1 + box["row_end"] - r0)
    cs = slice(x_2 + box["col_start"] - 1 - c0, x_2 + box["col_end"] - c0)
    if not (0 <= rs.start < rs.stop <= r1-r0 and 0 <= cs.start < cs.stop <= c1-c0):
        raise ValueError(f"ROI {box} falls outside saved crop {region}: {rs}, {cs}")
    return rs, cs


def recording_roi_set(meta, roi_set_dir=None, *, base_dir):
    """The ROI set one pixel dump is analysed with: (path, parsed YAML).

    `roi_set_dir=None` is the set the dump was made with (`meta["roi_set"]`); a
    relative `meta["roi_set"]` is resolved against `base_dir`.
    Otherwise it is `<roi_set_dir>/<day>_<animal>_<t#>.yaml`, the per-recording
    file the ROI editor and rebuild_roi_sets.py write. A missing file raises instead of falling back
    to the dump's own set: a silent fallback would run a whole batch on the wrong
    geometry without anything saying so.
    The set must share the dump's grid, Bregma and downsample, because box offsets
    are only meaningful relative to the Bregma the pixels were cropped around.
    """
    if roi_set_dir is None:
        atlas_path = Path(meta["roi_set"])
        if not atlas_path.is_absolute():
            atlas_path = Path(base_dir) / atlas_path
    else:
        atlas_path = Path(roi_set_dir) / f"{meta['day']}_{meta['animal']}_{meta['recording']}.yaml"
        if not atlas_path.is_file():
            raise FileNotFoundError(f"No ROI set for {meta['day']}_{meta['animal']}/{meta['recording']} "
                                    f"in {roi_set_dir}: expected {atlas_path}")
    atlas = yaml.safe_load(atlas_path.read_text(encoding="utf-8"))
    if list(atlas["grid"]) != list(meta["grid"]):
        raise ValueError(f"Atlas and pixel grid differ: {atlas_path}")
    for key in ("bregma_row", "bregma_col", "downsample"):
        if atlas[key] != meta[key]:
            raise ValueError(f"Atlas {key} differs from saved metadata: {atlas_path}")
    return atlas_path, atlas


def roi_output_suffix(roi_set_dir):
    """Output-folder suffix for a run on overridden ROI sets ("" for the dumps' own).

    The per-recording file names do not depend on the boxes, so without it a run
    on hand-selected ROIs would overwrite the tables and figures of the default run.
    """
    return "" if roi_set_dir is None else f"_roi_{Path(roi_set_dir).name}"
