"""Per-recording helpers shared by the extraction script and the ROI editor.

Holds only functions, no run configuration and no dataset tables:
    * ``apply_roi_set``        -- swap a ROI-set file's boxes and Bregma into a profile;
    * ``_load_limited_trial``  -- load the first N frames per channel (debug mode);
    * ``discover_recordings``  -- find every interleaved recording folder below a root;
    * ``_save_overlay``        -- ROI-placement overlay figure (debug mode);
    * ``_save_traces``         -- trial-averaged ROI trace figure.

Importing this module is inert: it has no ``__main__`` block.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np

from wfci import folder_frame_source

# The ROI-set file format (boxes + the Bregma they were drawn from) is owned by
# the editor utility; importing it here is inert (its run is __main__-guarded).
from roi_editor import load_roi_set


def apply_roi_set(profile, bregma_row: int, bregma_col: int, roi_set: str | None):
    """Swap in the geometry from a ROI-set file. Returns ``(profile, row, col)``.

    ``roi_set`` is a file written by ``roi_editor.py``: the ROI boxes plus the
    Bregma they were drawn from. The boxes replace the profile's atlas (via
    ``dataclasses.replace``, so the atlas's declared ``grid`` still guards against
    a layout drawn for another field of view), and the file's Bregma replaces the
    caller's -- the two are one measurement, and using boxes from one animal with
    the Bregma of another would move every ROI while still producing numbers.
    ``None`` leaves everything as it was.
    """
    if not roi_set:
        return profile, bregma_row, bregma_col

    atlas, file_row, file_col = load_roi_set(roi_set)
    profile = replace(profile, atlas=atlas)
    if file_row is not None:
        bregma_row, bregma_col = file_row, file_col
    print(f"ROI set: {roi_set} -> {len(atlas)} ROIs {list(atlas)}, grid {atlas.grid}, "
          f"Bregma row={bregma_row} col={bregma_col}")
    return profile, bregma_row, bregma_col


def _load_limited_trial(gcamp_files, emo_files, limit, rot90_k: int = 0):
    """Load only the first ``limit`` frames per channel into RAM (debug mode).

    Goes through the lazy folder sources and ``islice`` so exactly ``limit`` images
    per channel are decoded off the (network) disk -- not the whole 6000-file
    folder -- then stacks them to the ``[y, x, time]`` array the in-memory pipeline
    expects. Returns one ``(gcamp, emo)`` trial. ``rot90_k`` rotates each frame
    (see ``wfci.io.iter_folder_frames``).
    """
    gcamp_src = folder_frame_source(gcamp_files, rot90_k=rot90_k)
    emo_src = folder_frame_source(emo_files, rot90_k=rot90_k)
    gcamp = np.stack(list(itertools.islice(gcamp_src.open(), limit)), axis=-1)
    emo = np.stack(list(itertools.islice(emo_src.open(), limit)), axis=-1)
    return gcamp, emo


def discover_recordings(root: str | Path, pattern: str) -> list[Path]:
    """Find every interleaved recording folder at or below ``root``, sorted.

    A folder counts as a recording when it directly holds at least 2 images
    matching ``pattern`` -- the minimum an interleaved split needs. That single
    rule handles all three input levels of this dataset without hard-coding the
    ``day\\animal\\t#`` depth: a recording folder returns itself, an animal folder
    returns its t1..tn, a day folder returns every animal's recordings. Folders
    holding images are never descended into, so a stray subfolder inside a
    recording cannot turn into a second one.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    if len(sorted(root.glob(pattern))) >= 2:
        return [root]
    found: list[Path] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        found.extend(discover_recordings(sub, pattern))
    return found


def _save_overlay(dff_stack, cfg, profile, out_path):
    """Draw the step-2 ROI-placement overlay on a representative DFF frame.

    This is the Bregma sanity check: the 4 boxes are drawn from the same ``cfg``
    the pipeline averages, so if they land off the anatomy the coordinates are
    wrong. Only available in the RAM (debug) mode, which keeps ``dff_stack``.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt

    from wfci import show_roi_placement

    # overlay_frame may exceed a short debug clip; clamp to a valid frame index.
    frame_index = min(profile.overlay_frame, dff_stack.shape[2] - 1)
    ax = show_roi_placement(dff_stack, cfg, frame_index=frame_index, average_trials=False)
    ax.set_title(f"{profile.name}: ROI placement @ Bregma "
                 f"(frame {frame_index}) -- verify boxes sit on the anatomy")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def _save_traces(averaged_traces, labels, out_path):
    """Plot the trial-averaged ROI DFF traces -- a quick 'did signals come out' look."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for j, label in enumerate(labels):
        ax.plot(averaged_traces[:, j], label=label, lw=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel(r"$\Delta$F/F (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Trial-averaged ROI traces")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
