"""Batch resting-state FC over the BOTOX_RESTANI manifest, per recording.

This is a *study script* (P2 boundary: dataset policy lives here, mechanism lives
in ``wfci``). It reads the manifest produced by ``scan_botox_dataset.py`` and, for
each (day, animal) row, runs a cerebellar resting-state connectivity analysis over
that animal's ``t1..tn`` interleaved recordings.

SEPARATE (default, ``merge_recordings=False``). Each ``t#`` subfolder is its own
analysis: its own channel split, its own ``trim``, baseline, ROI time-series and
correlation matrix, written to its own output subfolder. Nothing is pooled across
recordings, so per-recording results stay comparable and one bad block cannot
contaminate the others.

MERGE = CONCATENATION (``merge_recordings=True``), for when the consecutive ``t#``
recordings really are one continuous session split across files. Each ``t#`` folder
is split into its two interleaved channels, the first ``trim`` frames of each block
are dropped (the per-acquisition onset transient), and the trimmed blocks are
concatenated frame-by-frame into one long gcamp stream + one long emo stream. That
single stream runs as ONE trial (with the pipeline's own ``trim`` set to 0, since
we already trimmed per block), so there is one baseline, one ROI time-series and
one correlation over the whole concatenation. Note this is concatenation, not
trial-averaging: averaging the blocks' correlation matrices would be right only if
they were separate acquisitions.

Per (day, animal) it writes, under ``<output_root>\\<day>_<animal>\\`` -- and, in
the default separate mode, under a further ``<t#>\\`` subfolder per recording:
    * connectivity[_debug].npz  -- temp_roi, R, R_mean, averaged_traces, labels,
                                   the recording path(s), Bregma, profile.
    * roi_fluorescence_*.csv    -- the INTERMEDIATE per-ROI fluorescence: temp_roi
                                   as plain text, one row per frame, one dF/F
                                   column per ROI. Same numbers the correlations
                                   are computed from, readable without numpy.
    * roi_traces[_debug].png    -- trial-averaged ROI traces.
    * roi_overlay_debug.png     -- ROI placement (debug/in-memory mode only).
``save_data`` is ON in RUN_CONFIG, so it also writes, under
``<save_data_root>`` -- ``pixel_data\\`` inside this repo, gitignored -- the
PER-PIXEL volumes the ROI means are reduced from:
    * pixels_dff_*.npy          -- corrected dF/F per pixel, [time, y, x], float16.
    * pixels_f_gcamp_*.npy      -- raw GCaMP fluorescence per pixel, float32.
    * pixels_f_emo_*.npy        -- raw reflectance fluorescence per pixel, float32.
    * pixels_meta_*.npz         -- mean_f/mean_r baselines, the crop region, Bregma.
Frames are cropped to a Bregma-relative window (``save_data_window``), so every
recording's volume has the same shape AND the same anatomical meaning. ~197 MB per
recording, ~70 GB for the whole manifest; the run refuses to start if the
estimate will not fit the disk. Pass ``--no-save-data`` to skip them. The
volumes are written as the recording streams, so they cost no extra pass and no
extra time, and the ROI traces are unchanged.
One summary row per recording (or per unit when merging) is appended to
``<output_root>\\batch_summary.csv``, flushed as each job lands.

EXECUTION. The supported default is ``workers=1``: every recording runs in the
main process, avoiding the native Windows DLL failures observed only in spawned
workers. Serial logs are printed live. ``workers > 1`` remains available as an
experimental ``ProcessPoolExecutor`` path, but is not recommended on this machine.
Parallel worker logs are buffered and replayed when each job lands; the parent
prints a heartbeat every 15 seconds. Ctrl+C cancels queued jobs and terminates
active workers; completed ``.npz`` results remain resumable in either mode.

RESUME. ``skip_existing`` (default True) skips any job whose ``.npz`` is already
under ``output_root``, so an interrupted batch can be restarted without redoing
finished recordings; the summary CSV is appended to rather than overwritten.
Pass ``--no-skip-existing`` to force a full recompute.

Two modes, mirroring ``run_intermingle_rs.py``:
    * debug = True  -- fast smoke test. Loads only ``debug_max_frames`` frames per
      channel per recording into RAM and runs ``run_profile``, optionally capping
      the number of recordings used (``max_recordings``). Keeps ``dff_stack`` so it
      can draw the ROI-placement overlay. Use to validate setup.
    * debug = False -- the real run. Streams every frame of every recording in
      constant memory (``run_streaming_profile``, two passes per trial). A single
      recording is ~3 GB and a full unit ~15 GB across 5, so streaming is
      mandatory. No overlay.

ROI geometry: each unit uses its profile's atlas and the manifest's Bregma unless a
ROI set drawn with ``roi_editor.py`` is configured -- ``roi_set_dir`` for per-unit
files (``<day>_<animal>.yaml``) or ``roi_set`` for one file shared by every unit.

Row selection:
    * ``select`` -- list of "day/animal" keys to run (e.g. ["260611/R1"]).
      Empty list = every row in the manifest.
    * ``max_rows`` -- cap the number of rows actually run (after ``select``).

Run it -- RUN_CONFIG below is already set up for the reliable real run (every unit
in the manifest, full streaming, the per-recording ROI sets in roi_sets\\rebuilt,
serial execution, per-pixel dumps into pixel_data\\), so with no flags at all
this does the whole batch:

    conda run -n letizia python run_botox_batch.py                  # editor mode, uses RUN_CONFIG
    conda run -n letizia python run_botox_batch.py --workers 1      # explicit reliable default
    conda run -n letizia python run_botox_batch.py --workers 2      # experimental on Windows
    conda run -n letizia python run_botox_batch.py --select 260611/R1 260611/R2
    conda run -n letizia python run_botox_batch.py --debug          # fast smoke test
    conda run -n letizia python run_botox_batch.py --no-skip-existing   # recompute everything
    conda run -n letizia python run_botox_batch.py --no-save-data       # ROI means only, no pixel volumes
    conda run -n letizia python run_botox_batch.py --merge-recordings   # one concatenated trial per animal

Edit ``RUN_CONFIG`` and run -- no CLI flags needed. Errors are left to surface.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from typing import Any

import numpy as np

from wfci import (
    VOLUME_NAMES,
    ChannelDecisionConfig,
    PixelDump,
    ROIConfig,
    get_profile,
    hemodynamic_correction,
    imresize_box,
    prepare_interleaved_trial,
    run_profile,
    run_streaming_profile,
    write_odd_even_dump,
)
from wfci.io import (
    BUFFER_MODES,
    LEVEL_REGIONS,
    LEVEL_STATISTICS,
    channel_stats_from_stacks,
    sorted_frame_files,
    split_odd_even,
)
from wfci.streaming import channel_choice_meta, choose_block_channels, roi_union_level_mask

# Reuse the single-folder helpers instead of duplicating them -- run_intermingle_rs
# guards its own run behind ``if __name__ == "__main__"``, so importing is inert.
from roi_editor import load_roi_set
from run_intermingle_rs import (
    _load_limited_trial,
    _save_overlay,
    _save_traces,
    apply_roi_set,
)


# Keep one stable schema for fresh, resumed, and reconstructed summary rows.
# ``elapsed_s`` is blank when a row has to be recovered from an older .npz,
# because processing time was not stored inside those result files.
SUMMARY_FIELDS = [
    'day', 'animal', 'group', 'profile', 'bregma_row', 'bregma_col',
    'roi_set', 'mode', 'recording', 'n_recordings', 'mean_offdiag_R',
    'elapsed_s', 'npz_path',
    # channel_verdict: certain / uncertain / not_interleaved / explicit per block
    # (;-joined when merged). status: "done"; "needs_channel_decision" for a
    # not_interleaved recording stored as odd/even; "skipped" when one was not stored.
    # Blank on rows written before the single-read channel selection existed.
    'channel_verdict', 'status',
]

# Returned by _run_trial for a recording stored by position, with no analysis.
NEEDS_CHANNEL_DECISION = "needs_channel_decision"

# On-disk dtypes offered for the pixel dumps. Storage only -- the numeric path is
# float64 throughout regardless.
DUMP_DTYPES = ("float16", "float32", "float64")

# Full recordings can take minutes when many workers contend for the same storage.
# Periodic parent-side messages distinguish useful work from a frozen process.
POOL_HEARTBEAT_SECONDS = 15.0

# ---------------------------------------------------------------------------
# Edit this section to run without CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # Manifest written by scan_botox_dataset.py.
    "manifest": r"manifests\botox_restani_manifest.csv",
    # Root under which per-unit output folders are created. Kept separate from the
    # older outputs\botox_restani (shared cortex22 geometry) so the two runs do not
    # overwrite each other.
    "output_root": r"outputs\botox_restani_rebuilt",
    # Which units to run: "day/animal" keys. Empty list = all rows in the manifest.
    "select": [],
    # Cap on rows actually run after `select` (None = no cap). Handy for a quick test.
    "max_rows": None,
    # ROI geometry (boxes + Bregma) drawn with roi_editor.py. Two ways to use it:
    #   * roi_set_dir -- per-unit files, <roi_set_dir>\<day>_<animal>.yaml. Each
    #     animal gets its own ROIs/Bregma. A unit with no file falls back to
    #     roi_set, and if that is None too, to the profile atlas + the manifest's
    #     Bregma. This is the "check every session" workflow.
    #   * roi_set     -- ONE file used for every unit ("the same ROIs for all
    #     sessions"), e.g. r"roi_sets\shared_roi_set.yaml".
    # Set both to None to keep the previous behaviour exactly.
    # roi_set is the 22-box cortical layout (wfci.atlases.CORTEX_22 -- the MATLAB's
    # img_av(y_1+.., x_2+..) boxes), shared by every unit. Only the ATLAS is
    # replaced: the manifest's profile column stays cerebellar_rs, so the pipeline
    # chain (trim, no mask, no GSR) is unchanged -- just 22 ROIs instead of 4.
    # Each file's own Bregma wins over the manifest's, so once you have drawn
    # per-animal sets with roi_editor.py, set roi_set_dir = r"roi_sets" and those
    # take precedence over this shared file.
    # roi_sets\rebuilt holds ONE file PER RECORDING, <day>_<animal>_<t#>.yaml (355
    # of them, one for every recording in the manifest), regenerated by
    # rebuild_roi_sets.py as a pure mirror-exact rescale of the cortex22 baseline.
    # resolve_roi_set() picks the per-recording file first, so each t# is analysed
    # with the boxes and Bregma drawn for it. cortex22 stays as the fallback for a
    # recording with no file -- with the current directory that never triggers.
    "roi_set_dir": r"roi_sets\rebuilt",
    "roi_set": r"atlas\cortex22_roi_set.yaml",
    # How the animal's t1..tn recordings are treated:
    #   False (default) -- each t# subfolder is its own analysis, with its own
    #     trim/baseline/correlation and its own <day>_<animal>\<t#>\ output folder.
    #   True            -- all t# are concatenated into ONE continuous trial
    #     (see the module docstring); one output folder per animal.
    "merge_recordings": False,
    # Debug smoke test (in-memory, limited frames) vs full streaming run.
    "debug": False,
    # Debug only: frames PER CHANNEL PER RECORDING to load into RAM.
    "debug_max_frames": 60,
    # Cap how many t# recordings are used (None = all). Applies to BOTH modes and
    # both merge settings -- e.g. 2 = only t1+t2. The real analysis uses all 5.
    "max_recordings": None,
    # Worker processes. Each job (one t# recording, or one merged animal) is fully
    # independent -- its own files, its own geometry, its own output folder -- so
    # they run in parallel with no shared state.
    #   1 (default)    -- reliable serial execution in the main process.
    #   None or <= 0  -- auto-select 75% of logical cores (experimental here).
    #   N > 1         -- process pool with N workers (experimental on Windows;
    #                    spawned workers have shown intermittent native DLL crashes).
    "workers": 1,
    # Resume: skip any analysis whose .npz already exists under output_root. Lets an
    # interrupted run be restarted without redoing finished recordings. Set to False
    # to force everything to be recomputed from scratch.
    "skip_existing": True,
    # --- per-pixel dumps (save_data) ---------------------------------------
    # True (the setting for this run) additionally writes, per recording, the
    # pixel-wise volumes the ROI means are computed from -- the corrected dF/F and
    # the raw fluorescence of BOTH channels -- as .npy files written frame by frame
    # while the recording streams. Costs ~197 MB per recording (see
    # save_data_window), ~70 GB across the full 355-recording manifest; the run
    # refuses to start if that will not fit. False keeps the pipeline as it was:
    # only the 22 ROI means survive each frame.
    "save_data": True,
    # Where the volumes go. None puts them beside the .npz in the analysis output
    # folder. A path mirrors the same <day>_<animal>\<t#>\ tree under it, which is
    # what you want when the pixels are far larger than the rest of the outputs.
    # "pixel_data" is relative to the repo root, so the volumes land in THIS
    # project folder, separate from outputs\ -- and .gitignore excludes it, since
    # ~70 GB of .npy must never enter git history.
    "save_data_root": r"pixel_data",
    # Crop, as (row_start, row_end, col_start, col_end) offsets RELATIVE TO BREGMA
    # on the final 128x128 grid -- the same coordinate system the ROI boxes use.
    # (-29, 47, -44, 43) is the measured union of every box across all 356 files in
    # roi_sets\rebuilt: a 76x87 window, 40.4% of the frame, that fits inside the
    # grid for every one of them. Being Bregma-relative it has the same anatomical
    # meaning in every animal, so dumps stack directly for a group analysis --
    # which a full-frame dump does not, since the frames are only aligned to the
    # camera. Set to None to write the whole 128x128 frame (2.5x the disk).
    "save_data_window": (-29, 47, -44, 43),
    # On-disk dtypes. These are a STORAGE choice; the numeric path stays float64.
    # Measured on real recordings: float16 costs dF/F at most 0.004 percentage
    # points against a signal sd of 0.66-0.97 pp, while float32 is *exactly*
    # lossless for raw F (the twice-box-downsampled values are k/16). float16 is
    # NOT safe for raw F: F_emo was measured at 63194 against float16's 65504
    # ceiling, so a brighter session would overflow -- PixelDump raises rather
    # than writing inf if you try.
    "save_data_dff_dtype": "float16",
    "save_data_f_dtype": "float32",
    # --- raw-frame handling, for cohorts not shaped like the 2026 one -------
    # Raw image glob inside each t# folder. The 2022 Basler files end in .tiff.
    "pattern": "*.tif",
    # Quarter turns applied to every frame right after decoding (np.rot90 sign,
    # -1 = clockwise). MUST equal the rotation the ROI sets were drawn on
    # (batch_roi_select.py rot90_k), or every box lands on the wrong anatomy.
    # The 2022 camera sits at 90 degrees: -1. Every other cohort: 0.
    "rot90_k": 0,
    # How acquisition order is read off the raw file names (wfci.io.sorted_frame_files).
    # "name" = lexicographic, right for zero-padded indices (every cohort before 2023).
    # "number" = by the trailing integer: the July 2023 cohort's files are ..._1.tif,
    # ..._2.tif, ..., ..._10.tif, and a name sort would mix the two channels. MUST
    # equal the order the ROI previews were built with (batch_roi_select.py).
    "frame_order": "name",
    # Onset frames dropped per channel. None = the profile's own trim (20 frames,
    # 2 s at the 2026 cohort's 10 Hz per channel). A cohort at another rate sets
    # the same DURATION in frames, e.g. 50 at the 2022 cohort's 25 Hz.
    "trim_frames": None,
    # --- single-read channel selection (channel_selection_spec.md) ----------
    # Every t# is read from disk ONCE: both interleaved groups are buffered
    # unlabelled, "auto" decides which is GCaMP from every frame, then the usual
    # baseline + correction run on the buffer. Where the buffer lives:
    #   "auto" -- RAM if 1.5x the estimate fits in free RAM, else disk in
    #             buffer_dir if set and 1.1x fits, else "none".
    #   "ram" / "disk" -- forced. "none" -- no buffer, the TIFFs are read twice.
    "channel_buffer": "auto",
    # Folder for "disk" buffers (float32 .npy, ~1.6 GB per 2026 recording). Should
    # be an SSD; None disables the disk option of "auto".
    "buffer_dir": None,
    # Keep the unlabelled half-res odd/even volumes next to the pixel dump as
    # pixels_f_odd_<suffix>.npy / pixels_f_even_<suffix>.npy (trimmed, uncropped).
    "keep_channel_buffer": False,
    # Decision thresholds (wfci.io.decide_channel_order). A block alternates when
    # the lag-1 autocorrelation of its interleaved frame medians is <= alternation_max,
    # and its channels separate when |level difference| / scatter >= separation_min.
    "alternation_max": -0.9,
    "separation_min": 10.0,
    # What happens to a recording whose brightness does not alternate (verdict
    # not_interleaved), where no GCaMP/emo label is defensible.
    #   False -- store its frames by position as pixels_f_odd/even_full.npy, with
    #            status needs_channel_decision and no dF/F or analysis. Needs
    #            save_data; a merged run has no single odd/even pair and still skips.
    #   True  -- write nothing for it (status skipped).
    "skip_not_interleaved": False,
    # Per-frame level the order is decided on: "median" (check_channel_assignment.py's
    # statistic; matched channel_assignment_2024.csv 9/9) or "top_percent" (mean of
    # the brightest top_percent% pixels; 3/9 wrong, the old auto's errors).
    "level_statistic": "median",
    "top_percent": 10.0,
    # Pixels the per-frame levels are taken over: "frame" (whole frame; what the
    # dimmer-is-GCaMP rule was validated on) or "roi_union" (only under this
    # recording's own ROI boxes). See wfci.io.ChannelDecisionConfig.level_region.
    "channel_level_region": "frame",
    "prefer_cli_args": True,
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Minimal CLI overrides for the knobs worth flipping from the terminal."""
    p = argparse.ArgumentParser(description="Batch RS-FC over the BOTOX manifest.")
    p.add_argument("--manifest", default=defaults["manifest"], help="Manifest CSV.")
    p.add_argument("--output-root", default=defaults["output_root"])
    p.add_argument("--select", nargs="*", default=defaults["select"],
                   metavar="DAY/ANIMAL[/T#]",
                   help="Units to run; omit for all. DAY/ANIMAL/T# picks a one-recording row.")
    p.add_argument("--max-rows", type=int, default=defaults["max_rows"])
    p.add_argument("--roi-set-dir", default=defaults["roi_set_dir"], metavar="DIR",
                   help="Per-unit ROI sets: DIR\\<day>_<animal>.yaml (roi_editor.py).")
    p.add_argument("--roi-set", default=defaults["roi_set"], metavar="PATH",
                   help="One ROI set for every unit (fallback when --roi-set-dir has no file).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", dest="debug", action="store_false", default=defaults["debug"],
                      help="Full streaming run over every frame of every recording.")
    mode.add_argument("--debug", dest="debug", action="store_true",
                      help="Debug smoke test on the first --debug-frames frames.")
    p.add_argument("--debug-frames", type=int, default=defaults["debug_max_frames"],
                   metavar="N", help="Debug mode: frames per channel per recording.")
    p.add_argument("--max-recordings", type=int, default=defaults["max_recordings"],
                   metavar="K", help="Cap the recordings used (both modes); e.g. 2 = t1+t2.")
    p.add_argument("--merge-recordings", action=argparse.BooleanOptionalAction,
                   default=defaults["merge_recordings"],
                   help="Concatenate an animal's t# recordings into ONE trial. "
                        "Default (--no-merge-recordings): analyse each t# separately.")
    p.add_argument("--workers", type=int, default=defaults["workers"], metavar="N",
                   help="Execution workers (default: 1, reliable serial path; "
                        "values >1 use the experimental process pool).")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction,
                   default=defaults["skip_existing"],
                   help="Skip analyses whose .npz already exists (resume). "
                        "--no-skip-existing recomputes everything.")
    p.add_argument("--save-data", action=argparse.BooleanOptionalAction,
                   default=defaults["save_data"],
                   help="Also write the per-pixel dF/F and raw-F volumes "
                        "(~197 MB per recording).")
    p.add_argument("--save-data-root", default=defaults["save_data_root"],
                   metavar="DIR",
                   help="Root for the per-pixel volumes. Default: beside the .npz. "
                        r"The <day>_<animal>\<t#> tree is mirrored under it, so this "
                        "is how you put ~70 GB of pixels on a different disk from "
                        "the analysis outputs.")
    p.add_argument("--save-data-dff-dtype", default=defaults["save_data_dff_dtype"],
                   choices=DUMP_DTYPES, metavar="DTYPE",
                   help=f"On-disk dtype for the dF/F volume {DUMP_DTYPES}. "
                        f"Default float16: measured cost <=0.004 percentage points "
                        f"against a 0.7-2.1 pp signal, and it halves the file.")
    p.add_argument("--save-data-f-dtype", default=defaults["save_data_f_dtype"],
                   choices=DUMP_DTYPES, metavar="DTYPE",
                   help=f"On-disk dtype for the two raw-F volumes {DUMP_DTYPES}. "
                        f"Default float32, which is EXACTLY lossless here. float16 "
                        f"halves them but has only ~3%% headroom over the measured "
                        f"peak and quantises F in steps of 32 counts at the bright "
                        f"end -- see CLAUDE.md 9.16.")
    p.add_argument("--pattern", default=defaults["pattern"],
                   help="Raw image glob in each t# folder (2022 cohort: *.tiff).")
    p.add_argument("--rot90-k", dest="rot90_k", type=int, default=defaults["rot90_k"],
                   help="Quarter turns per frame (np.rot90; -1 = clockwise). Must match "
                        "the rotation the ROI sets were drawn on.")
    p.add_argument("--frame-order", choices=["name", "number"],
                   default=defaults["frame_order"],
                   help="Order raw files by name (zero-padded indices) or by their "
                        "trailing number (July 2023 cohort).")
    p.add_argument("--trim-frames", type=int, default=defaults["trim_frames"],
                   metavar="N", help="Onset frames dropped per channel "
                                     "(default: the profile's trim).")
    p.add_argument("--channel-buffer", choices=BUFFER_MODES,
                   default=defaults["channel_buffer"],
                   help="Where the single-read channel buffer lives (auto/ram/disk/none).")
    p.add_argument("--buffer-dir", default=defaults["buffer_dir"], metavar="DIR",
                   help="Folder for disk buffers (an SSD).")
    p.add_argument("--keep-channel-buffer", action=argparse.BooleanOptionalAction,
                   default=defaults["keep_channel_buffer"],
                   help="Keep the unlabelled odd/even volumes next to the pixel dump.")
    p.add_argument("--alternation-max", type=float, default=defaults["alternation_max"])
    p.add_argument("--separation-min", type=float, default=defaults["separation_min"])
    p.add_argument("--skip-not-interleaved", action=argparse.BooleanOptionalAction,
                   default=defaults["skip_not_interleaved"],
                   help="Write nothing for a not_interleaved recording. Default "
                        "(--no-skip-not-interleaved): store it as odd/even frames.")
    p.add_argument("--top-percent", type=float, default=defaults["top_percent"])
    p.add_argument("--level-statistic", choices=LEVEL_STATISTICS,
                   default=defaults["level_statistic"])
    p.add_argument("--channel-level-region", choices=LEVEL_REGIONS,
                   default=defaults["channel_level_region"],
                   help="Pixels the channel levels are taken over: whole frame or ROI union.")
    ns = p.parse_args()
    # Not a flag: changing the window changes which pixels the array even contains,
    # so it belongs in RUN_CONFIG next to the comment recording how it was measured.
    ns.save_data_window = defaults["save_data_window"]
    return ns


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    """Prefer CLI args when any are given, else fall back to RUN_CONFIG (editor mode)."""
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        manifest=config["manifest"],
        output_root=config["output_root"],
        select=list(config["select"]),
        max_rows=config["max_rows"],
        roi_set_dir=config["roi_set_dir"],
        roi_set=config["roi_set"],
        debug=bool(config["debug"]),
        debug_frames=config["debug_max_frames"],
        max_recordings=config["max_recordings"],
        merge_recordings=bool(config["merge_recordings"]),
        workers=config["workers"],
        skip_existing=bool(config["skip_existing"]),
        save_data=bool(config["save_data"]),
        save_data_root=config["save_data_root"],
        save_data_window=config["save_data_window"],
        save_data_dff_dtype=config["save_data_dff_dtype"],
        save_data_f_dtype=config["save_data_f_dtype"],
        pattern=config["pattern"],
        rot90_k=int(config["rot90_k"]),
        frame_order=config["frame_order"],
        trim_frames=config["trim_frames"],
        channel_buffer=config["channel_buffer"],
        buffer_dir=config["buffer_dir"],
        keep_channel_buffer=bool(config["keep_channel_buffer"]),
        alternation_max=float(config["alternation_max"]),
        separation_min=float(config["separation_min"]),
        skip_not_interleaved=bool(config["skip_not_interleaved"]),
        top_percent=float(config["top_percent"]),
        level_statistic=config["level_statistic"],
        channel_level_region=config["channel_level_region"],
    )


def load_manifest(path: str) -> list[dict[str, str]]:
    """Read the manifest CSV into a list of row dicts (order preserved)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def select_rows(rows: list[dict[str, str]], keys: list[str],
                max_rows: int | None) -> list[dict[str, str]]:
    """Filter manifest rows by "day/animal" keys, then cap to ``max_rows``.

    A "day/animal/t#" key picks a one-recording row (the per-recording manifest
    resolve_channel_orders.py writes): its ``recordings`` cell is that single t#.
    """
    def row_keys(r: dict[str, str]) -> set[str]:
        unit = f"{r['day']}/{r['animal']}"
        return {unit, f"{unit}/{r['recordings']}"} if r.get("recordings") else {unit}

    if keys:
        wanted = set(keys)
        chosen = [r for r in rows if row_keys(r) & wanted]
        missing = wanted - set().union(*(row_keys(r) for r in chosen))
        if missing:
            raise ValueError(f"select keys not found in manifest: {sorted(missing)}")
    else:
        chosen = list(rows)
    if max_rows is not None:
        chosen = chosen[:max_rows]
    return chosen


def resolve_roi_set(day: str, animal: str, args: argparse.Namespace,
                    recording: str | None = None) -> str | None:
    """Which ROI-set file this analysis should use, or None for the profile default.

    Lookup order inside ``roi_set_dir``, most specific first:
      1. ``<day>_<animal>_<t#>.yaml`` -- per-RECORDING geometry. This is how
         ``roi_sets\\rebuilt\\`` is named: the editor saved one page per t#, so
         every recording carries the boxes/Bregma that were checked against its
         own anatomy.
      2. ``<day>_<animal>.yaml``      -- per-ANIMAL geometry, one file for all t#.
      3. the single shared ``roi_set``.

    A missing per-unit file is NOT an error when a shared file is configured --
    that is the intended "same ROIs for all sessions, except the ones I drew" mix
    -- but a ``roi_set_dir`` with no match and no shared fallback means this
    analysis silently keeps the preset geometry, so say so out loud.
    """
    if args.roi_set_dir:
        candidates = []
        if recording:
            candidates.append(os.path.join(args.roi_set_dir, f"{day}_{animal}_{recording}.yaml"))
        candidates.append(os.path.join(args.roi_set_dir, f"{day}_{animal}.yaml"))
        for path in candidates:
            if os.path.isfile(path):
                return path
        print(f"  no ROI set at {' or '.join(candidates)}"
              + (f" -- falling back to {args.roi_set}" if args.roi_set
                 else " -- using the profile's atlas and the manifest Bregma"))
    return args.roi_set


def _geometry_for(row: dict[str, str], args: argparse.Namespace,
                  recording: str | None):
    """Resolve (profile, cfg, roi_set, bregma) for one analysis.

    Called per RECORDING in the default separate mode, because ``roi_set_dir``
    may hold a different file per t#; called once with the first recording's name
    when merging, since a concatenation has a single geometry by construction.
    """
    profile = get_profile(row["profile"])
    bregma_row, bregma_col = int(row["bregma_row"]), int(row["bregma_col"])
    # An optional ROI set (drawn with roi_editor.py) overrides the profile's atlas
    # and the manifest's Bregma -- boxes and Bregma always travel together.
    roi_set = resolve_roi_set(row["day"], row["animal"], args, recording)
    profile, bregma_row, bregma_col = apply_roi_set(profile, bregma_row, bregma_col, roi_set)
    # The profile supplies the ROI atlas; cfg carries only the per-animal Bregma.
    cfg = ROIConfig.from_bregma(bregma_row, bregma_col, boxes=dict(profile.atlas))
    return profile, cfg, roi_set, bregma_row, bregma_col


def _frame_options(args: argparse.Namespace) -> tuple[str, int]:
    """``(pattern, rot90_k)`` for reading raw frames. getattr: namespaces built before
    these options existed (tests, older callers) mean the 2026 defaults."""
    return getattr(args, "pattern", "*.tif"), int(getattr(args, "rot90_k", 0))


def _frame_order(args: argparse.Namespace) -> str:
    """How raw files are ordered (wfci.io.sorted_frame_files). getattr: namespaces
    built before the option existed mean the 2026 default, ``"name"``."""
    return getattr(args, "frame_order", "name")


def _quiet_roi_set(row: dict[str, str], args: argparse.Namespace,
                   recording: str | None) -> str | None:
    """:func:`resolve_roi_set` without its "no ROI set" notice -- for the up-front
    disk estimate, which would otherwise print it once more for every job."""
    with contextlib.redirect_stdout(io.StringIO()):
        return resolve_roi_set(row["day"], row["animal"], args, recording)


def _with_trim(profile, args: argparse.Namespace):
    """The profile with ``trim_frames`` applied (None keeps the profile's own trim)."""
    trim_frames = getattr(args, "trim_frames", None)
    return profile if trim_frames is None else replace(profile, trim=int(trim_frames))


def _dump_dir(out_dir: str, args: argparse.Namespace) -> str:
    """Where this job's per-pixel volumes go.

    With ``save_data_root`` unset they sit beside the .npz, which keeps one
    analysis in one folder. With it set, the same ``<day>_<animal>\\<t#>`` tail is
    mirrored under that root -- the volumes are ~100x the size of everything else
    a job writes, so they usually belong on a different disk.
    """
    if not args.save_data_root:
        return out_dir
    tail = os.path.relpath(out_dir, args.output_root)
    return os.path.join(args.save_data_root, tail)


def _dump_region(cfg, args: argparse.Namespace) -> tuple[int, int, int, int] | None:
    """Resolve ``save_data_window`` to absolute rows/cols on the final grid.

    The window is stored as Bregma-relative half-open offsets, so the SAME window
    is a different rectangle in every animal -- that is the point: it names the
    same anatomy, not the same camera pixels, so dumps from different animals are
    directly comparable. Here it is anchored to this recording's own Bregma.

    Every ROI box is then checked to lie inside it. All 356 files in
    roi_sets\\rebuilt satisfy the configured window by construction (it is their
    measured union), but a different --roi-set could put a box outside it, and a
    dump that silently clipped away a box the correlations are computed from
    would be worse than no dump at all.
    """
    if not args.save_data_window:
        return None
    r0, r1, c0, c1 = args.save_data_window
    region = (cfg.y_1 + r0, cfg.y_1 + r1, cfg.x_2 + c0, cfg.x_2 + c1)
    outside = [
        name for name, b in cfg.boxes.items()
        # Box offsets are MATLAB 1-based inclusive; the half-open Python range is
        # [row_start - 1, row_end), which is the convention the window uses too.
        if b.row_start - 1 < r0 or b.row_end > r1
        or b.col_start - 1 < c0 or b.col_end > c1
    ]
    if outside:
        raise ValueError(
            f"save_data_window {args.save_data_window} (Bregma-relative) does not "
            f"contain these ROI boxes: {outside}. The window is never clipped, "
            f"because a dump missing a box the correlations use would be "
            f"misleading. Widen save_data_window or set it to None for the full "
            f"128x128 frame."
        )
    return region


def _make_pixel_dump(out_dir: str, args: argparse.Namespace, cfg, profile,
                     n_time: int, meta: dict[str, Any]) -> PixelDump | None:
    """Build this job's :class:`~wfci.dump.PixelDump`, or None when save_data is off."""
    if not args.save_data:
        return None
    dump_dir = _dump_dir(out_dir, args)
    region = _dump_region(cfg, args)
    dump = PixelDump(
        dump_dir,
        suffix="debug" if args.debug else "full",
        n_time=n_time,
        region=region,
        downsample=profile.downsample,
        dff_dtype=args.save_data_dff_dtype,
        f_dtype=args.save_data_f_dtype,
        metadata=_dump_metadata(args, cfg, profile, meta),
    )
    print(f"Pixels : {dump_dir}  (region {region}, {n_time} frames)")
    return dump


def _dump_metadata(args: argparse.Namespace, cfg, profile, meta: dict[str, Any]) -> dict[str, Any]:
    """Enough provenance to map any dumped pixel back to the full frame and to
    Bregma, and to recompute the per-channel ratios from the raw-F volumes."""
    return dict(
        profile=profile.name,
        roi_labels=np.array(profile.labels),
        bregma_row=meta["bregma"][0], bregma_col=meta["bregma"][1],
        y_1=cfg.y_1, x_2=cfg.x_2,
        grid=np.array(profile.atlas.grid),
        window_rel=np.array(args.save_data_window if args.save_data_window else []),
        **{k: v for k, v in meta.items() if k != "bregma"},
    )


def _keep_not_interleaved(args: argparse.Namespace, merged: bool) -> bool:
    """Is a not_interleaved recording stored as odd/even in this run?

    getattr: namespaces built before the option existed get the current default.
    """
    return (not getattr(args, "skip_not_interleaved", False)
            and bool(args.save_data) and not merged)


def _remove_labelled_outputs(out_dir: str, args: argparse.Namespace) -> list[str]:
    """Delete what an earlier run wrote for a recording now found not_interleaved.

    Those files carry GCaMP/emo labels the recording cannot support, and scripts
    that find recordings by file name would otherwise keep using them. The
    BL-pedestal caches are left in place (CLAUDE.md 9.72). Returns what was removed.
    """
    suffix = "debug" if args.debug else "full"
    dump_dir = _dump_dir(out_dir, args)
    paths = [os.path.join(dump_dir, f"pixels_{name}_{suffix}.npy") for name in VOLUME_NAMES]
    paths.append(os.path.join(dump_dir, f"pixels_meta_{suffix}.npz"))
    paths += [os.path.join(dump_dir, name) for name in
              (os.listdir(dump_dir) if os.path.isdir(dump_dir) else [])
              if name.startswith("pixels_median_dff_") and "blpedestal" not in name]
    paths += [os.path.join(out_dir, name) for name in
              (f"connectivity_{suffix}.npz", f"roi_fluorescence_{suffix}.csv",
               f"roi_traces_{suffix}.png", "roi_overlay_debug.png")]
    removed = [path for path in paths if os.path.isfile(path)]
    for path in removed:
        os.remove(path)
    return removed


def _store_odd_even(out_dir: str, args: argparse.Namespace, cfg, profile, meta: dict[str, Any],
                    odd_frames, even_frames, n_time: int, mean_odd, mean_even) -> None:
    """Replace any earlier outputs of this recording with its unlabelled odd/even volumes."""
    removed = _remove_labelled_outputs(out_dir, args)
    if removed:
        print(f"Removed: {len(removed)} earlier output file(s) of this recording")
    dump_dir = _dump_dir(out_dir, args)
    meta_path = write_odd_even_dump(
        dump_dir, "debug" if args.debug else "full", odd_frames, even_frames, n_time,
        mean_odd, mean_even, region=_dump_region(cfg, args), downsample=profile.downsample,
        f_dtype=args.save_data_f_dtype, metadata=_dump_metadata(args, cfg, profile, meta))
    print(f"Pixels : {meta_path}  (odd/even, channels not assigned, {n_time} frames)")


def _dump_in_memory(dump: PixelDump, trial, profile) -> None:
    """Feed an in-memory (debug) trial through the same writer as the streaming path.

    ``run_profile`` already returns the corrected stack, but not the raw channels
    at the resolution the correction used, so the first step of
    ``wfci.correction.build_dff_stack`` is repeated here: trim, then one 0.5x box
    downsample. Those are the very arrays the correction consumed, so the debug
    dump is the same quantity as the full one -- same files, same layout, just
    fewer frames.
    """
    gcamp_raw, emo_raw = trial
    g_half = imresize_box(gcamp_raw[:, :, profile.trim:], profile.downsample)
    e_half = imresize_box(emo_raw[:, :, profile.trim:], profile.downsample)
    # The correction itself is not re-derived here -- calling the library function
    # is what guarantees the dumped dF/F is the same quantity the ROI means came
    # from, including the profile's baseline window.
    dff_half = hemodynamic_correction(g_half, e_half, profile.baseline)
    dump.baselines(np.mean(g_half[:, :, profile.baseline], axis=2),
                   np.mean(e_half[:, :, profile.baseline], axis=2))
    try:
        for idx in range(g_half.shape[2]):
            dump.frame(idx, g_half[:, :, idx], e_half[:, :, idx],
                       imresize_box(dff_half[:, :, idx], profile.downsample))
    finally:
        dump.close()


def _decision_config(args: argparse.Namespace) -> ChannelDecisionConfig:
    """Channel-decision thresholds from the run options. getattr: namespaces built
    before these options existed (tests, older callers) get the spec defaults."""
    return ChannelDecisionConfig(
        top_percent=float(getattr(args, "top_percent", 10.0)),
        level_statistic=getattr(args, "level_statistic", "median"),
        alternation_max=float(getattr(args, "alternation_max", -0.9)),
        separation_min=float(getattr(args, "separation_min", 10.0)),
        level_region=getattr(args, "channel_level_region", "frame"),
    )


def _level_mask_for(cfg, profile, args: argparse.Namespace):
    """The ROI-union mask builder when the decision uses ``roi_union``, else None.

    Built from the SAME geometry pass 2 uses (run_streaming_profile: this
    recording's Bregma + the profile's atlas boxes, checked against its grid).
    """
    if getattr(args, "channel_level_region", "frame") != "roi_union":
        return None
    roi_cfg = ROIConfig(y_1=cfg.y_1, x_2=cfg.x_2, boxes=dict(profile.atlas))
    return roi_union_level_mask(roi_cfg, profile.atlas.grid, profile.downsample)


def _channel_buffer(args: argparse.Namespace) -> tuple[str, str | None, bool]:
    """``(channel_buffer, buffer_dir, keep_channel_buffer)`` with the spec defaults."""
    return (getattr(args, "channel_buffer", "auto"), getattr(args, "buffer_dir", None),
            bool(getattr(args, "keep_channel_buffer", False)))


def _block_files(folder: str, args: argparse.Namespace) -> list:
    """One block's images in acquisition order."""
    pattern, _ = _frame_options(args)
    files = sorted_frame_files(folder, pattern, _frame_order(args))
    if len(files) < 2:
        raise FileNotFoundError(f"Need at least 2 images matching {pattern!r} in "
                                f"{folder} to split two interleaved channels; "
                                f"found {len(files)}.")
    return files


def _debug_trial(folders: list[str], rec_names: list[str], configured: list[str],
                 profile, args: argparse.Namespace, merged: bool, level_mask_for=None):
    """In-memory (debug) trial: load the first N frames per channel, decide, label.

    The decision uses only the frames loaded (spec, "--debug"), trimmed and
    half-resolution like a full run. A single recording loads ``debug_frames``
    frames per channel and lets the profile trim them; a merged run loads
    ``debug_frames`` POST-trim frames per block and concatenates them, exactly as
    before. Returns ``(trial or None if skipped, choices)``; a recording to be
    stored by position returns ``(("unassigned", odd, even), choices)``.
    """
    _, rot90_k = _frame_options(args)
    decision = _decision_config(args)
    g_parts, e_parts, choices = [], [], []
    for folder, name, order in zip(folders, rec_names, configured):
        odd_files, even_files = split_odd_even(_block_files(folder, args))
        n = len(odd_files)
        if merged:
            if n <= profile.trim:
                raise ValueError(f"{folder}: only {n} frames/channel, at or below the "
                                 f"{profile.trim}-frame onset trim.")
            odd_files, even_files = odd_files[profile.trim:], even_files[profile.trim:]
            limit, stats_trim = min(args.debug_frames, len(odd_files)), 0
        else:
            limit, stats_trim = min(args.debug_frames, n), profile.trim
            if limit <= profile.trim:
                raise ValueError(
                    f"{folder}: debug frames={limit} is at or below the profile's "
                    f"{profile.trim}-frame trim; nothing left to correlate.")
        odd, even = _load_limited_trial(odd_files, even_files, limit, rot90_k=rot90_k)
        stats = channel_stats_from_stacks(odd, even, stats_trim, profile.downsample,
                                          decision.top_percent, level_mask_for)
        keep = _keep_not_interleaved(args, merged)
        choice = choose_block_channels(stats, order, "debug", name, decision,
                                       keep_not_interleaved=keep)
        choices.append(choice)
        if choice.verdict == "not_interleaved":
            return (("unassigned", odd, even) if keep else None), choices
        gcamp, emo = (odd, even) if choice.order == "gcamp_first" else (even, odd)
        g_parts.append(gcamp)
        e_parts.append(emo)
    # Concatenate along the time axis (axis=-1 of [y, x, time]) into one trial.
    return (np.concatenate(g_parts, axis=-1), np.concatenate(e_parts, axis=-1)), choices


def _buffer_keep_paths(out_dir: str, args: argparse.Namespace,
                       rec_names: list[str], merged: bool) -> list[tuple[str, str]]:
    """Where ``keep_channel_buffer`` puts each block's odd/even volumes: next to
    the pixel dump, ``pixels_f_odd_<suffix>.npy`` (``_<t#>`` added when merged)."""
    suffix = "debug" if args.debug else "full"
    dump_dir = _dump_dir(out_dir, args)
    paths = []
    for name in rec_names:
        tail = f"{suffix}_{name}" if merged else suffix
        paths.append((os.path.join(dump_dir, f"pixels_f_odd_{tail}.npy"),
                       os.path.join(dump_dir, f"pixels_f_even_{tail}.npy")))
    return paths


def _run_trial(folders: list[str], rec_names: list[str], cfg, profile,
               configured: list[str], args: argparse.Namespace, out_dir: str,
               meta: dict[str, Any], merged: bool):
    """Run one analysis -- one ``t#``, or an animal's ``t#`` concatenated (merged).

    Each block is read from disk ONCE (``wfci.prepare_interleaved_trial``): the two
    interleaved groups are buffered unlabelled, the channel order is decided from
    every frame, and the existing pass 1 / pass 2 math then runs on the buffer
    (``channel_buffer = none`` re-reads the TIFFs for pass 2 instead).

    A single recording keeps the profile's own ``trim``. A merged run trims each
    block's onset itself and runs the concatenation with ``trim = 0``, so it is
    not trimmed a second time at its start.

    The channel fields (``channel_order``, ``channel_verdict``, levels, ...) are
    written into ``meta`` before any output is created. Returns the result; None
    when an ``auto`` block was ``not_interleaved`` and nothing was stored; or
    NEEDS_CHANNEL_DECISION when it was stored as odd/even frames.
    """
    run_prof = replace(profile, trim=0) if merged else profile
    _, rot90_k = _frame_options(args)
    level_mask_for = _level_mask_for(cfg, profile, args)

    if args.debug:
        print(f"Mode   : DEBUG (in-memory) -- first {args.debug_frames} frames/channel "
              f"of {len(folders)} recording(s) (trim {profile.trim})")
        trial, choices = _debug_trial(folders, rec_names, configured, profile, args, merged,
                                      level_mask_for)
        meta.update(channel_choice_meta(choices))
        if trial is None:
            return None
        if isinstance(trial[0], str):  # ("unassigned", odd, even); a real trial holds arrays
            # Same steps as the streaming path: trim, then one 0.5x box downsample.
            odd_half = imresize_box(trial[1][:, :, profile.trim:], profile.downsample)
            even_half = imresize_box(trial[2][:, :, profile.trim:], profile.downsample)
            _store_odd_even(out_dir, args, cfg, run_prof, meta,
                            (odd_half[:, :, i] for i in range(odd_half.shape[2])),
                            (even_half[:, :, i] for i in range(even_half.shape[2])),
                            odd_half.shape[2],
                            np.mean(odd_half[:, :, profile.baseline], axis=2),
                            np.mean(even_half[:, :, profile.baseline], axis=2))
            return NEEDS_CHANNEL_DECISION
        result = run_profile([trial], cfg, run_prof)
        dump = _make_pixel_dump(out_dir, args, cfg, run_prof,
                                trial[0].shape[2] - run_prof.trim, meta)
        if dump is not None:
            _dump_in_memory(dump, trial, run_prof)
        return result

    channel_buffer, buffer_dir, keep = _channel_buffer(args)
    print(f"Mode   : FULL (streaming, single read) -- {len(folders)} recording(s), "
          f"onset trim {profile.trim}, channel buffer {channel_buffer}")
    prepared = prepare_interleaved_trial(
        [_block_files(f, args) for f in folders], configured,
        trim=profile.trim, pipeline_trim=run_prof.trim,
        downsample=profile.downsample, baseline_slice=profile.baseline,
        rot90_k=rot90_k, channel_buffer=channel_buffer, buffer_dir=buffer_dir,
        decision=_decision_config(args), block_names=rec_names,
        level_mask_for=level_mask_for,
        keep_not_interleaved=_keep_not_interleaved(args, merged),
    )
    meta.update(prepared.meta())
    try:
        if prepared.unassigned:
            block = prepared.blocks[0]
            _store_odd_even(out_dir, args, cfg, run_prof, meta,
                            block.odd.frames(), block.even.frames(), block.n_frames,
                            block.sum_odd / block.n_in_window,
                            block.sum_even / block.n_in_window)
            return NEEDS_CHANNEL_DECISION
        if prepared.skipped:
            return None
        dump = _make_pixel_dump(out_dir, args, cfg, run_prof, prepared.n_time, meta)
        result = run_streaming_profile(
            [(prepared.gcamp, prepared.emo)], cfg, run_prof,
            # One trial here, so the factory hands the single dump to trial 0.
            pixel_dump_factory=None if dump is None else (lambda i: dump),
            trial_baselines=[prepared.baselines],
        )
        if keep:
            kept = prepared.keep_buffers(_buffer_keep_paths(out_dir, args, rec_names, merged))
            if not all(kept):
                print("Buffer : keep_channel_buffer is on, but a block ran with "
                      "channel_buffer=none, so it has no buffer to keep.")
        return result
    finally:
        # Frees RAM and deletes disk buffers after success AND after an error.
        prepared.release()


def _save_roi_fluorescence(temp_roi: np.ndarray, labels, out_dir: str,
                           args: argparse.Namespace) -> str:
    """Write the per-ROI dF/F time series (TEMP_ROI) as a CSV. Returns its path.

    ``temp_roi`` is ``[time, n_rois, trial]``. Every analysis here runs a single
    trial (one t# recording, or one concatenation), but the trial axis is kept
    explicit so a multi-trial result is written unambiguously: rows are ordered
    trial-major, and the ``trial`` column says which block each frame came from.
    Values are dF/F after hemodynamic correction, averaged over each ROI box --
    exactly the numbers the correlation matrix is computed from.
    """
    n_time, n_rois, n_trial = temp_roi.shape
    path = os.path.join(out_dir, "roi_fluorescence_debug.csv" if args.debug
                        else "roi_fluorescence_full.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial", "frame", *labels])
        for t in range(n_trial):
            block = temp_roi[:, :, t]
            for i in range(n_time):
                writer.writerow([t, i, *(f"{v:.6g}" for v in block[i])])
    print(f"ROI dF/F: {path}  ({n_time} frames x {n_rois} ROIs x {n_trial} trial)")
    return path


def _save_outputs(result, out_dir: str, args: argparse.Namespace, cfg, profile,
                  meta: dict[str, Any]) -> str:
    """Write the overlay/traces/npz for one analysis. Returns the .npz path.

    Shared by the separate and merged paths so both write the same files with the
    same names -- only the directory (and ``meta``) differ.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = profile.labels

    if getattr(result, "dff_stack", None) is not None:
        # RAM (debug) mode only: the placement overlay needs the DFF stack.
        overlay_path = os.path.join(out_dir, "roi_overlay_debug.png")
        _save_overlay(result.dff_stack, cfg, profile, overlay_path)
        print(f"Overlay: {overlay_path}  <- CHECK the boxes sit on the anatomy")

    if args.debug:
        # Smoke tests are read by eye, so print the whole matrix. A full batch runs
        # hundreds of jobs and the matrices are in the .npz -- dumping 355 of them
        # would bury the log, so there only the single-number handle is printed.
        print(f"R_mean ({profile.n_rois}x{profile.n_rois}), ROI order {labels}:")
        print(np.array2string(result.R_mean, precision=4, suppress_small=True))
    else:
        print(f"R_mean : {profile.n_rois}x{profile.n_rois}, "
              f"mean off-diagonal {_mean_offdiag(result.R_mean):.4f}")

    traces_path = os.path.join(out_dir, "roi_traces_debug.png" if args.debug
                               else "roi_traces_full.png")
    _save_traces(result.averaged_traces, labels, traces_path)

    # The intermediate quantity everything downstream is built from: the per-ROI
    # dF/F fluorescence time series (TEMP_ROI). It already goes into the .npz, but
    # it is also written as a plain CSV so it can be re-analysed without loading
    # the pipeline -- one row per frame, one column per ROI.
    _save_roi_fluorescence(result.temp_roi, labels, out_dir, args)

    npz_path = os.path.join(out_dir, "connectivity_debug.npz" if args.debug
                            else "connectivity_full.npz")
    arrays = dict(
        temp_roi=result.temp_roi,
        R=result.R,
        R_mean=result.R_mean,
        averaged_traces=result.averaged_traces,
        roi_labels=np.array(labels),
        profile=np.array(profile.name),
        **{k: np.array(v) for k, v in meta.items()},
    )
    if getattr(result, "dff_stack", None) is not None:
        arrays["dff_stack"] = result.dff_stack
    np.savez(npz_path, **arrays)
    print(f"Saved  : {npz_path}")
    return npz_path


def _npz_name(args: argparse.Namespace) -> str:
    """File name ``_save_outputs`` will give the .npz -- also the resume marker."""
    return "connectivity_debug.npz" if args.debug else "connectivity_full.npz"


def _job_is_complete(job: dict[str, Any], args: argparse.Namespace) -> bool:
    """Has this job already produced everything the CURRENT flags ask for?

    The .npz alone is not the answer once ``--save-data`` is on. Every recording
    in this study already has its .npz, so keying resume on that file would skip
    all 355 jobs and write no pixels at all -- the flag would appear to work and
    produce nothing. A job counts as done only if the dump sidecar is there too;
    ``PixelDump.close`` writes it last, so its presence means the volumes were
    finished rather than interrupted halfway.
    """
    if not os.path.isfile(os.path.join(job["out_dir"], _npz_name(args))):
        return False
    if not getattr(args, "save_data", False):
        return True
    suffix = "debug" if args.debug else "full"
    return os.path.isfile(os.path.join(_dump_dir(job["out_dir"], args),
                                       f"pixels_meta_{suffix}.npz"))


def _summary_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    '''Identity of one summary row across interrupted/resumed batch runs.'''
    return (str(row['day']), str(row['animal']), str(row['mode']),
            str(row['recording']))


def _job_summary_key(job: dict[str, Any]) -> tuple[str, str, str, str]:
    '''Summary identity expected for a job before that job is executed.'''
    args = job['args']
    return (job['row']['day'], job['row']['animal'],
            'debug' if args.debug else 'full', job['label'])


def _summary_from_npz(job: dict[str, Any]) -> dict[str, Any]:
    '''Reconstruct a missing summary row from a completed job result file.

    All fields except processing time are saved in the npz itself. A blank
    elapsed time states that limitation honestly for reconstructed rows.
    '''
    args = job['args']
    npz_path = os.path.join(job['out_dir'], _npz_name(args))
    with np.load(npz_path, allow_pickle=False) as result:
        bregma = np.asarray(result['bregma']).tolist()
        return dict(
            day=str(result['day'].item()), animal=str(result['animal'].item()),
            group=str(result['group'].item()), profile=str(result['profile'].item()),
            bregma_row=int(bregma[0]), bregma_col=int(bregma[1]),
            roi_set=str(result['roi_set'].item()),
            mode='debug' if args.debug else 'full', recording=job['label'],
            n_recordings=len(job['folders']),
            mean_offdiag_R=_mean_offdiag(result['R_mean']), elapsed_s='',
            npz_path=npz_path,
            channel_verdict=(str(result['channel_verdict'].item())
                             if 'channel_verdict' in result.files else ''),
            status='done',
        )


def _read_summary_rows(path: str) -> list[dict[str, str]]:
    '''Read a prior non-empty summary so a resumed subset cannot erase it.'''
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


CHANNEL_ORDERS = ("auto", "gcamp_first", "emo_first")


def recording_channel_orders(row: dict[str, str]) -> list[str]:
    """One channel_order per entry of the row's ``recording_paths``, in that order.

    The manifest cell holds either ONE value, applied to every recording of the row
    (all older manifests), or a ``;``-separated list aligned with ``recordings`` /
    ``recording_paths`` (``t0;t1;...``), so each ``t#`` can carry its own order
    (resolve_channel_orders.py writes that form). A list of the wrong length or an
    unknown value raises rather than silently mislabelling a recording.
    """
    n = len([p for p in row["recording_paths"].split(";") if p])
    orders = [v.strip() for v in row["channel_order"].split(";")]
    if len(orders) == 1:
        orders = orders * n
    where = f"{row['day']}/{row['animal']}"
    if len(orders) != n:
        raise ValueError(f"{where}: channel_order lists {len(orders)} values for "
                         f"{n} recordings: {row['channel_order']!r}")
    bad = sorted(set(orders) - set(CHANNEL_ORDERS))
    if bad:
        raise ValueError(f"{where}: channel_order {bad} is not one of {CHANNEL_ORDERS}")
    return orders


def build_jobs(chosen: list[dict[str, str]],
               args: argparse.Namespace) -> list[dict[str, Any]]:
    """Flatten the selected manifest rows into independent units of work.

    One job = one thing that gets its own output folder: a single ``t#`` recording
    in the default separate mode, or a whole animal when ``--merge-recordings``.
    Jobs share nothing -- separate input folders, separate ROI geometry, separate
    output directory -- which is exactly what makes them safe to run in parallel.
    Everything a job needs is inside the dict so it can be pickled to a worker.
    """
    jobs: list[dict[str, Any]] = []
    for row in chosen:
        rec_paths = [p for p in row["recording_paths"].split(";") if p]
        # Aligned with rec_paths BEFORE the cap below, so each order stays with its t#.
        orders = recording_channel_orders(row)
        # Which recordings to use (cap applies to both modes; e.g. 2 -> t1+t2).
        used = rec_paths if args.max_recordings is None else rec_paths[:args.max_recordings]
        orders = orders[:len(used)]
        rec_names = [os.path.basename(p.rstrip("\\/")) or "recording" for p in used]
        unit_dir = os.path.join(args.output_root, f"{row['day']}_{row['animal']}")
        if args.merge_recordings:
            jobs.append(dict(row=row, args=args, folders=used, rec_names=rec_names,
                             channel_orders=orders, label="merged", out_dir=unit_dir))
        else:
            # Each recording keeps its own identity in the output tree: the t#
            # folder name becomes a subfolder, so t1 and t2 never overwrite.
            for folder, rec_name, order in zip(used, rec_names, orders):
                jobs.append(dict(row=row, args=args, folders=[folder],
                                 rec_names=[rec_name], channel_orders=[order],
                                 label=rec_name,
                                 out_dir=os.path.join(unit_dir, rec_name)))
    return jobs


def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one job to completion and write its outputs. Returns its summary row."""
    row, args = job["row"], job["args"]
    folders, rec_names = job["folders"], job["rec_names"]
    day, animal = row["day"], row["animal"]
    merged = job["label"] == "merged"

    print(f"=== {day}/{animal} / {job['label']} (group {row['group']}) ===")
    # Geometry is resolved PER JOB, not per animal: roi_set_dir may hold a separate
    # file per t# (roi_sets\rebuilt\<day>_<animal>_<t#>.yaml), each with its own
    # Bregma, so every recording gets the boxes drawn for it. When merging, the
    # blocks are one continuous session, so the first recording's set covers them.
    profile, cfg, roi_set, b_row, b_col = _geometry_for(row, args, rec_names[0])
    print(f"Bregma : row={b_row}, col={b_col} (downsampled y_1={cfg.y_1}, x_2={cfg.x_2})")
    profile = _with_trim(profile, args)
    pattern, rot90_k = _frame_options(args)

    # Common provenance saved into the .npz (and the pixel dump) alongside the arrays.
    # Rotation and pattern are recorded because the boxes only mean anything on
    # frames read the same way the ROI sets were drawn.
    # Per-recording channel order as configured (recording_channel_orders). An
    # "auto" is decided by _run_trial from EVERY frame of the block, while the block
    # is read once; the applied order, verdict and statistics land in meta there.
    configured = job.get("channel_orders") or recording_channel_orders(row)[:len(folders)]
    meta = dict(day=day, animal=animal, group=row["group"],
                bregma=[b_row, b_col], roi_set=roi_set or "",
                rot90_k=rot90_k, pattern=pattern, frame_order=_frame_order(args),
                trim=profile.trim, channel_order_configured=";".join(configured))
    if merged:
        meta["merged_recordings"] = folders
    else:
        print(f"Folder : {folders[0]}")
        meta["recording"] = rec_names[0]
        meta["recording_path"] = folders[0]

    t0 = time.perf_counter()
    result = _run_trial(folders, rec_names, cfg, profile, configured, args,
                        job["out_dir"], meta, merged)
    elapsed = time.perf_counter() - t0
    print(f"Time   : {elapsed:.1f}s for {len(folders)} recording(s)")

    summary = dict(day=day, animal=animal, group=row["group"], profile=profile.name,
                   bregma_row=b_row, bregma_col=b_col, roi_set=roi_set or "",
                   mode="debug" if args.debug else "full",
                   recording=job["label"], n_recordings=len(folders),
                   channel_verdict=meta["channel_verdict"],
                   elapsed_s=round(elapsed, 1))
    if result is NEEDS_CHANNEL_DECISION:
        print("Result : stored as odd/even frames -- channels to be decided")
        return dict(summary, status=NEEDS_CHANNEL_DECISION, mean_offdiag_R="", npz_path="")
    if result is None:
        # not_interleaved under "auto": no outputs, but listed in the summary so the
        # skip is visible and a resume does not read the recording again.
        print("Result : SKIPPED (not interleaved) -- no outputs written")
        return dict(summary, status="skipped", mean_offdiag_R="", npz_path="")
    npz_path = _save_outputs(result, job["out_dir"], args, cfg, profile, meta)
    return dict(summary, status="done", mean_offdiag_R=_mean_offdiag(result.R_mean),
                npz_path=npz_path)


def run_job(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Worker entrypoint: run a job with its printing captured.

    Workers run concurrently, so writing straight to the shared stdout would
    interleave half-lines from 15 processes into unreadable soup. Each job's
    output is buffered here and handed back to the parent, which prints it as one
    contiguous block when the job lands. Errors are NOT caught -- an exception
    propagates through the future and stops the batch, as everywhere else here.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary = execute_job(job)
    return buf.getvalue(), summary


def resolve_workers(requested: int | None) -> int:
    """How many worker processes to use.

    The configured default is 1. ``None`` (or <= 0) explicitly requests the old
    automatic choice: 75% of logical cores, rounded down, minimum 1. An explicit
    positive number is honoured but never exceeds the core count.
    """
    n_cores = os.cpu_count() or 1
    if requested is None or requested <= 0:
        return max(1, int(n_cores * 0.75))
    return max(1, min(int(requested), n_cores))


def _abort_pool(pool: ProcessPoolExecutor, futures: list) -> None:
    """Cancel queued work and terminate active workers without waiting for jobs.

    Python 3.11's process-pool context manager waits for submitted work while
    unwinding. After Ctrl+C that can leave the parent and workers alive despite a
    ``KeyboardInterrupt`` traceback. Result ``.npz`` files are written last, so an
    interrupted partial output has no resume marker and is safely recomputed.
    """
    for future in futures:
        future.cancel()
    processes = list((pool._processes or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    pool.shutdown(wait=False, cancel_futures=True)


def _mean_offdiag(R_mean: np.ndarray) -> float:
    """Mean of the off-diagonal correlations -- a crude single-number handle on
    connectivity strength, useful for eyeballing the batch summary at a glance."""
    n = R_mean.shape[0]
    return float(np.nanmean(R_mean[~np.eye(n, dtype=bool)]))


# Leave this much free after the dump. A disk filled to the last byte is its own
# failure mode, and the batch writes .npz/csv/png alongside the volumes.
DUMP_HEADROOM_BYTES = 5 * 1024**3


def _check_dump_space(jobs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Estimate the total dump size and refuse to start if it will not fit.

    A full manifest run is hours long and writes the volumes as it goes, so
    running out of disk at recording 200 would waste everything after it. The
    estimate is cheap and needs no decoding: the manifest already records
    ``total_frames`` across an animal's recordings, and both channels are
    interleaved in it, so one recording is ``total_frames / n_recordings / 2``
    frames per channel, minus the profile trim.
    """
    if not getattr(args, "save_data", False):
        return
    root = args.save_data_root or args.output_root
    os.makedirs(root, exist_ok=True)

    bytes_per_px = (np.dtype(args.save_data_dff_dtype).itemsize
                    + 2 * np.dtype(args.save_data_f_dtype).itemsize)

    total = 0
    for job in jobs:
        row = job["row"]
        profile = _with_trim(get_profile(row["profile"]), args)
        if args.save_data_window:
            r0, r1, c0, c1 = args.save_data_window
            n_px = (r1 - r0) * (c1 - c0)
        else:
            # The whole final grid: the job's ROI set records it (152x128 for the
            # rotated 2022 frames); 128x128 when there is none.
            roi_set = _quiet_roi_set(row, args, job["rec_names"][0])
            rows, cols = load_roi_set(roi_set)[0].grid if roi_set else (128, 128)
            n_px = rows * cols
        bytes_per_frame = n_px * bytes_per_px
        if args.debug:
            n_time = max(0, args.debug_frames - profile.trim) * len(job["folders"])
        else:
            per_rec = int(row["total_frames"]) // int(row["n_recordings"]) // 2
            n_time = max(0, per_rec - profile.trim) * len(job["folders"])
        total += n_time * bytes_per_frame

    free = shutil.disk_usage(root).free
    gb = 1024**3
    print(f"save_data: ~{total / gb:.1f} GB for {len(jobs)} job(s) into {root} "
          f"({free / gb:.1f} GB free)")
    if total + DUMP_HEADROOM_BYTES > free:
        raise RuntimeError(
            f"save_data needs ~{total / gb:.1f} GB plus "
            f"{DUMP_HEADROOM_BYTES / gb:.0f} GB headroom, but only "
            f"{free / gb:.1f} GB is free on {root}. Refusing to start rather than "
            f"filling the disk partway through. Narrow the run (--select, "
            f"--max-rows, --max-recordings) or point --save-data-root at a larger "
            f"disk."
        )


def main() -> None:
    args = build_runtime_args()
    rows = load_manifest(args.manifest)
    chosen = select_rows(rows, args.select, args.max_rows)
    all_jobs = build_jobs(chosen, args)

    keys = [f"{r['day']}/{r['animal']}" for r in chosen]
    print(f"Manifest : {args.manifest} ({len(rows)} rows)")
    print(f"Selected : {len(chosen)} unit(s) -> {keys if len(keys) <= 8 else str(keys[:8]) + ' ...'}")
    print(f"Mode     : {'DEBUG' if args.debug else 'FULL streaming'}")
    print("Recs     : " + ("merged into one trial per animal" if args.merge_recordings
                           else "kept separate (one analysis per t#)"))

    os.makedirs(args.output_root, exist_ok=True)
    summary_csv = os.path.join(args.output_root, "batch_summary.csv")

    # Resume: a finished job left its .npz behind, so its presence is the marker.
    # A job the summary lists as "skipped" (not_interleaved under "auto") left no
    # .npz on purpose; it counts as done too, so it is not read again -- unless this
    # run stores such recordings, which then takes precedence over any .npz an older
    # run left behind. A row marked needs_channel_decision is finished as it stands.
    n_total = len(all_jobs)
    skipped_jobs: list[dict[str, Any]] = []
    jobs = all_jobs
    if args.skip_existing:
        prior_status = {_summary_key(r): r.get("status")
                        for r in _read_summary_rows(summary_csv)}

        def already_handled(job: dict[str, Any]) -> bool:
            status = prior_status.get(_job_summary_key(job))
            if "auto" in job["channel_orders"]:
                if status == NEEDS_CHANNEL_DECISION:
                    return True
                if status == "skipped":
                    return not _keep_not_interleaved(args, job["label"] == "merged")
            return _job_is_complete(job, args)

        skipped_jobs = [job for job in all_jobs if already_handled(job)]
        skipped_ids = {id(job) for job in skipped_jobs}
        jobs = [job for job in all_jobs if id(job) not in skipped_ids]
        n_skipped = len(skipped_jobs)
        if n_skipped:
            print(f"Resume   : {n_skipped}/{n_total} job(s) already have "
                  f"{_npz_name(args)} -- skipping them")
    else:
        n_skipped = 0

    # Preserve prior rows, then reconstruct any skipped result that reached disk
    # before its row reached the CSV. This keeps resume markers and the index in sync.
    prior_rows = _read_summary_rows(summary_csv) if args.skip_existing else []
    rows_by_key = {_summary_key(row): row for row in prior_rows}
    n_recovered = 0
    for job in skipped_jobs:
        key = _job_summary_key(job)
        # A row marked skipped/needs_channel_decision always exists, so only a job
        # with a real .npz can reach the recovery below.
        if key not in rows_by_key:
            rows_by_key[key] = _summary_from_npz(job)
            n_recovered += 1
    if n_recovered:
        print(f'Summary  : recovered {n_recovered} missing row(s) from existing .npz files')

    # Checked after the resume filter, so an interrupted run is sized on what is
    # actually left to do rather than on the whole manifest.
    _check_dump_space(jobs, args)

    workers = resolve_workers(args.workers)
    if workers == 1:
        print(f"Jobs     : {len(jobs)} to run serially in the main process "
              f"({os.cpu_count()} logical cores available)")
    else:
        print(f"Jobs     : {len(jobs)} to run on {workers} worker process(es) "
              f"({os.cpu_count()} logical cores; experimental on Windows)")
    # Each worker is a whole process running its own numpy; letting each of them
    # also spawn a BLAS thread per core would oversubscribe the machine badly.
    # Children inherit this environment at spawn, before they import numpy.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    t_start = time.perf_counter()
    n_done = 0
    # Collected for the end-of-run report (they are also in the summary CSV).
    skipped_units: list[str] = []
    uncertain_units: list[str] = []
    needs_decision_units: list[str] = []
    new_summaries: list[dict[str, Any]] = []
    with open(summary_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for prior in rows_by_key.values():
            writer.writerow({field: prior.get(field, '') for field in SUMMARY_FIELDS})
        fh.flush()

        if not jobs:
            print('Nothing to do.')
            print(f'Batch summary ({len(rows_by_key)} existing row(s)) -> {summary_csv}')
            return

        def record(log_text: str, summary: dict[str, Any]) -> None:
            """Print one finished job's captured log and append its summary row."""
            nonlocal n_done
            n_done += 1
            verdicts = str(summary.get("channel_verdict", "")).split(";")
            key = f"{summary['day']}/{summary['animal']}/{summary['recording']}"
            new_summaries.append(summary)
            if summary.get("status") == "skipped":
                skipped_units.append(key)
            elif summary.get("status") == NEEDS_CHANNEL_DECISION:
                needs_decision_units.append(key)
            elif "uncertain" in verdicts:
                uncertain_units.append(key)
            elapsed = time.perf_counter() - t_start
            rate = elapsed / n_done
            print(f"\n[{n_done}/{len(jobs)}] done "
                  f"(elapsed {elapsed / 60:.1f} min, ~{rate * (len(jobs) - n_done) / 60:.0f} min left)")
            print(log_text, end="")
            writer.writerow(summary)
            fh.flush()

        if workers == 1:
            # Reliable Windows path: execute in the main process, with no spawn,
            # multiprocessing queue, or worker-side native DLL loading. Do not use
            # run_job here: its capture is only needed to prevent parallel logs
            # interleaving and would make a long serial recording look frozen.
            try:
                for job in jobs:
                    summary = execute_job(job)
                    record("", summary)
            except KeyboardInterrupt:
                print("\nInterrupted: serial execution stopped; completed results "
                      "remain resumable.", flush=True)
                raise SystemExit(130) from None
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            futures = []
            try:
                futures = [pool.submit(run_job, job) for job in jobs]
                pending = set(futures)
                print(f"Workers  : launched {min(workers, len(jobs))}; output is "
                      f"shown when each recording finishes", flush=True)
                while pending:
                    finished, pending = wait(
                        pending,
                        timeout=POOL_HEARTBEAT_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    if not finished:
                        elapsed = time.perf_counter() - t_start
                        # ``Future.running()`` includes calls staged in the pool's
                        # feeder queue (up to workers+1), not only executing child
                        # processes. Report occupied worker slots instead.
                        running = min(workers, len(pending))
                        queued = len(pending) - running
                        print(f"[{n_done}/{len(jobs)}] still working: {running} running, "
                              f"{queued} queued, elapsed {elapsed / 60:.1f} min",
                              flush=True)
                        continue
                    # Record every result that landed during this wait cycle.
                    for future in finished:
                        record(*future.result())
            except KeyboardInterrupt:
                print("\nInterrupted: terminating workers and leaving completed "
                      "results resumable.", flush=True)
                _abort_pool(pool, futures)
                raise SystemExit(130) from None
            except Exception:
                print("\nA worker failed: terminating the remaining pool; the "
                      "traceback follows.", flush=True)
                _abort_pool(pool, futures)
                raise
            else:
                pool.shutdown()

    # A rerun appended a fresh row after the one it replaced. The CSV stays append-only
    # during the run, so an interrupt loses nothing (a resume keeps the last row of
    # each key); once finished, the replaced rows are dropped, keeping the row order.
    replaced = [s for s in new_summaries if _summary_key(s) in rows_by_key]
    if replaced:
        rows_by_key.update({_summary_key(s): s for s in new_summaries})
        with open(summary_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            for row in rows_by_key.values():
                writer.writerow({field: row.get(field, '') for field in SUMMARY_FIELDS})

    if needs_decision_units:
        print(f"\nStored as odd/even, channels to be decided ({len(needs_decision_units)}):")
        for key in needs_decision_units:
            print(f"  {key}")
    if skipped_units:
        print(f"\nSkipped, not interleaved ({len(skipped_units)}): no outputs written")
        for key in skipped_units:
            print(f"  {key}")
    if uncertain_units:
        print(f"\nUncertain channel order ({len(uncertain_units)}): ran with the "
              f"dimmer group as GCaMP -- check before trusting amplitudes or events")
        for key in uncertain_units:
            print(f"  {key}")
    print(f"\nBatch summary ({n_done} new row(s)"
          + (f", {n_skipped} skipped" if n_skipped else "")
          + f") -> {summary_csv}")
    print(f"Total    : {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
