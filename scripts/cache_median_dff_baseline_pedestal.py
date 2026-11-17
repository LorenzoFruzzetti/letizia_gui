"""Median-baseline dF/F cache with every non-BL session's channel pedestal(s) put back to BL.

Why: when a channel's light level changes between sessions while its modulation in
raw counts stays constant, F/Fbar -- and with it the shared component and every
ROI-pair r -- is changed by the illumination, not by the brain. What changed is a
NON-modulating part of the light, so the matching correction is additive: take that
recording's own pedestal off and put the animal's BL pedestal back on.

For every recording whose timepoint is not `baseline_timepoint`, per pixel p:

    F'(t, p) = F(t, p) - smooth(mean_day)(p) + smooth(mean_BL)(p)

where mean_day is this recording's temporal mean of F, mean_BL the mean of the
temporal means of the animal's BL recordings (same Bregma-relative crop, so the
images stack), and smooth a Gaussian of `offset_smoothing_px` crop pixels (0 =
exact per pixel). Smoothing keeps the day's fine structure (vessels) and adopts
only BL's coarse level, so a 1-2 px session misalignment does not write one
session's vessel into the other's pedestal. `pedestal_channels` picks which
channel(s) get this correction: "gcamp" (the default), "emo" (reflectance,
R' = R - smooth(mean_day_R) + smooth(mean_BL_R)) or "both", for a cohort whose
reflectance level also moves between sessions.

Then exactly cache_median_dff.py's (F'/F'bar)/(R/Rbar) - 1 with the same window.
BL recordings are unchanged, so their cache equals the uncorrected one.

Two guards. Recordings a verdict table marks as split the wrong way round have their
two channel files swapped back first (`channel_assignment_csv`), otherwise the BL
GCaMP pedestal would be put under the reflectance signal; a mis-assigned BL
recording raises. And a handful of crop-edge pixels, where the sessions' pedestals
do not line up, would go nonpositive: those are left uncorrected and counted in the
meta, which is only allowed outside every ROI box (inside one it raises).

Output, next to each dump (window and tag in the name, so it never mixes):
    pixels_median_dff_<window>s_blpedestal_full.npy   float16, (n_time, rows, cols)
    pixels_median_dff_<window>s_blpedestal_meta.json  written LAST = done-marker

Example (wfci_gui env):
    python cache_median_dff_baseline_pedestal.py
    python cache_median_dff_baseline_pedestal.py --workers 1 --recording-limit 2
"""

from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from cache_median_dff import median_dff_volume
from pixel_helpers import crop_slices, median_window_frames
from roi_pixel_connectivity import read_meta, recording_boxes

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run the cache build without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "pixel_root": REPO_ROOT / "pixel_data",  # <date>_<animal>/<t#>/pixels_f_gcamp_full.npy
    "design_csv": REPO_ROOT / "pixel_data" / "experimental_design.csv",  # timepoint per recording
    "baseline_timepoint": "BL",  # Its recordings define each animal's reference pedestal.
    "offset_smoothing_px": 2.0,  # Gaussian sigma of the pedestal maps, crop pixels; 0 = exact per pixel.
    "window_s": 20.0,  # Same running-median window as the uncorrected cache (201 frames at 10 Hz).
    "sampling_rate_hz": 10.0,  # Same as the uncorrected cache, so only the pedestal differs.
    "tag": "blpedestal",  # File-name tag of this variant.
    # Channel(s) whose pedestal is put back to BL: "gcamp", "emo" or "both".
    "pedestal_channels": "gcamp",
    # Verdict table; recordings with verdict "certain" and agrees_with_auto False
    # have their two channel files swapped back. None: use the files as the batch wrote them.
    "channel_assignment_csv": None,
    "workers": 8,
    "recording_limit": None,  # None: all; positive integer: only the first N jobs (smoke test).
    "overwrite": False,
    "prefer_cli_args": True,
}


def cache_paths(folder: Path, window_s: float, tag: str) -> tuple[Path, Path]:
    stem = f"pixels_median_dff_{window_s:g}s_{tag}"
    return folder / f"{stem}_full.npy", folder / f"{stem}_meta.json"


# Dump file and pedestal-file label of each channel that can be corrected.
CHANNEL_FILES = {"gcamp": "pixels_f_gcamp_full.npy", "emo": "pixels_f_emo_full.npy"}
CHANNEL_LABELS = {"gcamp": "mean_f", "emo": "mean_r"}
CHANNEL_CHOICES = {"gcamp": ["gcamp"], "emo": ["emo"], "both": ["gcamp", "emo"]}


def temporal_mean(folder: Path, file_name: str) -> np.ndarray:
    """This recording's per-pixel temporal mean of one channel, float64 (rows, cols)."""
    volume = np.load(folder / file_name, mmap_mode="r")
    return np.asarray(volume, dtype=np.float64).mean(axis=0)


def smooth(image: np.ndarray, sigma_px: float) -> np.ndarray:
    return image if sigma_px <= 0 else gaussian_filter(image, sigma_px, mode="nearest")


def roi_union_mask(folder: Path) -> np.ndarray:
    """(rows, cols) bool: pixels of the crop inside any of the recording's own ROI boxes."""
    meta = read_meta(folder / "pixels_meta_full.npz")
    _, boxes = recording_boxes(meta)
    region = meta["region"]
    union = np.zeros((region[1] - region[0], region[3] - region[2]), dtype=bool)
    for box in boxes.values():
        row_slice, col_slice = crop_slices(box, int(meta["y_1"]), int(meta["x_2"]), region)
        union[row_slice, col_slice] = True
    return union


def build_one(job: dict[str, Any]) -> dict[str, Any]:
    """Correct one recording's GCaMP pedestal and write its median dF/F cache."""
    started = time.perf_counter()
    folder = Path(job["folder"])
    volume_path, meta_path = cache_paths(folder, job["window_s"], job["tag"])

    # channels: name -> (n_time, rows, cols). A recording the batch split the wrong way
    # round has the two files exchanged; each pedestal must go on the REAL
    # channel, so they are swapped back first.
    file_of = dict(CHANNEL_FILES)
    if job["swap_channels"]:
        file_of = {"gcamp": CHANNEL_FILES["emo"], "emo": CHANNEL_FILES["gcamp"]}
    channels = {}
    for name in ("gcamp", "emo"):
        volume = np.load(folder / file_of[name], mmap_mode="r")
        # Corrected channels are added to, so they need a float64 copy; the other stays mapped.
        channels[name] = np.array(volume, dtype=np.float64) if name in job["bl_mean_paths"] else volume

    union = roi_union_mask(folder)  # (rows, cols)
    levels, offsets, n_uncorrected = {}, {}, {}
    for name, bl_mean_path in job["bl_mean_paths"].items():
        mean_day = channels[name].mean(axis=0)  # (rows, cols)
        mean_bl = np.load(bl_mean_path)  # (rows, cols), the animal's BL pedestal
        if mean_bl.shape != mean_day.shape:
            raise ValueError(f"BL {name} pedestal {mean_bl.shape} vs recording {mean_day.shape}: {folder}")
        # offset: (rows, cols); zero for BL recordings by definition.
        if job["is_baseline"]:
            offset = np.zeros_like(mean_day)
        else:
            offset = smooth(mean_bl, job["sigma"]) - smooth(mean_day, job["sigma"])
        # A few crop-edge pixels, where the two sessions' pedestals do not line up, would
        # go nonpositive. They are left uncorrected; that is only allowed outside every
        # ROI box, because inside one it would change the analysed traces.
        nonpositive = (channels[name].min(axis=0) + offset) <= 0  # (rows, cols)
        if np.any(nonpositive & union):
            raise ValueError(f"Pedestal offset makes {name} nonpositive inside an ROI box: {folder}")
        offset[nonpositive] = 0.0
        channels[name] += offset
        levels[name] = (float(mean_day.mean()), float((mean_day + offset).mean()))
        offsets[name] = (float(offset.min()), float(offset.max()))
        n_uncorrected[name] = int(nonpositive.sum())
    fluorescence, reflectance = channels["gcamp"], channels["emo"]

    dff = median_dff_volume(fluorescence, reflectance, job["n_median"])
    stored = dff.astype(np.float16)
    storage_error = float(np.abs(stored.astype(np.float64) - dff).max())

    temporary_path = volume_path.with_name(volume_path.stem + ".partial.npy")
    np.save(temporary_path, stored)
    os.replace(temporary_path, volume_path)
    meta = dict(
        formula="X' = X + smooth(mean_BL) - smooth(mean_day) for each pedestal channel X (non-BL only); "
                "then (F'/F'bar)/(R'/R'bar) - 1",
        pedestal_channels=list(job["bl_mean_paths"]),
        unit="ratio (x100 for percent)", source=str(folder.resolve()),
        baseline_timepoint=job["baseline_timepoint"], is_baseline=job["is_baseline"],
        channels_swapped=job["swap_channels"], n_uncorrected_edge_pixels=n_uncorrected,
        bl_recordings=job["bl_recordings"], offset_smoothing_px=job["sigma"],
        window_s=job["window_s"], window_frames=job["n_median"], sampling_rate_hz=job["sampling_rate_hz"],
        level_before={name: level[0] for name, level in levels.items()},
        level_after={name: level[1] for name, level in levels.items()},
        offset_min={name: value[0] for name, value in offsets.items()},
        offset_max={name: value[1] for name, value in offsets.items()},
        shape=list(stored.shape), dtype="float16", axis_order="time,y,x",
        max_abs_storage_error=storage_error, seconds=round(time.perf_counter() - started, 2),
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dict(folder=str(folder), seconds=meta["seconds"], levels=levels)


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--design-csv", type=Path, default=defaults["design_csv"])
    parser.add_argument("--baseline-timepoint", default=defaults["baseline_timepoint"])
    parser.add_argument("--offset-smoothing-px", type=float, default=defaults["offset_smoothing_px"])
    parser.add_argument("--window-s", type=float, default=defaults["window_s"])
    parser.add_argument("--sampling-rate-hz", type=float, default=defaults["sampling_rate_hz"])
    parser.add_argument("--tag", default=defaults["tag"])
    parser.add_argument("--pedestal-channels", choices=sorted(CHANNEL_CHOICES), default=defaults["pedestal_channels"])
    # "none": no verdict file.
    parser.add_argument("--channel-assignment-csv", type=lambda value: None if value.lower() == "none" else Path(value),
                        default=defaults["channel_assignment_csv"])
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--recording-limit", type=int, default=defaults["recording_limit"])
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=defaults["overwrite"])
    return parser.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{key: value for key, value in config.items() if key != "prefer_cli_args"})


def main():
    args = build_runtime_args()
    pixel_root = Path(args.pixel_root)
    n_median = median_window_frames(args.window_s, args.sampling_rate_hz)
    # keep_default_na: a group may be the literal string "NA".
    design = pd.read_csv(args.design_csv, dtype={"day": str}, keep_default_na=False)
    design = design[[(pixel_root / pixel_dir / "pixels_f_gcamp_full.npy").exists()
                     for pixel_dir in design["pixel_dir"]]]

    pedestal_channels = CHANNEL_CHOICES[args.pedestal_channels]
    # Each animal's BL pedestal per corrected channel: mean of its BL recordings'
    # temporal means, saved once.
    pedestal_dir = pixel_root / f"_{args.tag}_pedestals"
    pedestal_dir.mkdir(parents=True, exist_ok=True)
    bl_info = {}
    for animal, rows in design.groupby("animal"):
        bl_rows = rows[rows["timepoint"] == args.baseline_timepoint]
        if bl_rows.empty:
            raise ValueError(f"{animal} has no {args.baseline_timepoint} recordings")
        bl_mean_paths = {}
        for name in pedestal_channels:
            bl_mean_path = pedestal_dir / f"{animal}_{CHANNEL_LABELS[name]}_{args.baseline_timepoint}.npy"
            if args.overwrite or not bl_mean_path.exists():
                # bl_means: (n_bl, rows, cols)
                bl_means = np.stack([temporal_mean(pixel_root / pixel_dir, CHANNEL_FILES[name])
                                     for pixel_dir in bl_rows["pixel_dir"]])
                np.save(bl_mean_path, bl_means.mean(axis=0))
            bl_mean_paths[name] = str(bl_mean_path)
            print(f"{animal}: BL {name} pedestal from {len(bl_rows)} recordings, mean level "
                  f"{np.load(bl_mean_path).mean():.0f}", flush=True)
        bl_info[animal] = (bl_mean_paths, list(bl_rows["recording_id"]))

    # Recordings whose two channel files must be exchanged, as "<unit>/<t#>".
    swapped = set()
    if args.channel_assignment_csv is not None:
        verdicts = pd.read_csv(args.channel_assignment_csv)
        wrong = verdicts[(verdicts["verdict"] == "certain") & (verdicts["agrees_with_auto"] == False)]  # noqa: E712
        swapped = set(wrong["recording_key"])
        baseline_keys = set(design.loc[design["timepoint"] == args.baseline_timepoint, "pixel_dir"])
        if swapped & baseline_keys:
            raise ValueError(f"A BL recording is mis-assigned; its pedestal would be wrong: {swapped & baseline_keys}")
    print(f"{len(swapped)} recordings get their channel files swapped back", flush=True)

    jobs = []
    for row in design.itertuples():
        folder = pixel_root / row.pixel_dir
        _, meta_path = cache_paths(folder, args.window_s, args.tag)
        if meta_path.exists() and not args.overwrite:
            continue
        bl_mean_paths, bl_recordings = bl_info[row.animal]
        jobs.append(dict(folder=str(folder), bl_mean_paths=bl_mean_paths, bl_recordings=bl_recordings,
                         is_baseline=row.timepoint == args.baseline_timepoint,
                         swap_channels=row.pixel_dir in swapped,
                         baseline_timepoint=args.baseline_timepoint, sigma=args.offset_smoothing_px,
                         window_s=args.window_s, n_median=n_median,
                         sampling_rate_hz=args.sampling_rate_hz, tag=args.tag))
    if args.recording_limit is not None:
        jobs = jobs[:args.recording_limit]
    print(f"{len(design)} recordings, {len(jobs)} to build ({n_median} frames window), "
          f"~{len(jobs) * 2980 * 76 * 87 * 2 / 1e9:.1f} GB, {args.workers} worker(s)", flush=True)
    if not jobs:
        return

    started = time.perf_counter()

    def report(i_job, result):
        elapsed = time.perf_counter() - started
        level_text = ", ".join(f"{name} {before:.0f} -> {after:.0f}"
                               for name, (before, after) in result["levels"].items())
        print(f"  [{i_job}/{len(jobs)}] {result['seconds']:.1f}s {level_text}  {result['folder']}  "
              f"(~{elapsed / i_job * (len(jobs) - i_job) / 60:.1f} min left)", flush=True)

    if args.workers == 1:
        for i_job, job in enumerate(jobs, start=1):
            report(i_job, build_one(job))
    else:
        with Pool(processes=min(args.workers, len(jobs))) as pool:
            for i_job, result in enumerate(pool.imap_unordered(build_one, jobs, chunksize=1), start=1):
                report(i_job, result)
    print(f"done: {len(jobs)} caches in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
