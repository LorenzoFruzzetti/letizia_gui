"""Cache the per-pixel median-baseline dF/F of every pixel dump, once.

For every `<date>_<animal>/<t#>/` of the cohort's pixel dumps (its `pixel_data/`, or the
cohort file's `pixel_root`) written by extract_cohort.py --save-data, computes over the
WHOLE saved crop

    dF/F(t, p) = (F(t, p) / Fbar(t, p)) / (R(t, p) / Rbar(t, p)) - 1

where Fbar/Rbar are each pixel's centred running median over `window_s`
seconds (`min_periods=1` at the ends). This is exactly the per-pixel quantity
the pixel event detector (`--signal-mode median_dff`) computes before averaging
inside a box: averaging the cached volume over a box reproduces its median-dF/F
ROI trace bit for bit. It is a ratio,
not a percentage.

It depends only on F, R and the baseline window, never on detection
parameters, so it is computed once and read many times. The window is part of
the file name, so a cache built with another window is never read by mistake:

    pixels_median_dff_<window>s_full.npy   (n_time, rows, cols), float16 by default
    pixels_median_dff_<window>s_meta.json  provenance; written LAST, it is the done-marker

float16 vs float32: float16 is half the size, and its rounding error is a small
fraction of a pixel's temporal SD, so ROI traces and thresholded maps are
practically unchanged. float32 is effectively exact. A median-dF/F ratio stays
far inside float16's +-65504, and the cast is range-checked anyway.

Interruption is safe: the volume is written to a temporary name and renamed,
and the meta JSON is written after it, so a recording without its JSON is
simply recomputed on the next run.

The cache is written next to each dump. The frame rate is the cohort's
acquisition.sampling_rate_hz.

Examples (in the wfci_gui environment; C is <cohort folder>/cohort.yaml):
    python cache_median_dff.py --cohort C
    python cache_median_dff.py --cohort C --workers 1 --recording-limit 2
    python cache_median_dff.py --cohort C --dtype float32 --window-s 30

Edit RUN_CONFIG below to run directly from the editor without CLI arguments.
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

from pixel_helpers import median_window_frames, running_median

# Edit this section to run the cache build without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    # The cohort file, <cohort folder>/cohort.yaml (or pass --cohort): its pixel dumps
    # (<date>_<animal>/<t#>/pixels_meta_full.npz) and its frame rate.
    "cohort_file": None,
    "window_s": 20.0,  # Running-median baseline, seconds (20 s = 201 frames at 10 Hz).
    "dtype": "float16",  # "float16" (half the disk) or "float32" (exact); see the docstring.
    # Parallel processes; 1 runs serially in this process. About 1.1 GB RAM per worker;
    # more workers help up to about the core count, and 8 leaves a desktop usable meanwhile.
    "workers": 8,
    "recording_limit": None,  # None: all dumps; positive integer: only the first N (smoke test).
    "overwrite": False,  # True: recompute even when a finished cache exists.
    "prefer_cli_args": True,  # False: ignore CLI arguments and use this block.
}

DTYPES = ("float16", "float32")


def cache_paths(output_folder: Path, window_s: float) -> tuple[Path, Path]:
    """Volume and meta paths; the window is in the name so caches never mix."""
    stem = f"pixels_median_dff_{window_s:g}s"
    return output_folder / f"{stem}_full.npy", output_folder / f"{stem}_meta.json"


def median_dff_volume(fluorescence: np.ndarray, reflectance: np.ndarray, n_median: int) -> np.ndarray:
    """(F/Fbar)/(R/Rbar) - 1 for every pixel of (n_time, rows, cols) volumes, float64.

    The operation order matches `median_dff_roi_trace` exactly, so the two agree
    bit for bit. Each channel's baseline is freed before the next is built, which
    keeps the peak near four float64 volumes (~630 MB for a 2980x76x87 dump).
    """
    if fluorescence.shape != reflectance.shape or fluorescence.ndim != 3:
        raise ValueError("Fluorescence and reflectance must be matching 3-D volumes")
    n_time, n_row, n_col = fluorescence.shape
    # f_ratio, r_ratio: (n_time, n_pixels). Explicit copies: they are divided in
    # place below, and asarray would hand back the caller's own float64 array.
    f_ratio = np.array(fluorescence, dtype=np.float64).reshape(n_time, -1)
    r_ratio = np.array(reflectance, dtype=np.float64).reshape(n_time, -1)
    if not np.isfinite(f_ratio).all() or not np.isfinite(r_ratio).all() or np.any(r_ratio <= 0):
        raise ValueError("Non-finite fluorescence/reflectance or nonpositive reflectance")
    for channel in (f_ratio, r_ratio):
        baseline = running_median(channel, n_median)  # (n_time, n_pixels)
        if np.any(baseline <= 0):
            raise ValueError("Running-median baseline is nonpositive; cannot form F/Fbar")
        np.divide(channel, baseline, out=channel)
        del baseline
    dff = f_ratio / r_ratio - 1.0
    return dff.reshape(n_time, n_row, n_col)


def build_one(job: dict[str, Any]) -> dict[str, Any]:
    """Compute and write one recording's cache. Top-level so worker processes can import it."""
    started = time.perf_counter()
    folder = Path(job["folder"])
    volume_path, meta_path = cache_paths(Path(job["output_folder"]), job["window_s"])
    volume_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(folder / "pixels_meta_full.npz", allow_pickle=False) as data:
        n_time, n_written = int(data["n_time"]), int(data["n_written"])
        axis_order = str(data["axis_order"])
    if axis_order != "time,y,x" or n_written != n_time or n_time <= 0:
        raise ValueError(f"Unsupported or incomplete pixel dump: {folder}")
    fluorescence = np.load(folder / "pixels_f_gcamp_full.npy", mmap_mode="r")
    reflectance = np.load(folder / "pixels_f_emo_full.npy", mmap_mode="r")
    if fluorescence.shape[0] != n_time:
        raise ValueError(f"Volume length disagrees with metadata: {folder}")

    dff = median_dff_volume(fluorescence, reflectance, job["n_median"])
    # Refuse a cast that would turn values into inf (float16 max is 65504).
    dtype_max = float(np.finfo(job["dtype"]).max)
    peak = float(np.abs(dff).max())
    if peak > dtype_max:
        raise ValueError(f"|dF/F| reaches {peak:g}, beyond {job['dtype']} ({dtype_max:g}): {folder}")
    stored = dff.astype(job["dtype"])
    storage_error = float(np.abs(stored.astype(np.float64) - dff).max())

    # Temporary name, then rename: a killed job never leaves a truncated volume under the real name.
    temporary_path = volume_path.with_name(volume_path.stem + ".partial.npy")
    np.save(temporary_path, stored)
    os.replace(temporary_path, volume_path)
    meta = dict(
        formula="(F/Fbar)/(R/Rbar) - 1 per pixel; Fbar/Rbar = centred running median, min_periods=1",
        unit="ratio (x100 for percent)",
        source=str(folder.resolve()),
        inputs=["pixels_f_gcamp_full.npy", "pixels_f_emo_full.npy"],
        window_s=job["window_s"], window_frames=job["n_median"],
        sampling_rate_hz=job["sampling_rate_hz"],
        shape=list(stored.shape), dtype=job["dtype"], axis_order="time,y,x",
        value_min=float(dff.min()), value_max=float(dff.max()),
        max_abs_storage_error=storage_error,
        seconds=round(time.perf_counter() - started, 2),
    )
    # Written last: its presence marks the cache as complete.
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dict(folder=str(folder), seconds=meta["seconds"], error=storage_error)


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", dest="cohort_file", default=defaults["cohort_file"],
                        help="The cohort file, <cohort folder>/cohort.yaml")
    parser.add_argument("--window-s", type=float, default=defaults["window_s"])
    parser.add_argument("--dtype", choices=DTYPES, default=defaults["dtype"])
    parser.add_argument("--workers", type=int, default=defaults["workers"],
                        help="Parallel processes; 1 runs serially in the main process")
    parser.add_argument("--recording-limit", type=int, default=defaults["recording_limit"])
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=defaults["overwrite"])
    return parser.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        args = parse_args(defaults=config)
    else:
        args = argparse.Namespace(**{key: value for key, value in config.items() if key != "prefer_cli_args"})
    # The dumps and the frame rate come from the cohort file.
    from wfci_gui.logic.cohort import acquisition_value, load_script_cohort
    cohort = load_script_cohort(args.cohort_file)
    args.pixel_root = cohort.layout.pixel_data
    args.sampling_rate_hz = float(acquisition_value(cohort, "sampling_rate_hz"))
    return args


def main():
    args = build_runtime_args()
    if args.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.recording_limit is not None and args.recording_limit <= 0:
        raise ValueError("recording_limit must be positive")
    pixel_root = Path(args.pixel_root)
    n_median = median_window_frames(args.window_s, args.sampling_rate_hz)

    folders = sorted(path.parent for path in pixel_root.glob("*/*/pixels_meta_full.npz"))
    if not folders:
        raise FileNotFoundError(f"No full pixel dumps found under {pixel_root}")
    if args.recording_limit is not None:
        folders = folders[:args.recording_limit]

    # One job per recording; the output folder mirrors <unit>/<t#>/.
    jobs = []
    for folder in folders:
        output_folder = folder
        _, meta_path = cache_paths(output_folder, args.window_s)
        if meta_path.exists() and not args.overwrite:
            continue
        jobs.append(dict(folder=str(folder), output_folder=str(output_folder),
                         window_s=args.window_s, n_median=n_median,
                         sampling_rate_hz=args.sampling_rate_hz, dtype=args.dtype))
    n_bytes = np.dtype(args.dtype).itemsize
    print(f"{len(folders)} dumps, {len(folders) - len(jobs)} already cached, {len(jobs)} to build "
          f"({args.window_s:g} s = {n_median} frames, {args.dtype}, "
          f"~{len(jobs) * 2980 * 76 * 87 * n_bytes / 1e9:.1f} GB), {args.workers} worker(s)", flush=True)
    if not jobs:
        return

    started = time.perf_counter()
    worst_error = 0.0

    def report(i_job, result):
        nonlocal worst_error
        worst_error = max(worst_error, result["error"])
        elapsed = time.perf_counter() - started
        remaining_min = elapsed / i_job * (len(jobs) - i_job) / 60
        print(f"  [{i_job}/{len(jobs)}] {result['seconds']:.1f}s {result['folder']}  "
              f"(elapsed {elapsed / 60:.1f} min, ~{remaining_min:.1f} min left)", flush=True)

    if args.workers == 1:
        for i_job, job in enumerate(jobs, start=1):
            report(i_job, build_one(job))
    else:
        # Pool's context exit calls terminate(), so Ctrl+C or a worker error stops
        # every worker instead of waiting for the queue.
        with Pool(processes=min(args.workers, len(jobs))) as pool:
            for i_job, result in enumerate(pool.imap_unordered(build_one, jobs, chunksize=1), start=1):
                report(i_job, result)
    print(f"done: {len(jobs)} caches in {(time.perf_counter() - started) / 60:.1f} min, "
          f"worst storage error {worst_error:.2e}")


if __name__ == "__main__":
    main()
