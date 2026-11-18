"""Write a small synthetic cohort into a workspace, to try every step on.

The example cohort is made up: a few animals, two sessions (timepoints BL and 3D), short
interleaved recordings of small frames. Every step of the GUI runs on it, it is the
cohort the tests and the acceptance run use, and a new lab member can practise on it.

What it writes, under ``<workspace>/<name>/``:
    cohort.yaml           acquisition facts filled in, not yet confirmed (step 2 confirms)
    tables/               sessions.csv, animals.csv, recordings.csv (empty until step 1)
    raw/<session>/<animal folder>/t<k>/frame_00001.tif ...
                          the recordings: odd frames GCaMP (dim, with the activity), even
                          frames reflectance (bright). Each TIFF header holds the time between
                          two frames (``Time_From_Last``), which step 2 reads.

The activity is built so the analysis has something to find: regions that belong together
(left and right motor areas, the somatosensory areas, the visual areas) share a slow random
signal, and at 3D the motor signal is weaker.

``--place-roi-sets`` stands in for hand drawing in step 3: after the scan (step 1), it
writes one reviewed ROI set per recording (the cortex22 atlas at the default Bregma) into
the cohort's ``roi_sets/reviewed``, as the ROI editor's Save would.

Run:
    conda run --no-capture-output -n wfci_gui python make_example_cohort.py --workspace D:\\WFCI
    ... make_example_cohort.py --place-roi-sets --cohort D:\\WFCI\\example\\cohort.yaml
In the GUI: Cohort > Create example cohort (it opens the new cohort).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfci_gui import APP_VERSION  # noqa: E402
from wfci_gui.logic.cohort import load_script_cohort, load_tables, used_recordings  # noqa: E402
from wfci_gui.logic.layout import COHORT_FILE_NAME  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[1]
ATLAS = CODE_ROOT / "atlas" / "cortex22_roi_set.yaml"

# Edit this section to run without CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "workspace": None,          # the workspace folder; the cohort goes to <workspace>/<name>
    "name": "example",
    "animals": 3,               # A1, A2, ...
    "recordings": 1,            # recordings per animal and session (t1, t2, ...)
    "seconds": 12.0,            # length of each recording
    "sampling_rate_hz": 5.0,    # frames per second per channel (the two channels interleave)
    "seed": 0,                  # random seed: the same arguments give the same files
    "place_roi_sets": False,    # True: write the reviewed ROI sets of an existing cohort
    "cohort_file": None,        # the cohort --place-roi-sets works on
    "prefer_cli_args": True,
}

# The two sessions: raw folder name -> (day, timepoint).
SESSIONS = {"session_1": ("200101", "BL"), "session_2": ("200104", "3D")}
FRAME_SHAPE = (512, 512)        # raw frames; the pipeline's two 0.5x downsamples give the 128 grid
# Bregma as the manifest and the ROI sets give it: in 256-grid units (2 x the 128 grid),
# the manifest default. In raw pixels it is twice that.
BREGMA = (121, 134)
RAW_PER_GRID = 4                # raw pixels per 128-grid pixel
TRIM_S = 2.0                    # onset trim per channel
LAMBDA_OFFSET = 55              # Bregma-Lambda distance the atlas is drawn at (no rescale)
# Regions that share one activity signal, by the start of their ROI label.
REGION_GROUPS = {"motor": ("M1", "M2"), "sensory": ("BFD", "Tr", "FL", "HL"),
                 "visual": ("V1",), "retrosplenial": ("RS",)}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a small synthetic cohort into a workspace.")
    p.add_argument("--workspace", default=defaults["workspace"], help="The workspace folder.")
    p.add_argument("--name", default=defaults["name"], help="Cohort name (its folder name).")
    p.add_argument("--animals", type=int, default=defaults["animals"])
    p.add_argument("--recordings", type=int, default=defaults["recordings"])
    p.add_argument("--seconds", type=float, default=defaults["seconds"])
    p.add_argument("--sampling-rate-hz", type=float, default=defaults["sampling_rate_hz"])
    p.add_argument("--seed", type=int, default=defaults["seed"])
    p.add_argument("--place-roi-sets", action="store_true", default=defaults["place_roi_sets"],
                   help="Write one reviewed ROI set per recording of an existing, scanned cohort.")
    p.add_argument("--cohort", dest="cohort_file", default=defaults["cohort_file"],
                   help="With --place-roi-sets: the cohort file.")
    return p.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{k: v for k, v in config.items() if k != "prefer_cli_args"})


# ----------------------------------------------------------------------------- images
def atlas_data() -> dict[str, Any]:
    return yaml.safe_load(ATLAS.read_text(encoding="utf-8"))


def region_maps(shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(head mask, {region group: weight map}) in raw pixels.

    Each weight map is a sum of Gaussian blobs at the centres of that group's atlas boxes
    (box offsets are 128-grid rows/columns from Bregma).
    """
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    bregma_row, bregma_col = 2 * BREGMA[0], 2 * BREGMA[1]          # raw pixels
    mask = (((rows - bregma_row - 40) / 220) ** 2 + ((cols - bregma_col) / 210) ** 2) <= 1.0
    maps = {group: np.zeros(shape) for group in REGION_GROUPS}
    for label, box in atlas_data()["boxes"].items():
        group = next((g for g, starts in REGION_GROUPS.items() if label.startswith(starts)), None)
        if group is None:
            continue
        centre_row = bregma_row + RAW_PER_GRID * (box["row_start"] + box["row_end"]) / 2
        centre_col = bregma_col + RAW_PER_GRID * (box["col_start"] + box["col_end"]) / 2
        maps[group] += np.exp(-((rows - centre_row) ** 2 + (cols - centre_col) ** 2) / (2 * 18.0 ** 2))
    return mask, {group: np.clip(m, 0, 1) * mask for group, m in maps.items()}


def slow_signal(rng: np.random.Generator, n: int, smoothing: float = 0.8) -> np.ndarray:
    """A slow random signal (AR(1)), unit standard deviation."""
    values = np.zeros(n)
    noise = rng.normal(size=n)
    for i in range(1, n):
        values[i] = smoothing * values[i - 1] + noise[i]
    return (values - values.mean()) / values.std()


def recording_frames(rng: np.random.Generator, n_per_channel: int, timepoint: str,
                     mask: np.ndarray, maps: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Interleaved frames: GCaMP, reflectance, GCaMP, ... (uint16)."""
    gcamp_base = np.where(mask, 2000.0, 300.0)
    reflectance_base = np.where(mask, 6000.0, 500.0)
    shared = slow_signal(rng, n_per_channel)
    amplitude = {"motor": 0.06 if timepoint == "BL" else 0.02, "sensory": 0.05,
                 "visual": 0.05, "retrosplenial": 0.04}
    signals = {group: slow_signal(rng, n_per_channel) for group in maps}
    frames = []
    for t in range(n_per_channel):
        activity = 0.01 * shared[t] + sum(amplitude[g] * maps[g] * signals[g][t] for g in maps)
        # Little pixel noise: the 4x4 downsample averages it away, and the files compress.
        gcamp = gcamp_base * (1.0 + activity) + rng.normal(0, 4, FRAME_SHAPE)
        reflectance = reflectance_base * (1.0 + 0.004 * shared[t]) + rng.normal(0, 4, FRAME_SHAPE)
        frames += [np.clip(gcamp, 0, 65535).astype(np.uint16),
                   np.clip(reflectance, 0, 65535).astype(np.uint16)]
    return frames


def write_recording(folder: Path, frames: list[np.ndarray], sampling_rate_hz: float) -> None:
    """One TIFF per frame, zero-padded numbers, the frame interval in each header."""
    folder.mkdir(parents=True)
    interval = 1.0 / (2.0 * sampling_rate_hz)     # two interleaved channels
    description = f"Time_From_Last = 0:00:{interval:07.4f}"
    for index, frame in enumerate(frames, start=1):
        tifffile.imwrite(folder / f"frame_{index:05d}.tif", frame, description=description,
                         metadata=None, compression="zlib")


# ----------------------------------------------------------------------------- cohort
def cohort_data(name: str, n_files: int, sampling_rate_hz: float) -> dict[str, Any]:
    return {
        "app_version": APP_VERSION,
        "name": name,
        "raw_root": "raw",
        "timepoints": [timepoint for _, timepoint in SESSIONS.values()],
        "acquisition": {
            "sampling_rate_hz": sampling_rate_hz, "frame_order": "name", "trim_s": TRIM_S,
            "rot90_k": 0, "grid": [128, 128], "file_glob": "*.tif", "expected_files": n_files,
            "confirmed": False,
        },
        "exclusions": [],
        "exclusions_confirmed": False,
        "overrides": {},
    }


def make_cohort(args: argparse.Namespace) -> Path:
    if not args.workspace:
        raise ValueError("set workspace or pass --workspace")
    folder = Path(args.workspace) / args.name
    if folder.exists():
        raise FileExistsError(f"{folder} already exists: choose another --name or delete it first")
    n_per_channel = int(round(args.seconds * args.sampling_rate_hz))
    if n_per_channel * 1.0 / args.sampling_rate_hz <= 2 * TRIM_S:
        raise ValueError(f"--seconds must be longer than {2 * TRIM_S:g} s (the onset trim is {TRIM_S:g} s)")
    rng = np.random.default_rng(args.seed)
    mask, maps = region_maps(FRAME_SHAPE)
    animals = [f"A{i}" for i in range(1, args.animals + 1)]
    for session, (_, timepoint) in SESSIONS.items():
        for animal in animals:
            for k in range(1, args.recordings + 1):
                frames = recording_frames(rng, n_per_channel, timepoint, mask, maps)
                write_recording(folder / "raw" / session / f"mouse_{animal.lower()}" / f"t{k}",
                                frames, args.sampling_rate_hz)
    tables = folder / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame([dict(session_folder=s, day=d, timepoint=t, day_zero="false", include="true")
                  for s, (d, t) in SESSIONS.items()]).to_csv(tables / "sessions.csv", index=False)
    pd.DataFrame([dict(raw_name=f"mouse_{a.lower()}", animal=a, group="example",
                       mouse_line="example_line", include="true")
                  for a in animals]).to_csv(tables / "animals.csv", index=False)
    (tables / "recordings.csv").write_text(
        "day,animal,recording,path,n_files,status,use,channel_order\n", encoding="utf-8")
    path = folder / COHORT_FILE_NAME
    data = cohort_data(args.name, 2 * n_per_channel, args.sampling_rate_hz)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    n_recordings = len(SESSIONS) * len(animals) * args.recordings
    print(f"Example cohort: {path}")
    print(f"  {len(animals)} animals x {len(SESSIONS)} sessions x {args.recordings} recording(s) = "
          f"{n_recordings} recordings of {2 * n_per_channel} frames ({args.seconds:g} s at "
          f"{args.sampling_rate_hz:g} Hz per channel)")
    return path


def place_roi_sets(cohort_file: str | Path | None) -> list[Path]:
    """One reviewed ROI set per used recording: the atlas at the default Bregma."""
    from roi_editor import save_roi_set
    from wfci import load_atlas
    cohort = load_script_cohort(cohort_file)
    used = used_recordings(load_tables(cohort))
    if used.empty:
        raise ValueError("no runnable recordings in recordings.csv: run the scan (step 1) first")
    atlas = load_atlas(ATLAS)
    written = []
    for row in used.itertuples():
        key = f"{row.day}_{row.animal}_{row.recording}"
        written.append(save_roi_set(
            cohort.layout.reviewed_roi_sets / f"{key}.yaml", dict(atlas), BREGMA[0], BREGMA[1],
            (128, 128), key, "placed by make_example_cohort.py (the atlas at the default Bregma)",
            lambda_offset=LAMBDA_OFFSET))
    print(f"Placed {len(written)} ROI set(s) in {cohort.layout.reviewed_roi_sets}")
    return written


def main() -> None:
    args = build_runtime_args()
    if args.place_roi_sets:
        place_roi_sets(args.cohort_file)
    else:
        make_cohort(args)


if __name__ == "__main__":
    main()
