"""Bake ROI-editor alignment previews headlessly, in parallel.

``roi_editor.ROIEditor.bake_all_images`` already caches one 128x128 float32 preview
per session, but it runs inside the Qt startup path: serial, and only while the
editor window is open. Decoding a cohort off a USB disk that way means staring at a
progress dialog for an hour before the first box can be drawn.

Everything comes from the cohort file (``--cohort``): the recordings are the cohort's
junction view (``<cohort folder>/links``), the frames are read with its acquisition facts
(file_glob, rot90_k, frame_order), and the previews go where the new ROI editor
(``python -m wfci_gui.roi_editor``) reads them:
``<cohort folder>/roi_sets/alignment_images/<key>_odd.tif`` and ``<key>_even.tif`` (float32, 128 grid,
the mean of the first ``frames_per_parity`` odd / even frames) plus ``<key>_odd_even.json``
recording the settings, through the editor's own writer (wfci_gui.roi_editor.sources),
so a baked pair is exactly what the editor caches itself.
``--legacy-preview`` writes the old editor's single ``<key>.tif`` instead (same
``preview_image`` call as ``ROIEditor.bake_all_images``). Both use write-then-rename
``.partial`` files; already-baked images are skipped, so it is resumable and safe to rerun.

Keys are the editor's own: ``folder_key`` takes the LAST THREE path components, which is
why the junction tree ``links/<yymmdd>/<animal>/<t#>`` is read, not the raw paths, whose
session folder name would otherwise land in the key.

Run:
    conda run --no-capture-output -n wfci_gui python bake_alignment_images.py --cohort <cohort folder>/cohort.yaml
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from roi_editor import folder_key, preview_image

# The new editor's cache writer lives in src/wfci_gui (no Qt is imported by it).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Edit this section to run without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    # The cohort file, <cohort folder>/cohort.yaml (or pass --cohort). Recordings, frame
    # settings and the preview folder come from it (cohort_settings).
    "cohort_file": None,
    # Only these day folders (names under the junction view); None = every day found.
    "days": None,
    # The odd / even pair (new editor): frames of each parity averaged. Must match the
    # editor's "frames per parity" (default 10), or the editor treats the pair as stale.
    "frames_per_parity": 10,
    # True: write the old editor's single <key>.tif instead (channel_order + preview_frames).
    "legacy_preview": False,
    "channel_order": "auto",
    "preview_frames": 1,
    # 8 leaves headroom on this 20-core machine; each worker holds one frame plus the
    # two downsamples, so memory is not the limit -- the USB disk is.
    "workers": 8,
    # Report what would be decoded and exit.
    "dry_run": False,
    "prefer_cli_args": True,
}


def find_recordings(root: Path, days: list[str] | None) -> list[Path]:
    """Every ``<root>/<yymmdd>/<animal>/<t#>`` folder, sorted.

    Only ``t``-prefixed leaves are returned: an animal folder can hold other folders
    (for example a single hemodynamics image) that are not recordings.
    """
    folders = []
    for day_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if days is not None and day_dir.name not in days:
            continue
        for animal_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            for rec_dir in sorted(p for p in animal_dir.iterdir() if p.is_dir()):
                if rec_dir.name.startswith("t"):
                    folders.append(rec_dir)
    return folders


def pair_settings(args):
    """The editor settings the odd / even pair is written with (its sidecar)."""
    from wfci_gui.roi_editor.logic import EditorSettings
    return EditorSettings(pattern=args.pattern, frame_order=args.frame_order, rot90_k=int(args.rot90_k),
                          roi_set_dir=str(args.roi_set_dir), image_dir=str(args.image_dir),
                          frames_per_parity=int(args.frames_per_parity))


def is_baked(key: str, image_dir: Path, args) -> bool:
    if args.legacy_preview:
        return (image_dir / f"{key}.tif").is_file()
    from wfci_gui.roi_editor.sources import read_cache
    return read_cache(pair_settings(args), key) is not None


def bake_pair(job) -> tuple[str, float, bool]:
    """Read one recording's odd / even pair and cache it. ``(key, seconds, skipped)``."""
    from wfci_gui.roi_editor.sources import cache_page
    key, folder, args = job
    start = time.perf_counter()
    result = cache_page(key, folder, pair_settings(args))
    return key, time.perf_counter() - start, result == "already cached"


def bake_one(job: tuple[str, str, str, int, str]) -> tuple[str, float, bool]:
    """Decode one legacy preview and cache it. Returns ``(key, seconds, skipped)``.

    Module level and tuple-argued so ``multiprocessing.Pool`` can pickle it.
    """
    folder, image_path_str, channel_order, preview_frames, frame_order = job
    image_path = Path(image_path_str)
    key = image_path.stem
    if image_path.is_file():
        return key, 0.0, True

    start = time.perf_counter()
    image = preview_image(folder, channel_order, preview_frames, frame_order=frame_order)
    # Write-then-rename, exactly as bake_all_images does: a run killed mid-write
    # leaves only a .partial, which the next run ignores and overwrites.
    partial_path = image_path.with_suffix(".tif.partial")
    tifffile.imwrite(partial_path, image.astype(np.float32))
    partial_path.replace(image_path)
    return key, time.perf_counter() - start, False


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", dest="cohort_file", default=defaults["cohort_file"],
                   help="The cohort file, <cohort folder>/cohort.yaml.")
    p.add_argument("--days", nargs="*", default=defaults["days"],
                   help="Only these day folders (default: all).")
    p.add_argument("--frames-per-parity", type=int, default=defaults["frames_per_parity"],
                   help="Odd / even frames averaged into the pair (the editor's setting).")
    p.add_argument("--legacy-preview", action="store_true", default=defaults["legacy_preview"],
                   help="Write the old editor's single <key>.tif instead of the odd / even pair.")
    p.add_argument("--channel-order", default=defaults["channel_order"])
    p.add_argument("--preview-frames", type=int, default=defaults["preview_frames"])
    p.add_argument("--workers", type=int, default=defaults["workers"])
    p.add_argument("--dry-run", action="store_true", default=defaults["dry_run"])
    return p.parse_args()


def cohort_settings(cohort_file: str | None) -> dict[str, Any]:
    """Folders and frame settings from the cohort file (raises without one)."""
    from wfci_gui.logic.cohort import acquisition_value, load_script_cohort
    cohort = load_script_cohort(cohort_file)
    layout = cohort.layout
    return dict(root=str(layout.links), roi_set_dir=str(layout.reviewed_roi_sets),
                image_dir=str(layout.alignment_images),
                pattern=acquisition_value(cohort, "file_glob"),
                rot90_k=int(acquisition_value(cohort, "rot90_k")),
                frame_order=acquisition_value(cohort, "frame_order"))


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        args = parse_args(defaults=config)
    else:
        args = argparse.Namespace(
            cohort_file=config["cohort_file"],
            days=config["days"],
            frames_per_parity=config["frames_per_parity"],
            legacy_preview=bool(config["legacy_preview"]),
            channel_order=config["channel_order"],
            preview_frames=config["preview_frames"],
            workers=config["workers"],
            dry_run=config["dry_run"],
        )
    for name, value in cohort_settings(args.cohort_file).items():
        setattr(args, name, value)
    return args


def main() -> None:
    args = build_runtime_args()
    root = Path(args.root)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    folders = find_recordings(root, args.days)
    # Empty folders are not recordings and would die in the channel split with no
    # frames to read (a recording folder can exist with nothing in it).
    runnable = [f for f in folders if any(f.glob("*.tif"))]
    empty = [f for f in folders if f not in runnable]

    if args.legacy_preview:
        jobs = [(str(f), str(image_dir / f"{folder_key(f)}.tif"),
                 args.channel_order, args.preview_frames, args.frame_order) for f in runnable]
        todo = [j for j in jobs if not Path(j[1]).is_file()]
        worker, key_of = bake_one, (lambda job: Path(job[1]).stem)
    else:
        jobs = [(folder_key(f), str(f), args) for f in runnable]
        todo = [j for j in jobs if not is_baked(j[0], image_dir, args)]
        worker, key_of = bake_pair, (lambda job: job[0])
    what = ("single previews (legacy)" if args.legacy_preview
            else f"odd / even pairs, {args.frames_per_parity} frame(s) per parity")

    print(f"Root      : {root}")
    print(f"Previews  : {image_dir}  ({what})")
    print(f"Recordings: {len(jobs)} runnable, {len(jobs) - len(todo)} already cached, "
          f"{len(todo)} to decode")
    if empty:
        print(f"Skipped   : {len(empty)} empty folder(s), e.g. "
              f"{folder_key(empty[0])}")
    if args.dry_run or not todo:
        for job in todo[:20]:
            print(f"  would decode {key_of(job)}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return

    print(f"Workers   : {args.workers}\n")
    start = time.perf_counter()
    decoded = 0
    # Pool, not ProcessPoolExecutor: its context exit terminates workers, so an
    # interrupt does not leave the pool running. No BLAS here.
    with multiprocessing.Pool(processes=args.workers) as pool:
        for key, seconds, skipped in pool.imap_unordered(worker, todo):
            if skipped:
                continue
            decoded += 1
            elapsed = time.perf_counter() - start
            eta = elapsed / decoded * (len(todo) - decoded)
            print(f"[{decoded}/{len(todo)}] {key} in {seconds:.1f} s | "
                  f"elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)

    print(f"\nDone: {decoded} preview(s) decoded in "
          f"{(time.perf_counter() - start) / 60:.1f} min")
    print(f"Total cached now: {len(list(image_dir.glob('*.tif')))} in {image_dir}")


if __name__ == "__main__":
    main()
