"""Where the ROI editor's pages come from, and their odd / even images. No Qt.

roi_editor_spec.md 6.1 and 5.3.

Pages
-----
* a cohort file: its manifest (``manifest:``) when that file exists, else the folder scan
  of ``link_root`` / ``raw_root``; settings from the cohort;
* a manifest: one page per recording in ``recording_paths``, key ``<day>_<animal>_<t#>``
  from the row, group and ``channel_order`` per row; settings from the cohort whose
  ``manifest:`` names this file (``cohort_for_manifest``), else the caller asks the user;
* a folder (Settings dialog): every subfolder holding at least two images.

Every page is seeded exactly as in the old editor (``roi_editor._seed_layout``: its own
saved set, else the seed folder, else ``start_from``, else the profile atlas); a page
seeded from its own set also takes that set's ``channel_rectangle``. Pages are ordered
animal -> day -> t#. A page whose recording folder is missing (raw drive not mounted) is
kept when it has a cached image, else left out (and listed in ``LoadedData.notes``).

Images (``<roi_set_dir>/alignment_images``)
-------------------------------------------
``<key>_odd.tif`` / ``<key>_even.tif``: float32, 128 grid, the mean of the first
*frames per parity* odd (even) frames after ``rot90_k`` and two 0.5x box downsamples;
``<key>_odd_even.json`` records the settings they were made with, and a cache whose
sidecar does not match the current settings is treated as missing. Each file is written
as ``<name>.partial`` and renamed. The old single ``<key>.tif`` preview is only read
(the odd panel shows it, there is no even image and no decision), never written.

The rectangle decision needs the 256-grid means, read from the raw frames. A decision
from the cached 128 images (upsampled 2x) disagreed with the raw one on 27 of 496
recordings of the 2023/2024 dumps (spec 5.3 open point, 2026-11-16), so when only the
cache is available the page has no decision.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .logic import EDITOR_DEFAULTS, REPO_ROOT, EditorSettings, resolve_settings
from .model import PageModel

from channel_rectangle import parity_of_order, read_roi_set_channel_order, read_roi_set_rectangle  # noqa: E402
from roi_editor import MIN_LAMBDA_OFFSET, _safe, _seed_layout, folder_key, odd_even_means  # noqa: E402

CACHE_FOLDER = "alignment_images"
SOURCE_KINDS = ("cohort", "manifest", "folder")


# ----------------------------------------------------------------------------- data
@dataclass
class LoadedData:
    """Everything a load produced: the pages and how they were made."""

    pages: list[PageModel]
    args: argparse.Namespace
    profile_name: str
    settings: EditorSettings
    kind: str                         # cohort | manifest | folder
    path: str | None                  # the cohort / manifest file (None for a folder)
    cohort: str | None                # the cohort the settings came from, if any
    pixel_root: Path | None           # the cohort's pixel cache (verdict bar's "library" line)
    preview_only: bool = False        # no raw folder at all: every page is a cached image
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        if self.path:
            return Path(self.path).name
        return ", ".join(Path(folder).name for folder in self.settings.folder) or "folder"


def editor_args(settings: EditorSettings) -> argparse.Namespace:
    """The namespace roi_editor's seeding and the model read: the editor's geometry
    defaults (logic.EDITOR_DEFAULTS) plus the settings."""
    args = argparse.Namespace(
        **EDITOR_DEFAULTS, folder=list(settings.folder), pattern=settings.pattern,
        frame_order=settings.frame_order, rot90_k=settings.rot90_k, roi_set_dir=settings.roi_set_dir,
        roi_seed_dir=settings.roi_seed_dir, start_from=settings.start_from,
        frames_per_parity=settings.frames_per_parity, manifest=None)
    if args.lambda_offset < MIN_LAMBDA_OFFSET:
        raise ValueError(f"lambda_offset={args.lambda_offset} is below the minimum {MIN_LAMBDA_OFFSET} px")
    if args.roi_seed_dir and not Path(args.roi_seed_dir).is_dir():
        raise FileNotFoundError(f"roi_seed_dir {args.roi_seed_dir!r} does not exist.")
    if args.start_from and not Path(args.start_from).is_file():
        raise FileNotFoundError(f"start_from file {args.start_from!r} does not exist.")
    return args


def _profile(args):
    from wfci import get_profile
    return get_profile(args.profile)


# ----------------------------------------------------------------------------- seeding
def seed_origin(key: str, args) -> tuple[Path | None, bool]:
    """The file ``_seed_layout`` takes the page from and whether it is the page's own
    set (same candidate order: own saved set, seed folder, start_from); None = atlas."""
    own_path = Path(args.roi_set_dir) / f"{key}.yaml"
    seed_path = Path(args.roi_seed_dir) / f"{key}.yaml" if args.roi_seed_dir else None
    candidates = []
    if args.load_existing:
        candidates.append(own_path)
        if seed_path is not None and seed_path != own_path:
            candidates.append(seed_path)
    if args.start_from:
        candidates.append(Path(args.start_from))
    for path in candidates:
        if path.is_file():
            return path, path in (own_path, seed_path)
    return None, False


def make_page(key: str, folder: str, args, default_boxes, bregma_row: int, bregma_col: int,
              group: str = "", channel_order: str = "") -> PageModel:
    """One page, seeded exactly as the old editor seeds it, plus its own rectangle."""
    y_1, x_2, lam, boxes, own, sign, angle = _seed_layout(key, args, default_boxes, bregma_row, bregma_col)
    path, _own = seed_origin(key, args)
    rectangle = read_roi_set_rectangle(path) if own and path is not None else None
    order = read_roi_set_channel_order(path) if own and path is not None else None
    choice = parity_of_order(order) if order else None
    origin = ("own set" if own else f"seeded from {path.name}" if path is not None
              else f"seeded from the {args.profile} atlas")
    return PageModel(key=key, folder=str(folder), y_1=y_1, x_2=x_2, boxes=boxes, lambda_offset=lam,
                     lambda_sign=sign, lambda_angle=angle, has_own_set=own, group=group,
                     channel_order=channel_order, rectangle=rectangle, seed_rectangle=rectangle,
                     gcamp_choice=choice, seed_gcamp_choice=choice, seed_source=origin)


def page_order(page: PageModel) -> tuple[str, str, str]:
    """animal -> day -> t# (D24): 'c' then copies between sessions of one animal."""
    return page.animal, page.day, page.recording


def has_frames(folder: str | Path, pattern: str) -> bool:
    """A recording folder that is there and holds at least two images."""
    folder = Path(folder)
    return folder.is_dir() and len(list(itertools.islice(folder.glob(pattern), 2))) == 2


# ----------------------------------------------------------------------------- manifest
def manifest_recordings(path: str | Path) -> list[dict[str, Any]]:
    """One dict per recording of a manifest (rows with several recordings are split)."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    recordings = []
    for row in rows:
        names = [n for n in row["recordings"].split(";") if n]
        folders = [p for p in row["recording_paths"].split(";") if p]
        if len(names) != len(folders):
            raise ValueError(f"{path}: {row['day']} {row['animal']} lists {len(names)} recordings "
                             f"but {len(folders)} recording_paths")
        for name, folder in zip(names, folders):
            recordings.append(dict(
                key=_safe(f"{row['day']}_{row['animal']}_{name}"), folder=folder,
                group=row.get("group", ""), channel_order=row.get("channel_order", ""),
                bregma_row=int(row["bregma_row"]) if row.get("bregma_row") else None,
                bregma_col=int(row["bregma_col"]) if row.get("bregma_col") else None))
    return recordings


def cohort_for_manifest(manifest: str | Path, cohort_dir: str | Path = REPO_ROOT / "cohorts") -> Path | None:
    """The cohort file whose ``manifest:`` is this file; None when no cohort names it.

    Two cohorts naming the same manifest is ambiguous and raises.
    """
    import yaml
    from wfci_gui.logic.cohort import resolve_path
    target = Path(manifest).resolve()
    matches = []
    for path in sorted(Path(cohort_dir).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        value = data.get("manifest") if isinstance(data, dict) else None
        if value and resolve_path(value).resolve() == target:
            matches.append(path)
    if len(matches) > 1:
        raise ValueError(f"{Path(manifest).name} is named by {len(matches)} cohort files "
                         f"({', '.join(p.name for p in matches)}): ambiguous, open one of them instead")
    return matches[0] if matches else None


def pages_from_manifest(manifest: str | Path, settings: EditorSettings) -> tuple[list[PageModel], Any, str, bool, list[str]]:
    """Pages of a manifest: ``(pages, args, profile name, preview_only, notes)``."""
    args = editor_args(settings)
    profile = _profile(args)
    pages, notes = [], []
    for entry in manifest_recordings(manifest):
        raw = has_frames(entry["folder"], settings.pattern)
        if not raw and cache_kind(settings, entry["key"]) is None:
            notes.append(f"{entry['key']}: no recording folder and no cached image ({entry['folder']})")
            continue
        pages.append(make_page(
            entry["key"], entry["folder"], args, dict(profile.atlas),
            entry["bregma_row"] if entry["bregma_row"] is not None else args.bregma_row,
            entry["bregma_col"] if entry["bregma_col"] is not None else args.bregma_col,
            group=entry["group"], channel_order=entry["channel_order"]))
    pages.sort(key=page_order)
    preview_only = bool(pages) and not any(has_frames(p.folder, settings.pattern) for p in pages)
    return pages, args, profile.name, preview_only, notes


# ----------------------------------------------------------------------------- folders
def cohort_recordings(cohort_path: str | Path) -> list[tuple[str, str, str]]:
    """``(day, animal, t#)`` of every used recording of a cohort (recordings.csv)."""
    from wfci_gui.logic.cohort import load_cohort, load_tables, used_recordings
    used = used_recordings(load_tables(load_cohort(cohort_path)))
    return [(str(r.day), str(r.animal), str(r.recording)) for r in used.itertuples()]


def pages_from_folders(settings: EditorSettings, cohort: str | None = None
                       ) -> tuple[list[PageModel], Any, str, bool, list[str]]:
    """Pages of a folder scan: ``(pages, args, profile name, preview_only, notes)``.

    When NO root holds a recording (raw drive not mounted) and a cohort is given, the
    pages are the cohort's recordings that have a cached image.
    """
    from run_intermingle_rs import discover_recordings
    args = editor_args(settings)
    profile = _profile(args)
    found_by_root = {root: discover_recordings(root, args.pattern) if Path(root).is_dir() else []
                     for root in args.folder}
    empty = [root for root, found in found_by_root.items() if not found]
    if len(empty) == len(found_by_root) and cohort:
        pages = []
        for day, animal, recording in cohort_recordings(cohort):
            key = f"{day}_{animal}_{recording}"
            if cache_kind(settings, key) is not None:
                pages.append(make_page(key, str(Path(args.folder[0]) / day / animal / recording), args,
                                       dict(profile.atlas), args.bregma_row, args.bregma_col))
        if not pages:
            raise FileNotFoundError(
                f"No folder holding at least 2 {args.pattern!r} images found at or below {empty} "
                f"(is the raw drive mounted?), and no cached image of this cohort in "
                f"{Path(args.roi_set_dir) / CACHE_FOLDER}.")
        pages.sort(key=page_order)
        return pages, args, profile.name, True, []
    if empty:
        raise FileNotFoundError(f"No folder holding at least 2 {args.pattern!r} images found at "
                                f"or below {empty[0]!r} (is the raw drive mounted?).")
    recordings = [Path(r) for found in found_by_root.values() for r in found]
    pages = [make_page(folder_key(r), str(r), args, dict(profile.atlas), args.bregma_row, args.bregma_col)
             for r in recordings]
    pages.sort(key=page_order)
    return pages, args, profile.name, False, []


# ----------------------------------------------------------------------------- loading
def load_cohort(cohort: str | Path, flags: dict[str, Any] | None = None) -> LoadedData:
    """Open a cohort file: its manifest when it exists, else the folder scan."""
    from wfci_gui.logic.cohort import load_cohort as read_cohort
    cohort = str(cohort)
    settings = resolve_settings(cohort, dict(flags or {}))
    data = read_cohort(cohort)
    manifest = data.path_of("manifest")
    if manifest is not None and manifest.is_file():
        pages, args, profile, preview_only, notes = pages_from_manifest(manifest, settings)
    else:
        pages, args, profile, preview_only, notes = pages_from_folders(settings, cohort)
    return LoadedData(pages, args, profile, settings, "cohort", cohort, cohort, data.path_of("pixel_root"),
                      preview_only, notes)


def manifest_settings(manifest: str | Path, flags: dict[str, Any] | None = None
                      ) -> tuple[EditorSettings | None, str | None]:
    """The settings of a manifest from the cohort that names it: ``(settings, cohort)``.
    ``(None, None)`` when no cohort names it: the caller asks for them (D10)."""
    cohort = cohort_for_manifest(manifest)
    if cohort is None:
        return None, None
    return resolve_settings(str(cohort), dict(flags or {}), need_folder=False), str(cohort)


def load_manifest(manifest: str | Path, settings: EditorSettings, cohort: str | None = None) -> LoadedData:
    pages, args, profile, preview_only, notes = pages_from_manifest(manifest, settings)
    pixel_root = None
    if cohort:
        from wfci_gui.logic.cohort import load_cohort as read_cohort
        pixel_root = read_cohort(cohort).path_of("pixel_root")
    return LoadedData(pages, args, profile, settings, "manifest", str(manifest), cohort, pixel_root,
                      preview_only, notes)


def load_folders(settings: EditorSettings) -> LoadedData:
    pages, args, profile, preview_only, notes = pages_from_folders(settings)
    return LoadedData(pages, args, profile, settings, "folder", None, None, None, preview_only, notes)


def push_recent(entries: list[tuple[str, str]], kind: str, path: str, limit: int = 10) -> list[tuple[str, str]]:
    """The Recent list with (kind, path) first, without duplicates, at most ``limit``."""
    if kind not in ("cohort", "manifest"):
        raise ValueError(f"Recent entries are cohorts or manifests, not {kind!r}")
    entry = (kind, str(Path(path).resolve()))
    return [entry] + [item for item in entries if item != entry][:limit - 1]


# ----------------------------------------------------------------------------- images
def cache_paths(roi_set_dir: str | Path, key: str) -> dict[str, Path]:
    folder = Path(roi_set_dir) / CACHE_FOLDER
    return {"odd": folder / f"{key}_odd.tif", "even": folder / f"{key}_even.tif",
            "sidecar": folder / f"{key}_odd_even.json", "legacy": folder / f"{key}.tif"}


def cache_meta(settings: EditorSettings) -> dict[str, Any]:
    """What a cached pair must have been made with to be used."""
    return {"frames_per_parity": int(settings.frames_per_parity), "pattern": settings.pattern,
            "rot90_k": int(settings.rot90_k), "frame_order": settings.frame_order}


def read_cache(settings: EditorSettings, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The cached (odd, even) 128 images; None when missing or made with other settings."""
    import tifffile
    paths = cache_paths(settings.roi_set_dir, key)
    if not all(paths[name].is_file() for name in ("odd", "even", "sidecar")):
        return None
    if json.loads(paths["sidecar"].read_text(encoding="utf-8")) != cache_meta(settings):
        return None
    return (np.asarray(tifffile.imread(paths["odd"]), dtype=np.float64),
            np.asarray(tifffile.imread(paths["even"]), dtype=np.float64))


def cache_kind(settings: EditorSettings, key: str) -> str | None:
    """'cache' (a valid pair), 'legacy' (only the old <key>.tif) or None."""
    if read_cache(settings, key) is not None:
        return "cache"
    return "legacy" if cache_paths(settings.roi_set_dir, key)["legacy"].is_file() else None


def write_cache(roi_set_dir: str | Path, key: str, odd: np.ndarray, even: np.ndarray,
                meta: dict[str, Any]) -> None:
    """Write the pair, then the sidecar (its presence means both images are complete).
    Each through ``.partial`` + rename, so an interrupted write leaves nothing usable."""
    import tifffile
    paths = cache_paths(roi_set_dir, key)
    paths["odd"].parent.mkdir(parents=True, exist_ok=True)
    for name, image in (("odd", odd), ("even", even)):
        partial = paths[name].with_suffix(".tif.partial")
        tifffile.imwrite(partial, np.asarray(image, dtype=np.float32))
        partial.replace(paths[name])
    partial = paths["sidecar"].with_suffix(".json.partial")
    partial.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    partial.replace(paths["sidecar"])


def to_128(image_256: np.ndarray) -> np.ndarray:
    """The 256-grid mean on the 128 grid: the pipeline's second 0.5x box downsample."""
    from wfci import imresize_box
    return imresize_box(image_256, 0.5)


def read_raw(folder: str | Path, settings: EditorSettings) -> tuple[np.ndarray, np.ndarray]:
    """(odd, even) means of the first frames_per_parity frames, 256 grid."""
    return odd_even_means(folder, int(settings.frames_per_parity), pattern=settings.pattern,
                          rot90_k=settings.rot90_k, frame_order=settings.frame_order)


def load_images(page: PageModel, settings: EditorSettings) -> None:
    """Fill a page's images the first time it is opened.

    Raw frames when the folder is there (the decision needs their 256 means; the pair is
    cached on the way), else the cached pair, else the old single preview.
    """
    import tifffile
    if has_frames(page.folder, settings.pattern):
        odd_256, even_256 = read_raw(page.folder, settings)
        page.odd_even = (odd_256, even_256)
        page.odd_even_128 = (to_128(odd_256), to_128(even_256))
        page.image_source = "raw"
        if read_cache(settings, page.key) is None:
            write_cache(settings.roi_set_dir, page.key, *page.odd_even_128, cache_meta(settings))
    else:
        cached = read_cache(settings, page.key)
        legacy = cache_paths(settings.roi_set_dir, page.key)["legacy"]
        if cached is not None:
            page.odd_even_128 = cached
            page.image_source = "cache"
        elif legacy.is_file():
            page.odd_even_128 = (np.asarray(tifffile.imread(legacy), dtype=np.float64), None)
            page.image_source = "legacy"
        else:
            raise FileNotFoundError(f"{page.key}: no recording folder ({page.folder}) and no cached "
                                    f"image in {legacy.parent}")
    page.image = page.odd_even_128[0]


def cache_page(key: str, folder: str, settings: EditorSettings) -> str:
    """Cache all: write one page's pair from its raw frames. Returns what happened."""
    if read_cache(settings, key) is not None:
        return "already cached"
    if not has_frames(folder, settings.pattern):
        return "no recording folder"
    odd_256, even_256 = read_raw(folder, settings)
    write_cache(settings.roi_set_dir, key, to_128(odd_256), to_128(even_256), cache_meta(settings))
    return "cached"


def library_order(pixel_root: Path | None, key: str) -> str | None:
    """The pixel cache's channel_order of a recording, None when it has no labelled cache."""
    if pixel_root is None:
        return None
    unit, recording = key.rsplit("_", 1)
    path = Path(pixel_root) / unit / recording / "pixels_meta_full.npz"
    if not path.is_file():
        return None
    return str(np.load(path)["channel_order"])
