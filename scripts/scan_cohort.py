"""Scan a cohort's raw folders from its cohort file: inventory, manifest, junction view.

Generic: everything about the cohort's folder layout is in the cohort file
and its tables:

  * ``sessions.csv``  session folder -> day (yymmdd) and timepoint; ``include`` false skips it;
  * ``animals.csv``   animal folder name -> animal (aliases are rows with the same animal);
  * the cohort file   raw_root, link_root, acquisition.file_glob / frame_order / expected_files.

Writes:
  * ``recordings.csv`` (cohort table): one row per recording folder, runnable or not, with a
    ``status`` (ok / empty / not_a_recording / frame_gap / wrong_count / parity_break). The
    user's ``use`` and ``channel_order`` edits of an earlier scan are kept;
  * the manifest (extract_cohort layout; logic/manifest.py), paths through the junction
    view when ``manifest_paths: link``;
  * the junction view ``<link_root>\\<yymmdd>\\<animal>\\<t#>`` -> raw folder (Windows only):
    the ROI editor keys a page by the last three path parts.

Run it:
    conda run --no-capture-output -n wfci_gui python -u scan_cohort.py --cohort <cohort folder>/cohort.yaml
    ... --no-walk     rebuild manifest + junctions from recordings.csv only (no raw disk read)

Errors surface (no try/except): an unplugged disk or an unknown folder name stops the scan.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfci_gui.logic.cohort import (  # noqa: E402
    Cohort, load_cohort, load_tables, parse_bool, validate_tables, write_table, write_text_atomic,
)
from wfci_gui.logic.manifest import (  # noqa: E402
    MANIFEST_DEFAULTS, link_path, manifest_rows, manifest_text,
)

# Edit this section to run without CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "cohort": None,  # path to the cohort file (or pass --cohort)
    "walk": True,               # False: no raw-disk read; manifest + junctions from recordings.csv
    "make_links": True,         # False: create no junctions
    # Read the first and last image pair of each recording to check that the bright
    # (reflectance) image keeps its parity: a dropped frame mid-recording swaps the channels.
    "check_parity": True,
    "center_fraction": 0.5,     # central part of each image compared for the parity check
    # Recording folder name -> recording: "t3" and "t3_LEDv40_..." both give t3.
    "recording_pattern": r"^(t\d+)(?:_.*)?$",
    "default_bregma_row": MANIFEST_DEFAULTS["default_bregma_row"],
    "default_bregma_col": MANIFEST_DEFAULTS["default_bregma_col"],
    "profile": MANIFEST_DEFAULTS["profile"],
    "prefer_cli_args": True,
}
# Trailing frame number of a raw file: "..._bin48_37.tif" -> 37.
FRAME_NUMBER_RE = re.compile(r"(\d+)\.[^.]+$")


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan a cohort's raw folders into recordings.csv, manifest, junctions.")
    p.add_argument("--cohort", default=defaults["cohort"], help="Cohort file (cohorts/<name>.yaml).")
    p.add_argument("--walk", action=argparse.BooleanOptionalAction, default=defaults["walk"],
                   help="Read the raw folders (--no-walk: rebuild from recordings.csv).")
    p.add_argument("--make-links", action=argparse.BooleanOptionalAction, default=defaults["make_links"])
    p.add_argument("--check-parity", action=argparse.BooleanOptionalAction, default=defaults["check_parity"])
    p.add_argument("--center-fraction", type=float, default=defaults["center_fraction"])
    p.add_argument("--recording-pattern", default=defaults["recording_pattern"])
    p.add_argument("--default-bregma-row", type=int, default=defaults["default_bregma_row"])
    p.add_argument("--default-bregma-col", type=int, default=defaults["default_bregma_col"])
    p.add_argument("--profile", default=defaults["profile"])
    return p.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{k: v for k, v in config.items() if k != "prefer_cli_args"})


# ----------------------------------------------------------------------------- walking
def matching_files(folder: Path, pattern: str) -> list[str]:
    """Names of the raw files (one scandir, no stat: stat-ing 6000 files over USB is slow)."""
    return [e.name for e in os.scandir(folder) if e.is_file() and fnmatch.fnmatch(e.name.lower(), pattern.lower())]


def ordered_files(names: list[str], frame_order: str) -> tuple[list[str], list[int] | None]:
    """Files in acquisition order, and their frame numbers when ordered by number."""
    if frame_order == "name":
        return sorted(names), None
    numbered = []
    for name in names:
        match = FRAME_NUMBER_RE.search(name)
        if not match:
            raise ValueError(f"no trailing frame number in {name!r} (frame_order: number)")
        numbered.append((int(match.group(1)), name))
    numbered.sort()
    return [n for _, n in numbered], [i for i, _ in numbered]


def bright_parity(paths: list[Path], center_fraction: float) -> int:
    """0 or 1: position within the pair of the brighter (reflectance) image."""
    import numpy as np
    import tifffile
    medians = []
    for path in paths[:2]:
        image = tifffile.imread(str(path))
        rows, cols = image.shape[-2:]
        margin_r = int(round(rows * (1 - center_fraction) / 2))
        margin_c = int(round(cols * (1 - center_fraction) / 2))
        medians.append(float(np.median(image[..., margin_r:rows - margin_r, margin_c:cols - margin_c])))
    return int(medians[1] > medians[0])


def walk(cohort: Cohort, tables: dict, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    """One row per recording folder of every included session; plus notes for the log."""
    acquisition = cohort.acquisition
    pattern = acquisition.get("file_glob") or "*.tif"
    frame_order = acquisition.get("frame_order") or "name"
    expected = acquisition.get("expected_files")
    recording_re = re.compile(args.recording_pattern)
    raw_root = Path(cohort.get("raw_root"))
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root {raw_root} does not exist (is the disk plugged in?)")
    sessions = tables["sessions"]
    animals = tables["animals"]
    animal_of = dict(zip(animals["raw_name"], animals["animal"]))
    include_animal = dict(zip(animals["raw_name"], animals["include"].map(parse_bool)))
    notes = []
    listed = set(sessions["session_folder"])
    unlisted = sorted(p.name for p in raw_root.iterdir() if p.is_dir() and p.name not in listed)
    if unlisted:
        notes.append(f"session folders not in sessions.csv (ignored): {unlisted}")
    unknown_animals: list[str] = []
    rows: list[dict[str, Any]] = []
    for session in sessions.itertuples(index=False):
        if not parse_bool(session.include):
            notes.append(f"session {session.session_folder}: include = false, skipped")
            continue
        session_dir = raw_root / session.session_folder
        if not session_dir.is_dir():
            raise FileNotFoundError(f"session folder {session_dir} listed in sessions.csv does not exist")
        for animal_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            if animal_dir.name not in animal_of:
                unknown_animals.append(f"{session.session_folder}/{animal_dir.name}")
                continue
            if not include_animal[animal_dir.name]:
                continue
            animal = animal_of[animal_dir.name]
            for recording_dir in sorted(p for p in animal_dir.iterdir() if p.is_dir()):
                match = recording_re.match(recording_dir.name)
                row = dict(day=session.day, animal=animal,
                           recording=match.group(1) if match else recording_dir.name,
                           path=str(recording_dir), n_files="", status="ok")
                if not match:
                    row["status"] = "not_a_recording"
                    rows.append(row)
                    continue
                names = matching_files(recording_dir, pattern)
                row["n_files"] = len(names)
                if len(names) < 4:
                    row["status"] = "empty"
                    rows.append(row)
                    continue
                files, numbers = ordered_files(names, frame_order)
                if numbers is not None and numbers != list(range(numbers[0], numbers[0] + len(numbers))):
                    row["status"] = "frame_gap"
                elif expected is not None and len(files) != expected:
                    row["status"] = "wrong_count"
                elif args.check_parity:
                    # The last pair starts at an even position so it is compared like the first.
                    last = (len(files) - 2) // 2 * 2
                    start = bright_parity([recording_dir / f for f in files[:2]], args.center_fraction)
                    end = bright_parity([recording_dir / f for f in files[last:last + 2]], args.center_fraction)
                    if start != end:
                        row["status"] = "parity_break"
                rows.append(row)
    if unknown_animals:
        raise ValueError("animal folders not in animals.csv (add a row for each, nothing is "
                         f"guessed): {unknown_animals}")
    duplicates = {}
    for row in rows:
        key = (row["day"], row["animal"], row["recording"])
        duplicates[key] = duplicates.get(key, 0) + 1
    clashes = [k for k, n in duplicates.items() if n > 1]
    if clashes:
        raise ValueError(f"two folders give the same recording (fix animals.csv aliases or untick a "
                         f"session): {clashes}")
    return rows, notes


def merge_user_columns(rows: list[dict[str, Any]], previous) -> list[dict[str, Any]]:
    """Keep the user's ``use`` / ``channel_order`` from the previous recordings.csv."""
    kept = {(r.day, r.animal, r.recording): (r.use, r.channel_order) for r in previous.itertuples()}
    for row in rows:
        use, order = kept.get((row["day"], row["animal"], row["recording"]),
                              ("true" if row["status"] == "ok" else "false", "auto"))
        # A recording that is no longer ok cannot stay in use.
        row["use"] = use if row["status"] == "ok" else "false"
        row["channel_order"] = order
    return rows


def make_links(cohort: Cohort, recordings) -> int:
    """Create missing junctions; refuse one that points somewhere else."""
    if os.name != "nt":
        raise NotImplementedError("junction creation is Windows-only")
    import _winapi
    made = 0
    for row in recordings.itertuples():
        if row.status != "ok" or not parse_bool(row.use):
            continue
        link, target = link_path(cohort, row.day, row.animal, row.recording), Path(row.path)
        # lexists: true for a junction even when its target disk is unplugged. A junction
        # one level up (an animal-folder view) is fine when it resolves to
        # the same folder.
        if os.path.lexists(link):
            if os.path.normcase(os.path.realpath(link)) != os.path.normcase(os.path.realpath(target)):
                raise FileExistsError(f"{link} already points at {os.path.realpath(link)}, not {target}")
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        _winapi.CreateJunction(str(target), str(link))
        made += 1
    return made


def main() -> None:
    args = build_runtime_args()
    cohort = load_cohort(args.cohort)
    tables = load_tables(cohort)
    issues = validate_tables(cohort, tables)
    for kind in ("sessions", "animals"):
        if issues[kind]:
            raise ValueError(f"{kind}.csv is invalid:\n" + "\n".join(f"  {i}" for i in issues[kind]))
    import pandas as pd
    if args.walk:
        rows, notes = walk(cohort, tables, args)
        rows = merge_user_columns(rows, tables["recordings"])
        recordings = pd.DataFrame(rows, columns=["day", "animal", "recording", "path", "n_files",
                                                 "status", "use", "channel_order"]).astype(str)
        write_table(cohort.table_path("recordings"), "recordings", recordings)
    else:
        notes = ["--no-walk: recordings.csv reused as it is"]
        recordings = tables["recordings"]
        if recordings.empty:
            raise ValueError("recordings.csv is empty: run the scan with --walk first")
    settings = dict(default_bregma_row=args.default_bregma_row,
                    default_bregma_col=args.default_bregma_col, profile=args.profile)
    manifest = manifest_rows(cohort, recordings, tables["animals"], settings)
    write_text_atomic(cohort.path_of("manifest"), manifest_text(manifest))
    made = make_links(cohort, recordings) if args.make_links and cohort.get("link_root") else 0

    print(f"Cohort    : {cohort.path}")
    print(f"Raw root  : {cohort.get('raw_root')}")
    counts = recordings["status"].value_counts().to_dict()
    print(f"Recordings: {cohort.table_path('recordings')} ({len(recordings)} folders: {counts})")
    print(f"Manifest  : {cohort.path_of('manifest')} ({len(manifest)} rows, "
          f"{sum(r['n_recordings'] for r in manifest)} recordings)")
    print(f"Junctions : {made} created under {cohort.get('link_root')}" if cohort.get("link_root")
          else "Junctions : none (no link_root)")
    for note in notes:
        print(f"  note: {note}")
    for row in recordings.itertuples():
        if row.status != "ok":
            print(f"  not runnable: {row.day}/{row.animal}/{row.recording}: {row.status} "
                  f"({row.n_files} files) {row.path}")


if __name__ == "__main__":
    main()
