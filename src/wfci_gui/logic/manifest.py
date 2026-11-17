"""Build the extract_cohort manifest from a cohort's recordings table.

Shared by ``scan_cohort.py`` (which writes it) and the step-1 status check (which
compares it with the manifest on disk), so "the manifest matches the tables" means
exactly "scan_cohort.py would write the same bytes".

Layout: the same columns as every existing manifest (scan_botox_dataset.FIELDS), one
row per (day, animal) unit, sorted by day then animal, recordings sorted by number,
CRLF line endings (what pandas and the csv module write on Windows, so the existing
2023/2024 manifests are reproduced byte for byte).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pandas as pd

from .cohort import Cohort, parse_bool

MANIFEST_FIELDS = [
    "day", "animal", "group", "n_recordings", "recordings", "total_frames",
    "bregma_row", "bregma_col", "profile", "channel_order", "animal_path",
    "recording_paths",
]
# Pipeline placeholders every existing scanner writes. Bregma is overridden by each
# recording's own ROI set (extract_cohort: the file's Bregma wins, CLAUDE.md 9.4).
MANIFEST_DEFAULTS = {
    "default_bregma_row": 121,
    "default_bregma_col": 134,
    "profile": "cerebellar_rs",
}


def recording_number(name: str) -> int:
    """``t12`` -> 12, for numeric ordering (a text sort puts t10 before t2)."""
    return int(name[1:])


def link_path(cohort: Cohort, day: str, animal: str, recording: str | None = None) -> Path:
    """Junction-view path ``<link_root>\\<yymmdd>\\<animal>[\\<t#>]``."""
    path = Path(cohort.get("link_root")) / day / animal
    return path / recording if recording else path


def manifest_rows(cohort: Cohort, recordings: pd.DataFrame, animals: pd.DataFrame,
                  settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per unit of recordings with status ok and use true.

    A unit whose recordings need different explicit channel orders is written as one
    row per recording, the only way the manifest can express a per-recording order.
    """
    settings = {**MANIFEST_DEFAULTS, **(settings or {})}
    groups = dict(zip(animals["animal"], animals["group"])) if not animals.empty else {}
    style = cohort.get("manifest_paths")
    ok = recordings[(recordings["status"] == "ok") & recordings["use"].map(parse_bool)]
    units: dict[tuple[str, str], list[Any]] = {}
    for row in ok.itertuples(index=False):
        units.setdefault((row.day, row.animal), []).append(row)
    rows = []
    for (day, animal), recs in sorted(units.items()):
        recs = sorted(recs, key=lambda r: recording_number(r.recording))
        orders = {r.channel_order for r in recs}
        batches = [recs] if len(orders) == 1 else [[r] for r in recs]
        for batch in batches:
            if style == "link":
                paths = [str(link_path(cohort, day, animal, r.recording)) for r in batch]
                animal_path = str(link_path(cohort, day, animal))
            else:
                paths = [r.path for r in batch]
                animal_path = str(Path(batch[0].path).parent)
            rows.append(dict(
                day=day, animal=animal, group=groups.get(animal, "NA"), n_recordings=len(batch),
                recordings=";".join(r.recording for r in batch),
                total_frames=sum(int(r.n_files) for r in batch),
                bregma_row=settings["default_bregma_row"], bregma_col=settings["default_bregma_col"],
                profile=settings["profile"], channel_order=batch[0].channel_order,
                animal_path=animal_path, recording_paths=";".join(paths)))
    return rows


def manifest_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def manifest_matches(cohort: Cohort, tables: dict[str, pd.DataFrame],
                     settings: dict[str, Any] | None = None) -> tuple[bool, str]:
    """(matches, reason): is the manifest on disk what the tables would produce?"""
    path = cohort.path_of("manifest")
    if not path.exists():
        return False, f"no manifest at {path}"
    if tables["recordings"].empty:
        return False, "recordings.csv is empty"
    expected = manifest_text(manifest_rows(cohort, tables["recordings"], tables["animals"], settings))
    on_disk = path.read_bytes().decode("utf-8")
    if on_disk != expected:
        return False, "the manifest differs from what the tables produce (rerun the scan)"
    return True, ""
