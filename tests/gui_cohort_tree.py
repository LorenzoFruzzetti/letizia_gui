"""A small fake cohort on disk for the wfci_gui tests, in the per-cohort workspace layout.

raw/SESSION_A/{m1_a,m2_a}/{t1,t2}  and  raw/SESSION_B/{m1_b,m2_b}/{t1,t2}: 6 tiny TIFFs
each, numbered 1..6, the bright (reflectance) image second in every pair. ``m1_a`` and
``m1_b`` are aliases of one animal, M1 (and likewise M2).

The cohort is ``workspace/fake/cohort.yaml`` with ``tables/`` beside it; every other
location (manifest, links, ROI sets, pixel data, outputs) follows from that folder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROI_SET = REPO_ROOT / "atlas" / "cortex22_roi_set.yaml"
SESSIONS = {"SESSION_A": ("200101", "BL"), "SESSION_B": ("200104", "3D")}
ANIMAL_FOLDERS = {"m1_a": "M1", "m2_a": "M2", "m1_b": "M1", "m2_b": "M2"}


def write_raw(root: Path, n_files: int = 6) -> None:
    for session, (_, _) in SESSIONS.items():
        suffix = "a" if session.endswith("A") else "b"
        for folder in (f"m1_{suffix}", f"m2_{suffix}"):
            for recording in ("t1", "t2"):
                target = root / session / folder / recording
                target.mkdir(parents=True)
                for i in range(1, n_files + 1):
                    level = 3000 if i % 2 else 9000   # odd frames dim (GCaMP), even bright
                    tifffile.imwrite(target / f"frame_{i}.tif", np.full((8, 8), level, np.uint16))


def make_cohort(tmp_path: Path, **changes) -> Path:
    """Write ``tmp_path/workspace/fake/cohort.yaml`` and its tables; return the cohort file.

    ``changes`` replace top-level keys of the cohort file (e.g. ``pixel_root=...``).
    """
    raw = tmp_path / "raw"
    write_raw(raw)
    folder = tmp_path / "workspace" / "fake"
    tables = folder / "tables"
    tables.mkdir(parents=True)
    data = {
        "name": "fake",
        "raw_root": str(raw),
        "timepoints": ["BL", "3D"],
        "acquisition": {"sampling_rate_hz": 25, "frame_order": "number", "trim_s": 2.0, "rot90_k": 0,
                        "grid": [128, 128], "file_glob": "*.tif", "expected_files": 6, "confirmed": True},
        "exclusions": [],
        "exclusions_confirmed": False,
        "overrides": {},
        **changes,
    }
    path = folder / "cohort.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    pd.DataFrame([dict(session_folder=s, day=d, timepoint=t, day_zero="false", include="true")
                  for s, (d, t) in SESSIONS.items()]).to_csv(tables / "sessions.csv", index=False)
    pd.DataFrame([dict(raw_name=f, animal=a, group="P", mouse_line="line_a", include="true")
                  for f, a in ANIMAL_FOLDERS.items()]).to_csv(tables / "animals.csv", index=False)
    (tables / "recordings.csv").write_text(
        "day,animal,recording,path,n_files,status,use,channel_order\n", encoding="utf-8")
    return path


def recording_keys() -> list[str]:
    return [f"{d}_{a}/{r}" for d, _ in SESSIONS.values() for a in ("M1", "M2") for r in ("t1", "t2")]


def write_rebuilt_sets(folder: Path, keys: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(TEMPLATE_ROI_SET.read_text(encoding="utf-8"))
    for key in keys:
        unit, recording = key.split("/")
        data = dict(template, name=f"{unit}_{recording}", grid=[128, 128], bregma_row=120, bregma_col=134,
                    lambda_row_offset=55, downsample=0.5)
        (folder / f"{unit}_{recording}.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_extraction(out: Path, pixel: Path, keys: list[str], needs_decision: tuple[str, ...] = ()) -> None:
    """Fake batch outputs: npz marker, dump files, batch_summary.csv."""
    rows = []
    for key in keys:
        unit, recording = key.split("/")
        day, animal = unit.split("_", 1)
        status = "needs_channel_decision" if key in needs_decision else "done"
        rows.append(dict(day=day, animal=animal, recording=recording, status=status))
        if status == "done":
            (out / unit / recording).mkdir(parents=True, exist_ok=True)
            (out / unit / recording / "connectivity_full.npz").write_bytes(b"x")
            (pixel / unit / recording).mkdir(parents=True, exist_ok=True)
            for name in ("pixels_dff_full.npy", "pixels_f_gcamp_full.npy", "pixels_median_dff_20s_full.npy"):
                (pixel / unit / recording / name).write_bytes(b"x")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "batch_summary.csv", index=False)


def copy_template_set(target: Path) -> None:
    shutil.copy(TEMPLATE_ROI_SET, target)
