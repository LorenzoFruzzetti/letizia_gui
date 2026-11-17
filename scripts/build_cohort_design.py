"""Write a cohort's experimental design table from its cohort file and tables.

``design_csv`` (default ``<pixel_root>/experimental_design.csv``): one row per used recording that is not
excluded, with the columns the analysis scripts read:

    recording_id, day, animal, group, mouse_line, day_index, animal_day, days_since_day1,
    timepoint, recording, recording_index, n_days_for_animal, n_recordings_for_animal, pixel_dir

  * group and mouse_line come from animals.csv, timepoint and day 0 from sessions.csv;
  * day_index / days_since_day1 / recording_index use the functions of design_helpers.py,
    so the numbering is the same in every design table;
  * a missing group, mouse line or timepoint raises (on purpose).

Adding a session later shifts that animal's day_index: rerun everything keyed on it.

    conda run --no-capture-output -n wfci_gui python -u build_cohort_design.py --cohort <cohort folder>/cohort.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from design_helpers import (  # noqa: E402
    add_days_since_day1, add_recording_index, add_within_animal_day_index,
)
from wfci_gui.logic.cohort import (  # noqa: E402
    load_cohort, load_tables, parse_bool, used_recordings, write_text_atomic,
)

RUN_CONFIG: dict[str, Any] = {
    "cohort": None,  # path to the cohort file (or pass --cohort)
    "output": None,             # None = the cohort's design_csv (<pixel_root>/experimental_design.csv)
    "prefer_cli_args": True,
}
DESIGN_FIELDS = [
    "recording_id", "day", "animal", "group", "mouse_line", "day_index", "animal_day",
    "days_since_day1", "timepoint", "recording", "recording_index", "n_days_for_animal",
    "n_recordings_for_animal", "pixel_dir",
]


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a cohort's experimental_design.csv.")
    p.add_argument("--cohort", default=defaults["cohort"])
    p.add_argument("--output", default=defaults["output"], help="Design CSV path (default: the cohort's).")
    return p.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{k: v for k, v in config.items() if k != "prefer_cli_args"})


def build_design(cohort, tables) -> pd.DataFrame:
    used = used_recordings(tables, cohort, drop_excluded=True)
    if used.empty:
        raise ValueError("no used, non-excluded recordings")
    sessions = tables["sessions"]
    animals = tables["animals"].drop_duplicates("animal")
    timepoint = dict(zip(sessions["day"], sessions["timepoint"]))
    group = dict(zip(animals["animal"], animals["group"]))
    mouse_line = dict(zip(animals["animal"], animals["mouse_line"]))
    day_zero_days = sorted(sessions.loc[sessions["day_zero"].map(parse_bool), "day"])
    design = pd.DataFrame({
        "recording_id": [f"{r.day}_{r.animal}_{r.recording}" for r in used.itertuples()],
        "day": used["day"], "animal": used["animal"], "recording": used["recording"],
        "timepoint": used["day"].map(timepoint),
        "pixel_dir": [f"{r.day}_{r.animal}/{r.recording}" for r in used.itertuples()],
    })
    design["group"] = design["animal"].map(group)
    design["mouse_line"] = design["animal"].map(mouse_line)
    for column in ("timepoint", "group", "mouse_line"):
        blank = design.loc[design[column].isna() | (design[column] == ""), "recording_id"]
        if not blank.empty:
            raise ValueError(f"no {column} for: {sorted(blank)[:10]} (fill the sessions/animals table)")
    design = add_within_animal_day_index(design, day_zero_days=day_zero_days)
    design = add_days_since_day1(design)
    design = add_recording_index(design)
    # t# sorted numerically (a text sort puts t10 before t2); recording_index from 1.
    design["recording_number"] = design["recording"].str[1:].astype(int)
    design = design.sort_values(["animal", "day_index", "recording_number"])
    design["recording_index"] = design.groupby(["animal", "day"]).cumcount() + 1
    return design.loc[:, DESIGN_FIELDS].reset_index(drop=True)


def main() -> None:
    args = build_runtime_args()
    cohort = load_cohort(args.cohort)
    tables = load_tables(cohort)
    design = build_design(cohort, tables)
    out = Path(args.output) if args.output else cohort.path_of("design_csv")
    # CRLF, as pandas writes on Windows and as the existing design tables are.
    write_text_atomic(out, design.to_csv(index=False, lineterminator="\r\n"))
    print(f"Design: {out} ({len(design)} recordings, {design['animal'].nunique()} animals, "
          f"{len(cohort.excluded_keys())} excluded)")
    print("Sessions per animal (day_index: timepoint, days_since_day1):")
    for animal, rows in design.drop_duplicates(["animal", "day"]).groupby("animal"):
        print(f"  {animal} [{rows['group'].iloc[0]}]: " + ", ".join(
            f"{r.day_index}:{r.timepoint}({r.days_since_day1})" for r in rows.itertuples()))


if __name__ == "__main__":
    main()
