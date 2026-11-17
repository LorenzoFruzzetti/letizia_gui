"""Write the experimental design table for the pixel dataset.

One row per RECORDING (`pixel_data/<day>_<animal>/<t#>/`), carrying the
experimental factors needed to model it:

    recording_id, day, animal, group, mouse_line, day_index, animal_day,
    days_since_day1, timepoint, recording, recording_index, n_days_for_animal,
    n_recordings_for_animal, pixel_dir

`days_since_day1` is the number of calendar days from that animal's day1 session
(day1 = 0; day1 = 260618, day2 = 260621 gives 3 for day2; a day 0 session is negative).
`timepoint` groups those day counts into classes (BL, 3D, 15D, 30D, 40D, 60D, 77D) via
`timepoint_classes`, for the PV animals only; other animals are left blank.

`group` comes from `outputs/epileptic_groups/batch_summary.csv` (built by
`build_epileptic_groups.py` from both manifests) and is relabelled for reporting:
the summary's `P` is written as `PV`, matching the `PV*` animal names; `R` and `T`
are unchanged.

`mouse_line` is the transgenic line, which does NOT follow the group: R and T each
contain both lines. It is a hand-recorded fact with no source in the dataset, so it
lives in MOUSE_LINES below, keyed per animal. An animal missing from that map is an
error rather than a blank -- a new cohort must have its line recorded deliberately.

`day_index` / `animal_day` are the WITHIN-ANIMAL session number: each animal's
recording days are sorted chronologically and numbered from 1, so an animal seen
on 260611 and 260615 gets day1 and day2. The numbering is per animal, not global --
day1 is a different calendar day for different animals.

A day listed in `day_zero_days` is a pre-baseline session: its sessions get
day_index 0 (`day0`) and the animal's other days are numbered from 1 as usual.
It must be the animal's first recording day.

Output: pixel_data/experimental_design.csv (next to the dumps it describes).
With `animals` set, a subset table instead, e.g. pixel_data/experimental_design_PV.csv
for PV3-PV8.

Edit RUN_CONFIG below to run directly from the editor without CLI arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    # Unit folders named <day>_<animal>, each holding per-recording <t#> folders.
    "pixel_root": REPO_ROOT / "pixel_data",
    # day,animal,group table; written by build_epileptic_groups.py.
    "groups_path": REPO_ROOT / "outputs/epileptic_groups/batch_summary.csv",
    "output_path": REPO_ROOT / "pixel_data/experimental_design.csv",
    # Days whose sessions are day_index 0 (pre-baseline); the rest count from 1.
    # 260519: PV3 (the PV3_bis acquisition) and PV4, the day before their day 1.
    "day_zero_days": ["260519"],
    # None: every animal under pixel_root. A list keeps only those animals; every other
    # column is per animal, so the kept rows are identical to the full table's.
    # PV-only manifest:  --animals PV3 PV4 PV5 PV6 PV7 PV8
    #                    --output-path pixel_data/experimental_design_PV.csv
    "animals": None,
    # Time-point class per days_since_day1, written to the `timepoint` column for the
    # animals in timepoint_animals only (blank for every other animal). A value with
    # no class for one of these animals is an error, so a new session must be mapped.
    "timepoint_classes": {
        "BL": [0],
        "3D": [2, 3, 5],
        "15D": [14, 15],
        "30D": [29, 30, 31],
        "40D": [38, 42, 47],
        "60D": [57, 58],
        "77D": [77],
    },
    "timepoint_animals": ["PV3", "PV4", "PV5", "PV6", "PV7", "PV8"],
    "prefer_cli_args": True,  # False: ignore CLI arguments and use this block.
}

# Reporting labels for the single-letter group codes in batch_summary.csv.
GROUP_LABELS = {"P": "PV", "R": "R", "T": "T"}

# Transgenic line per animal, recorded by hand: it is not derivable from the animal
# name or the group, and both R and T contain animals of each line. Every animal
# under pixel_root must appear here; see `attach_mouse_line`.
MOUSE_LINES = {
    # Every PV* animal of the P/PV group is PV-CRE.
    "PV3": "PV-CRE",
    "PV4": "PV-CRE",
    "PV5": "PV-CRE",
    "PV6": "PV-CRE",
    "PV7": "PV-CRE",
    "PV8": "PV-CRE",
    "R1": "C57",
    "R2": "C57",
    "R3": "C57",
    "R4": "C57",
    "R6": "PV-CRE",
    "R7": "PV-CRE",
    "R8": "C57",
    "R9": "C57",
    # Unnamed 261012 animal (files R101_*, probably R10); line confirmed by the user.
    "Nan": "C57",
    "T4": "C57",
    "T5": "C57",
    "T7": "C57",
    "T8": "C57",
    "T9": "PV-CRE",
    "T10": "PV-CRE",
    "T11": "PV-CRE",
    "T12": "PV-CRE",
    "T13": "PV-CRE",
    "T14": "PV-CRE",
    "T15": "C57",
    "T16": "C57",
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--groups-path", type=Path, default=defaults["groups_path"])
    parser.add_argument("--output-path", type=Path, default=defaults["output_path"])
    parser.add_argument("--day-zero-days", nargs="*", default=defaults["day_zero_days"])
    parser.add_argument("--animals", nargs="+", default=defaults["animals"],
                        help="Keep only these animals (default: all).")
    args = parser.parse_args()
    # The class map is a nested structure, so it is only settable in RUN_CONFIG.
    args.timepoint_classes = dict(defaults["timepoint_classes"])
    args.timepoint_animals = list(defaults["timepoint_animals"])
    return args


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        pixel_root=Path(config["pixel_root"]),
        groups_path=Path(config["groups_path"]),
        output_path=Path(config["output_path"]),
        day_zero_days=list(config["day_zero_days"]),
        animals=None if config.get("animals") is None else list(config["animals"]),
        timepoint_classes=dict(config["timepoint_classes"]),
        timepoint_animals=list(config["timepoint_animals"]),
    )


def scan_recordings(pixel_root: Path) -> pd.DataFrame:
    """One row per <day>_<animal>/<t#> folder found under pixel_root."""
    rows = []
    for unit_dir in sorted(p for p in pixel_root.iterdir() if p.is_dir()):
        # Unit folders are <day>_<animal>; animal names never contain "_".
        day, _, animal = unit_dir.name.partition("_")
        if not animal:
            raise ValueError(f"unit folder is not <day>_<animal>: {unit_dir}")
        for recording_dir in sorted(p for p in unit_dir.iterdir() if p.is_dir()):
            rows.append(
                {
                    "recording_id": f"{unit_dir.name}_{recording_dir.name}",
                    "day": day,
                    "animal": animal,
                    "recording": recording_dir.name,
                    "pixel_dir": f"{unit_dir.name}/{recording_dir.name}",
                }
            )
    if not rows:
        raise ValueError(f"no <day>_<animal>/<t#> folders under {pixel_root}")
    return pd.DataFrame(rows)


def attach_groups(design: pd.DataFrame, groups_path: Path) -> pd.DataFrame:
    """Add the relabelled group column, requiring a group for every unit."""
    groups = pd.read_csv(groups_path, dtype=str)
    missing_columns = {"day", "animal", "group"} - set(groups.columns)
    if missing_columns:
        raise ValueError(f"{groups_path} is missing columns: {sorted(missing_columns)}")

    duplicated = groups.duplicated(subset=["day", "animal"])
    if duplicated.any():
        raise ValueError(
            f"{groups_path} has repeated day/animal rows: "
            f"{groups.loc[duplicated, ['day', 'animal']].to_dict('records')}"
        )

    unknown_codes = sorted(set(groups["group"]) - set(GROUP_LABELS))
    if unknown_codes:
        raise ValueError(f"unknown group codes in {groups_path}: {unknown_codes}")
    groups["group"] = groups["group"].map(GROUP_LABELS)

    merged = design.merge(groups, on=["day", "animal"], how="left")
    ungrouped = merged.loc[merged["group"].isna(), "pixel_dir"].tolist()
    if ungrouped:
        raise ValueError(f"no group in {groups_path} for: {ungrouped}")

    # An animal must stay in one group across its sessions, or the factor is meaningless.
    per_animal_groups = merged.groupby("animal")["group"].nunique()
    inconsistent = per_animal_groups[per_animal_groups > 1].index.tolist()
    if inconsistent:
        raise ValueError(f"animal recorded under more than one group: {inconsistent}")
    return merged


def attach_mouse_line(design: pd.DataFrame) -> pd.DataFrame:
    """Add the transgenic line, requiring MOUSE_LINES to cover every animal."""
    unmapped = sorted(set(design["animal"]) - set(MOUSE_LINES))
    if unmapped:
        raise ValueError(f"no entry in MOUSE_LINES for: {unmapped}")
    design["mouse_line"] = design["animal"].map(MOUSE_LINES)
    return design


def add_within_animal_day_index(design: pd.DataFrame, day_zero_days: list[str]) -> pd.DataFrame:
    """Number each animal's recording days 1..n in chronological order.

    Days are YYMMDD strings of fixed width, so lexicographic order is chronological.
    Sessions on a `day_zero_days` day get 0 and are skipped by the 1..n count.
    """
    sessions = design.drop_duplicates(subset=["animal", "day"]).sort_values(["animal", "day"])
    is_day_zero = sessions["day"].isin(day_zero_days)
    # A day 0 after a counted day would make day_index non-chronological.
    first_day = sessions.groupby("animal")["day"].transform("min")
    late_day_zero = sessions.loc[is_day_zero & (sessions["day"] != first_day), ["animal", "day"]]
    if not late_day_zero.empty:
        raise ValueError(f"day_zero_days that are not the animal's first day: "
                         f"{late_day_zero.to_dict('records')}")
    counted = sessions[~is_day_zero].assign(
        day_index=lambda frame: frame.groupby("animal").cumcount() + 1)
    day_index = pd.concat([counted, sessions[is_day_zero].assign(day_index=0)]).loc[
        :, ["animal", "day", "day_index"]]
    design = design.merge(day_index, on=["animal", "day"], how="left")
    design["animal_day"] = "day" + design["day_index"].astype(str)
    return design


def add_days_since_day1(design: pd.DataFrame) -> pd.DataFrame:
    """Calendar days from the animal's day 1 session (day1 = 0; day 0 sessions are negative).

    E.g. day1 = 260618 and day2 = 260621 gives 0 and 3.
    """
    session_date = pd.to_datetime(design["day"], format="%y%m%d")
    day1_date = session_date.where(design["day_index"] == 1).groupby(design["animal"]).transform("min")
    design["days_since_day1"] = (session_date - day1_date).dt.days
    return design


def add_timepoint(design: pd.DataFrame, classes: dict[str, list[int]],
                  animals: list[str]) -> pd.DataFrame:
    """Class label of days_since_day1 (e.g. 31 -> "30D") for `animals`; blank for the rest."""
    day_to_class = {days: label for label, days_list in classes.items() for days in days_list}
    if len(day_to_class) != sum(len(days_list) for days_list in classes.values()):
        raise ValueError(f"a days_since_day1 value is listed under two classes: {classes}")

    in_scope = design["animal"].isin(animals)
    design["timepoint"] = ""
    design.loc[in_scope, "timepoint"] = design.loc[in_scope, "days_since_day1"].map(day_to_class)
    unmapped = design.loc[in_scope & design["timepoint"].isna(), ["animal", "day", "days_since_day1"]]
    if not unmapped.empty:
        raise ValueError("days_since_day1 with no timepoint class (add it to timepoint_classes): "
                         f"{unmapped.drop_duplicates().to_dict('records')}")
    return design


def add_recording_index(design: pd.DataFrame) -> pd.DataFrame:
    """Order of the recording within its session, and per-animal totals."""
    design = design.sort_values(["animal", "day", "recording"]).reset_index(drop=True)
    design["recording_index"] = design.groupby(["animal", "day"]).cumcount() + 1
    design["n_days_for_animal"] = design.groupby("animal")["day"].transform("nunique")
    design["n_recordings_for_animal"] = design.groupby("animal")["recording_id"].transform("size")
    return design


def main() -> None:
    args = build_runtime_args()

    design = scan_recordings(args.pixel_root)
    if args.animals is not None:
        # A misspelt name would silently shrink the table, so it is an error instead.
        absent = sorted(set(args.animals) - set(design["animal"]))
        if absent:
            raise ValueError(f"animals not found under {args.pixel_root}: {absent}")
        design = design[design["animal"].isin(args.animals)].reset_index(drop=True)
    design = attach_groups(design, args.groups_path)
    design = attach_mouse_line(design)
    design = add_within_animal_day_index(design, args.day_zero_days)
    design = add_days_since_day1(design)
    design = add_timepoint(design, args.timepoint_classes, args.timepoint_animals)
    design = add_recording_index(design)

    design = (
        design.loc[
            :,
            [
                "recording_id",
                "day",
                "animal",
                "group",
                "mouse_line",
                "day_index",
                "animal_day",
                "days_since_day1",
                "timepoint",
                "recording",
                "recording_index",
                "n_days_for_animal",
                "n_recordings_for_animal",
                "pixel_dir",
            ],
        ]
        .sort_values(["group", "animal", "day_index", "recording"])
        .reset_index(drop=True)
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output_path, index=False)

    print(f"Wrote {args.output_path}")
    print(f"  recordings: {len(design)}")
    print(f"  animal-days (units): {design[['animal', 'day']].drop_duplicates().shape[0]}")
    print(f"  animals: {design['animal'].nunique()}")
    print("  recordings per group:")
    for group, count in design["group"].value_counts().sort_index().items():
        animals = design.loc[design["group"] == group, "animal"].nunique()
        print(f"    {group}: {count} recordings, {animals} animals")
    print("  animals per group x mouse line:")
    animals = design.drop_duplicates("animal")
    crossed = animals.pivot_table(
        index="group", columns="mouse_line", values="animal", aggfunc="size", fill_value=0
    )
    for line in crossed.to_string().splitlines():
        print(f"    {line}")
    print("  animals per number of recording days:")
    days_per_animal = design.drop_duplicates("animal").set_index("animal")["n_days_for_animal"]
    for n_days, count in days_per_animal.value_counts().sort_index().items():
        print(f"    {n_days} day(s): {count} animals")


if __name__ == "__main__":
    main()
