"""Experimental-design helpers used by build_cohort_design.py.

Holds only functions, no run configuration and no dataset tables:
    * ``add_within_animal_day_index`` -- each animal's session number (day 0 allowed);
    * ``add_days_since_day1``         -- calendar days from the animal's day 1;
    * ``add_recording_index``         -- recording order in its session, per-animal totals.

Importing this module is inert: it has no ``__main__`` block.
"""

from __future__ import annotations

import pandas as pd


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

    E.g. day1 = 200101 and day2 = 200104 gives 0 and 3.
    """
    session_date = pd.to_datetime(design["day"], format="%y%m%d")
    day1_date = session_date.where(design["day_index"] == 1).groupby(design["animal"]).transform("min")
    design["days_since_day1"] = (session_date - day1_date).dt.days
    return design


def add_recording_index(design: pd.DataFrame) -> pd.DataFrame:
    """Order of the recording within its session, and per-animal totals."""
    design = design.sort_values(["animal", "day", "recording"]).reset_index(drop=True)
    design["recording_index"] = design.groupby(["animal", "day"]).cumcount() + 1
    design["n_days_for_animal"] = design.groupby("animal")["day"].transform("nunique")
    design["n_recordings_for_animal"] = design.groupby("animal")["recording_id"].transform("size")
    return design
