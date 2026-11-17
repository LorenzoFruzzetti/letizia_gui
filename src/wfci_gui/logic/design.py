"""Analysis-A completeness check: is the animal x timepoint design complete?

The statistics of analysis A (paired tests across animals, test_pv_detrended_stats.py)
need every chosen animal at every chosen timepoint. This check says whether the choice
is complete and, if not, the two ways to complete it: drop the animals that miss a
timepoint, or drop the timepoints some animal misses (never the baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DesignCheck:
    complete: bool
    missing: list[tuple[str, str]] = field(default_factory=list)      # (animal, timepoint)
    drop_animals: list[str] = field(default_factory=list)             # option 1
    drop_timepoints: list[str] = field(default_factory=list)          # option 2
    baseline_missing: list[str] = field(default_factory=list)         # animals without baseline
    message: str = ""


def check_design(design: pd.DataFrame, baseline: str, timepoints: list[str],
                 animals: list[str]) -> DesignCheck:
    """``design``: experimental_design.csv rows (animal, timepoint), excluded rows removed."""
    if baseline not in timepoints:
        return DesignCheck(False, message=f"the baseline {baseline!r} must be one of the chosen timepoints")
    if len(timepoints) < 2:
        return DesignCheck(False, message="choose the baseline and at least one later timepoint")
    if len(animals) < 2:
        return DesignCheck(False, message="choose at least two animals")
    present = set(zip(design["animal"], design["timepoint"]))
    missing = [(a, t) for a in animals for t in timepoints if (a, t) not in present]
    if not missing:
        return DesignCheck(True, message=f"complete: {len(animals)} animals x {len(timepoints)} timepoints")
    drop_animals = sorted({a for a, _ in missing})
    drop_timepoints = [t for t in timepoints if t != baseline and any(mt == t for _, mt in missing)]
    baseline_missing = sorted({a for a, t in missing if t == baseline})
    lines = [f"incomplete: {len(missing)} animal x timepoint cell(s) have no recording "
             f"({', '.join(f'{a} at {t}' for a, t in missing[:8])}{' ...' if len(missing) > 8 else ''})."]
    kept = len(animals) - len(drop_animals)
    lines.append(f"To complete it, drop the animals {drop_animals} ({kept} would remain)"
                 if kept >= 2 else "Dropping animals would leave fewer than two.")
    if baseline_missing:
        lines.append(f"{baseline_missing} have no {baseline} recording, so dropping timepoints "
                     "cannot fix them.")
    elif drop_timepoints and len(timepoints) - len(drop_timepoints) >= 2:
        lines.append(f"or drop the timepoints {drop_timepoints}.")
    return DesignCheck(False, missing, drop_animals, drop_timepoints, baseline_missing, " ".join(lines))
