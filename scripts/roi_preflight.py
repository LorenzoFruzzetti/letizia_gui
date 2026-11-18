"""Pre-flight of a cohort's rebuilt ROI sets: every box inside the dump window and the grid.

extract_cohort.py crops each pixel dump to a Bregma-relative window
(``save_data_window`` = rows -29..47, cols -44..43 around Bregma) and refuses a box outside
it only at run time, recording by recording; it never checks that the window itself fits
the grid before the crop. This checks every set of the cohort up
front, with the batch's own conventions:

  * Bregma on the final grid: ``y_1 = bregma_row // 2``, ``x_2 = bregma_col // 2``;
  * a box's half-open rows are ``[row_start - 1, row_end)`` relative to ``y_1`` (MATLAB
    1-based inclusive offsets), the same convention the window uses;
  * a box must lie inside the window, the window inside the grid, and the box inside
    the grid.

A set drawn with Lambda above Bregma (``lambda_above_bregma``) puts V1 at negative rows and
fails here by design: the dump window assumes Lambda below.

Reads ``<cohort folder>/roi_sets/rebuilt``; writes ``<cohort folder>/outputs/roi_preflight.csv`` (one row per recording) and exits with code 1
when any recording fails, so the job shows as failed.

    conda run --no-capture-output -n wfci_gui python -u roi_preflight.py --cohort <cohort folder>/cohort.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from wfci_gui.logic.cohort import load_script_cohort, load_tables, used_recordings, write_text_atomic  # noqa: E402

RUN_CONFIG: dict[str, Any] = {
    "cohort_file": None,  # <cohort folder>/cohort.yaml (or pass --cohort)
    # Bregma-relative half-open window (rows start, rows end, cols start, cols end); must
    # equal extract_cohort.py's save_data_window.
    "window": [-29, 47, -44, 43],
    "prefer_cli_args": True,
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check every rebuilt ROI set of a cohort against the dump window.")
    p.add_argument("--cohort", dest="cohort_file", default=defaults["cohort_file"])
    p.add_argument("--window", type=int, nargs=4, default=defaults["window"],
                   metavar=("ROW0", "ROW1", "COL0", "COL1"))
    return p.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{k: v for k, v in config.items() if k != "prefer_cli_args"})


def check_set(data: dict[str, Any], window: list[int], grid: list[int]) -> list[str]:
    """Every violation of one ROI set, as readable sentences (empty = passes)."""
    r0, r1, c0, c1 = window
    rows, cols = data.get("grid", grid)
    problems = []
    if list(data.get("grid", grid)) != list(grid):
        problems.append(f"grid {data.get('grid')} is not the cohort grid {grid}")
    if data.get("lambda_above_bregma"):
        problems.append("Lambda above Bregma (flipped set): the dump window assumes Lambda below")
    y_1, x_2 = int(data["bregma_row"]) // 2, int(data["bregma_col"]) // 2
    if y_1 + r0 < 0 or y_1 + r1 > rows or x_2 + c0 < 0 or x_2 + c1 > cols:
        problems.append(f"dump window rows {y_1 + r0}..{y_1 + r1}, cols {x_2 + c0}..{x_2 + c1} "
                        f"does not fit the {rows}x{cols} grid")
    for name, box in data["boxes"].items():
        b_r0, b_r1 = box["row_start"] - 1, box["row_end"]
        b_c0, b_c1 = box["col_start"] - 1, box["col_end"]
        if b_r0 < r0 or b_r1 > r1 or b_c0 < c0 or b_c1 > c1:
            problems.append(f"{name} outside the dump window")
        if y_1 + b_r0 < 0 or y_1 + b_r1 > rows or x_2 + b_c0 < 0 or x_2 + b_c1 > cols:
            problems.append(f"{name} outside the grid")
    return problems


def main() -> None:
    args = build_runtime_args()
    cohort = load_script_cohort(args.cohort_file)
    tables = load_tables(cohort)
    used = used_recordings(tables)
    if used.empty:
        raise ValueError("no runnable recordings in recordings.csv")
    grid = cohort.acquisition.get("grid") or [128, 128]
    rebuilt = cohort.layout.rebuilt_roi_sets
    rows = []
    for recording in used.itertuples():
        name = f"{recording.day}_{recording.animal}_{recording.recording}.yaml"
        path = rebuilt / name
        if not path.exists():
            rows.append(dict(roi_set=name, recording=recording.key, exists=False, bregma_row="",
                             bregma_col="", violations=1, detail="no rebuilt ROI set"))
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        problems = check_set(data, list(args.window), list(grid))
        rows.append(dict(roi_set=name, recording=recording.key, exists=True,
                         bregma_row=data["bregma_row"], bregma_col=data["bregma_col"],
                         violations=len(problems), detail="; ".join(problems)))
    frame = pd.DataFrame(rows)
    out = cohort.layout.outputs / "roi_preflight.csv"
    write_text_atomic(out, frame.to_csv(index=False, lineterminator="\n"))
    failed = frame[frame["violations"] > 0]
    print(f"ROI sets checked: {len(frame)} (window {args.window}, grid {grid})")
    print(f"Passed: {len(frame) - len(failed)}   Failed: {len(failed)}")
    for row in failed.itertuples():
        print(f"  FAIL {row.roi_set}: {row.detail}")
    print(f"Written: {out}")
    sys.exit(1 if len(failed) else 0)


if __name__ == "__main__":
    main()
