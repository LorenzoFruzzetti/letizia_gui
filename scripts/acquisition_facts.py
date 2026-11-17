"""Read a cohort's acquisition facts from its raw files (step 2 of the workflow).

For the first runnable recording of every session it reports:
  * the per-channel frame rate, from the TIFF header's ``Time_From_Last`` (the time between
    two interleaved frames: 0.0200 s -> 50 frames/s -> 25 Hz per channel, CLAUDE.md 9.55);
  * whether the frame numbers in the file names are zero-padded (if not, a name sort mixes
    the channels and ``frame_order`` must be ``number``, 9.64);
  * the image size, file extension and file count;
  * LED-setting tokens in the file names (``LEDB100_LEDV20``): a change between sessions is
    an illumination change that creates fake connectivity changes (9.54-9.56).

It writes ``<output_root>/acquisition_facts.json`` and prints a table. It does not change the
cohort file: the user checks the values on the Acquisition page and confirms them there.

Run it:
    conda run --no-capture-output -n wfci_gui python -u acquisition_facts.py --cohort cohorts\\botox_2023.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tifffile  # noqa: E402

from wfci_gui.logic.cohort import load_cohort, load_tables, used_recordings, write_text_atomic  # noqa: E402

RUN_CONFIG: dict[str, Any] = {
    "cohort": r"cohorts\botox_2023.yaml",
    "headers_per_recording": 3,   # TIFF headers read per recording (Time_From_Last is per frame)
    "prefer_cli_args": True,
}
TIME_FROM_LAST_RE = re.compile(r"Time_From_Last\s*=\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
LED_TOKEN_RE = re.compile(r"(LED[A-Za-z]*\d+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"(\d+)\.[^.]+$")


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read frame rate, frame order and LED settings of a cohort.")
    p.add_argument("--cohort", default=defaults["cohort"])
    p.add_argument("--headers-per-recording", type=int, default=defaults["headers_per_recording"])
    return p.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{k: v for k, v in config.items() if k != "prefer_cli_args"})


def interval_s(description: str) -> float | None:
    match = TIME_FROM_LAST_RE.search(description or "")
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def recording_facts(folder: Path, pattern: str, n_headers: int) -> dict[str, Any]:
    names = sorted(p.name for p in folder.glob(pattern))
    if not names:
        raise FileNotFoundError(f"no {pattern} files in {folder}")
    numbers = [NUMBER_RE.search(n) for n in names]
    widths = {len(m.group(1)) for m in numbers if m}
    # Zero-padded numbers all have one width; "_1", "_10", "_100" have several.
    padded = len(widths) == 1
    intervals = []
    shape = None
    for name in names[1:1 + n_headers]:
        with tifffile.TiffFile(str(folder / name)) as tif:
            page = tif.pages[0]
            shape = list(page.shape)
            intervals.append(interval_s(page.description))
    found = [i for i in intervals if i]
    interleaved_hz = round(1.0 / (sum(found) / len(found)), 3) if found else None
    leds = sorted({t for name in names[:2] for t in LED_TOKEN_RE.findall(name)})
    return dict(
        folder=str(folder), n_files=len(names), extension=Path(names[0]).suffix.lower(),
        image_shape=shape, numbers_zero_padded=padded,
        suggested_frame_order="name" if padded else "number",
        interleaved_rate_hz=interleaved_hz,
        sampling_rate_hz=round(interleaved_hz / 2, 3) if interleaved_hz else None,
        led_tokens=leds, first_file=names[0])


def main() -> None:
    args = build_runtime_args()
    cohort = load_cohort(args.cohort)
    tables = load_tables(cohort)
    used = used_recordings(tables)
    if used.empty:
        raise ValueError("no runnable recordings in recordings.csv: run the scan (step 1) first")
    pattern = cohort.acquisition.get("file_glob") or "*.tif"
    timepoint_of = dict(zip(tables["sessions"]["day"], tables["sessions"]["timepoint"]))
    sessions = []
    for day, rows in used.groupby("day", sort=True):
        first = rows.iloc[0]
        facts = recording_facts(Path(first["path"]), pattern, args.headers_per_recording)
        facts.update(day=day, timepoint=timepoint_of.get(day, ""), animal=first["animal"],
                     recording=first["recording"])
        sessions.append(facts)
    rates = Counter(s["sampling_rate_hz"] for s in sessions)
    orders = Counter(s["suggested_frame_order"] for s in sessions)
    led_sets = {tuple(s["led_tokens"]) for s in sessions}
    summary = dict(
        sampling_rate_hz=rates.most_common(1)[0][0], frame_order=orders.most_common(1)[0][0],
        file_glob=f"*{sessions[0]['extension']}", image_shape=sessions[0]["image_shape"],
        rate_consistent=len(rates) == 1, frame_order_consistent=len(orders) == 1,
        led_settings_change=len(led_sets) > 1)
    out = cohort.path_of("output_root") / "acquisition_facts.json"
    write_text_atomic(out, json.dumps(dict(summary=summary, sessions=sessions), indent=2) + "\n")

    print(f"{'day':8}{'timepoint':10}{'rate Hz':>9}  {'frame order':12}{'shape':14}{'files':>6}  LED tokens")
    for s in sessions:
        print(f"{s['day']:8}{s['timepoint']:10}{s['sampling_rate_hz'] or '?':>9}  "
              f"{s['suggested_frame_order']:12}{str(s['image_shape']):14}{s['n_files']:>6}  "
              f"{' '.join(s['led_tokens']) or '-'}")
    print(f"Suggested: sampling_rate_hz={summary['sampling_rate_hz']}, frame_order={summary['frame_order']}, "
          f"file_glob={summary['file_glob']}")
    if not summary["rate_consistent"]:
        print("WARNING: the frame rate differs between sessions")
    if summary["led_settings_change"]:
        print("WARNING: the LED tokens in the file names change between sessions: an illumination "
              "change creates fake connectivity changes (CLAUDE.md 9.54-9.56)")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
