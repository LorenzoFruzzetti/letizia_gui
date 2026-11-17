"""Per-animal stack of the ROI-union raw-F traces, one row per timepoint, shared y-scale.

For every animal in the design CSV (excluded timepoints left out) draw one figure with
one row per timepoint, sorted by ``days_since_day1``. Each row is the raw-F panel of that
day's ``<day>_<animal>\\plots_roi_union\\channel_traces.png``: t1..t5 joined end to end,
per-frame ROI-union mean of each channel, faint full rate + bold 1 s moving mean.

Channels follow the union plots exactly: done recordings are labelled by ``plotted_order``
from that day's ``plots_roi_union\\channel_means.csv`` (stored volumes swapped where it
differs from ``stored_order``); needs_channel_decision recordings are drawn as odd/even.

Memory: each recording's volumes are memory-mapped, reduced to 1-D traces, drawn and
released before the next one. The y-limits are applied to every row after all rows are
drawn. Every expected t1..t5 cache must exist; anything missing raises.

Output: ``<pixel_root>\\plots_roi_union_by_animal\\<animal>.png`` plus one CSV,
``p99_minus_p1_by_recording.csv``, with one row per drawn line (animal x timepoint x
recording x line): mean, SD, P1, P99 and P99-P1 of the full-rate trace in raw counts.
``identity`` is gcamp/emo as plotted, or ``undecided`` for not_interleaved (odd/even)
recordings. With --select, only the selected animals' rows are replaced in the CSV.

    python plot_channel_traces_by_animal.py
    python plot_channel_traces_by_animal.py --no-plots   # CSV only

Cohorts whose days do not all have t1..t5: ``--recordings-from-design`` takes each
day's recordings from the design CSV instead of a fixed t1..t5; every listed recording must
still have its cache, and a day missing one the animal has on other days says so in its label.

    python plot_channel_traces_by_animal.py --pixel-root pixel_data ^
        --design-csv pixel_data\\experimental_design.csv --recordings-from-design
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_channel_traces import (CHANNELS, COLORS, INK, META_NAME, MUTED, NAMES, NEEDS_DECISION,
                                 moving_mean, roi_union_traces)

REPO_ROOT = Path(__file__).resolve().parent
RUN_CONFIG = {
    "pixel_root": REPO_ROOT / "pixel_data",
    "design_csv": REPO_ROOT / "pixel_data" / "experimental_design.csv",
    "out_dir": "plots_roi_union_by_animal",
    "stats_csv": "p99_minus_p1_by_recording.csv",
    "exclude_timepoints": ("BL2",),
    "recordings": ("t1", "t2", "t3", "t4", "t5"),
    # Per-channel frame rate of the dumps, and the moving-mean window.
    "sampling_rate_hz": 25.0,
    "smooth_s": 1.0,
}
SHORT = {"gcamp": "GCaMP", "emo": "emo", "odd": "odd", "even": "even"}
IDENTITY = {"gcamp": "gcamp", "emo": "emo", "odd": "undecided", "even": "undecided"}
LINE_ORDER = ("gcamp", "emo", "odd", "even")
STATS_FIELDS = ("animal", "timepoint", "day", "days_since_day1", "recording", "channel_verdict",
                "line", "identity", "mean", "sd", "p1", "p99", "p99_minus_p1")


def read_days(design_csv: Path, exclude: tuple[str, ...]) -> dict[str, list[dict[str, str]]]:
    """{animal: [one row per day, sorted by days_since_day1]} without excluded timepoints.

    Each day row also carries ``design_recordings``: that day's recordings in the design CSV,
    sorted by number (t0, t1, ..., t10).
    """
    days: dict[str, dict[str, dict[str, str]]] = {}
    recordings: dict[tuple[str, str], set[str]] = {}
    with open(design_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["timepoint"] in exclude:
                continue
            days.setdefault(row["animal"], {})[row["day"]] = row
            recordings.setdefault((row["animal"], row["day"]), set()).add(row["recording"])
    out = {}
    for animal, by_day in sorted(days.items()):
        rows = []
        for day, row in by_day.items():
            recs = sorted(recordings[(animal, day)], key=lambda r: int(r.lstrip("t")))
            rows.append({**row, "design_recordings": tuple(recs)})
        out[animal] = sorted(rows, key=lambda r: int(r["days_since_day1"]))
    return out


def day_recordings(day: dict, all_days: list[dict], cfg: argparse.Namespace
                   ) -> tuple[tuple[str, ...], list[str]]:
    """(recordings to draw for this day, recordings the animal has on other days but not here)."""
    if not cfg.recordings_from_design:
        return cfg.recordings, []
    animal_recs = {r for d in all_days for r in d["design_recordings"]}
    missing = sorted(animal_recs - set(day["design_recordings"]), key=lambda r: int(r.lstrip("t")))
    return day["design_recordings"], missing


def read_channel_means(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing (run plot_channel_traces.py --level-region roi_union)")
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["recording"]: r for r in csv.DictReader(fh)}


def recording_status(rec: Path, row: dict[str, str]) -> str:
    status = row.get("status") or "done"
    if status not in CHANNELS:
        raise RuntimeError(f"{rec}: status {status!r} has no cached volumes to draw")
    for channel in CHANNELS[status]:
        for name in (META_NAME[status], f"pixels_f_{channel}_full.npy"):
            if not (rec / name).is_file():
                raise FileNotFoundError(f"{rec / name}: missing cache")
    return status


def trace_stats(trace: np.ndarray) -> dict[str, float]:
    """Mean, SD, P1, P99 and P99-P1 of one full-rate trace (raw counts)."""
    p1, p99 = np.percentile(trace, (1, 99))
    return {"mean": float(trace.mean()), "sd": float(trace.std()), "p1": float(p1),
            "p99": float(p99), "p99_minus_p1": float(p99 - p1)}


def draw_row(ax, unit: Path, means: dict[str, dict[str, str]], recordings: tuple[str, ...],
             rate: float, n_smooth: int, stats: list[dict[str, object]]
             ) -> tuple[float, float, int]:
    """Draw one day's recordings end to end on `ax` (None: measure only) and append one
    stats row per line to `stats`; returns (y_min, y_max, n_frames)."""
    label_transform = ax.get_xaxis_transform() if ax is not None else None
    lo, hi, start = np.inf, -np.inf, 0
    for name in recordings:
        rec = unit / name
        if name not in means:
            raise RuntimeError(f"{unit}: {name} missing from channel_means.csv")
        row = means[name]
        status = recording_status(rec, row)
        traces, _ = roi_union_traces(rec, status)
        if status == "done" and row["plotted_order"] != row["stored_order"]:
            traces = {"gcamp": traces["emo"], "emo": traces["gcamp"]}
        n = len(next(iter(traces.values())))
        line_stats = {channel: trace_stats(trace) for channel, trace in traces.items()}
        for channel, values in line_stats.items():
            stats.append({"recording": name, "channel_verdict": row["channel_verdict"],
                          "line": channel, "identity": IDENTITY[channel], **values})
        if ax is not None:
            t = (start + np.arange(n)) / rate
            for channel, trace in traces.items():
                smooth = moving_mean(trace, n_smooth)
                ax.plot(t, trace, color=COLORS[channel], lw=0.5, alpha=0.25)
                ax.plot(t, smooth, color=COLORS[channel], lw=1.6, label=NAMES[channel])
                lo, hi = min(lo, float(trace.min())), max(hi, float(trace.max()))
            ax.axvline(start / rate, color=MUTED, lw=0.8, ls="--")
            mid = (start + n / 2) / rate
            ax.text(mid, 1.25, f"{name} [{row['channel_verdict']}]",
                    transform=label_transform, ha="center", va="bottom", fontsize=7, color=INK)
            # SD and P99-P1 of each full-rate (unsmoothed) trace over this recording,
            # one line per channel.
            for channel, y in zip(traces, (1.13, 1.01)):
                ax.text(mid, y, f"{SHORT[channel]}  SD {line_stats[channel]['sd']:.0f} · "
                                f"P99−P1 {line_stats[channel]['p99_minus_p1']:.0f}",
                        transform=label_transform, ha="center", va="bottom", fontsize=7,
                        color=COLORS[channel], fontweight="bold")
            del t
        start += n
        del traces
        gc.collect()
    return lo, hi, start


def day_fields(animal: str, day: dict) -> dict[str, object]:
    return {"animal": animal, "timepoint": day["timepoint"], "day": day["day"],
            "days_since_day1": int(day["days_since_day1"])}


def animal_stats(animal: str, days: list[dict], cfg: argparse.Namespace
                 ) -> list[dict[str, object]]:
    """Stats rows only, no drawing (--no-plots)."""
    rate = cfg.sampling_rate_hz
    n_smooth = max(1, round(cfg.smooth_s * rate))
    rows: list[dict[str, object]] = []
    for day in days:
        unit = cfg.pixel_root / f"{day['day']}_{animal}"
        means = read_channel_means(unit / "plots_roi_union" / "channel_means.csv")
        recordings, _ = day_recordings(day, days, cfg)
        day_stats: list[dict[str, object]] = []
        draw_row(None, unit, means, recordings, rate, n_smooth, day_stats)
        rows += [day_fields(animal, day) | r for r in day_stats]
        print(f"    {unit.name} ({day['timepoint']}) measured", flush=True)
    return rows


def plot_animal(animal: str, days: list[dict[str, str]], cfg: argparse.Namespace
                ) -> tuple[Path, list[dict[str, object]]]:
    rate = cfg.sampling_rate_hz
    n_smooth = max(1, round(cfg.smooth_s * rate))
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(len(days), 1, figsize=(16, 2.4 * len(days) + 0.9), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    lo, hi, longest = np.inf, -np.inf, 0
    for ax, day in zip(axes, days):
        unit = cfg.pixel_root / f"{day['day']}_{animal}"
        means = read_channel_means(unit / "plots_roi_union" / "channel_means.csv")
        recordings, missing = day_recordings(day, days, cfg)
        day_stats: list[dict[str, object]] = []
        r_lo, r_hi, n = draw_row(ax, unit, means, recordings, rate, n_smooth, day_stats)
        rows += [day_fields(animal, day) | r for r in day_stats]
        lo, hi, longest = min(lo, r_lo), max(hi, r_hi), max(longest, n)
        label = f"{day['timepoint']} · {day['day']}\nday {day['days_since_day1']}"
        if missing:
            label += f"\nno {', '.join(missing)}"
        ax.set_ylabel(label, color=INK, fontsize=9, rotation=0, ha="right", va="center")
        print(f"    {unit.name} ({day['timepoint']}) drawn", flush=True)
    pad = 0.03 * (hi - lo)
    for ax in axes:
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, longest / rate)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[-1].set_xlabel("time (s)", color=MUTED)
    fig.text(0.004, 0.5, "raw F (counts), ROI-union mean", rotation=90, va="center",
             color=MUTED, fontsize=9)
    handles: dict[str, object] = {}
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    order = [NAMES[c] for c in ("gcamp", "emo", "odd", "even") if NAMES[c] in handles]
    fig.legend([handles[k] for k in order], order, loc="upper right", ncol=len(order),
               frameon=False, fontsize=9, labelcolor=INK)
    if cfg.recordings_from_design:
        recs = sorted({r for d in days for r in d["design_recordings"]}, key=lambda r: int(r.lstrip("t")))
    else:
        recs = cfg.recordings
    fig.suptitle(f"{animal} — ROI-union raw F per timepoint, recordings {recs[0]}–{recs[-1]} joined end to end; "
                 f"same y-scale in every row\nSD = standard deviation, P99−P1 = 99th minus 1st "
                 f"percentile, both of the full-rate trace over each recording", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0.015, 0, 1, 1 - 0.55 / fig.get_figheight()))
    out = cfg.pixel_root / cfg.out_dir
    out.mkdir(exist_ok=True)
    path = out / f"{animal}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    gc.collect()
    return path, rows


def write_stats(path: Path, rows: list[dict[str, object]], replaced: set[str]) -> None:
    """Write `rows`, keeping existing rows of animals not in `replaced`."""
    kept: list[dict[str, object]] = []
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as fh:
            kept = [r for r in csv.DictReader(fh) if r["animal"] not in replaced]
    all_rows = kept + rows
    all_rows.sort(key=lambda r: (r["animal"], int(r["days_since_day1"]),
                                 int(str(r["recording"]).lstrip("t")), LINE_ORDER.index(r["line"])))
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATS_FIELDS)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: (f"{v:.1f}" if isinstance(v, float) else v) for k, v in r.items()})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pixel-root", type=Path, default=RUN_CONFIG["pixel_root"])
    p.add_argument("--design-csv", type=Path, default=RUN_CONFIG["design_csv"])
    p.add_argument("--out-dir", default=RUN_CONFIG["out_dir"])
    p.add_argument("--stats-csv", default=RUN_CONFIG["stats_csv"], help="File name inside --out-dir.")
    p.add_argument("--no-plots", action="store_true", help="Only write the stats CSV.")
    p.add_argument("--exclude-timepoints", nargs="*", default=list(RUN_CONFIG["exclude_timepoints"]))
    p.add_argument("--recordings", nargs="*", default=list(RUN_CONFIG["recordings"]))
    p.add_argument("--recordings-from-design", action="store_true",
                   help="Draw each day's recordings as listed in the design CSV (ignores --recordings).")
    p.add_argument("--sampling-rate-hz", type=float, default=RUN_CONFIG["sampling_rate_hz"])
    p.add_argument("--smooth-s", type=float, default=RUN_CONFIG["smooth_s"])
    p.add_argument("--select", nargs="*", default=None, help="Animals, e.g. A1.")
    args = p.parse_args()
    args.recordings = tuple(args.recordings)
    return args


def main() -> None:
    args = parse_args()
    animals = read_days(args.design_csv, tuple(args.exclude_timepoints))
    if args.select:
        missing = set(args.select) - set(animals)
        if missing:
            raise ValueError(f"--select animals not found: {sorted(missing)}")
        animals = {a: d for a, d in animals.items() if a in args.select}
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for i, (animal, days) in enumerate(animals.items(), 1):
        print(f"[{i}/{len(animals)}] {animal}: {len(days)} timepoints "
              f"({', '.join(d['timepoint'] for d in days)})", flush=True)
        if args.no_plots:
            rows += animal_stats(animal, days, args)
        else:
            path, animal_rows = plot_animal(animal, days, args)
            rows += animal_rows
            print(f"  -> {path}", flush=True)
    stats_path = args.pixel_root / args.out_dir / args.stats_csv
    write_stats(stats_path, rows, set(animals))
    print(f"  -> {stats_path} ({len(rows)} rows for {len(animals)} animals)")
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
