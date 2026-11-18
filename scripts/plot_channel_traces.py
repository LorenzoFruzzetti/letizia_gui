"""GCaMP and reflectance (emo) channel traces over time, per session folder of a pixel-dump root.

Study script. For every ``<pixel data>\\<day>_<animal>\\`` folder of the cohort (``--cohort``:
its ``pixel_data``, or the cohort file's ``pixel_root``; frame rate from its acquisition facts) it reads each
recording's raw-F dumps (``pixels_f_gcamp_full.npy``, ``pixels_f_emo_full.npy``), reduces
every frame to the mean over the union of that recording's ROI boxes (the same union
roi_pixel_connectivity.py uses), and writes into ``<day>_<animal>\\plots\\``:

  * ``channel_traces.png``     all ``t#`` in numeric order, end to end (dashed lines at the
                               boundaries; the gaps between recordings are not known).
                               Top: raw F, both channels on one axis. Bottom: each
                               channel as % from its own mean over the DONE recordings.
                               A skipped recording is an empty light-grey segment of
                               ``skipped_gap_frames`` frames labelled
                               ``t# - skipped (not_interleaved)``; the title lists them.
  * ``channel_traces_t#.png``  one done recording, same layout, % from the recording's own
                               mean; the title carries the tag and the two ROI-union means.
                               The PNG of a skipped recording is deleted, not redrawn.
  * ``channel_means.csv``      recording, mean_gcamp, mean_emo, channel_verdict, vs_verdict.
                               Skipped recordings get a row with empty means and
                               channel_verdict ``not_interleaved``.

A unit whose recordings were ALL skipped gets its ``plots\\`` emptied and a one-line
``README.txt`` instead.

``--level-region roi_union`` re-labels the channels from the cache instead of trusting the
stored order: per done recording, the level of each stored volume is the median over
frames of its per-frame median over the ROI-union pixels, and the dimmer one is called
GCaMP (the pipeline rule, ``wfci.io.decide_channel_order``). Where that disagrees with
the stored ``channel_order`` the two traces are swapped and the tag gets
``relabelled roi_union``; ``vs_verdict`` is computed for the plotted order. Output goes to
``<day>_<animal>\\plots_roi_union\\`` (``plots\\`` is untouched) and the CSV gains
``stored_order``, ``plotted_order``, ``roi_level_gcamp``, ``roi_level_emo`` (levels of the
STORED volumes). This approximates the pipeline's ``--channel-level-region roi_union``:
the cache holds final-grid pixels (2x2 means of the half-res pixels the pipeline takes
the median over), so a recording with near-equal levels could come out differently.
The disagreement count check describes the stored order, so it is not applied in this mode.

Every trace is the full frame rate drawn faint plus a 1 s moving average drawn bold.
Channels are plotted AS STORED. Every dump of the single-read extraction records its
own ``channel_order`` and ``channel_verdict``; the tag is:
  * ``certain`` / ``uncertain``  the dump's ``channel_verdict``;
  * ``, disagrees verdict``       appended when a verdict table (``--channel-assignment-csv``) made a
                                  decision and the new ``channel_order`` is not the one it
                                  asks for: ``correct`` -> the OLD order
                                  (``--old-manifest-csv``), ``swapped`` -> the flipped old
                                  order, ``inseparable`` -> no comparison.
``vs_verdict`` in the CSV is ``agree`` / ``disagrees`` / ``inseparable``. The total number of
disagreements over every done recording must equal ``expected_disagreements`` (when
set) or the script raises.

The cohort's ``outputs\\batch_summary.csv`` (status ``done``/``skipped`` and verdict per
recording; ``--batch-summary-csv``, default ``cohort``) drives the recording list, the tags and
the gaps. Passing ``none`` to both ``--batch-summary-csv`` and ``--old-manifest-csv`` turns
tags and gaps off: every ``t#`` folder with dumps is drawn.

A recording with status ``needs_channel_decision`` (stored by position) is
drawn like a done one but from ``pixels_f_odd_full.npy`` / ``pixels_f_even_full.npy``,
named **odd** and **even**, with the alternation and separation in its title. In a mixed
unit the joined figure shows GCaMP/emo for the done recordings and odd/even for the others,
each as % from the mean of its own kind. The CSV then gains ``status``, ``mean_odd`` and
``mean_even``. ``--only-needs-decision`` redraws only the units holding such a recording.

Run:
    $env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
    & "$env:USERPROFILE\\anaconda3\\condabin\\conda.bat" run --no-capture-output -n wfci_gui python -u plot_channel_traces.py --cohort <cohort folder>\\cohort.yaml
    ... plot_channel_traces.py --cohort <...>\\cohort.yaml --batch-summary-csv none --old-manifest-csv none --channel-assignment-csv none --expected-disagreements none
"""
from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pixel_helpers import crop_slices
from roi_pixel_connectivity import read_meta, recording_boxes

# Edit this section to run without CLI flags.
RUN_CONFIG: dict[str, Any] = {
    # The cohort file, <cohort folder>/cohort.yaml (or pass --cohort): its pixel dumps,
    # its batch summary and its frame rate.
    "cohort_file": None,
    "channel_assignment_csv": "none",  # verdict table (recording_key, verdict); "none" = no comparison
    # "cohort" = the cohort's outputs/batch_summary.csv; "none" turns tags and gaps off; or a path.
    "batch_summary_csv": "cohort",
    "old_manifest_csv": "none",  # manifest with the OLD per-recording channel_order; "none" = no comparison
    "expected_disagreements": None,  # None skips the check
    "skipped_gap_frames": 2950,  # frames drawn for a skipped recording (118 s at 25 Hz)
    # "frame": channels as stored. "roi_union": re-decide the order from the cached
    # ROI-union levels, swap where it differs, write to plots_roi_union\ (see docstring).
    "level_region": "frame",
    "smooth_s": 1.0,
    "workers": 8,
}

# Categorical slots 1-2 of the dataviz reference palette; text stays in neutral ink.
# odd/even get their own hues so an unlabelled pair is never read as GCaMP/emo.
COLORS = {"gcamp": "#2a78d6", "emo": "#eb6834", "odd": "#1b9e77", "even": "#8e44ad"}
NAMES = {"gcamp": "GCaMP", "emo": "reflectance (emo)", "odd": "odd frames", "even": "even frames"}
INK, MUTED, GAP_FILL = "#0b0b0b", "#52514e", "#e6e5e1"
FLIP = {"gcamp_first": "emo_first", "emo_first": "gcamp_first"}
NEEDS_DECISION = "needs_channel_decision"
# Which volumes and sidecar each drawable status reads.
CHANNELS = {"done": ("gcamp", "emo"), NEEDS_DECISION: ("odd", "even")}
META_NAME = {"done": "pixels_meta_full.npz", NEEDS_DECISION: "pixels_meta_odd_even_full.npz"}


def is_none(value: Any) -> bool:
    return value is None or str(value).lower() == "none"


def t_number(name: str) -> int:
    """Numeric part of 't#', so t10 sorts after t9."""
    return int(name[1:])


def roi_union_traces(rec: Path, status: str = "done") -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Per-frame ROI-union mean of both raw-F volumes ({'gcamp': (n_time,), 'emo': ...},
    or 'odd'/'even' for an unassigned recording), and each volume's level: median over
    frames of the per-frame ROI-union median."""
    meta = read_meta(rec / META_NAME[status])
    _, boxes = recording_boxes(meta)
    region = meta["region"]
    union = np.zeros((region[1] - region[0], region[3] - region[2]), dtype=bool)
    for box in boxes.values():
        rows, cols = crop_slices(box, int(meta["y_1"]), int(meta["x_2"]), region)
        union[rows, cols] = True
    traces, levels = {}, {}
    for channel in CHANNELS[status]:
        volume = np.load(rec / f"pixels_f_{channel}_full.npy", mmap_mode="r")
        if volume.shape[1:] != union.shape:
            raise ValueError(f"{rec}: {channel} volume {volume.shape} does not match the region")
        pixels = np.asarray(volume[:, union], dtype=np.float64)  # (n_time, n_union_pixels)
        traces[channel] = pixels.mean(axis=1)
        levels[channel] = float(np.median(np.median(pixels, axis=1)))
    return traces, levels


def read_meta_channel(rec: Path, status: str = "done") -> dict[str, str]:
    """The dump's own channel fields ('' when the dump predates them)."""
    with np.load(rec / META_NAME[status], allow_pickle=False) as meta:
        return {key: str(meta[key]) if key in meta.files else ""
                for key in ("channel_order", "channel_verdict",
                            "channel_alternation", "channel_separation")}


def moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving mean, same length; edges average over what is available."""
    kernel = np.ones(n)
    return np.convolve(x, kernel, "same") / np.convolve(np.ones_like(x), kernel, "same")


def draw(axes, t: np.ndarray, traces: dict[str, np.ndarray], reference: dict[str, float],
         n_smooth: int) -> None:
    """Raw F on axes[0], % from `reference` on axes[1]; faint full rate + bold 1 s mean."""
    for channel, trace in traces.items():
        for ax, y in ((axes[0], trace), (axes[1], (trace / reference[channel] - 1.0) * 100.0)):
            ax.plot(t, y, color=COLORS[channel], lw=0.5, alpha=0.25)
            ax.plot(t, moving_mean(y, n_smooth), color=COLORS[channel], lw=2,
                    label=NAMES[channel] if ax is axes[0] else None)


def style(fig, axes, title: str, pct_label: str) -> None:
    axes[0].set_ylabel("raw F (counts)", color=MUTED)
    axes[1].set_ylabel(pct_label, color=MUTED)
    axes[1].set_xlabel("time (s)", color=MUTED)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    for ax in axes:
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    # One entry per channel (the joined figure draws every recording with a label), in
    # the title row so it never sits on the data.
    handles = dict(zip(*reversed(axes[0].get_legend_handles_labels())))
    fig.legend(handles.values(), handles.keys(), loc="upper right", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))


def write_all_skipped(folder: Path, records: list[dict[str, Any]], plots_dir: str) -> str:
    """No done recording: empty the plots folder and leave a one-line README in its place."""
    out = folder / plots_dir
    out.mkdir(exist_ok=True)
    for path in out.iterdir():
        if not path.is_file():
            raise RuntimeError(f"{path}: unexpected sub-folder in {plots_dir}/, not deleting it")
        path.unlink()
    names = [r["recording"] for r in records]
    span = names[0] if len(names) == 1 else f"{names[0]}-{names[-1]}"
    verdicts = sorted({r["channel_verdict"] for r in records})
    (out / "README.txt").write_text(
        f"All recordings ({span}) skipped as {'/'.join(verdicts)} in the "
        f"single-read extraction; no dumps, no plots.\n", encoding="utf-8")
    return f"{folder.name}: all {len(records)} recordings skipped, README.txt written"


def check_disk_against_records(folder: Path, records: list[dict[str, Any]]) -> None:
    """Every t# on disk must agree with batch_summary: labelled dumps <=> done, odd/even
    dumps <=> needs_channel_decision, no dumps <=> skipped."""
    status = {r["recording"]: r["status"] for r in records}
    for rec in sorted(folder.glob("t*"), key=lambda p: t_number(p.name)):
        if not rec.is_dir():
            continue
        found = ("done" if (rec / META_NAME["done"]).is_file()
                 else NEEDS_DECISION if (rec / META_NAME[NEEDS_DECISION]).is_file()
                 else "skipped")
        listed = status.get(rec.name)
        if found != listed:
            raise RuntimeError(f"{rec}: dumps on disk look {found!r} but batch_summary "
                               f"lists it as {listed!r}")
    for name, value in status.items():
        if value in META_NAME and not (folder / name / META_NAME[value]).is_file():
            raise RuntimeError(f"{folder / name}: listed as {value} but has no dumps")


def make_tag(record: dict[str, Any]) -> str:
    """Title tag: verdict (when tags are on), then the relabel and verdict-table notes."""
    parts = [record["channel_verdict"]] if record["show_verdict"] else []
    if record["status"] == NEEDS_DECISION:
        parts.append(f"alternation {float(record['alternation']):+.2f}, "
                     f"separation {float(record['separation']):.1f}")
    if record.get("relabelled"):
        parts.append("relabelled roi_union")
    if record["vs_verdict"] == "disagrees":
        parts.append("disagrees verdict")
    return ", ".join(parts)


def plot_folder(job: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Draw one unit. Returns (log line, relabelled keys, keys disagreeing with the verdict table)."""
    started = time.perf_counter()
    folder = Path(job["folder"])
    rate, n_smooth = job["sampling_rate_hz"], max(1, round(job["smooth_s"] * job["sampling_rate_hz"]))
    records: list[dict[str, Any]] = job["records"]  # numeric t# order, done and skipped
    if job["check_disk"]:
        check_disk_against_records(folder, records)
    done = [r for r in records if r["status"] in CHANNELS]  # everything with volumes to draw
    skipped = [r for r in records if r["status"] == "skipped"]
    if not done:
        return write_all_skipped(folder, records, job["plots_dir"]), [], []
    has_unassigned = any(r["status"] == NEEDS_DECISION for r in records)
    roi_mode = job["level_region"] == "roi_union"

    out = folder / job["plots_dir"]
    out.mkdir(exist_ok=True)
    # A skipped recording keeps no per-recording PNG: one from an earlier run would be stale.
    for r in skipped:
        (out / f"channel_traces_{r['recording']}.png").unlink(missing_ok=True)

    traces_by_rec: dict[str, dict[str, np.ndarray]] = {}
    means_by_rec: dict[str, dict[str, float]] = {}
    for r in done:
        traces, levels = roi_union_traces(folder / r["recording"], r["status"])
        r["roi_levels"], r["plotted_order"] = levels, r["order"]
        if roi_mode and r["status"] == "done" and levels["gcamp"] > levels["emo"]:
            # Over the ROI union the stored GCaMP volume is the brighter one: under
            # dimmer-is-GCaMP the labels swap (equal levels keep the stored order).
            traces = {"gcamp": traces["emo"], "emo": traces["gcamp"]}
            r["relabelled"], r["plotted_order"] = True, FLIP[r["order"]]
            r["vs_verdict"] = compare_to_table(r["plotted_order"], r["table_verdict"], r["old_order"])
        r["tag"] = make_tag(r)
        means = {c: float(tr.mean()) for c, tr in traces.items()}
        traces_by_rec[r["recording"]], means_by_rec[r["recording"]] = traces, means

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        n_frames = len(next(iter(traces.values())))
        draw(axes, np.arange(n_frames) / rate, traces, means, n_smooth)
        tag = f" [{r['tag']}]" if r["tag"] else ""
        first, second = CHANNELS[r["status"]]
        short = {"gcamp": "GCaMP", "emo": "emo", "odd": "odd", "even": "even"}
        style(fig, axes, f"{folder.name} {r['recording']}{tag} — mean {short[first]} "
                         f"{means[first]:.0f}, mean {short[second]} {means[second]:.0f}",
              "% from recording mean")
        fig.savefig(out / f"channel_traces_{r['recording']}.png", dpi=120)
        plt.close(fig)

    with open(out / "channel_means.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["recording", "mean_gcamp", "mean_emo", "channel_verdict", "vs_verdict"]
        if roi_mode:
            header += ["stored_order", "plotted_order", "roi_level_gcamp", "roi_level_emo"]
        if has_unassigned:
            header += ["status", "mean_odd", "mean_even"]
        writer.writerow(header)

        def cell(values: dict[str, float] | None, channel: str) -> str:
            """One channel's value, blank when this recording has no such channel."""
            return f"{values[channel]:.2f}" if values and channel in values else ""

        for r in records:
            means = means_by_rec.get(r["recording"])
            row = [r["recording"], cell(means, "gcamp"), cell(means, "emo"),
                   r["channel_verdict"], r["vs_verdict"]]
            if roi_mode:
                # Levels are those of the STORED volumes (stored_order labels).
                levels = r.get("roi_levels")
                row += [r["order"], r.get("plotted_order", ""),
                        cell(levels, "gcamp"), cell(levels, "emo")]
            if has_unassigned:
                row += [r["status"], cell(means, "odd"), cell(means, "even")]
            writer.writerow(row)

    # Joined figure: every recording in t# order; % from the mean over the recordings that
    # have that channel (GCaMP/emo over the done ones, odd/even over the unassigned ones).
    joined: dict[str, list[np.ndarray]] = {}
    for r in done:
        for channel, trace in traces_by_rec[r["recording"]].items():
            joined.setdefault(channel, []).append(trace)
    folder_mean = {c: float(np.concatenate(parts).mean()) for c, parts in joined.items()}
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    label_transform = axes[0].get_xaxis_transform()
    start = 0
    for r in records:
        if r["status"] in CHANNELS:
            traces = traces_by_rec[r["recording"]]
            n = len(next(iter(traces.values())))
            draw(axes, (start + np.arange(n)) / rate, traces, folder_mean, n_smooth)
            label = f"{r['recording']}\n[{r['tag']}]" if r["tag"] else r["recording"]
            color = INK
        else:
            # Empty segment: no data, shaded, both edges dashed.
            n = job["skipped_gap_frames"]
            for ax in axes:
                ax.axvspan(start / rate, (start + n) / rate, color=GAP_FILL, lw=0, zorder=0)
                ax.axvline((start + n) / rate, color=MUTED, lw=0.8, ls="--")
            label = f"{r['recording']} - skipped ({r['channel_verdict']})"
            color = MUTED
        for ax in axes:
            ax.axvline(start / rate, color=MUTED, lw=0.8, ls="--")
        axes[0].text((start + n / 2) / rate, 1.01, label, transform=label_transform,
                     ha="center", va="bottom", fontsize=8, color=color)
        start += n
    for ax in axes:
        ax.set_xlim(0, start / rate)
    skipped_note = f"; skipped: {', '.join(r['recording'] for r in skipped)}" if skipped else ""
    if roi_mode:
        skipped_note += "; channels labelled by ROI-union level"
    style(fig, axes, f"{folder.name} — ROI-union mean per channel, recordings joined end to end "
                     f"(gaps between recordings not shown){skipped_note}",
          "% from mean over recordings of that channel" if has_unassigned
          else "% from mean of done recordings")
    fig.subplots_adjust(top=0.86)
    fig.savefig(out / "channel_traces.png", dpi=120)
    plt.close(fig)
    relabelled = [f"{folder.name}/{r['recording']}" for r in done if r.get("relabelled")]
    disagreeing = [f"{folder.name}/{r['recording']}" for r in done if r["vs_verdict"] == "disagrees"]
    return (f"{folder.name}: {len(done)} done, {len(skipped)} skipped, {len(relabelled)} relabelled "
            f"in {time.perf_counter() - started:.1f}s"), relabelled, disagreeing


def read_verdict_table(path: Any) -> dict[str, str]:
    """Verdict-table verdict per recording_key ('<day>_<animal>/t#'): correct / swapped / inseparable."""
    if is_none(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["recording_key"]: row["verdict"] for row in csv.DictReader(fh)}


def read_old_orders(path: Any) -> dict[str, str]:
    """OLD channel order per recording key from the explicit manifest (';'-joined columns)."""
    if is_none(path):
        return {}
    orders = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            recs, recs_orders = row["recordings"].split(";"), row["channel_order"].split(";")
            if len(recs) != len(recs_orders):
                raise ValueError(f"{path}: {row['day']}_{row['animal']} has {len(recs)} recordings "
                                 f"but {len(recs_orders)} channel orders")
            for rec, order in zip(recs, recs_orders):
                orders[f"{row['day']}_{row['animal']}/{rec}"] = order
    return orders


def read_summary(path: Any) -> dict[str, list[dict[str, str]]]:
    """batch_summary rows per unit '<day>_<animal>', each list in numeric t# order."""
    units: dict[str, list[dict[str, str]]] = {}
    seen = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            unit, key = f"{row['day']}_{row['animal']}", f"{row['day']}_{row['animal']}/{row['recording']}"
            if key in seen:
                raise ValueError(f"{path}: {key} appears twice")
            if row["status"] not in ("done", "skipped", NEEDS_DECISION):
                raise ValueError(f"{path}: {key} has status {row['status']!r}")
            seen.add(key)
            units.setdefault(unit, []).append(row)
    for rows in units.values():
        rows.sort(key=lambda row: t_number(row["recording"]))
    return units


def lookup_table_verdict(key: str, table_verdicts: dict[str, str],
               old_orders: dict[str, str]) -> tuple[str, str]:
    """(verdict-table verdict, OLD order) for one recording; ('', '') when the comparison is off."""
    if not table_verdicts or not old_orders:
        return "", ""
    return table_verdicts[key], old_orders[key]


def compare_to_table(new_order: str, table_verdict: str, old_order: str) -> str:
    """'agree' / 'disagrees' / 'inseparable' against the verdict table; '' when the comparison is off."""
    if not table_verdict or not old_order:
        return ""
    if table_verdict == "inseparable":
        return "inseparable"
    intended = {"correct": old_order, "swapped": FLIP[old_order]}[table_verdict]
    return "agree" if new_order == intended else "disagrees"


def build_records(pixel_root: Path, args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    """Per unit: one record per recording (status, verdict, vs_verdict, tag), numeric t# order."""
    table_verdicts = read_verdict_table(args.channel_assignment_csv)
    old_orders = read_old_orders(args.old_manifest_csv)
    units: dict[str, list[dict[str, Any]]] = {}

    if is_none(args.batch_summary_csv):
        # Tags and gaps off: every t# folder with dumps, as stored.
        for folder in sorted(f for f in pixel_root.iterdir() if f.is_dir()):
            recs = sorted((p for p in folder.glob("t*") if (p / "pixels_meta_full.npz").is_file()),
                          key=lambda p: t_number(p.name))
            if not recs:
                continue
            units[folder.name] = []
            for rec in recs:
                meta = read_meta_channel(rec)
                table_verdict, old_order = lookup_table_verdict(f"{folder.name}/{rec.name}",
                                                    table_verdicts, old_orders)
                units[folder.name].append(dict(
                    recording=rec.name, status="done", channel_verdict=meta["channel_verdict"],
                    show_verdict=False, order=meta["channel_order"], table_verdict=table_verdict,
                    old_order=old_order,
                    vs_verdict=compare_to_table(meta["channel_order"], table_verdict, old_order)))
        return units

    for unit, rows in read_summary(args.batch_summary_csv).items():
        units[unit] = []
        for row in rows:
            key = f"{unit}/{row['recording']}"
            record = dict(recording=row["recording"], status=row["status"],
                          channel_verdict=row["channel_verdict"], show_verdict=True,
                          vs_verdict="", order="")
            if row["status"] == NEEDS_DECISION:
                meta = read_meta_channel(pixel_root / unit / row["recording"], NEEDS_DECISION)
                if meta["channel_verdict"] != row["channel_verdict"]:
                    raise ValueError(f"{key}: dump verdict {meta['channel_verdict']!r} != "
                                     f"batch_summary {row['channel_verdict']!r}")
                record.update(order=meta["channel_order"], table_verdict="", old_order="",
                              alternation=meta["channel_alternation"],
                              separation=meta["channel_separation"])
            if row["status"] == "done":
                meta = read_meta_channel(pixel_root / unit / row["recording"])
                if meta["channel_verdict"] != row["channel_verdict"]:
                    raise ValueError(f"{key}: dump verdict {meta['channel_verdict']!r} != "
                                     f"batch_summary {row['channel_verdict']!r}")
                table_verdict, old_order = lookup_table_verdict(key, table_verdicts, old_orders)
                record.update(order=meta["channel_order"], table_verdict=table_verdict,
                              old_order=old_order,
                              vs_verdict=compare_to_table(meta["channel_order"], table_verdict, old_order))
            units[unit].append(record)
    return units


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cohort", dest="cohort_file", default=RUN_CONFIG["cohort_file"],
                   help="The cohort file, <cohort folder>/cohort.yaml.")
    p.add_argument("--channel-assignment-csv", default=RUN_CONFIG["channel_assignment_csv"],
                   help="Verdict table (recording_key, verdict); 'none' to skip.")
    p.add_argument("--batch-summary-csv", default=RUN_CONFIG["batch_summary_csv"],
                   help="status + channel_verdict per recording: 'cohort' (the cohort's), 'none' "
                        "(tags and gaps off) or a path.")
    p.add_argument("--old-manifest-csv", default=RUN_CONFIG["old_manifest_csv"],
                   help="Manifest with the OLD per-recording channel_order; 'none' to skip.")
    p.add_argument("--expected-disagreements", default=RUN_CONFIG["expected_disagreements"],
                   help="Raise unless exactly this many done recordings disagree with the verdict table; "
                        "'none' to skip.")
    p.add_argument("--level-region", choices=("frame", "roi_union"),
                   default=RUN_CONFIG["level_region"],
                   help="roi_union: re-label channels from the cached ROI-union levels, "
                        "into plots_roi_union/.")
    p.add_argument("--skipped-gap-frames", type=int, default=RUN_CONFIG["skipped_gap_frames"])
    p.add_argument("--smooth-s", type=float, default=RUN_CONFIG["smooth_s"])
    p.add_argument("--workers", type=int, default=RUN_CONFIG["workers"])
    p.add_argument("--select", nargs="*", default=None, help="Folder names, e.g. 200101_A1.")
    p.add_argument("--only-needs-decision", action="store_true",
                   help="Redraw only the units holding a needs_channel_decision recording.")
    args = p.parse_args()
    # The dumps, the batch summary and the frame rate come from the cohort file.
    from wfci_gui.logic.cohort import acquisition_value, load_script_cohort
    cohort = load_script_cohort(args.cohort_file)
    args.pixel_root = cohort.layout.pixel_data
    args.sampling_rate_hz = float(acquisition_value(cohort, "sampling_rate_hz"))
    if str(args.batch_summary_csv).lower() == "cohort":
        args.batch_summary_csv = cohort.layout.batch_summary
    return args


def main() -> None:
    args = parse_args()
    units = build_records(args.pixel_root, args)

    # The verdict-table comparison is checked over EVERY done recording, before --select narrows.
    disagreeing = [f"{unit}/{r['recording']}" for unit, records in units.items()
                   for r in records if r["vs_verdict"] == "disagrees"]
    print(f"{len(disagreeing)} recordings (stored order) disagree with the verdict table: {', '.join(disagreeing)}")
    roi_mode = args.level_region == "roi_union"
    # The expected count describes the STORED order; roi_union re-labels, so skip it there.
    if (not roi_mode and not is_none(args.expected_disagreements)
            and len(disagreeing) != int(args.expected_disagreements)):
        raise RuntimeError(f"expected {args.expected_disagreements} recordings disagreeing with the verdict table, "
                           f"found {len(disagreeing)}")

    names = sorted(units)
    if args.select:
        missing = set(args.select) - set(names)
        if missing:
            raise ValueError(f"--select names not found: {sorted(missing)}")
        names = [n for n in names if n in set(args.select)]
    if args.only_needs_decision:
        names = [n for n in names if any(r["status"] == NEEDS_DECISION for r in units[n])]
    jobs = [dict(folder=str(args.pixel_root / name), records=units[name],
                 check_disk=not is_none(args.batch_summary_csv),
                 level_region=args.level_region,
                 plots_dir="plots_roi_union" if roi_mode else "plots",
                 skipped_gap_frames=args.skipped_gap_frames,
                 sampling_rate_hz=args.sampling_rate_hz, smooth_s=args.smooth_s) for name in names]
    n_skipped = sum(r["status"] == "skipped" for name in names for r in units[name])
    print(f"{len(jobs)} folders under {args.pixel_root}; {n_skipped} skipped recordings")
    started = time.perf_counter()
    relabelled_all, disagreeing_plotted = [], []
    with Pool(min(args.workers, len(jobs))) as pool:
        for i, (msg, relabelled, disagree) in enumerate(pool.imap_unordered(plot_folder, jobs), 1):
            relabelled_all += relabelled
            disagreeing_plotted += disagree
            print(f"  [{i}/{len(jobs)}] {msg}", flush=True)
    if roi_mode:
        print(f"{len(relabelled_all)} relabelled by ROI-union level: "
              f"{', '.join(sorted(relabelled_all))}")
        print(f"{len(disagreeing_plotted)} (plotted order) disagree with the verdict table: "
              f"{', '.join(sorted(disagreeing_plotted))}")
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
