"""Quality-control table of a cohort: one row per recording, with a SUGGESTED verdict.

Each check looks for a problem that can occur in real data and that the batch itself
does not report:

  completeness   4 dump files + connectivity_full.npz, no ``.partial`` leftovers, volume shape
  geometry       the dump's Bregma and ROI set are the recording's OWN rebuilt set (a shared
                 fallback would give one Bregma for all)
  dump vs traces box means of pixels_dff_full.npy reproduce roi_fluorescence_full.csv
                 (~1e-3 pp, the float16 cost; the dump is already in %)
  channels       the batch's verdict and odd/even separation (certain / uncertain /
                 not_interleaved / needs_channel_decision)
  glitch         max |field mean| of the median dF/F (ratio; > 1 = a one-frame
                 reflectance collapse, exclude; 0.2-0.3 check the R first)
  levels         brightest-half mean of mean_f / mean_r, also per session: a step in GCaMP
                 level with flat reflectance is an acquisition change
  stale          outputs left from an earlier run for a recording the batch now skips
  rectangle      the midline-rectangle channel rule (channel_rectangle.py) on the dump's odd/even
                 mean images: which parity it calls GCaMP, and whether the batch's order agrees.
                 For a recording that needs a channel decision it is the proposed order. The
                 rectangle is the recording's own when its ROI set has a ``channel_rectangle``
                 (moved in the ROI editor), else the cohort's channel_rectangle_params.json; the
                 ``rectangle_source`` column says which (``roi_set`` / ``params``).
  chosen order   ``channel_order`` the user set in the ROI editor for this recording, if any
                 (``chosen_order``), and whether the batch's order agrees with it.

NOTHING IS EXCLUDED HERE: ``suggest`` is ``exclude`` / ``check`` / ``ok``
with a reason; the user decides on the QC page.

Writes ``qc_table.csv`` and ``qc_levels_by_session.csv`` in ``<cohort folder>/outputs``.

    conda run --no-capture-output -n wfci_gui python -u qc_cohort.py --cohort <cohort folder>/cohort.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from wfci_gui.logic.cohort import load_script_cohort, load_tables, used_recordings, write_text_atomic  # noqa: E402
from wfci_gui.logic.steps import read_batch_summary, summary_status_by_key  # noqa: E402

import channel_rectangle as rectangle  # noqa: E402

RUN_CONFIG: dict[str, Any] = {
    "cohort_file": None,  # <cohort folder>/cohort.yaml (or pass --cohort)
    "glitch_threshold": 1.0,       # max |field mean| (ratio) above which exclusion is suggested
    "glitch_check": 0.2,           # above this, suggest checking R against the session-mates
    "trace_tolerance_pp": 0.01,    # dump-vs-CSV max difference (pp) above which to check
    "median_cache": "pixels_median_dff_20s_full.npy",  # glitch scan volume (ratio)
    "check_traces": True,          # False skips the dump-vs-CSV reduction (reads 40 MB per recording)
    "prefer_cli_args": True,
}
DUMP_FILES = ("pixels_dff_full.npy", "pixels_f_gcamp_full.npy", "pixels_f_emo_full.npy",
              "pixels_meta_full.npz")


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the QC table of a cohort (suggestions only).")
    p.add_argument("--cohort", dest="cohort_file", default=defaults["cohort_file"])
    p.add_argument("--glitch-threshold", type=float, default=defaults["glitch_threshold"])
    p.add_argument("--glitch-check", type=float, default=defaults["glitch_check"])
    p.add_argument("--trace-tolerance-pp", type=float, default=defaults["trace_tolerance_pp"])
    p.add_argument("--median-cache", default=defaults["median_cache"])
    p.add_argument("--check-traces", action=argparse.BooleanOptionalAction, default=defaults["check_traces"])
    return p.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{k: v for k, v in config.items() if k != "prefer_cli_args"})


def brightest_half_mean(image: np.ndarray) -> float:
    """Mean of the pixels at or above the median: cortex, not the dark surround."""
    values = np.asarray(image, dtype=np.float64).ravel()
    return float(values[values >= np.median(values)].mean())


def trace_difference(pixel_dir: Path, output_dir: Path, meta, boxes: dict) -> float:
    """Worst |box mean of the dumped dF/F - roi_fluorescence_full.csv| over frames and ROIs (pp)."""
    volume = np.load(pixel_dir / "pixels_dff_full.npy", mmap_mode="r")  # (time, rows, cols), %
    traces = pd.read_csv(output_dir / "roi_fluorescence_full.csv")
    r_crop, c_crop = int(meta["region"][0]), int(meta["region"][2])
    y_1, x_2 = int(meta["y_1"]), int(meta["x_2"])
    worst = 0.0
    for label, box in boxes.items():
        if label not in traces:
            continue
        rows = slice(y_1 + box["row_start"] - 1 - r_crop, y_1 + box["row_end"] - r_crop)
        cols = slice(x_2 + box["col_start"] - 1 - c_crop, x_2 + box["col_end"] - c_crop)
        # float64 before reducing: float16 sums overflow.
        dump_trace = np.asarray(volume[:, rows, cols], dtype=np.float64).mean(axis=(1, 2))
        csv_trace = traces[label].to_numpy(dtype=np.float64)
        n = min(len(dump_trace), len(csv_trace))
        worst = max(worst, float(np.nanmax(np.abs(dump_trace[:n] - csv_trace[:n]))))
    return worst


def glitch_value(pixel_dir: Path, cache_name: str) -> tuple[float | None, str]:
    """Max |field mean| over frames of the median dF/F (ratio), and the volume used."""
    path = pixel_dir / cache_name
    scale = 1.0
    if not path.exists():
        # No cache yet: the mean-baseline dump in % (divided to a ratio). Its slow drift
        # makes the value larger, so it is a coarser screen than the cache.
        path, scale = pixel_dir / "pixels_dff_full.npy", 0.01
        if not path.exists():
            return None, ""
    volume = np.load(path, mmap_mode="r")
    field_mean = np.nanmean(np.asarray(volume, dtype=np.float32).reshape(volume.shape[0], -1), axis=1)
    return float(np.nanmax(np.abs(field_mean)) * scale), path.name


def rectangle_columns(pixel_dir: Path, roi_set: Path, params) -> dict[str, Any]:
    """The rectangle rule on the dump's mean images; empty when there is nothing to read.

    A labelled dump stores mean_f / mean_r and its channel_order says which parity each
    is; a not-interleaved one (needs_channel_decision) stores mean_odd /
    mean_even. Both are (256, 256), the grid the rule is defined on.
    """
    labelled, odd_even = pixel_dir / "pixels_meta_full.npz", pixel_dir / "pixels_meta_odd_even_full.npz"
    meta_path = labelled if labelled.exists() else odd_even if odd_even.exists() else None
    if meta_path is None or not roi_set.exists():
        return {}
    meta = np.load(meta_path)
    order = str(meta["channel_order"]) if "channel_order" in meta.files else ""
    if order in ("gcamp_first", "emo_first"):
        mean_f = np.asarray(meta["mean_f"], dtype=np.float64)
        mean_r = np.asarray(meta["mean_r"], dtype=np.float64)
        odd, even = (mean_f, mean_r) if order == "gcamp_first" else (mean_r, mean_f)
    else:
        odd = np.asarray(meta["mean_odd"], dtype=np.float64)
        even = np.asarray(meta["mean_even"], dtype=np.float64)
        order = ""
    data = yaml.safe_load(roi_set.read_text(encoding="utf-8"))
    # The recording's own rectangle when its ROI set has one.
    params, source = rectangle.rectangle_params_for(roi_set, params)
    bregma = rectangle.bregma_point(int(data["bregma_row"]), int(data["bregma_col"]))
    lambda_ = rectangle.lambda_point(bregma, float(data["lambda_row_offset"]),
                                     float(data.get("lambda_angle_deg", 0.0)),
                                     bool(data.get("lambda_above_bregma", False)))
    corners = rectangle.params_corners(bregma, lambda_, params)
    problem = rectangle.decision_problem(odd.shape, corners, params.statistic)
    if problem:
        return dict(rect_channel_order="", rect_ratio="", rect_agrees="", rect_note=problem,
                    rectangle_source=source)
    decision = rectangle.decide(odd, even, bregma, lambda_, params)
    return dict(rect_channel_order=decision.channel_order, rect_ratio=round(decision.ratio, 3),
                rect_agrees=("" if not order else decision.channel_order == order), rect_note="",
                rectangle_source=source)


def qc_row(key: str, cohort, summary_status: str | None, summary_row: dict, args) -> dict[str, Any]:
    unit, recording = key.split("/")
    day, animal = unit.split("_", 1)
    output_dir = cohort.layout.outputs / unit / recording
    pixel_dir = cohort.layout.pixel_data / unit / recording
    roi_set = cohort.layout.rebuilt_roi_sets / f"{unit}_{recording}.yaml"
    row: dict[str, Any] = dict(recording=key, day=day, animal=animal, rec=recording,
                               batch_status=summary_status or "not run",
                               channel_verdict=summary_row.get("channel_verdict", ""),
                               channel_order_source=summary_row.get("channel_order_source", ""),
                               mean_offdiag_R=summary_row.get("mean_offdiag_R", ""))
    reasons_exclude, reasons_check = [], []
    npz = (output_dir / "connectivity_full.npz").exists()
    missing = [f for f in DUMP_FILES if not (pixel_dir / f).exists()]
    partial = sorted(p.name for p in pixel_dir.glob("*.partial")) if pixel_dir.exists() else []
    row.update(has_npz=npz, missing_files=";".join(missing), partial_files=";".join(partial))
    if summary_status == "needs_channel_decision":
        reasons_exclude.append("channels do not alternate (needs a channel decision)")
        stale = [p.name for p in (pixel_dir.glob("pixels_median_dff_*") if pixel_dir.exists() else [])]
        if npz or stale:
            row["stale_outputs"] = ";".join((["connectivity_full.npz"] if npz else []) + stale)
            reasons_check.append("stale outputs from an earlier run are still on disk")
    elif summary_status is None:
        reasons_exclude.append("not extracted")
    elif missing or not npz or partial:
        reasons_exclude.append("incomplete: " + ", ".join(missing + (["connectivity_full.npz"] if not npz else [])
                                                          + partial))
    if row["channel_verdict"] == "not_interleaved":
        reasons_exclude.append("verdict not_interleaved")
    elif row["channel_verdict"] == "uncertain":
        reasons_check.append("channel verdict uncertain")

    meta_path = pixel_dir / "pixels_meta_full.npz"
    if meta_path.exists():
        meta = np.load(meta_path)
        row.update(bregma_row=int(meta["bregma_row"]), bregma_col=int(meta["bregma_col"]),
                   shape="x".join(str(v) for v in meta["shape"]),
                   channel_separation=round(float(meta["channel_separation"]), 2)
                   if "channel_separation" in meta else "",
                   gcamp_level=round(brightest_half_mean(meta["mean_f"]), 1),
                   emo_level=round(brightest_half_mean(meta["mean_r"]), 1))
        if roi_set.exists():
            data = yaml.safe_load(roi_set.read_text(encoding="utf-8"))
            same_bregma = (int(data["bregma_row"]), int(data["bregma_col"])) == (row["bregma_row"], row["bregma_col"])
            same_set = Path(str(meta["roi_set"])).name == roi_set.name
            row["geometry_ok"] = same_bregma and same_set
            if not row["geometry_ok"]:
                reasons_exclude.append("dump geometry is not its own ROI set")
            if args.check_traces and not missing and (output_dir / "roi_fluorescence_full.csv").exists():
                row["trace_diff_pp"] = round(trace_difference(pixel_dir, output_dir, meta, data["boxes"]), 5)
                if row["trace_diff_pp"] > args.trace_tolerance_pp:
                    reasons_check.append(f"dump differs from the ROI traces by {row['trace_diff_pp']} pp")
        else:
            row["geometry_ok"] = False
            reasons_exclude.append("no rebuilt ROI set")
        glitch, source = glitch_value(pixel_dir, args.median_cache)
        row.update(glitch_max=round(glitch, 4) if glitch is not None else "", glitch_source=source)
        if glitch is not None and glitch > args.glitch_threshold:
            reasons_exclude.append(f"glitch {glitch:.2f} > {args.glitch_threshold:g}")
        elif glitch is not None and glitch > args.glitch_check:
            reasons_check.append(f"glitch {glitch:.2f}: compare its R with its session-mates")
    row.update(rectangle_columns(pixel_dir, roi_set, args.rectangle_params))
    chosen = rectangle.read_roi_set_channel_order(roi_set) if roi_set.exists() else None
    row["chosen_order"] = chosen or ""
    if chosen and meta_path.exists():
        batch_order = str(np.load(meta_path)["channel_order"])
        if batch_order in ("gcamp_first", "emo_first") and batch_order != chosen:
            reasons_check.append(f"chosen in the ROI editor: {chosen}, the batch used {batch_order}")
    if row.get("rect_agrees") is False:
        reasons_check.append(f"channel rectangle says {row['rect_channel_order']} "
                             f"(x{row['rect_ratio']}), the batch used the other order")
    if summary_status == "needs_channel_decision" and row.get("rect_channel_order"):
        reasons_check.append(f"channel rectangle proposes {row['rect_channel_order']} "
                             f"(x{row['rect_ratio']})")
    suggest = "exclude" if reasons_exclude else "check" if reasons_check else "ok"
    row.update(suggest=suggest, reason="; ".join(reasons_exclude + reasons_check))
    return row


def main() -> None:
    args = build_runtime_args()
    cohort = load_script_cohort(args.cohort_file)
    # The cohort's global rectangle parameters (the viewer's Save); missing = the defaults.
    args.rectangle_params = rectangle.params_or_default(cohort.layout.channel_rectangle_params)
    tables = load_tables(cohort)
    used = used_recordings(tables)
    if used.empty:
        raise ValueError("no runnable recordings in recordings.csv")
    summary = read_batch_summary(cohort)
    status = summary_status_by_key(summary)
    summary_rows = {f"{r['day']}_{r['animal']}/{r['recording']}": r for r in summary.to_dict("records")}
    rows = []
    for i, key in enumerate(used["key"], start=1):
        rows.append(qc_row(key, cohort, status.get(key), summary_rows.get(key, {}), args))
        print(f"[{i}/{len(used)}] {key}: {rows[-1]['suggest']} {rows[-1]['reason']}", flush=True)
    table = pd.DataFrame(rows)
    timepoint = dict(zip(tables["sessions"]["day"], tables["sessions"]["timepoint"]))
    table.insert(1, "timepoint", table["day"].map(timepoint))
    out = cohort.layout.outputs / "qc_table.csv"
    write_text_atomic(out, table.to_csv(index=False, lineterminator="\n"))

    levels = (table.dropna(subset=["gcamp_level"]).groupby(["day", "timepoint"], as_index=False)
              .agg(n=("recording", "size"), gcamp_level=("gcamp_level", "median"),
                   emo_level=("emo_level", "median")) if "gcamp_level" in table else pd.DataFrame())
    if not levels.empty:
        levels["gcamp_vs_first"] = (levels["gcamp_level"] / levels["gcamp_level"].iloc[0]).round(3)
        levels["emo_vs_first"] = (levels["emo_level"] / levels["emo_level"].iloc[0]).round(3)
    levels_out = cohort.layout.outputs / "qc_levels_by_session.csv"
    write_text_atomic(levels_out, levels.to_csv(index=False, lineterminator="\n"))

    print(f"\nQC table: {out}")
    print(table["suggest"].value_counts().to_string())
    print(f"\nIllumination level per session (brightest-half mean; a GCaMP step with flat "
          f"reflectance is an acquisition change):\n{levels.to_string(index=False)}")
    print("Nothing was excluded: confirm the exclusions on the QC page.")


if __name__ == "__main__":
    main()
