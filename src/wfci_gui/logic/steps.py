"""Step definitions, prerequisites and status read from files on disk.

Status is never stored by the GUI: every call recomputes it from the cohort's files,
so opening a cohort shows what is really there, including stale outputs left by an
earlier run. The job history only adds "last run" notes.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from .cohort import Cohort, acquisition_missing, parse_bool, recording_key, used_recordings
from .manifest import manifest_matches

# Disk needed per recording (workflow step 0): pixel dumps, and one cache variant.
DUMP_MB_PER_RECORDING = 200
CACHE_MB_PER_VARIANT = 40
# The four connectivity variants of analysis A (workflow A1), plus the optional BL-pedestal one.
CONNECTIVITY_VARIANTS = (
    "pixels_dff_full", "pixels_dff_full_gsr",
    "pixels_median_dff_20s_full", "pixels_median_dff_20s_full_gsr",
)
DEFAULT_CACHE_WINDOW_S = 20.0


@dataclass(frozen=True)
class StepDef:
    key: str
    number: str
    title: str
    prerequisites: tuple[str, ...]
    help_html: str


@dataclass
class StepStatus:
    done: bool
    detail: str
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


STEPS: tuple[StepDef, ...] = (
    StepDef("setup", "0", "Setup", (), """
<h3>0 · Setup</h3>
<p>Checks that this computer can run the analysis: conda is installed, the <b>wfci_gui</b>
environment exists with every package, the analysis library is loaded from <i>this</i>
folder, the tests pass, the raw-data disk is plugged in and the dump disk has enough space.</p>
<p><b>Run checks</b> tests the environment (a minute; add the test suite for a few more).
<b>Repair</b> runs <code>setup_env.ps1</code>, which installs only what is missing.</p>
<p>The very first install must be done by hand: open PowerShell in
the repository folder and run <code>.\\env\\setup_env.ps1</code>.</p>"""),
    StepDef("scan", "1", "Scan raw folders", ("setup",), """
<h3>1 · Scan the raw folders</h3>
<p>Lists every recording folder of the cohort, checks it is complete, and writes the
<b>recordings table</b>, the <b>manifest</b> (the batch's input) and the <b>junction view</b>
(short folder names the ROI editor needs).</p>
<p>Before scanning, fill the <b>Tables</b> tab: which session folder is which date and
timepoint (sessions), and which folder name is which mouse (animals). Write these out;
folder names are irregular, so nothing is guessed. Each full folder name is one mouse unless
you map two names to the same animal. <b>Trust the folders, not the file names</b>.</p>
<p>After the scan you can untick <i>use</i> for a recording or force its channel order.</p>"""),
    StepDef("acquisition", "2", "Acquisition facts", ("scan",), """
<h3>2 · Acquisition facts</h3>
<p>Values that silently change every result if wrong: the <b>frame rate per channel</b>
(for example 10 Hz or 25 Hz), whether frame numbers are zero-padded
(<b>frame order</b>), the onset trim, the image rotation and the grid.</p>
<p><b>Read facts</b> reads them from the TIFF headers and file names of one recording per
session (also the LED settings in the file names: a change of LED setting between sessions
creates fake connectivity changes). Check the values, then <b>Confirm</b>.</p>"""),
    StepDef("rois", "3", "ROI placement", ("acquisition",), """
<h3>3 · ROI placement</h3>
<p>One set of 22 cortical boxes per recording, placed by Bregma (position) and Lambda (scale).</p>
<ol><li><b>Bake previews</b>: prepares the anatomy images (fast, can be resumed).</li>
<li><b>Open ROI editor</b>: place Bregma and Lambda on each page and save.</li>
<li><b>Rebuild</b>: regenerates this cohort's new sets as an exact rescale of the atlas
(only this cohort's files are rebuilt; check any header the report says was outvoted).</li>
<li><b>Pre-flight</b>: every box must be inside the dump window and the grid.</li></ol>"""),
    StepDef("extract", "4", "Extraction (batch)", ("rois",), """
<h3>4 · Extraction</h3>
<p>The long run (1.5-4 minutes per recording): hemodynamic correction, the 22 ROI traces,
the correlation matrix and the per-pixel volumes every later step reads.</p>
<p>Run the <b>smoke test</b> on one animal-session first. The full run can be stopped at any
time and resumed: finished recordings are skipped. Recordings whose two channels do not
alternate are saved without dF/F and marked <i>needs a channel decision</i> (step 5).</p>"""),
    StepDef("qc", "5", "Quality control", ("extract",), """
<h3>5 · Quality control</h3>
<p>Builds one row per recording: are all files there, does the geometry match its ROI set,
does the dump reproduce the ROI traces, how sure is the channel split, are there glitch
frames, and the illumination level per session.</p>
<p>Recordings with a glitch above 1 or with channels that do not alternate are <b>ticked as
suggestions</b>. Nothing is excluded until you confirm, and each exclusion needs a reason.
Agree the list with the person who acquired the data.</p>"""),
    StepDef("caches", "6", "Derived caches", ("qc",), """
<h3>6 · Derived caches</h3>
<p>Per-pixel dF/F with a running-median baseline (20 s by default), computed once and read by
most analyses. The frame rate from step 2 is passed automatically; with the wrong rate a
"20 s" window is 8 s.</p>
<p>The optional <b>BL-pedestal</b> variant corrects an illumination change between sessions; use
it only if QC showed one, and only for an additive change.</p>"""),
    StepDef("design", "7", "Design and groups", ("scan", "qc"), """
<h3>7 · Experimental design</h3>
<p>Writes the cohort's <code>design.csv</code>: for every recording that is not excluded, its
animal, group, mouse line, timepoint and session number. Groups and mouse lines come from the
animals table, timepoints from the sessions table.</p>
<p>Adding a session later changes every later session number of that animal: rerun the
analyses after it.</p>"""),
    StepDef("analysis_a", "A", "Connectivity", ("caches", "design"), """
<h3>A · Functional connectivity</h3>
<p>ROI x ROI correlation for each chosen variant (whole-recording or 20 s median baseline,
with and without global-signal regression), then the statistics over time.</p>
<p>Pick the baseline timepoint, the timepoints and the animals. The tests need every animal at
every timepoint: the check below says what to drop when the design is incomplete, and the run
button stays disabled until it is complete.</p>
<p><b>Read results with care</b>: without GSR, mean correlation follows the shared global
signal, so illumination and drift move it. Report a change only if it survives GSR.</p>"""),
)
STEP_KEYS = tuple(step.key for step in STEPS)


def step_def(key: str) -> StepDef:
    for step in STEPS:
        if step.key == key:
            return step
    raise KeyError(key)


# ----------------------------------------------------------------------------- helpers
def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def read_batch_summary(cohort: Cohort) -> pd.DataFrame:
    path = cohort.layout.batch_summary
    if not path.exists():
        return pd.DataFrame(columns=["day", "animal", "recording", "status"])
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "status" not in frame:
        frame["status"] = "done"
    return frame


def summary_status_by_key(summary: pd.DataFrame) -> dict[str, str]:
    """Latest batch_summary status per recording key (a resumed run appends rows)."""
    return {recording_key(r.day, r.animal, r.recording): r.status for r in summary.itertuples()}


def recording_dirs(cohort: Cohort, key: str) -> tuple[Path, Path]:
    """(outputs dir, pixel dir) of a recording key ``<day>_<animal>/<t#>``."""
    unit, recording = key.split("/")
    return (cohort.layout.outputs / unit / recording,
            cohort.layout.pixel_data / unit / recording)


def cache_windows(cohort: Cohort) -> list[float]:
    """Cache windows the user chose (cache_median.window_s; a list builds several)."""
    value = cohort.overrides("cache_median").get("window_s", DEFAULT_CACHE_WINDOW_S)
    return [float(v) for v in (value if isinstance(value, list) else [value])]


def cache_meta_name(window_s: float, tag: str = "") -> str:
    """cache_median_dff.py's marker: ``pixels_median_dff_20s_meta.json``."""
    tag = f"_{tag}" if tag else ""
    return f"pixels_median_dff_{window_s:g}s{tag}_meta.json"


def chosen_variants(cohort: Cohort) -> list[str]:
    return list(cohort.overrides("analysis_a").get("variants", CONNECTIVITY_VARIANTS))


def design_path(cohort: Cohort) -> Path:
    return cohort.layout.design_csv


def roi_preflight_path(cohort: Cohort) -> Path:
    return cohort.layout.outputs / "roi_preflight.csv"


def qc_table_path(cohort: Cohort) -> Path:
    return cohort.layout.outputs / "qc_table.csv"


def setup_check_path(config_dir: Path) -> Path:
    return Path(config_dir) / "setup_check.json"


def free_gb(path: Path) -> float | None:
    """Free space on the drive holding ``path`` (its first existing ancestor), in GB."""
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return shutil.disk_usage(candidate).free / 1e9
    return None


def needed_dump_gb(n_recordings: int, n_cache_variants: int) -> float:
    return n_recordings * (DUMP_MB_PER_RECORDING + n_cache_variants * CACHE_MB_PER_VARIANT) / 1000


# ----------------------------------------------------------------------------- statuses
def status_setup(cohort: Cohort, tables, setup_result: dict | None, quick_checks: list[dict]) -> StepStatus:
    checks = list(quick_checks) + list((setup_result or {}).get("checks", []))
    failed = [c for c in checks if not c["ok"]]
    if setup_result is None:
        return StepStatus(False, "environment not checked yet: press Run checks",
                          [f"{c['name']}: {c['detail']}" for c in failed])
    if not any(c["name"] == "tests pass" for c in checks):
        failed.append({"name": "tests pass", "ok": False, "detail": "test suite not run yet"})
    when = setup_result.get("checked", "?")
    if failed:
        return StepStatus(False, f"{len(failed)} check(s) fail (checked {when})",
                          [f"{c['name']}: {c['detail']}" for c in failed])
    return StepStatus(True, f"all {len(checks)} checks pass (checked {when})")


def status_scan(cohort: Cohort, tables) -> StepStatus:
    for kind in ("sessions", "animals"):
        if tables[kind].empty:
            return StepStatus(False, f"{kind} table is empty: fill it in the Tables tab")
    ok, reason = manifest_matches(cohort, tables, cohort.overrides("scan"))
    recordings = tables["recordings"]
    counts = recordings["status"].value_counts().to_dict() if not recordings.empty else {}
    detail = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "not scanned"
    if not ok:
        return StepStatus(False, reason, counts=counts)
    return StepStatus(True, f"{len(recordings)} folders: {detail}", counts=counts)


def status_acquisition(cohort: Cohort, tables) -> StepStatus:
    missing = acquisition_missing(cohort.acquisition)
    if missing:
        return StepStatus(False, f"not filled: {', '.join(missing)}")
    if not cohort.acquisition.get("confirmed"):
        return StepStatus(False, "filled, not confirmed yet")
    a = cohort.acquisition
    return StepStatus(True, f"{a['sampling_rate_hz']:g} Hz, frame order {a['frame_order']}, "
                            f"trim {a['trim_s']:g} s, rot90 {a['rot90_k']}")


def status_rois(cohort: Cohort, tables) -> StepStatus:
    used = used_recordings(tables)
    if used.empty:
        return StepStatus(False, "no recordings to place ROIs on (scan first)")
    names = [f"{r.day}_{r.animal}_{r.recording}.yaml" for r in used.itertuples()]
    reviewed_dir = cohort.layout.reviewed_roi_sets
    rebuilt_dir = cohort.layout.rebuilt_roi_sets
    drawn = sum((reviewed_dir / n).exists() for n in names)
    rebuilt_paths = [rebuilt_dir / n for n in names]
    rebuilt = sum(p.exists() for p in rebuilt_paths)
    report = roi_preflight_path(cohort)
    passed, warnings = 0, []
    if report.exists():
        frame = pd.read_csv(report, dtype=str, keep_default_na=False)
        good = set(frame.loc[frame["violations"] == "0", "roi_set"])
        newest = max((_mtime(p) for p in rebuilt_paths if p.exists()), default=0)
        if _mtime(report) < newest:
            warnings.append("ROI sets changed after the last pre-flight: run it again")
        else:
            passed = sum(n in good for n in names)
    total = len(names)
    counts = dict(total=total, drawn=drawn, rebuilt=rebuilt, passed=passed)
    detail = f"{drawn} of {total} drawn / {rebuilt} rebuilt / {passed} passed pre-flight"
    return StepStatus(rebuilt == total and passed == total, detail, warnings, counts)


# The two values a ROI set's channel_order may hold (channel_rectangle.CHOSEN_ORDERS).
CHOSEN_ORDERS = ("gcamp_first", "emo_first")


def chosen_channel_order(cohort: Cohort, key: str) -> str | None:
    """The channel order chosen in the ROI editor for a recording key (``channel_order``
    of the ROI set the extraction resolves for it), None when none was chosen."""
    unit, recording = key.split("/")
    for name in (f"{unit}_{recording}.yaml", f"{unit}.yaml"):
        path = cohort.layout.rebuilt_roi_sets / name
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            value = data.get("channel_order") if isinstance(data, dict) else None
            if value is not None and value not in CHOSEN_ORDERS:
                raise ValueError(f"{path}: channel_order must be one of {CHOSEN_ORDERS}, got {value!r}")
            return value
    return None


def extracted_channel_order(npz_path: Path) -> str:
    """The channel order a recording's outputs were extracted with ("" when not recorded)."""
    with np.load(npz_path, allow_pickle=False) as result:
        return str(result["channel_order"].item()) if "channel_order" in result.files else ""


def status_extract(cohort: Cohort, tables) -> StepStatus:
    used = used_recordings(tables)
    if used.empty:
        return StepStatus(False, "no recordings")
    by_key = summary_status_by_key(read_batch_summary(cohort))
    done = needs_decision = missing_row = 0
    stale, order_changed = [], []
    for key in used["key"]:
        output_dir, pixel_dir = recording_dirs(cohort, key)
        npz_path = output_dir / "connectivity_full.npz"
        npz = npz_path.exists()
        status = by_key.get(key)
        if status is None:
            missing_row += 1
            continue
        chosen = chosen_channel_order(cohort, key)
        if status == "needs_channel_decision":
            if chosen is not None:
                order_changed.append(key)     # decided in the ROI editor: extract again
                continue
            needs_decision += 1
            if npz:
                stale.append(key)
        elif npz:
            if chosen is not None and extracted_channel_order(npz_path) != chosen:
                order_changed.append(key)
            else:
                done += 1
    total = len(used)
    warnings = []
    if stale:
        warnings.append(f"{len(stale)} recording(s) marked 'needs a channel decision' still have "
                        f"outputs from an earlier run (stale): {', '.join(stale[:6])}"
                        f"{' ...' if len(stale) > 6 else ''}")
    if order_changed:
        warnings.append(f"{len(order_changed)} recording(s) have a channel order chosen in the ROI "
                        f"editor that their outputs were not extracted with (stale; run step 4 "
                        f"again): {', '.join(order_changed[:6])}{' ...' if len(order_changed) > 6 else ''}")
    counts = dict(total=total, done=done, needs_decision=needs_decision, missing=missing_row,
                  order_changed=len(order_changed))
    detail = f"{done} of {total} extracted, {needs_decision} need a channel decision"
    if order_changed:
        detail += f", {len(order_changed)} stale (channel order changed)"
    return StepStatus(done + needs_decision == total, detail, warnings, counts)


def status_qc(cohort: Cohort, tables) -> StepStatus:
    table = qc_table_path(cohort)
    excluded = len(cohort.exclusions)
    if not table.exists():
        return StepStatus(False, "QC table not built yet")
    batch = cohort.layout.batch_summary
    warnings = []
    if batch.exists() and _mtime(table) < _mtime(batch):
        warnings.append("the batch ran again after the QC table was built: rebuild it")
    if not cohort.get("exclusions_confirmed"):
        return StepStatus(False, f"QC table built; exclusion list not confirmed ({excluded} listed)",
                          warnings)
    return StepStatus(not warnings, f"{excluded} recording(s) excluded, list confirmed", warnings)


def status_caches(cohort: Cohort, tables) -> StepStatus:
    used = used_recordings(tables, cohort, drop_excluded=True)
    by_key = summary_status_by_key(read_batch_summary(cohort))
    # Recordings without dF/F (needs_channel_decision) get no cache.
    keys = [k for k in used["key"] if by_key.get(k) != "needs_channel_decision"] if not used.empty else []
    if not keys:
        return StepStatus(False, "no extracted recordings")
    windows = cache_windows(cohort)
    parts, warnings, complete = [], [], True
    for window in windows:
        have = stale = 0
        for key in keys:
            _, pixel_dir = recording_dirs(cohort, key)
            meta = pixel_dir / cache_meta_name(window)
            source = pixel_dir / "pixels_f_gcamp_full.npy"
            if meta.exists():
                have += 1
                if source.exists() and _mtime(meta) < _mtime(source):
                    stale += 1
        parts.append(f"{window:g} s: {have} of {len(keys)}")
        if stale:
            warnings.append(f"{window:g} s cache: {stale} older than their dumps (stale)")
        complete &= have == len(keys) and not stale
    return StepStatus(complete, "; ".join(parts), warnings)


def status_design(cohort: Cohort, tables) -> StepStatus:
    path = design_path(cohort)
    if not path.exists():
        return StepStatus(False, f"no {path.name} yet")
    design = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Every used, non-excluded recording, as build_cohort_design.py writes it. Recordings
    # still marked "needs a channel decision" stay in until QC excludes them.
    used = used_recordings(tables, cohort, drop_excluded=True)
    expected = set(used["key"]) if not used.empty else set()
    covered = {recording_key(r.day, r.animal, r.recording) for r in design.itertuples()
               if str(r.group).strip() and str(r.timepoint).strip()}
    missing = expected - covered
    extra = {recording_key(r.day, r.animal, r.recording) for r in design.itertuples()} - expected
    warnings = []
    if extra:
        warnings.append(f"{len(extra)} design row(s) for recordings that are excluded or not "
                        f"in use: {', '.join(sorted(extra)[:6])}{' ...' if len(extra) > 6 else ''}")
    by_key = summary_status_by_key(read_batch_summary(cohort))
    undecided = sorted(k for k in expected if by_key.get(k) == "needs_channel_decision")
    if undecided:
        warnings.append(f"{len(undecided)} design row(s) are recordings without dF/F (needs a "
                        "channel decision); exclude them in QC or decide their channels")
    if design_path(cohort).exists() and used_recordings(tables).shape[0] and \
            cohort.table_path("recordings").exists() and \
            _mtime(design_path(cohort)) < _mtime(cohort.path) and cohort.exclusions:
        warnings.append("the cohort file (exclusions) changed after the design was built: rebuild it")
    if missing:
        return StepStatus(False, f"{len(missing)} recording(s) without a design row, group or "
                                 "timepoint", warnings)
    return StepStatus(True, f"{len(design)} recordings, {design['animal'].nunique()} animals",
                      warnings)


def variant_volume(variant: str) -> str:
    """``pixels_median_dff_20s_full_gsr`` -> the volume it reads, ``pixels_median_dff_20s_full.npy``."""
    return f"{variant[:-4] if variant.endswith('_gsr') else variant}.npy"


def stale_connectivity(cohort: Cohort, variant: str) -> tuple[int, int, list[str]]:
    """(results, stale results, examples) of one variant: a per-recording result is stale
    when its input volume was rewritten after it, or no longer exists."""
    folder = cohort.layout.connectivity / variant
    results = sorted(folder.glob("*/*/roi_pixel_connectivity.npz"))
    stale = []
    for result in results:
        unit, recording = result.parent.parent.name, result.parent.name
        source = cohort.layout.pixel_data / unit / recording / variant_volume(variant)
        if not source.exists() or _mtime(result) < _mtime(source):
            stale.append(f"{unit}/{recording}")
    return len(results), len(stale), stale


def status_analysis_a(cohort: Cohort, tables) -> StepStatus:
    root = cohort.layout.connectivity
    variants = chosen_variants(cohort)
    present = [v for v in variants if (root / v).is_dir()]
    warnings = []
    any_stale = False
    for variant in present:
        n_results, n_stale, examples = stale_connectivity(cohort, variant)
        if n_stale:
            any_stale = True
            warnings.append(f"{variant}: {n_stale} of {n_results} results are older than their input "
                            f"volume or have none (stale), e.g. {', '.join(examples[:3])}")
    detail = f"{len(present)} of {len(variants)} variants built"
    if any_stale:
        detail += ", some stale"
    return StepStatus(len(present) == len(variants) and not any_stale, detail, warnings)


STATUS_FUNCTIONS: dict[str, Callable[..., StepStatus]] = {
    "scan": status_scan, "acquisition": status_acquisition, "rois": status_rois,
    "extract": status_extract, "qc": status_qc, "caches": status_caches,
    "design": status_design, "analysis_a": status_analysis_a,
}


def compute_statuses(cohort: Cohort, tables: dict[str, pd.DataFrame],
                     setup_result: dict | None, quick_checks: list[dict]) -> dict[str, StepStatus]:
    """Status of every step, from the files on disk only."""
    statuses = {"setup": status_setup(cohort, tables, setup_result, quick_checks)}
    for key, function in STATUS_FUNCTIONS.items():
        statuses[key] = function(cohort, tables)
    return statuses


def blocking_steps(key: str, statuses: dict[str, StepStatus]) -> list[str]:
    """Prerequisites of ``key`` that are not done (empty list: the step may run)."""
    return [p for p in step_def(key).prerequisites if not statuses[p].done]


# ----------------------------------------------------------------------------- setup checks
def quick_checks(cohort: Cohort | None, conda_path: Path | None, env_path: Path | None,
                 n_recordings: int = 0) -> list[dict[str, Any]]:
    """Checks the GUI can make without running anything: conda, env, drives, space."""
    checks = [
        dict(name="conda found", ok=conda_path is not None, detail=str(conda_path or "not found")),
        dict(name="env exists", ok=bool(env_path and env_path.is_dir()), detail=str(env_path or "")),
    ]
    if cohort is not None:
        raw = cohort.raw_root
        checks.append(dict(name="raw drive mounted", ok=raw.exists(), detail=str(raw)))
        pixel_root = cohort.layout.pixel_data
        free = free_gb(pixel_root)
        needed = needed_dump_gb(n_recordings, len(cache_windows(cohort)) + 1)
        checks.append(dict(
            name="free space on the pixel drive", ok=free is not None and free >= needed,
            detail=f"{free:.0f} GB free, about {needed:.0f} GB needed for {n_recordings} recordings"
            if free is not None else f"{pixel_root}: no such drive"))
    return checks


def read_setup_result(config_dir: Path) -> dict | None:
    path = setup_check_path(config_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def history_notes(history: list[dict]) -> dict[str, str]:
    """'last run ... (failed)' per step from the job history."""
    notes = {}
    for record in history:
        when = record.get("end") or record.get("start") or ""
        stamp = when[:16].replace("T", " ")
        failed = record.get("exit_code") not in (0, None)
        notes[record["step"]] = f"last run {stamp}{' — FAILED' if failed else ''}"
    return notes


def is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and parse_bool(value))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
