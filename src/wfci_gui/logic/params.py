"""Jobs the GUI runs, how their parameters are passed, and the command each one becomes.

How each kind of script gets its values:
  * argparse scripts get their existing CLI flags;
  * environment-variable scripts (``CONNECTIVITY_STATS_*``) get those variables;
  * flat scripts run through ``run_with_overrides.py <script> --set name=value``.

Every script takes ``--cohort <cohort.yaml>`` (parameter ``cohort_file``) and derives its
folders and acquisition facts from it, so a job passes the cohort file, not paths.

Every job has three kinds of value:
  * ``fixed``    -- the cohort file, and values the GUI chooses per run (variant, the
                    analysis-A choices). Shown read-only: they are edited where they
                    belong (the cohort file / the page);
  * ``defaults`` -- what the GUI passes unless the user changes it (e.g. no built-in exclusions);
  * overrides    -- what the user changed, stored in the cohort file under ``overrides``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from . import REPO_ROOT
from .cohort import Cohort, used_recordings
from .conda import python_command
from .defaults import Parameter, ScriptParameters, format_literal, read_script_parameters
from .jobs import Job, log_file_name, timestamp
from .steps import cache_windows, chosen_variants

SCRIPTS_DIR = "scripts"  # the pipeline scripts, relative to the repository root
FLAT_RUNNER = f"{SCRIPTS_DIR}/run_with_overrides.py"


@dataclass(frozen=True)
class Hint:
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""


@dataclass
class JobSpec:
    key: str                                    # override group (cohort.OVERRIDE_GROUPS)
    step: str                                   # step key (steps.STEPS)
    label: str
    script: str                                 # repo-relative script file
    flat: bool = False                          # run through run_with_overrides.py
    # (cohort, tables, variant) -> {name: value}; names are parameter names (dest / env / var).
    fixed: Callable[..., dict[str, Any]] = lambda cohort, tables, variant: {}
    defaults: Callable[..., dict[str, Any]] = lambda cohort, tables, variant: {}
    help: str = ""


# A fixed value that cannot be known yet is shown in the form as this text, and
# build_arguments refuses to build a command containing it.
NOT_SET = "<not set: "


def not_set(what: str) -> str:
    return f"{NOT_SET}{what}>"


def _acq(cohort: Cohort, key: str) -> Any:
    value = cohort.acquisition.get(key)
    return not_set(f"acquisition.{key}, fill step 2") if value is None else value


def missing_values(values: dict[str, Any]) -> list[str]:
    return [f"{name}: {value[len(NOT_SET):-1]}" for name, value in values.items()
            if isinstance(value, str) and value.startswith(NOT_SET)]


def _cohort_arg(cohort: Cohort, tables, variant) -> dict[str, Any]:
    return {"cohort_file": str(cohort.path)}


def _needs_acquisition(cohort: Cohort, *keys: str) -> dict[str, Any]:
    """The cohort file, plus a not-set marker per acquisition fact the script reads and
    step 2 has not filled (so the form says so and the job is refused before it runs)."""
    values = _cohort_arg(cohort, None, None)
    for key in keys:
        if cohort.acquisition.get(key) is None:
            values[f"acquisition.{key}"] = _acq(cohort, key)
    return values


def _connectivity_fixed(cohort: Cohort, tables, variant) -> dict[str, Any]:
    variant = variant or chosen_variants(cohort)[0]  # the form shows the first variant
    gsr = variant.endswith("_gsr")
    volume = variant[:-4] if gsr else variant
    return {**_cohort_arg(cohort, tables, variant), "volume_file": f"{volume}.npy", "gsr": gsr}


def _stats_fixed(cohort: Cohort, tables, variant) -> dict[str, Any]:
    """The analysis-A choices of the page, passed as connectivity_stats.py's flags."""
    choice = cohort.overrides("analysis_a")
    variant = variant or chosen_variants(cohort)[0]
    timepoints = list(choice.get("timepoints") or cohort.timepoints)
    baseline = choice.get("baseline") or timepoints[0]
    values = {
        **_cohort_arg(cohort, tables, variant),
        "baseline": baseline,
        "timepoints": timepoints,
        "variant": variant,
        "focus": choice.get("focus", "SSp"),
        "CONNECTIVITY_STATS_TIME_AXIS": "timepoint",
    }
    if choice.get("animals"):
        values["animals"] = list(choice["animals"])     # unset = every animal of the design
    return values


EXTRACT_ACQUISITION = ("frame_order", "rot90_k", "trim_s", "sampling_rate_hz", "file_glob")

JOBS: dict[str, JobSpec] = {spec.key: spec for spec in (
    JobSpec("scan", "scan", "Scan raw folders", "scan_cohort.py", fixed=_cohort_arg),
    JobSpec("facts", "acquisition", "Read acquisition facts", "acquisition_facts.py",
            fixed=_cohort_arg),
    JobSpec("bake", "rois", "Bake previews", "bake_alignment_images.py",
            fixed=lambda c, t, v: _needs_acquisition(c, "frame_order", "rot90_k", "file_glob")),
    JobSpec("rebuild", "rois", "Rebuild ROI sets", "rebuild_roi_sets.py",
            # roi_set_dir is replaced by the temporary copy when the job is built; the
            # rebuilt sets go to the cohort's roi_sets/rebuilt.
            fixed=lambda c, t, v: {**_cohort_arg(c, t, v),
                                   "roi_set_dir": "<copy of this cohort's new reviewed sets>"}),
    JobSpec("preflight", "rois", "ROI pre-flight", "roi_preflight.py", fixed=_cohort_arg),
    JobSpec("extract", "extract", "Extraction", "extract_cohort.py",
            fixed=lambda c, t, v: _needs_acquisition(c, *EXTRACT_ACQUISITION)),
    JobSpec("qc", "qc", "Quality control table", "qc_cohort.py", fixed=_cohort_arg),
    JobSpec("cache_median", "caches", "Median dF/F cache", "cache_median_dff.py",
            fixed=lambda c, t, v: {**_needs_acquisition(c, "sampling_rate_hz"),
                                   "window_s": float(v) if v is not None else cache_windows(c)[0]}),
    JobSpec("cache_blpedestal", "caches", "BL-pedestal cache", "cache_median_dff_baseline_pedestal.py",
            fixed=lambda c, t, v: _needs_acquisition(c, "sampling_rate_hz")),
    JobSpec("design", "design", "Experimental design", "build_cohort_design.py", fixed=_cohort_arg),
    JobSpec("connectivity", "analysis_a", "Connectivity", "roi_pixel_connectivity.py",
            fixed=_connectivity_fixed),
    JobSpec("stats", "analysis_a", "Connectivity statistics", "connectivity_stats.py",
            flat=True, fixed=_stats_fixed),
)}

# Unit, range and help per parameter, for the form. A parameter without a hint
# is still shown, as a plain field.
HINTS: dict[tuple[str, str], Hint] = {
    ("extract", "workers"): Hint("", 1, 32, "Parallel workers. 1 is the reliable default on "
                                 "Windows (process pools there fail intermittently); more is experimental."),
    ("extract", "channel_buffer"): Hint(help="How each recording is read once: auto picks RAM "
                                        "or a disk buffer."),
    ("extract", "buffer_dir"): Hint(help="SSD folder for the channel buffer when it does not fit in RAM."),
    ("extract", "skip_existing"): Hint(help="Skip recordings that already have results (resume)."),
    ("extract", "save_data"): Hint(help="Write the per-pixel volumes (~200 MB per recording). "
                                   "Every later step needs them."),
    ("extract", "alternation_max"): Hint("", -1, 1, "Channels count as alternating below this "
                                         "lag-1 autocorrelation of frame medians."),
    ("extract", "separation_min"): Hint("", 0, 1000, "Minimum odd/even level gap, in units of "
                                        "within-channel noise, for a 'certain' verdict."),
    ("cache_median", "workers"): Hint("", 1, 32, "Parallel workers (about 1.1 GB RAM each)."),
    ("cache_median", "dtype"): Hint(help="float16 halves the disk; float32 is exact."),
    ("cache_median", "overwrite"): Hint(help="Recompute caches that already exist."),
    ("cache_blpedestal", "offset_smoothing_px"): Hint("px", 0, 20, "Gaussian smoothing of the pedestal maps."),
    ("connectivity", "workers"): Hint("", 1, 32, "Parallel workers."),
    ("connectivity", "overwrite"): Hint(help="Recompute recordings that already have results."),
    ("bake", "workers"): Hint("", 1, 32, "Parallel workers."),
    ("rebuild", "reference_lambda"): Hint("rows", 1, 200, "Bregma-Lambda distance the atlas was "
                                          "drawn at (55). Do not change."),
    ("stats", "fdr_alpha"): Hint("", 0.001, 0.5, "Benjamini-Hochberg level."),
    ("stats", "uncorrected_alpha"): Hint("", 0.001, 0.5, "Level that marks edges in the figures."),
    ("stats", "nbs_primary_threshold"): Hint("t", 0.5, 10, "NBS edge cut-off on the paired t."),
    ("stats", "nbs_alpha"): Hint("", 0.001, 0.5, "NBS component FWER level."),
    ("qc", "glitch_threshold"): Hint("dF/F ratio", 0.01, 100, "Max |field mean| above which a "
                                     "recording is suggested for exclusion."),
}

# Parameters never offered in a form: they only make sense when editing a script by hand.
HIDDEN = {"prefer_cli_args"}


@dataclass
class FormField:
    """One row of a step's parameter form."""
    name: str
    parameter: Parameter
    value: Any                  # effective value
    default: Any                # script default (or GUI default)
    source: str                 # "default" | "gui default" | "override" | "cohort file" | "computed"
    editable: bool
    hint: Hint = field(default_factory=Hint)


_PARAMETER_CACHE: dict[tuple[str, float], ScriptParameters] = {}


def script_parameters(spec: JobSpec, repo: Path = REPO_ROOT) -> ScriptParameters:
    """Parameters of the job's script, re-read whenever the file changes."""
    path = repo / SCRIPTS_DIR / spec.script
    key = (str(path), path.stat().st_mtime)
    if key not in _PARAMETER_CACHE:
        _PARAMETER_CACHE[key] = read_script_parameters(path)
    return _PARAMETER_CACHE[key]


def offered_parameters(spec: JobSpec, parameters: ScriptParameters) -> list[Parameter]:
    """Flags and env vars for argparse scripts; for flat ones also their module-level
    variables (a flat script's flags, such as ``--cohort``, follow ``--`` on its command)."""
    chosen = [*parameters.flags, *parameters.env]
    if spec.flat:
        # A module-level variable with the name of a flag or env var holds the value the
        # script derived from it (``timepoints = choice("timepoints") ...``): the flag wins.
        taken = {p.name for p in chosen}
        chosen += [p for p in parameters.assigns if p.name not in taken]
    return [p for p in chosen if p.name not in HIDDEN]


def form_fields(job: str, cohort: Cohort, tables, variant: Any = None,
                repo: Path = REPO_ROOT) -> tuple[list[FormField], str]:
    """Rows of the form and a message naming the fixed values that are not set yet."""
    spec = JOBS[job]
    parameters = script_parameters(spec, repo)
    fixed = spec.fixed(cohort, tables, variant)
    gui_defaults = spec.defaults(cohort, tables, variant)
    missing = missing_values(fixed)
    error = ("Not set yet: " + "; ".join(missing)) if missing else ""
    overrides = cohort.overrides(job)
    rows = []
    for parameter in offered_parameters(spec, parameters):
        name = parameter.name
        hint = HINTS.get((job, name), Hint(help=parameter.help))
        if name in fixed:
            rows.append(FormField(name, parameter, fixed[name], parameter.default, "cohort file", False, hint))
        elif not parameter.overridable:
            rows.append(FormField(name, parameter, None, None, "computed", False, hint))
        else:
            default = gui_defaults.get(name, parameter.default)
            source = "gui default" if name in gui_defaults else "default"
            value = overrides.get(name, default)
            rows.append(FormField(name, parameter, value, default,
                                  "override" if name in overrides else source, True, hint))
    return rows, error


def unknown_overrides(job: str, cohort: Cohort, repo: Path = REPO_ROOT) -> list[str]:
    """Overrides that name no parameter of the script (renamed since they were saved)."""
    if job == "analysis_a":
        return []
    names = {p.name for p in offered_parameters(JOBS[job], script_parameters(JOBS[job], repo))}
    return sorted(set(cohort.overrides(job)) - names)


# ----------------------------------------------------------------------------- commands
def flag_arguments(parameter: Parameter, value: Any) -> list[str]:
    """CLI tokens that make argparse store ``value`` for this parameter."""
    flag = next(iter(parameter.flags))
    if parameter.action == "store_true":
        return [flag] if value else []
    if parameter.action == "store_false":
        return [] if value else [flag]
    if parameter.action == "bool_pair":
        for option, stores in parameter.flags.items():
            if stores is bool(value):
                return [option]
        raise ValueError(f"{parameter.name}: no option stores {value!r}")
    if parameter.action == "boolean_optional":
        return [flag] if value else [f"--no-{flag[2:]}"]
    if value is None:
        if parameter.type_name == "custom":
            return [flag, "none"]  # the repo's scripts parse "none" as None
        raise ValueError(f"{parameter.name}: None cannot be passed on the command line")
    if isinstance(value, (list, tuple)):
        if parameter.nargs in (None, "?"):
            raise ValueError(f"{parameter.name} takes one value, got a list")
        return [flag, *(str(v) for v in value)]
    return [flag, str(value)]


@dataclass
class BuiltCommand:
    arguments: list[str]            # after "python -u"
    env: dict[str, str]             # job-specific environment variables
    overridden: set[str]            # names the user changed (marked in the preview)
    passed: dict[str, Any]          # every value passed, for the history


def build_arguments(job: str, cohort: Cohort, tables, variant: Any = None,
                    extra: dict[str, Any] | None = None, repo: Path = REPO_ROOT) -> BuiltCommand:
    """Script arguments of a job: fixed values, then GUI defaults, overrides and ``extra``.

    ``extra`` holds per-run values the page adds (``--debug --select`` of the smoke test).
    Only values that differ from the script default are passed, except fixed ones.
    """
    spec = JOBS[job]
    parameters = script_parameters(spec, repo)
    by_name = {p.name: p for p in offered_parameters(spec, parameters)}
    fixed = spec.fixed(cohort, tables, variant)
    missing = missing_values(fixed)
    if missing:
        raise ValueError(f"{spec.label} cannot run yet. Not set: " + "; ".join(missing))
    values: dict[str, Any] = dict(spec.defaults(cohort, tables, variant))
    overrides = cohort.overrides(job)
    values.update(overrides)
    values.update(extra or {})
    values.update(fixed)
    unknown = sorted(set(values) - set(by_name))
    if unknown:
        raise ValueError(f"{spec.script} has no parameter(s) {unknown}")
    arguments: list[str] = []
    sets: list[str] = []
    env: dict[str, str] = {}
    for name, value in values.items():
        parameter = by_name[name]
        if name not in fixed and parameter.literal and value == parameter.default:
            continue  # same as the script's own default: nothing to pass
        if parameter.kind == "env":
            env[name] = str(value)
        elif parameter.kind == "assign":
            if not parameter.overridable:
                raise ValueError(f"{name} is computed in {spec.script} and cannot be overridden")
            sets += ["--set", f"{name}={format_literal(value)}"]
        else:
            arguments += flag_arguments(parameter, value)
    # Full script paths: a job runs in the cohort folder, never in the code folder (M8).
    script = str(Path(repo) / SCRIPTS_DIR / spec.script)
    if spec.flat:
        arguments = [str(Path(repo) / FLAT_RUNNER), script, *sets,
                     *(["--", *arguments] if arguments else [])]
    else:
        arguments = [script, *arguments]
    overridden = set(overrides) | set(extra or {})
    return BuiltCommand(arguments, env, overridden, values)


def build_job(job: str, cohort: Cohort, tables, conda: Path, variant: Any = None,
              extra: dict[str, Any] | None = None, label: str | None = None,
              repo: Path = REPO_ROOT) -> Job:
    spec = JOBS[job]
    before = None
    if job == "rebuild":
        copy_dir, sources = rebuild_inputs(cohort, tables)
        before = lambda: copy_reviewed_sets(sources, copy_dir)
    built = build_arguments(job, cohort, tables, variant, extra, repo)
    if job == "rebuild":
        index = built.arguments.index("--roi-set-dir")
        built.arguments[index + 1] = str(copy_dir)
    tag = f"{job}_{variant}" if variant is not None else job
    return Job(
        step=spec.step, key=job, label=label or (f"{spec.label}: {variant}" if variant is not None else spec.label),
        command=python_command(conda, built.arguments), cwd=cohort.layout.folder, env=built.env,
        overrides={k: v for k, v in built.passed.items() if k in built.overridden},
        log_path=cohort.layout.logs / log_file_name(tag),
        history_path=cohort.layout.job_history, before=before)


# ----------------------------------------------------------------------------- step 3 copy
def rebuild_inputs(cohort: Cohort, tables) -> tuple[Path, list[Path]]:
    """Reviewed sets to rebuild: every set of each session that has a new or changed one.

    A whole session is copied, not only the changed file, because rebuild_roi_sets.py
    normalises Bregma/Lambda by the t1-t5 majority within a session.
    Raises when nothing is new.
    """
    reviewed = cohort.layout.reviewed_roi_sets
    rebuilt = cohort.layout.rebuilt_roi_sets
    used = used_recordings(tables)
    sessions: dict[str, list[Path]] = {}
    changed: set[str] = set()
    for row in used.itertuples():
        name = f"{row.day}_{row.animal}_{row.recording}.yaml"
        source = reviewed / name
        if not source.exists():
            continue
        session = f"{row.day}_{row.animal}"
        sessions.setdefault(session, []).append(source)
        target = rebuilt / name
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            changed.add(session)
    sources = sorted(p for s in changed for p in sessions[s])
    if not sources:
        raise ValueError("no new or changed reviewed ROI sets for this cohort: nothing to rebuild")
    stamp = timestamp().replace(":", "").replace("-", "")
    return cohort.layout.outputs / "tmp" / f"rebuild_input_{stamp}", sources


def copy_reviewed_sets(sources: list[Path], copy_dir: Path) -> None:
    copy_dir.mkdir(parents=True, exist_ok=False)
    for source in sources:
        shutil.copy2(source, copy_dir / source.name)


def smoke_unit(tables) -> str | None:
    """First used unit, as ``<yymmdd>/<animal>`` for ``--select`` (step 4 smoke test)."""
    used = used_recordings(tables)
    if used.empty:
        return None
    first = used.sort_values(["day", "animal"]).iloc[0]
    return f"{first['day']}/{first['animal']}"


def unit_frame(tables) -> pd.DataFrame:
    return used_recordings(tables)
