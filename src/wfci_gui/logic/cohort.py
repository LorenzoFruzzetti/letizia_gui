"""Cohort file (``<workspace>/<cohort>/cohort.yaml``) and its three tables: load, validate,
save, diff.

The cohort file holds what only the user knows about a cohort: its name, where the raw
data is, the timepoints, the acquisition facts, the exclusions and the parameter
overrides. Every other location is fixed by the cohort folder (``layout.CohortLayout``).

Rules:
  * unknown keys are rejected on load (a typo must not be silently ignored), and so are
    the path keys of the older per-repository layout (``REMOVED_KEYS``);
  * relative paths (``raw_root``, ``pixel_root``) are relative to the cohort folder;
  * saves are atomic (write a temporary file next to the target, then rename), and the
    GUI shows ``diff_text`` against the file on disk before it saves;
  * a save writes ``app_version`` (the version doing the save); ``newer_app_version``
    tells when a file was saved by a newer version than this one.

Tables are kept as DataFrames of strings (``dtype=str``), so what the user typed is
what is saved; booleans are normalised to ``true`` / ``false`` on import.
"""

from __future__ import annotations

import copy
import difflib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .. import APP_VERSION
from .layout import TABLE_KINDS, CohortLayout, require_cohort_file, resolve_in_cohort

# ----------------------------------------------------------------------------- schema
# Top-level keys, in the order they are written back. Required keys have no default.
REQUIRED_KEYS = ("name", "raw_root", "timepoints", "acquisition")
OPTIONAL_KEYS = {
    "app_version": None,                     # version that last saved the file (written on save)
    "pixel_root": None,                      # None = <cohort folder>/pixel_data
    "exclusions": [],                        # written by the QC page
    "exclusions_confirmed": False,           # step 5 is done only once this is true
    "overrides": {},                         # per job: only values that differ from defaults
}
KEY_ORDER = (
    "app_version", "name", "raw_root", "pixel_root", "timepoints", "acquisition",
    "exclusions", "exclusions_confirmed", "overrides",
)
# Path keys of the older layout; their locations are now fixed by the cohort folder.
REMOVED_KEYS = (
    "link_root", "output_root", "manifest", "manifest_paths", "roi_set_dir",
    "reviewed_roi_set_dir", "connectivity_root", "stats_root", "design_csv", "tables",
)
# Acquisition facts (step 2). Every field may be null until step 2 fills it.
ACQUISITION_KEYS = {
    "sampling_rate_hz": (int, float),        # per channel frame rate, e.g. 10 or 25
    "frame_order": (str,),                   # "name" | "number"
    "trim_s": (int, float),                  # onset trim per channel, seconds
    "rot90_k": (int,),
    "grid": (list,),                         # [rows, cols] of the final grid
    "file_glob": (str,),
    "expected_files": (int,),                # files per complete recording; null = no check
    "confirmed": (bool,),                    # the user confirmed the facts on the page
}
FRAME_ORDERS = ("name", "number")
# Groups of overrides, one per job the GUI runs (logic/params.py uses the same keys).
OVERRIDE_GROUPS = (
    "scan", "facts", "bake", "rebuild", "preflight", "extract", "qc",
    "cache_median", "cache_blpedestal", "design", "connectivity", "stats", "analysis_a",
)
# The analysis-A page's choices (overrides.analysis_a); anything else is a typo.
ANALYSIS_A_KEYS = ("variants", "baseline", "timepoints", "animals", "focus")
RECORDING_KEY_RE = re.compile(r"^\d{6}_[^/\\]+/[^/\\]+$")    # 200101_A1_000/t3
VERSION_RE = re.compile(r"^\d+(\.\d+)*$")


class CohortError(ValueError):
    """Invalid cohort file; the message names the file and the field."""


@dataclass
class Cohort:
    """A validated cohort file. ``data`` is what is saved (defaults are not written in)."""

    data: dict[str, Any]
    path: Path | None = None

    # --- plain values ---------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.data["name"]

    def get(self, key: str) -> Any:
        """Value of a top-level key, falling back to its documented default."""
        if key in self.data:
            return self.data[key]
        return copy.deepcopy(OPTIONAL_KEYS[key])

    @property
    def acquisition(self) -> dict[str, Any]:
        return self.data["acquisition"]

    @property
    def timepoints(self) -> list[str]:
        return list(self.data["timepoints"])

    @property
    def exclusions(self) -> list[dict[str, str]]:
        return list(self.get("exclusions"))

    def excluded_keys(self) -> set[str]:
        return {entry["recording"] for entry in self.exclusions}

    def overrides(self, group: str) -> dict[str, Any]:
        return dict(self.get("overrides").get(group, {}))

    # --- paths ----------------------------------------------------------------------
    @property
    def layout(self) -> CohortLayout:
        """Every fixed location of this cohort (derived from the cohort file's folder)."""
        if self.path is None:
            raise ValueError(f"cohort {self.name!r} has no file yet, so it has no folder")
        return CohortLayout(self.path, self.data.get("pixel_root"))

    @property
    def raw_root(self) -> Path:
        """The raw data folder (a relative value is relative to the cohort folder)."""
        return resolve_in_cohort(self.layout.folder, self.data["raw_root"])

    def table_path(self, kind: str) -> Path:
        """``<cohort folder>/tables/<kind>.csv``."""
        return self.layout.table(kind)


# ----------------------------------------------------------------------------- validate
def _fail(where: str, message: str) -> None:
    raise CohortError(f"{where}: {message}")


def validate_cohort(data: Any, where: str = "cohort file") -> dict[str, Any]:
    """Return ``data`` unchanged if valid; raise CohortError naming the first problem."""
    if not isinstance(data, dict):
        _fail(where, "the file must contain a mapping of keys")
    removed = [key for key in REMOVED_KEYS if key in data]
    if removed:
        _fail(where, f"key(s) {removed} are no longer used: every location except raw_root and "
                     "pixel_root is fixed by the cohort folder; remove them")
    unknown = set(data) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
    if unknown:
        _fail(where, f"unknown key(s) {sorted(unknown)}; allowed: {list(KEY_ORDER)}")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        _fail(where, f"missing required key(s) {missing}")
    for key in ("name", "raw_root"):
        if not isinstance(data[key], str) or not data[key].strip():
            _fail(where, f"'{key}' must be a non-empty text value")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", data["name"]):
        _fail(where, "'name' may contain only letters, digits, '_' and '-'")
    pixel_root = data.get("pixel_root")
    if pixel_root is not None and (not isinstance(pixel_root, str) or not pixel_root.strip()):
        _fail(where, "'pixel_root' must be a non-empty text value or null")
    version = data.get("app_version")
    if version is not None and (not isinstance(version, str) or not VERSION_RE.match(version)):
        _fail(where, f"'app_version' must be a version like '1.2.0', not {version!r}")
    timepoints = data["timepoints"]
    if (not isinstance(timepoints, list) or not timepoints
            or not all(isinstance(t, str) and t for t in timepoints)
            or len(set(timepoints)) != len(timepoints)):
        _fail(where, "'timepoints' must be a non-empty list of distinct names, e.g. [BL, 3D]")
    _validate_acquisition(data["acquisition"], where)
    exclusions = data.get("exclusions", [])
    if not isinstance(exclusions, list):
        _fail(where, "'exclusions' must be a list")
    seen = set()
    for i, entry in enumerate(exclusions):
        if not isinstance(entry, dict) or set(entry) != {"recording", "reason"}:
            _fail(where, f"exclusions[{i}] must have exactly 'recording' and 'reason'")
        if not RECORDING_KEY_RE.match(str(entry["recording"])):
            _fail(where, f"exclusions[{i}].recording {entry['recording']!r} is not <yymmdd>_<animal>/<t#>")
        if not str(entry["reason"]).strip():
            _fail(where, f"exclusions[{i}] ({entry['recording']}) needs a reason")
        if entry["recording"] in seen:
            _fail(where, f"exclusions: {entry['recording']} is listed twice")
        seen.add(entry["recording"])
    if not isinstance(data.get("exclusions_confirmed", False), bool):
        _fail(where, "'exclusions_confirmed' must be true or false")
    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict):
        _fail(where, "'overrides' must be a mapping of job -> {parameter: value}")
    for group, values in overrides.items():
        if group not in OVERRIDE_GROUPS:
            _fail(where, f"overrides: unknown job {group!r}; allowed: {list(OVERRIDE_GROUPS)}")
        if not isinstance(values, dict) or not all(isinstance(k, str) for k in values):
            _fail(where, f"overrides.{group} must be a mapping of parameter -> value")
    unknown_choices = set(overrides.get("analysis_a", {})) - set(ANALYSIS_A_KEYS)
    if unknown_choices:
        _fail(where, f"overrides.analysis_a: unknown choice(s) {sorted(unknown_choices)}; "
                     f"allowed: {list(ANALYSIS_A_KEYS)}")
    return data


def _validate_acquisition(block: Any, where: str) -> None:
    if not isinstance(block, dict):
        _fail(where, "'acquisition' must be a mapping")
    unknown = set(block) - set(ACQUISITION_KEYS)
    if unknown:
        _fail(where, f"acquisition: unknown key(s) {sorted(unknown)}; allowed: {list(ACQUISITION_KEYS)}")
    for key, types in ACQUISITION_KEYS.items():
        value = block.get(key)
        if value is None:
            continue
        # bool is an int subclass; only "confirmed" may be a bool.
        if (isinstance(value, bool) and bool not in types) or not isinstance(value, types):
            _fail(where, f"acquisition.{key} has the wrong type ({value!r})")
    if block.get("frame_order") not in (None, *FRAME_ORDERS):
        _fail(where, f"acquisition.frame_order must be one of {FRAME_ORDERS}")
    grid = block.get("grid")
    if grid is not None and (len(grid) != 2 or not all(isinstance(v, int) and v > 0 for v in grid)):
        _fail(where, "acquisition.grid must be [rows, cols] of positive integers")
    rate = block.get("sampling_rate_hz")
    if rate is not None and rate <= 0:
        _fail(where, "acquisition.sampling_rate_hz must be positive")


def acquisition_missing(block: dict[str, Any]) -> list[str]:
    """Acquisition fields still empty (expected_files and confirmed are not facts)."""
    return [key for key in ACQUISITION_KEYS
            if key not in ("expected_files", "confirmed") and block.get(key) is None]


def acquisition_value(cohort: Cohort, key: str) -> Any:
    """One acquisition fact a script needs; raises when step 2 has not filled it."""
    value = cohort.acquisition.get(key)
    if value is None:
        raise ValueError(f"{cohort.path}: acquisition.{key} is not set (fill step 2, Acquisition facts)")
    return value


# ----------------------------------------------------------------------------- I/O
def load_cohort(path: str | Path) -> Cohort:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Cohort(validate_cohort(data, str(path)), path)


def load_script_cohort(cohort_file: str | Path | None) -> Cohort:
    """The cohort a script was given (``--cohort`` / ``RUN_CONFIG["cohort_file"]``).

    Raises ``set cohort_file or pass --cohort`` when it was given none.
    """
    return load_cohort(require_cohort_file(cohort_file))


def dump_cohort(data: dict[str, Any]) -> str:
    """YAML text in the canonical key order (comments of a hand-edited file are not kept)."""
    validate_cohort(data)
    ordered = {key: data[key] for key in KEY_ORDER if key in data}
    return yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, default_flow_style=None,
                          width=100)


def diff_text(old: str, new: str, name: str = "cohort") -> str:
    """Unified diff shown to the user before a save; empty when nothing changes."""
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"{name} (on disk)", tofile=f"{name} (to be saved)"))


def write_text_atomic(path: Path, text: str) -> None:
    """Write next to the destination, flush to disk, then rename over it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    # finally only removes a leftover temporary file; errors still propagate.
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_cohort(cohort: Cohort) -> None:
    """Write the cohort file, stamped with the version doing the save."""
    cohort.data["app_version"] = APP_VERSION
    write_text_atomic(cohort.path, dump_cohort(cohort.data))


def pending_diff(cohort: Cohort) -> str:
    """Diff between the file on disk and what save_cohort would write."""
    old = cohort.path.read_text(encoding="utf-8") if cohort.path.exists() else ""
    return diff_text(old, dump_cohort({**cohort.data, "app_version": APP_VERSION}), cohort.path.name)


def version_tuple(version: str) -> tuple[int, ...]:
    """'1.10.2' -> (1, 10, 2), compared number by number (so 1.10 is newer than 1.9)."""
    if not VERSION_RE.match(str(version)):
        raise ValueError(f"{version!r} is not a version like '1.2.0'")
    return tuple(int(part) for part in str(version).split("."))


def newer_app_version(cohort: Cohort, current: str = APP_VERSION) -> str | None:
    """The cohort file's ``app_version`` when it is newer than ``current``, else None.

    A file saved by a newer version may use keys or values this version does not know,
    so the GUI refuses to open it (a file without ``app_version`` predates the key).
    """
    saved = cohort.data.get("app_version")
    if saved is None:
        return None
    return saved if version_tuple(saved) > version_tuple(current) else None


def new_cohort_data(name: str, raw_root: str, sampling_rate_hz: float | None,
                    frame_order: str | None, timepoints: list[str]) -> dict[str, Any]:
    """Data of a new cohort file as the New-cohort wizard fills it (locations are fixed)."""
    data = {
        "app_version": APP_VERSION,
        "name": name,
        "raw_root": raw_root,
        "timepoints": list(timepoints),
        "acquisition": {
            "sampling_rate_hz": sampling_rate_hz, "frame_order": frame_order, "trim_s": 2.0,
            "rot90_k": 0, "grid": [128, 128], "file_glob": "*.tif", "expected_files": None,
            "confirmed": False,
        },
        "exclusions": [],
        "exclusions_confirmed": False,
        "overrides": {},
    }
    return validate_cohort(data, f"new cohort {name}")


# ----------------------------------------------------------------------------- tables
TABLE_COLUMNS = {
    "sessions": ["session_folder", "day", "timepoint", "day_zero", "include"],
    "animals": ["raw_name", "animal", "group", "mouse_line", "include"],
    "recordings": ["day", "animal", "recording", "path", "n_files", "status", "use",
                   "channel_order"],
}
BOOL_COLUMNS = {"sessions": ("day_zero", "include"), "animals": ("include",),
                "recordings": ("use",)}
# Columns the user may edit in the recordings table; the rest are written by scan_cohort.
EDITABLE_RECORDING_COLUMNS = ("use", "channel_order")
CHANNEL_ORDERS = ("auto", "gcamp_first", "emo_first")
UNIQUE_COLUMNS = {"sessions": ("session_folder",), "animals": ("raw_name",),
                  "recordings": ("day", "animal", "recording")}
_TRUE = {"true", "1", "yes", "y"}
_FALSE = {"false", "0", "no", "n"}


@dataclass(frozen=True)
class TableIssue:
    row: int          # 1-based data row (as the user sees it in a spreadsheet); 0 = header
    reason: str

    def __str__(self) -> str:
        return f"row {self.row}: {self.reason}" if self.row else f"header: {self.reason}"


class TableError(ValueError):
    """An import refused by validation; ``issues`` lists every problem found."""

    def __init__(self, kind: str, source: str, issues: list[TableIssue]):
        self.issues = issues
        shown = "\n".join(f"  {issue}" for issue in issues[:20])
        more = f"\n  ... and {len(issues) - 20} more" if len(issues) > 20 else ""
        super().__init__(f"{kind} table {source} refused:\n{shown}{more}")


def parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{text!r} is not a boolean (use true/false)")


def empty_table(kind: str) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=str) for column in TABLE_COLUMNS[kind]})


def read_table(path: str | Path, kind: str) -> pd.DataFrame:
    """Read a table as strings (``000`` stays ``000``); a missing file is an empty table."""
    path = Path(path)
    if not path.exists():
        return empty_table(kind)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_table(kind: str, frame: pd.DataFrame, timepoints: list[str] | None = None,
                   known_animals: set[str] | None = None) -> list[TableIssue]:
    """Every problem in the table; an empty list means it is valid."""
    columns = TABLE_COLUMNS[kind]
    missing = [c for c in columns if c not in frame.columns]
    unknown = [c for c in frame.columns if c not in columns]
    issues = []
    if missing:
        issues.append(TableIssue(0, f"missing column(s) {missing}"))
    if unknown:
        issues.append(TableIssue(0, f"unknown column(s) {unknown}"))
    if issues:
        return issues
    for i, row in enumerate(frame.itertuples(index=False), start=1):
        values = row._asdict()
        for column in BOOL_COLUMNS[kind]:
            if str(values[column]).strip().lower() not in _TRUE | _FALSE:
                issues.append(TableIssue(i, f"{column} = {values[column]!r} is not true/false"))
        if "day" in values and not re.fullmatch(r"\d{6}", str(values["day"])):
            issues.append(TableIssue(i, f"day = {values['day']!r} is not six digits (yymmdd)"))
        if kind == "sessions":
            if not str(values["session_folder"]).strip():
                issues.append(TableIssue(i, "session_folder is empty"))
            if timepoints is not None and values["timepoint"] not in timepoints:
                issues.append(TableIssue(
                    i, f"timepoint {values['timepoint']!r} is not in the cohort's timepoints {timepoints}"))
        if kind == "animals":
            for column in ("raw_name", "animal"):
                if not str(values[column]).strip():
                    issues.append(TableIssue(i, f"{column} is empty"))
            if re.fullmatch(r"\d+", str(values["animal"])):
                issues.append(TableIssue(i, f"animal {values['animal']!r} is all digits; prefix it "
                                            "(e.g. A1_000) so no CSV reader turns it into a number"))
        if kind == "recordings":
            if values["channel_order"] not in CHANNEL_ORDERS:
                issues.append(TableIssue(
                    i, f"channel_order = {values['channel_order']!r} is not one of {CHANNEL_ORDERS}"))
            if str(values["n_files"]).strip() and not str(values["n_files"]).isdigit():
                issues.append(TableIssue(i, f"n_files = {values['n_files']!r} is not a whole number"))
            if known_animals is not None and values["animal"] not in known_animals:
                issues.append(TableIssue(i, f"animal {values['animal']!r} is not in animals.csv"))
    keys = list(UNIQUE_COLUMNS[kind])
    duplicated = frame.duplicated(subset=keys, keep="first")
    for i in [int(j) + 1 for j in frame.index[duplicated]]:
        issues.append(TableIssue(i, f"duplicate {'/'.join(keys)}"))
    return issues


def normalize_table(kind: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Columns in canonical order, booleans as 'true'/'false', surrounding spaces removed."""
    frame = frame.loc[:, TABLE_COLUMNS[kind]].astype(str).apply(lambda column: column.str.strip())
    for column in BOOL_COLUMNS[kind]:
        frame[column] = frame[column].map(lambda value: "true" if parse_bool(value) else "false")
    return frame.reset_index(drop=True)


def import_table(source: str | Path, kind: str, timepoints: list[str] | None = None,
                 known_animals: set[str] | None = None) -> pd.DataFrame:
    """Read a CSV for import; refuse it (TableError with every row and reason) if invalid."""
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    issues = validate_table(kind, frame, timepoints, known_animals)
    if issues:
        raise TableError(kind, str(source), issues)
    return normalize_table(kind, frame)


def table_to_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def write_table(path: str | Path, kind: str, frame: pd.DataFrame) -> None:
    """Validate the columns and write atomically."""
    issues = validate_table(kind, frame)
    if issues:
        raise TableError(kind, str(path), issues)
    write_text_atomic(Path(path), table_to_csv(normalize_table(kind, frame)))


def load_tables(cohort: Cohort) -> dict[str, pd.DataFrame]:
    return {kind: read_table(cohort.table_path(kind), kind) for kind in TABLE_KINDS}


def validate_tables(cohort: Cohort, tables: dict[str, pd.DataFrame]) -> dict[str, list[TableIssue]]:
    """Cross-table validation: timepoints against the cohort, recordings against animals."""
    animals = set(tables["animals"]["animal"]) if "animal" in tables["animals"] else set()
    return {
        "sessions": validate_table("sessions", tables["sessions"], cohort.timepoints),
        "animals": validate_table("animals", tables["animals"]),
        "recordings": validate_table("recordings", tables["recordings"], known_animals=animals),
    }


# ----------------------------------------------------------------------------- helpers
def recording_key(day: str, animal: str, recording: str) -> str:
    """``<yymmdd>_<animal>/<t#>``: the recording id used by exclusions and every script."""
    return f"{day}_{animal}/{recording}"


def used_recordings(tables: dict[str, pd.DataFrame], cohort: Cohort | None = None,
                    drop_excluded: bool = False) -> pd.DataFrame:
    """Rows of recordings.csv with ``use`` true and status ok (optionally minus exclusions)."""
    frame = tables["recordings"]
    if frame.empty:
        return frame
    mask = frame["use"].map(parse_bool) & (frame["status"] == "ok")
    used = frame[mask].copy()
    used["key"] = [recording_key(r.day, r.animal, r.recording) for r in used.itertuples()]
    if drop_excluded and cohort is not None:
        used = used[~used["key"].isin(cohort.excluded_keys())]
    return used.reset_index(drop=True)


def trim_frames(acquisition: dict[str, Any]) -> int:
    """Onset trim in frames per channel: trim_s x sampling rate (2 s = 50 frames at 25 Hz)."""
    return int(round(float(acquisition["trim_s"]) * float(acquisition["sampling_rate_hz"])))
