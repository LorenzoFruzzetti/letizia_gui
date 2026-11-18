"""Fixed locations inside a cohort folder of the workspace. No Qt, no pandas: scripts import it.

Every cohort lives in its own folder of the workspace, and every file a step reads or
writes has a fixed place in it, derived from the folder of ``cohort.yaml``:

    <workspace>/
    |-- <cohort>/
    |   |-- cohort.yaml
    |   |-- tables/      sessions.csv  animals.csv  recordings.csv
    |   |-- manifest.csv     written by step 1
    |   |-- design.csv       written by step 7
    |   |-- links/       junction view of the raw folders
    |   |-- roi_sets/    reviewed/  rebuilt/  alignment_images/
    |   |-- pixel_data/  unless cohort.yaml's pixel_root redirects it
    |   |-- outputs/     per-step results, logs/, gui_jobs.jsonl
    |   |-- channel_rectangle_params.json   optional (missing = channel_rectangle.py defaults)
    |   `-- presets/
    `-- presets/         shared across cohorts

Only ``raw_root`` and the optional ``pixel_root`` are paths in the cohort file; a relative
value is relative to the cohort folder (``resolve_in_cohort``).

A script gets its cohort from ``--cohort <cohort.yaml>`` (or ``RUN_CONFIG["cohort_file"]``);
``require_cohort_file`` stops a run that has neither with ``NO_COHORT_MESSAGE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

COHORT_FILE_NAME = "cohort.yaml"
NO_COHORT_MESSAGE = "set cohort_file or pass --cohort"
TABLE_KINDS = ("sessions", "animals", "recordings")
PIXEL_DATA_FOLDER = "pixel_data"
PRESETS_FOLDER = "presets"


def require_cohort_file(value: str | Path | None) -> Path:
    """The cohort file a script was given, as an absolute path; raises when there is none."""
    if value is None or not str(value).strip():
        raise ValueError(NO_COHORT_MESSAGE)
    return Path(value).resolve()


def resolve_in_cohort(folder: str | Path, value: str | Path) -> Path:
    """Absolute path of a cohort-file value; a relative value is relative to the cohort folder."""
    path = Path(value)
    return path if path.is_absolute() else Path(folder) / path


def cohort_files(workspace: str | Path) -> list[Path]:
    """Every ``<workspace>/<cohort>/cohort.yaml``, sorted by cohort folder name."""
    workspace = Path(workspace)
    return sorted(workspace.glob(f"*/{COHORT_FILE_NAME}")) if workspace.is_dir() else []


def cohort_file_in(workspace: str | Path, name: str) -> Path:
    """Where the cohort called ``name`` keeps its file: ``<workspace>/<name>/cohort.yaml``."""
    return Path(workspace) / name / COHORT_FILE_NAME


@dataclass(frozen=True)
class CohortLayout:
    """Every fixed location of one cohort, from the path of its cohort file.

    ``pixel_root`` is the cohort file's optional ``pixel_root`` value (None = the
    ``pixel_data`` folder of the cohort).
    """

    cohort_file: Path
    pixel_root: str | Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "cohort_file", Path(self.cohort_file).resolve())

    # --- the cohort folder and the workspace around it --------------------------------
    @property
    def folder(self) -> Path:
        return self.cohort_file.parent

    @property
    def workspace(self) -> Path:
        return self.folder.parent

    @property
    def workspace_presets(self) -> Path:
        """Presets shared by every cohort of the workspace."""
        return self.workspace / PRESETS_FOLDER

    # --- inputs and tables ------------------------------------------------------------
    @property
    def tables(self) -> Path:
        return self.folder / "tables"

    def table(self, kind: str) -> Path:
        if kind not in TABLE_KINDS:
            raise ValueError(f"no table {kind!r}; the tables are {list(TABLE_KINDS)}")
        return self.tables / f"{kind}.csv"

    @property
    def manifest(self) -> Path:
        return self.folder / "manifest.csv"

    @property
    def design_csv(self) -> Path:
        return self.folder / "design.csv"

    @property
    def links(self) -> Path:
        """Junction view ``links/<yymmdd>/<animal>/<t#>`` of the raw recording folders."""
        return self.folder / "links"

    # --- ROI sets ---------------------------------------------------------------------
    @property
    def roi_sets(self) -> Path:
        return self.folder / "roi_sets"

    @property
    def reviewed_roi_sets(self) -> Path:
        """Sets saved by the ROI editor."""
        return self.roi_sets / "reviewed"

    @property
    def rebuilt_roi_sets(self) -> Path:
        """Sets written by rebuild_roi_sets.py; the extraction reads these."""
        return self.roi_sets / "rebuilt"

    @property
    def alignment_images(self) -> Path:
        """The editor's preview cache (<key>_odd.tif, <key>_even.tif, <key>_odd_even.json)."""
        return self.roi_sets / "alignment_images"

    # --- data and results -------------------------------------------------------------
    @property
    def pixel_data(self) -> Path:
        if self.pixel_root is None or not str(self.pixel_root).strip():
            return self.folder / PIXEL_DATA_FOLDER
        return resolve_in_cohort(self.folder, self.pixel_root)

    @property
    def outputs(self) -> Path:
        return self.folder / "outputs"

    @property
    def logs(self) -> Path:
        return self.outputs / "logs"

    @property
    def job_history(self) -> Path:
        return self.outputs / "gui_jobs.jsonl"

    @property
    def batch_summary(self) -> Path:
        """One row per extracted recording (extract_cohort.py)."""
        return self.outputs / "batch_summary.csv"

    @property
    def connectivity(self) -> Path:
        """roi_pixel_connectivity.py output, one subfolder per variant."""
        return self.outputs / "roi_pixel_connectivity"

    @property
    def connectivity_stats(self) -> Path:
        """connectivity_stats.py output."""
        return self.outputs / "connectivity_stats"

    @property
    def channel_rectangle_params(self) -> Path:
        """The cohort's channel-rectangle parameters (the viewer's Save); optional."""
        return self.folder / "channel_rectangle_params.json"

    @property
    def presets(self) -> Path:
        """Presets of this cohort."""
        return self.folder / PRESETS_FOLDER
