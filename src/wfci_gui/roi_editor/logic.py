"""ROI editor logic with no Qt: undo snapshots, page status, settings, rebuild inputs.

Everything geometric is imported from ``roi_editor.py`` (repo root), which is not modified
Pages are built in ``sources.py``; the
edits themselves are in ``model.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
# The old editor and its helpers are pipeline scripts in scripts/.
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ----------------------------------------------------------------------------- undo
# What one undo step restores. `dirty`, `has_own_set` and `saved_to` are left out on
# purpose: they describe the file on disk, which an undo does not change. `rectangle` is
# the page's own channel rectangle (model.PageModel; None = params.json's) and
# `gcamp_choice` the user's call of which parity is GCaMP. A plain roi_editor.Session has
# neither attribute and snapshots them as None.
PAGE_ONLY_FIELDS = ("rectangle", "gcamp_choice")
GEOMETRY_FIELDS = ("y_1", "x_2", "lambda_offset", "lambda_sign", "lambda_angle", "base_lambda",
                   "seed_angle", *PAGE_ONLY_FIELDS)


def snapshot(session) -> tuple:
    """The geometry of one page, copied so later edits cannot change it."""
    y_1, x_2, lam, seed_boxes = session.seed
    return (tuple(getattr(session, name, None) for name in GEOMETRY_FIELDS), dict(session.boxes),
            dict(session.base_boxes), (y_1, x_2, lam, dict(seed_boxes)))


def restore(session, saved: tuple) -> None:
    values, boxes, base_boxes, (y_1, x_2, lam, seed_boxes) = saved
    for name, value in zip(GEOMETRY_FIELDS, values):
        if name in PAGE_ONLY_FIELDS and not hasattr(session, name):
            continue                     # a plain Session: nothing to restore
        setattr(session, name, value)
    session.boxes = dict(boxes)
    session.base_boxes = dict(base_boxes)
    session.seed = (y_1, x_2, lam, dict(seed_boxes))


def capture(sessions) -> list[tuple]:
    return [snapshot(session) for session in sessions]


def changes(before: list[tuple], after: list[tuple]) -> dict[int, tuple[tuple, tuple]]:
    """``{page index: (before, after)}`` for the pages whose geometry differs."""
    return {index: (old, new) for index, (old, new) in enumerate(zip(before, after)) if old != new}


# ----------------------------------------------------------------------------- page status
STATUSES = ("not opened", "opened", "edited", "saved", "rebuilt", "failed pre-flight")


def file_times(key: str, reviewed_dir: Path, rebuilt_dir: Path | None) -> tuple[float | None, float | None]:
    """Modification times of the page's reviewed and rebuilt sets (None = no file)."""
    reviewed = Path(reviewed_dir) / f"{key}.yaml"
    rebuilt = Path(rebuilt_dir) / f"{key}.yaml" if rebuilt_dir else None
    return (reviewed.stat().st_mtime if reviewed.is_file() else None,
            rebuilt.stat().st_mtime if rebuilt is not None and rebuilt.is_file() else None)


def page_status(dirty: bool, visited: bool, reviewed_time: float | None, rebuilt_time: float | None,
                problems: list[str] | None) -> str:
    """One status per page; the first that applies, most urgent first.

    ``problems`` is the page's pre-flight result (None = not run, [] = passed). A reviewed
    set newer than its rebuilt copy is "saved": it still has to be rebuilt.
    """
    if dirty:
        return "edited"
    if problems:
        return "failed pre-flight"
    if rebuilt_time is not None and (reviewed_time is None or rebuilt_time >= reviewed_time):
        return "rebuilt"
    if reviewed_time is not None:
        return "saved"
    return "opened" if visited else "not opened"


def preflight_page(key: str, rebuilt_dir: Path, window: list[int], grid: list[int]) -> list[str]:
    """roi_preflight.py's check on one page's rebuilt set (empty list = passes)."""
    from roi_preflight import check_set
    path = Path(rebuilt_dir) / f"{key}.yaml"
    if not path.is_file():
        return ["no rebuilt ROI set"]
    return check_set(yaml.safe_load(path.read_text(encoding="utf-8")), list(window), list(grid))


def rebuild_sources(keys: list[str], reviewed_dir: Path, rebuilt_dir: Path) -> list[Path]:
    """Reviewed sets to rebuild: every set of each session that has a new or changed one.

    Same rule as wfci_gui.logic.params.rebuild_inputs: a whole session (``<day>_<animal>``,
    the key without its ``_t#``) is rebuilt together, because rebuild_roi_sets.py
    normalises Bregma/Lambda by the majority within a session.
    """
    sessions: dict[str, list[Path]] = {}
    changed: set[str] = set()
    for key in keys:
        source = Path(reviewed_dir) / f"{key}.yaml"
        if not source.is_file():
            continue
        session = key.rsplit("_", 1)[0]
        sessions.setdefault(session, []).append(source)
        target = Path(rebuilt_dir) / source.name
        if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
            changed.add(session)
    return sorted(path for session in changed for path in sessions[session])


# ----------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class EditorSettings:
    """Where the pages come from and how their frames are read (Settings dialog)."""

    folder: tuple[str, ...] = ()
    pattern: str = "*.tif"
    frame_order: str = "name"
    rot90_k: int = field(default=0, metadata=dict(
        doc="Quarter turns applied to every frame (np.rot90; -1 = clockwise)", unit="x 90 deg",
        minimum=-3, maximum=3))
    roi_set_dir: str = "roi_sets/reviewed"
    roi_seed_dir: str | None = "roi_sets/rebuilt"
    rebuilt_dir: str | None = None
    start_from: str | None = None
    # Folder of the cached odd / even images; None = <roi_set_dir>/alignment_images.
    # A cohort's is its roi_sets/alignment_images.
    image_dir: str | None = None
    # The global channel-rectangle parameters (a cohort's channel_rectangle_params.json);
    # None or a missing file = channel_rectangle.py's defaults.
    rectangle_params: str | None = None
    # One setting for both the two panels and the rectangle decision:
    # the first N odd and the first N even frames are averaged.
    frames_per_parity: int = field(default=10, metadata=dict(
        doc="Frames of each parity (odd / even) averaged into the two images and the "
            "rectangle decision", unit="frames", minimum=1, maximum=200))

    @property
    def cache_dir(self) -> Path:
        """Where the odd / even image pairs are cached."""
        return Path(self.image_dir) if self.image_dir else Path(self.roi_set_dir) / "alignment_images"

    @property
    def rebuilt(self) -> str | None:
        """Where rebuild_roi_sets.py writes: rebuilt_dir, else the read-only seed folder."""
        return self.rebuilt_dir or self.roi_seed_dir


def make_settings(values: dict[str, Any], need_folder: bool = True) -> EditorSettings:
    """Validate a dict (dialog, flags or cohort file) into EditorSettings; errors propagate.

    ``need_folder=False`` for a manifest, whose rows carry their own recording paths.
    """
    folder = values["folder"] or ()
    folder = (folder,) if isinstance(folder, str) else tuple(str(f) for f in folder if str(f).strip())
    if need_folder and not folder:
        raise ValueError("Give at least one folder of recordings")
    if values["frame_order"] not in ("name", "number"):
        raise ValueError(f"frame_order must be 'name' or 'number', got {values['frame_order']!r}")
    if not -3 <= int(values["rot90_k"]) <= 3:
        raise ValueError("Choose the rotation (0 = none)")
    if not 1 <= int(values["frames_per_parity"]) <= 200:
        raise ValueError("Frames per parity must be between 1 and 200")
    return EditorSettings(
        folder=folder, pattern=str(values["pattern"]), frame_order=values["frame_order"],
        rot90_k=int(values["rot90_k"]), roi_set_dir=str(values["roi_set_dir"]),
        roi_seed_dir=values.get("roi_seed_dir") or None, rebuilt_dir=values.get("rebuilt_dir") or None,
        start_from=values.get("start_from") or None, image_dir=values.get("image_dir") or None,
        rectangle_params=values.get("rectangle_params") or None,
        frames_per_parity=int(values["frames_per_parity"]))


def default_values() -> dict[str, Any]:
    """The EditorSettings defaults as a dict (no hidden script config)."""
    from dataclasses import asdict
    return asdict(EditorSettings())


# Geometry defaults every page is seeded with: the values roi_editor.py ships with,
# so a page opens exactly as in the old editor.
EDITOR_DEFAULTS: dict[str, Any] = {
    "profile": "cortical_gsr",           # atlas = CORTEX_22, downsample 0.5 (128 grid)
    "bregma_row": 121,                   # fallback Bregma (2 x grid units) for a page with no set
    "bregma_col": 134,
    "lambda_offset": 55,                 # scale reference of the atlas
    "lambda_above_bregma": False,
    "lambda_scales_box_size": False,
    "mirror_lock": True,
    "rotate_with_lambda": True,
    "load_existing": True,
    "shared_name": "shared_roi_set.yaml",
}


def cohort_values(cohort_path: str | Path) -> dict[str, Any]:
    """The values the main GUI passes to step 3, read from a cohort file."""
    from wfci_gui.logic.cohort import load_cohort
    cohort = load_cohort(cohort_path)
    acquisition = cohort.acquisition
    layout = cohort.layout
    return dict(folder=[str(layout.links)],
                pattern=acquisition["file_glob"], frame_order=acquisition["frame_order"],
                rot90_k=acquisition["rot90_k"],
                roi_set_dir=str(layout.reviewed_roi_sets),
                roi_seed_dir=str(layout.rebuilt_roi_sets),
                rebuilt_dir=str(layout.rebuilt_roi_sets),
                image_dir=str(layout.alignment_images),
                rectangle_params=str(layout.channel_rectangle_params))


def resolve_settings(cohort: str | None, flags: dict[str, Any], need_folder: bool = True) -> EditorSettings:
    """Defaults < cohort file < explicit flags (a flag left at None changes nothing)."""
    values = default_values()
    if cohort:
        # An acquisition fact not filled in yet (a new cohort) keeps the default.
        values.update({key: value for key, value in cohort_values(cohort).items() if value is not None})
    values.update({key: value for key, value in flags.items() if value is not None})
    return make_settings(values, need_folder=need_folder)


def rebuild_options(cohort: str | None) -> list[str]:
    """Extra rebuild_roi_sets.py flags from the cohort file's ``overrides.rebuild``."""
    if not cohort:
        return []
    from wfci_gui.logic.cohort import load_cohort
    overrides = load_cohort(cohort).overrides("rebuild")
    options: list[str] = []
    for name in ("reference_lambda", "baseline"):
        if name in overrides:
            options += [f"--{name.replace('_', '-')}", str(overrides[name])]
    return options
