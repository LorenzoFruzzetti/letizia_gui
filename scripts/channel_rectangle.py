"""Midline-rectangle channel rule: which interleaved group (odd / even) is GCaMP?

Pure numpy, no Qt. Shared by experiments/channel_rectangle_viewer.py (where the rule was
tuned, channel_rectangle_viewer_spec.md) and roi_editor.py (which draws the rectangle on
every page and states the decision while the ROIs are placed).

Hypothesis: along the Bregma-Lambda axis (superior sagittal sinus) the reflectance image
shows more vessel contrast than the GCaMP image. A rectangle centred on the axis is
compared between the odd and even mean images with one statistic.

Geometry (256 grid, continuous coordinates, pixel (r, c) covers [r, r+1) x [c, c+1)):
B = Bregma, L = Lambda, d = |L - B|, u = unit vector from L toward B. The rectangle is
centred on the axis, starts at L + start_offset_factor * d * u, is length_factor * d long
and width_factor * d wide. Every parameter is a factor of d, so one parameter set applies
at every scale and on any grid: the 128-grid editor uses the same corners divided by 2.

Defaults are the parameters saved from the viewer in
outputs/channel_rectangle/params.json (2026-11-12); that file, when present, wins.

Per recording (roi_editor_spec.md D14/D15): a ROI set may carry its own
``channel_rectangle: {length_factor, width_factor, start_offset_factor}``, written by the
ROI editor (wfci_gui.roi_editor) for a page whose rectangle was moved. The statistic stays
global. ``rectangle_params_for`` gives the parameters a recording is decided with.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PARAMS_PATH = REPO_ROOT / "outputs" / "channel_rectangle" / "params.json"
STATISTICS = ("median_level", "centre_flank", "spatial_cv")
PARITIES = ("odd", "even")
UNASSIGNED = "unassigned"


# ---------------------------------------------------------------------------
# Parameters (global, factors of d; saved to params.json)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RectangleParams:
    # Defaults = outputs/channel_rectangle/params.json as saved from the viewer.
    length_factor: float = field(default=0.7444891064863796, metadata=dict(
        doc="Rectangle length along the axis, as a multiple of the Bregma-Lambda distance d",
        unit="d", minimum=0.05, maximum=5.0))
    width_factor: float = field(default=0.1468939228518398, metadata=dict(
        doc="Total rectangle width across the axis (symmetric about it), as a multiple of d",
        unit="d", minimum=0.01, maximum=2.0))
    start_offset_factor: float = field(default=1.1166427822870573, metadata=dict(
        doc="Start of the rectangle, measured from Lambda toward Bregma, as a multiple of d "
            "(0 = at Lambda, 1 = at Bregma, negative = beyond Lambda)",
        unit="d", minimum=-1.0, maximum=2.0))
    # centre_flank is the vessel-contrast hypothesis the tool was built to look at.
    statistic: str = "centre_flank"


def make_params(values: dict) -> RectangleParams:
    """Validate a dict (dialog or params.json) into RectangleParams; errors propagate."""
    names = {item.name for item in fields(RectangleParams)}
    if set(values) != names:
        raise ValueError(f"Rectangle parameters need exactly {sorted(names)}, got {sorted(values)}")
    for item in fields(RectangleParams):
        if "minimum" not in item.metadata:
            continue
        value = float(values[item.name])
        if not item.metadata["minimum"] <= value <= item.metadata["maximum"]:
            raise ValueError(f"{item.name} = {value} is outside "
                             f"[{item.metadata['minimum']}, {item.metadata['maximum']}]")
    if values["statistic"] not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {values['statistic']!r}")
    return RectangleParams(
        length_factor=float(values["length_factor"]),
        width_factor=float(values["width_factor"]),
        start_offset_factor=float(values["start_offset_factor"]),
        statistic=str(values["statistic"]),
    )


def save_params(path: Path, params: RectangleParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(params), indent=2) + "\n", encoding="utf-8")


def load_params(path: Path) -> RectangleParams:
    return make_params(json.loads(Path(path).read_text(encoding="utf-8")))


def params_or_default(path: str | Path | None = DEFAULT_PARAMS_PATH) -> RectangleParams:
    """The saved parameters when the file exists, else the defaults above."""
    if path is not None:
        path = Path(path)
        path = path if path.is_absolute() else REPO_ROOT / path
        if path.is_file():
            return load_params(path)
    return RectangleParams()


# ---------------------------------------------------------------------------
# Per-recording rectangle (roi_editor_spec.md D14/D15): the three factors only; the
# statistic stays global in params.json. Stored in the recording's ROI set as one line,
#     channel_rectangle: {length_factor: ..., width_factor: ..., start_offset_factor: ...}
# written only for a page whose rectangle was moved in the ROI editor.
# ---------------------------------------------------------------------------
FACTOR_NAMES = ("length_factor", "width_factor", "start_offset_factor")
ROI_SET_KEY = "channel_rectangle"


@dataclass(frozen=True)
class RectangleFactors:
    length_factor: float
    width_factor: float
    start_offset_factor: float


def factor_limits(name: str) -> tuple[float, float]:
    """(minimum, maximum) of one factor: the RectangleParams field limits."""
    metadata = next(item.metadata for item in fields(RectangleParams) if item.name == name)
    return float(metadata["minimum"]), float(metadata["maximum"])


def make_factors(values: dict) -> RectangleFactors:
    """Validate a dict into RectangleFactors: exactly the three keys, each within its limits."""
    if set(values) != set(FACTOR_NAMES):
        raise ValueError(f"{ROI_SET_KEY} needs exactly {list(FACTOR_NAMES)}, got {sorted(values)}")
    for name in FACTOR_NAMES:
        low, high = factor_limits(name)
        value = values[name]
        # bool is an int: a YAML `true` must not pass as 1.0.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{ROI_SET_KEY}.{name} must be a number, got {value!r}")
        if not low <= float(value) <= high:
            raise ValueError(f"{ROI_SET_KEY}.{name} = {value} is outside [{low}, {high}]")
    return RectangleFactors(**{name: float(values[name]) for name in FACTOR_NAMES})


def factors_of(params: RectangleParams) -> RectangleFactors:
    return RectangleFactors(params.length_factor, params.width_factor, params.start_offset_factor)


def params_with(params: RectangleParams, factors: RectangleFactors | None) -> RectangleParams:
    """``params`` with a recording's own factors (None = the global ones); statistic kept."""
    if factors is None:
        return params
    return RectangleParams(length_factor=factors.length_factor, width_factor=factors.width_factor,
                           start_offset_factor=factors.start_offset_factor, statistic=params.statistic)


def clip_factor(name: str, value: float) -> float:
    low, high = factor_limits(name)
    return min(max(float(value), low), high)


def factors_from_roi(bregma, lambda_, corner, length: float, width: float) -> RectangleFactors:
    """A dragged rectangle read back as factors (the viewer's rule, D16).

    ``corner`` is the ROI's local origin (row, col) and ``length`` / ``width`` its size,
    all in the same grid as ``bregma`` / ``lambda_``. Only the component along the axis is
    kept, so the rectangle stays centred on it; each factor is clipped to its limits.
    """
    d, u, n = axis_frame(bregma, lambda_)
    length, width = abs(float(length)), abs(float(width))
    start = np.asarray(corner, dtype=float) + 0.5 * width * n
    offset = float((start - np.asarray(lambda_, dtype=float)) @ u)
    return RectangleFactors(length_factor=clip_factor("length_factor", length / d),
                            width_factor=clip_factor("width_factor", width / d),
                            start_offset_factor=clip_factor("start_offset_factor", offset / d))


def roi_state(bregma, lambda_, factors: RectangleFactors) -> dict:
    """pyqtgraph ROI state of the rectangle: pos = corner 0 as (x = col, y = row), size =
    (length, width) in grid units, angle of the Lambda -> Bregma axis in degrees."""
    corners = rectangle_corners(bregma, lambda_, factors.length_factor, factors.width_factor,
                                factors.start_offset_factor)
    d, u, _ = axis_frame(bregma, lambda_)
    return {"pos": (float(corners[0][1]), float(corners[0][0])),
            "size": (factors.length_factor * d, factors.width_factor * d),
            "angle": math.degrees(math.atan2(u[0], u[1]))}


def read_roi_set_rectangle(path: str | Path) -> RectangleFactors | None:
    """The ROI set's own rectangle, or None when it has none (= use params.json).

    A partial or out-of-range entry raises: it is a hand-edited or corrupt file, and
    silently falling back to the global rectangle would hide that.
    """
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    value = data.get(ROI_SET_KEY) if isinstance(data, dict) else None
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {ROI_SET_KEY} must be a mapping, got {value!r}")
    return make_factors(value)


def roi_set_rectangle_line(factors: RectangleFactors) -> str:
    """The one-line YAML block, values at full precision (yaml's float repr round-trips)."""
    import yaml
    payload = {ROI_SET_KEY: {name: float(getattr(factors, name)) for name in FACTOR_NAMES}}
    # A huge width keeps the flow mapping on one line; yaml writes repr(), which reads back
    # to the identical float (and adds the '.0' PyYAML needs for an exponent-only value).
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=None, width=4096)


def append_roi_set_rectangle(path: str | Path, factors: RectangleFactors) -> None:
    """Append the block to a ROI set ``roi_editor.save_roi_set`` has just written.

    Written as bytes with the file's own line ending (save_roi_set writes in text mode, so
    CRLF on Windows; CLAUDE.md 9.28), so the rest of the file is byte-identical to what
    save_roi_set wrote. A file that already carries the key is refused.
    """
    path = Path(path)
    content = path.read_bytes()
    if read_roi_set_rectangle(path) is not None:
        raise ValueError(f"{path} already has a {ROI_SET_KEY} entry")
    if not content.endswith(b"\n"):
        raise ValueError(f"{path} does not end with a newline: not a file save_roi_set wrote")
    newline = b"\r\n" if content.endswith(b"\r\n") else b"\n"
    line = roi_set_rectangle_line(factors).rstrip("\n").encode("utf-8")
    path.write_bytes(content + line + newline)


# The user's own call of which parity is GCaMP, made in the ROI editor (one line,
#     channel_order: gcamp_first        (odd frames = GCaMP; emo_first = even frames)
# written only for a page where it was chosen). Same append-only mechanism as the rectangle.
CHANNEL_KEY = "channel_order"
CHOSEN_ORDERS = ("gcamp_first", "emo_first")


def read_roi_set_channel_order(path: str | Path) -> str | None:
    """The ROI set's chosen channel_order, None when none was chosen. Other values raise."""
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    value = data.get(CHANNEL_KEY) if isinstance(data, dict) else None
    if value is None:
        return None
    if value not in CHOSEN_ORDERS:
        raise ValueError(f"{path}: {CHANNEL_KEY} must be one of {CHOSEN_ORDERS}, got {value!r}")
    return value


def append_roi_set_channel_order(path: str | Path, channel_order: str) -> None:
    """Append ``channel_order: <order>`` to a ROI set, as bytes, in the file's own line ending."""
    if channel_order not in CHOSEN_ORDERS:
        raise ValueError(f"channel_order must be one of {CHOSEN_ORDERS}, got {channel_order!r}")
    path = Path(path)
    content = path.read_bytes()
    if read_roi_set_channel_order(path) is not None:
        raise ValueError(f"{path} already has a {CHANNEL_KEY} entry")
    if not content.endswith(b"\n"):
        raise ValueError(f"{path} does not end with a newline: not a file save_roi_set wrote")
    newline = b"\r\n" if content.endswith(b"\r\n") else b"\n"
    path.write_bytes(content + f"{CHANNEL_KEY}: {channel_order}".encode("utf-8") + newline)


def parity_of_order(channel_order: str) -> str:
    return {"gcamp_first": "odd", "emo_first": "even"}[channel_order]


def rectangle_params_for(roi_set_path: str | Path, global_params: RectangleParams
                         ) -> tuple[RectangleParams, str]:
    """The rectangle a recording is decided with, and where it came from (D18).

    ``("roi_set", ...)``: the ROI set's own factors with the global statistic;
    ``("params", ...)``: the set has no ``channel_rectangle``, so params.json's rectangle.
    """
    factors = read_roi_set_rectangle(roi_set_path)
    if factors is None:
        return global_params, "params"
    return params_with(global_params, factors), "roi_set"


# ---------------------------------------------------------------------------
# Agreement between systems (moved here from experiments/channel_rectangle_viewer.py so
# the ROI editor's verdict bar uses the same marks).
# ---------------------------------------------------------------------------
def gcamp_parity_from_order(channel_order: str) -> str:
    """`gcamp_first` -> the odd frames are GCaMP; `emo_first` -> the even ones.

    `odd_even_unassigned` is the cache of a recording the pipeline judged not interleaved
    (CLAUDE.md 9.72): the library gives no parity for it. The manifest leaves the column
    empty for its `uncertain` / `not_interleaved` verdicts, and `auto` names no parity
    either (the brightness heuristic decides at run time): no parity.
    """
    if channel_order == "gcamp_first":
        return "odd"
    if channel_order == "emo_first":
        return "even"
    if channel_order in ("odd_even_unassigned", "", "auto"):
        return UNASSIGNED
    raise ValueError(f"Unknown channel_order {channel_order!r}")


def agreement_marks(labels: dict[str, str]) -> dict[str, str]:
    """✓ for a system that agrees with every other labelled one, ⚠ otherwise.

    A system with no label ("unassigned") gets "–" and is left out of the others' check.
    """
    labelled = {name: label for name, label in labels.items() if label != UNASSIGNED}
    return {name: "–" if name not in labelled else
            "✓" if all(label == other for key, other in labelled.items() if key != name) else "⚠"
            for name, label in labels.items()}


def agreement_summary(labels: dict[str, str]) -> str:
    """Table column: ⚠ any labelled disagreement, ? a system has no label, ✓ all agree."""
    labelled = {label for label in labels.values() if label != UNASSIGNED}
    if len(labelled) > 1:
        return "⚠"
    return "?" if UNASSIGNED in labels.values() else "✓"


# ---------------------------------------------------------------------------
# Geometry. Points are (row, col) on the 256 grid unless said otherwise.
# ---------------------------------------------------------------------------
def bregma_point(bregma_row: int, bregma_col: int) -> np.ndarray:
    """Bregma in continuous 256-grid coordinates.

    ROI-set `bregma_row/col` are in 2x-grid units; the pipeline places Bregma on the
    128-grid pixel y_1 = bregma_row // 2 (1-based, MATLAB), i.e. 0-based 128 pixel
    y_1 - 1, which covers 256 rows [2 y_1 - 2, 2 y_1). Its centre is 2 y_1 - 1.
    """
    return np.array([2 * (bregma_row // 2) - 1.0, 2 * (bregma_col // 2) - 1.0])


def lambda_point(bregma: np.ndarray, lambda_row_offset: float, angle_deg: float = 0.0,
                 lambda_above_bregma: bool = False) -> np.ndarray:
    """Lambda on the 256 grid: `lambda_row_offset` 128-grid rows (= 2x on 256) from Bregma
    along the axis, direction (cos a, -sin a) in (row, col) as roi_editor.lambda_position
    and check_roi_symmetry.reflect_offsets; sign flipped when Lambda is above Bregma."""
    radians = math.radians(angle_deg)
    distance = (-1.0 if lambda_above_bregma else 1.0) * 2.0 * float(lambda_row_offset)
    return np.asarray(bregma, dtype=float) + distance * np.array([math.cos(radians), -math.sin(radians)])


def axis_frame(bregma: np.ndarray, lambda_: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(d, u, n): Bregma-Lambda distance, unit vector Lambda -> Bregma, and its normal.

    n = (u_col, -u_row) is what the rectangle's local "width" axis maps to when the
    rectangle is drawn as a pyqtgraph ROI rotated to u (view x = col, y = row).
    """
    delta = np.asarray(bregma, dtype=float) - np.asarray(lambda_, dtype=float)
    d = float(np.hypot(*delta))
    if d == 0:
        raise ValueError("Bregma and Lambda coincide: the axis is undefined")
    u = delta / d
    return d, u, np.array([u[1], -u[0]])


def rectangle_corners(bregma, lambda_, length_factor: float, width_factor: float,
                      start_offset_factor: float) -> np.ndarray:
    """(4, 2) corners in (row, col): start-left, start-right, end-right, end-left.

    "left/right" are -n / +n; corner 0 is the pyqtgraph ROI's local origin.
    """
    d, u, n = axis_frame(bregma, lambda_)
    start = np.asarray(lambda_, dtype=float) + start_offset_factor * d * u
    end = start + length_factor * d * u
    half = 0.5 * width_factor * d * n
    return np.array([start - half, start + half, end + half, end - half])


def params_corners(bregma, lambda_, params: RectangleParams) -> np.ndarray:
    return rectangle_corners(bregma, lambda_, params.length_factor, params.width_factor,
                             params.start_offset_factor)


def rectangle_fractions(shape: tuple[int, int], corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel (along, across) position in the rectangle's own frame, both (rows, cols).

    0..1 along = start..end edge, 0..1 across = corner 0..corner 1 side; pixel centres
    (r + 0.5, c + 0.5) are used. Outside the rectangle the values leave [0, 1].
    """
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    points = np.stack([rows + 0.5, cols + 0.5], axis=-1) - corners[0]  # (rows, cols, 2)
    along_edge = corners[3] - corners[0]
    across_edge = corners[1] - corners[0]
    along = points @ along_edge / float(along_edge @ along_edge)
    across = points @ across_edge / float(across_edge @ across_edge)
    return along, across


def rectangle_mask(shape: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    """Boolean (rows, cols) mask of the pixels whose centre lies inside the rectangle."""
    along, across = rectangle_fractions(shape, corners)
    return (along >= 0) & (along <= 1) & (across >= 0) & (across <= 1)


# ---------------------------------------------------------------------------
# Statistics: one value per image; returns which parity the rule calls GCaMP
# ---------------------------------------------------------------------------
def rectangle_statistic(image: np.ndarray, corners: np.ndarray, statistic: str) -> float:
    along, across = rectangle_fractions(image.shape, corners)
    inside = (along >= 0) & (along <= 1) & (across >= 0) & (across <= 1)
    if not inside.any():
        raise ValueError("The rectangle contains no pixel")
    values = image[inside]
    if statistic == "median_level":
        return float(np.median(values))
    if statistic == "spatial_cv":
        return float(np.std(values) / np.mean(values))
    if statistic == "centre_flank":
        # Middle third of the width (the sinus) against the two outer thirds.
        middle = inside & (across >= 1 / 3) & (across <= 2 / 3)
        flanks = inside & ~middle
        if not middle.any() or not flanks.any():
            raise ValueError("Rectangle too narrow for centre_flank: a third has no pixel")
        return float(abs(np.median(image[middle]) / np.median(image[flanks]) - 1.0))
    raise ValueError(f"Unknown statistic {statistic!r}")


def decision_problem(shape: tuple[int, int], corners: np.ndarray, statistic: str) -> str:
    """Why the rectangle cannot decide on an image of ``shape`` ('' when it can).

    The cases rectangle_statistic raises on, checked first so an interactive tool (the
    ROI editor, while Lambda is dragged) can say so instead of stopping.
    """
    along, across = rectangle_fractions(shape, corners)
    inside = (along >= 0) & (along <= 1) & (across >= 0) & (across <= 1)
    if not inside.any():
        return "the rectangle contains no pixel of the image"
    if statistic == "centre_flank":
        middle = inside & (across >= 1 / 3) & (across <= 2 / 3)
        if not middle.any() or not (inside & ~middle).any():
            return "the rectangle is too narrow for centre_flank"
    return ""


def gcamp_parity_from_statistic(odd_value: float, even_value: float, statistic: str) -> str:
    """median_level: lower = GCaMP. centre_flank / spatial_cv: higher = reflectance."""
    if statistic in STATISTICS:
        return "odd" if odd_value < even_value else "even"
    raise ValueError(f"Unknown statistic {statistic!r}")


def value_ratio(first: float, second: float) -> float:
    """Higher / lower of the two statistic values (>= 1): how clearly the rule separates
    odd from even. inf when the lower one is 0 (e.g. centre_flank on a flat image)."""
    high, low = max(first, second), min(first, second)
    return math.inf if low == 0 else high / low


def order_from_parity(parity: str) -> str:
    """GCaMP parity -> the pipeline's channel_order (odd = gcamp_first)."""
    return {"odd": "gcamp_first", "even": "emo_first"}[parity]


@dataclass(frozen=True)
class RectangleDecision:
    gcamp_parity: str          # "odd" | "even" by the rectangle rule
    odd_value: float
    even_value: float
    ratio: float               # >= 1, how clearly the rule separates the two
    level_parity: str          # the dimmer group over the whole image ("auto"'s rule)

    @property
    def channel_order(self) -> str:
        return order_from_parity(self.gcamp_parity)

    @property
    def agrees_with_level(self) -> bool:
        return self.gcamp_parity == self.level_parity


def decide(odd: np.ndarray, even: np.ndarray, bregma: np.ndarray, lambda_: np.ndarray,
           params: RectangleParams) -> RectangleDecision:
    """The rectangle's decision for one recording from its odd and even mean images.

    ``bregma`` / ``lambda_`` are in the images' own continuous coordinates.
    """
    corners = params_corners(bregma, lambda_, params)
    odd_value = rectangle_statistic(odd, corners, params.statistic)
    even_value = rectangle_statistic(even, corners, params.statistic)
    return RectangleDecision(
        gcamp_parity=gcamp_parity_from_statistic(odd_value, even_value, params.statistic),
        odd_value=odd_value, even_value=even_value, ratio=value_ratio(odd_value, even_value),
        level_parity="odd" if np.median(odd) < np.median(even) else "even")
