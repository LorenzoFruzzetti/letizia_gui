"""TIFF loading for wide-field imaging stacks.

The original MATLAB reader consumed *multi-page* TIFFs (one page per time
frame) with ``imread(NAME, ii)`` in a loop. It also warned that files larger
than ~4 GB are read incorrectly; ``tifffile`` used here has no such limit.

Two acquisition layouts are supported:
  * a single multi-page TIFF file  -> :func:`load_stack`
  * a folder of single-page TIFFs   -> :func:`load_frame_folder`
    (for example img_00001.tif, img_00002.tif, ..., each one 512x512 frame,
    forming one channel's time series when sorted.)

Every loader returns a float64 array shaped ``[y, x, time]`` to match the
MATLAB convention (rows, cols, frames).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import psutil
import tifffile

from .resize import imresize_box


# Valid values of every ``frame_order`` argument below.
FRAME_ORDERS = ("name", "number")
# Trailing integer of a file stem: "..._bin48_37" -> 37, "img_00001" -> 1.
_TRAILING_NUMBER_RE = re.compile(r"(\d+)$")


def sorted_frame_files(folder: str | Path, pattern: str = "*.tif",
                       frame_order: str = "name") -> list[Path]:
    """Files of one recording folder in acquisition order.

    ``frame_order`` says how that order is read off the file names:

    ``"name"`` (default)
        Lexicographic sort. Correct for zero-padded frame indices (img_00001,
        img_00002, ...).
    ``"number"``
        Sort by the integer at the END of the file stem. Needed when the index is
        NOT zero-padded (``..._1.tif``, ``..._2.tif``, ...,
        ``..._10.tif``), where a name sort gives 1, 10, 100, 1000, 1001, ... and
        mixes the two interleaved channels. A file with no trailing number raises
        (it would have no place in the order), and so does a repeated number.

    For zero-padded names with one shared prefix the two orders are identical.
    """
    if frame_order not in FRAME_ORDERS:
        raise ValueError(f"frame_order={frame_order!r} is not one of {FRAME_ORDERS}.")
    files = sorted(Path(folder).glob(pattern))
    if frame_order == "name":
        return files
    numbered = []
    for f in files:
        match = _TRAILING_NUMBER_RE.search(f.stem)
        if match is None:
            raise ValueError(f"frame_order='number': no trailing frame number in {f}")
        numbered.append((int(match.group(1)), f))
    numbered.sort(key=lambda pair: pair[0])
    numbers = [n for n, _ in numbered]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"frame_order='number': repeated frame numbers in {folder}")
    return [f for _, f in numbered]


def tiff_frame_count(path: str | Path) -> int:
    """Number of pages (time frames) in a multi-page TIFF, without reading them.

    Only the IFD headers are parsed, so this is cheap even for multi-GB files.
    Used by the streaming path to size the baseline window before any frame is
    loaded into memory.
    """
    with tifffile.TiffFile(str(path)) as tif:
        return len(tif.pages)


def iter_tiff_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield one ``[y, x]`` float64 frame at a time from a multi-page TIFF.

    This is the constant-memory alternative to :func:`load_stack`: pages are
    decoded lazily as the generator is consumed, so a recording of any length
    (including the >4 GB files the MATLAB reader mishandled) never has more than
    one frame resident at a time.
    """
    with tifffile.TiffFile(str(path)) as tif:
        for page in tif.pages:
            yield np.asarray(page.asarray(), dtype=np.float64)


@dataclass
class FrameSource:
    """A *re-iterable* sequence of ``[y, x]`` float64 frames plus its length.

    This is the storage-format-agnostic input the streaming pipeline consumes.
    Streaming makes TWO passes over each channel (baseline mean, then the
    per-frame correction), so it must be able to restart the sequence: ``open()``
    returns a *fresh* iterator each call. ``count`` is the number of frames,
    known without decoding any pixels, so the baseline window can be sized up
    front while memory stays constant.

    Decoupling the frame source from the on-disk layout is what lets any storage
    format (single multi-page TIFF, folder of single-page TIFFs, or one channel's
    half of an interleaved folder) feed the same constant-memory math -- i.e. the
    "how the channels are stored" axis is now independent of the "stream vs load
    it all" axis.
    """

    count: int
    _open: Callable[[], Iterator[np.ndarray]]

    def open(self) -> Iterator[np.ndarray]:
        """Return a fresh ``[y, x]`` frame iterator for one pass over the data."""
        return self._open()


def tiff_frame_source(path: str | Path) -> FrameSource:
    """A :class:`FrameSource` backed by one multi-page TIFF (one page per frame).

    ``count`` comes from the IFD headers only (:func:`tiff_frame_count`) and each
    pass lazily decodes pages (:func:`iter_tiff_frames`), so nothing beyond a
    single frame is ever resident.
    """
    return FrameSource(tiff_frame_count(path), lambda: iter_tiff_frames(path))


def iter_folder_frames(files: list[str | Path], rot90_k: int = 0) -> Iterator[np.ndarray]:
    """Yield ``[y, x]`` float64 frames from an ordered list of single-page TIFFs.

    The folder equivalent of :func:`iter_tiff_frames`: one file is decoded per
    step, so a channel stored as thousands of single-page TIFFs streams in
    constant memory just like a multi-page file.

    ``rot90_k`` rotates every frame by ``k`` quarter turns right after decoding
    (``np.rot90``: positive is counter-clockwise, -1 is clockwise), for a camera
    mounted at 90 degrees to the atlas: with the midline horizontal and anterior
    on the left, -1 puts anterior on top. It is a rotation, never a
    mirror, so the hemispheres keep their handedness. 0 = frames unchanged.
    """
    for f in files:
        frame = np.asarray(tifffile.imread(str(f)), dtype=np.float64)
        if rot90_k % 4:
            # np.rot90 returns a strided view; copy so downstream code gets C order.
            frame = np.ascontiguousarray(np.rot90(frame, rot90_k))
        yield frame


def folder_frame_source(files: list[str | Path], rot90_k: int = 0) -> FrameSource:
    """A :class:`FrameSource` backed by an explicit, ordered list of TIFF files.

    Used for one channel of an interleaved folder (the odd- or even-positioned
    images already split out by :func:`interleaved_channel_files`), where the
    frame list is not a whole directory. ``count`` is just ``len(files)``.
    ``rot90_k`` is passed to :func:`iter_folder_frames`.
    """
    files = list(files)
    return FrameSource(len(files), lambda: iter_folder_frames(files, rot90_k))


def frame_folder_source(folder: str | Path, pattern: str = "*.tif",
                        frame_order: str = "name") -> FrameSource:
    """A :class:`FrameSource` backed by a folder of single-page TIFFs (one channel).

    Streaming counterpart of :func:`load_frame_folder`: same sort
    (:func:`sorted_frame_files`), but frames are decoded one at a time instead of
    stacked into RAM.
    """
    files = sorted_frame_files(folder, pattern, frame_order)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")
    return folder_frame_source(files)


def load_stack(path: str | Path) -> np.ndarray:
    """Load a multi-page TIFF as a ``[y, x, time]`` float64 array."""
    arr = tifffile.imread(str(path))  # -> [time, y, x] or [y, x] if single page
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # tifffile yields [time, y, x]; MATLAB convention is [y, x, time].
    return np.moveaxis(arr, 0, -1)


def load_frame_folder(folder: str | Path, pattern: str = "*.tif",
                      frame_order: str = "name") -> np.ndarray:
    """Assemble single-page TIFFs in a folder into one ``[y, x, time]`` stack.

    Files are ordered by :func:`sorted_frame_files`: by name (img_00001,
    img_00002, ..., the acquisition order for zero-padded frame indices) unless
    ``frame_order="number"``.
    """
    folder = Path(folder)
    files = sorted_frame_files(folder, pattern, frame_order)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")
    frames = [np.asarray(tifffile.imread(str(f)), dtype=np.float64) for f in files]
    # Stack along a new time axis -> [y, x, time].
    return np.stack(frames, axis=-1)


def _top_percent_mean(image: np.ndarray, top_percent: float = 10.0) -> float:
    """Mean of the brightest ``top_percent``% of pixels in one image.

    Same signal proxy used by ``src/inspect_channel_intensity.py``: taking only
    the top pixels compares the two channels on their illuminated part rather
    than on background, which is what separates the two interleaved groups
    cleanly. On this rig the reflectance (emo) channel is the *brighter* of the
    two and GCaMP the dimmer, so the group with the LOWER value here is gcamp.
    """
    pixels = np.asarray(image, dtype=np.float64).ravel()
    threshold = np.percentile(pixels, 100.0 - top_percent)
    return float(pixels[pixels >= threshold].mean())


def auto_channel_order(
    folder: str | Path,
    pattern: str = "*.tif",
    top_percent: float = 10.0,
    frame_order: str = "name",
) -> tuple[str, float, float]:
    """What ``channel_order="auto"`` resolves to for ONE folder, made explicit.

    Reads the first two images in acquisition order (the first of the odd and of the
    even group) and returns ``(order, odd_intensity, even_intensity)``: ``order`` is
    ``"gcamp_first"`` when the odd image is the dimmer (or equal) one, else
    ``"emo_first"``. This is the single place the ``auto`` rule lives;
    :func:`interleaved_channel_files` calls it, so an explicit order written from it
    reproduces an ``auto`` run exactly.
    """
    files = sorted_frame_files(folder, pattern, frame_order)
    if len(files) < 2:
        raise FileNotFoundError(
            f"Need at least 2 images matching {pattern!r} in {folder} to split "
            f"two interleaved channels; found {len(files)}."
        )
    odd_intensity = _top_percent_mean(tifffile.imread(str(files[0])), top_percent)
    even_intensity = _top_percent_mean(tifffile.imread(str(files[1])), top_percent)
    order = "gcamp_first" if odd_intensity <= even_intensity else "emo_first"
    return order, odd_intensity, even_intensity


def interleaved_channel_files(
    folder: str | Path,
    pattern: str = "*.tif",
    top_percent: float = 10.0,
    channel_order: str = "auto",
    frame_order: str = "name",
) -> tuple[list[Path], list[Path]]:
    """Split ONE interleaved folder into ordered ``(gcamp_files, emo_files)`` lists.

    This is the *split-and-identify* half of the interleaved layout, done without
    decoding a single full frame's worth of the recording: sorted by name, the
    odd-positioned images (1st, 3rd, 5th, ...) are one channel and the
    even-positioned images (2nd, 4th, ...) the other -- exactly the odd/even
    split that ``src/inspect_channel_intensity.py`` inspects.

    ``channel_order`` decides which group is which:

    ``"auto"`` (default)
        Decide by intensity: the DIMMER group is GCaMP, the brighter one is emo
        (the reflectance channel, which on this rig comes back stronger than the
        GCaMP fluorescence). We compare the first image of each group (i.e. the
        first two images in the folder) by the mean of their top
        ``top_percent``% pixels; the dimmer one's group becomes gcamp. Only those
        two images are read, so this stays cheap even for a folder too large to
        load.
    ``"gcamp_first"`` / ``"emo_first"``
        Assign by position instead, reading nothing at all. This is what the
        MATLAB scripts do implicitly (the cerebellar ones take GCaMP first, the
        cortical ones emo first) -- and getting it wrong silently swaps the
        channels, inverting the hemodynamic correction. Use these only when you
        know the layout and the brightness heuristic misfires (e.g. an unusually
        bright GCaMP recording, where the two channels are close enough that the
        dimmer-is-gcamp rule can pick the wrong group).

    ``frame_order`` is how acquisition order is read off the file names
    (:func:`sorted_frame_files`); the odd/even split is only right in that order.

    The two lists are truncated to equal length (odd total -> groups differ by
    one) so the channels stay in lockstep for the per-frame hemodynamic
    correction. :func:`load_interleaved_folder` stacks these into memory; the
    streaming path feeds each list to :func:`folder_frame_source` instead, so the
    same split powers both the load-it-all and constant-memory variants.
    """
    valid_orders = ("auto", "gcamp_first", "emo_first")
    if channel_order not in valid_orders:
        raise ValueError(
            f"channel_order={channel_order!r} is not one of {valid_orders}."
        )

    folder = Path(folder)
    files = sorted_frame_files(folder, pattern, frame_order)
    if len(files) < 2:
        raise FileNotFoundError(
            f"Need at least 2 images matching {pattern!r} in {folder} to split "
            f"two interleaved channels; found {len(files)}."
        )

    # 1-based position: odd group = 1st, 3rd, ... -> files[0::2];
    #                   even group = 2nd, 4th, ... -> files[1::2].
    odd_files = files[0::2]
    even_files = files[1::2]

    if channel_order == "auto":
        # Decide which group is the dimmer (GCaMP) channel from the first image
        # of each group -- "the first two images" the user inspects.
        channel_order, _, _ = auto_channel_order(folder, pattern, top_percent, frame_order)
    if channel_order == "gcamp_first":
        gcamp_files, emo_files = odd_files, even_files
    else:
        gcamp_files, emo_files = even_files, odd_files

    # Keep the two channels the same length (odd total -> groups differ by one).
    n = min(len(gcamp_files), len(emo_files))
    return gcamp_files[:n], emo_files[:n]


def load_interleaved_folder(
    folder: str | Path,
    pattern: str = "*.tif",
    top_percent: float = 10.0,
    channel_order: str = "auto",
    frame_order: str = "name",
) -> tuple[np.ndarray, np.ndarray]:
    """Split ONE folder of interleaved single-page TIFFs into ``(gcamp, emo)``.

    Full-load variant: :func:`interleaved_channel_files` decides the odd/even
    split and channel identity, then each group is stacked into a ``[y, x, time]``
    float64 array ready to be used as one trial by the in-memory pipeline. For a
    recording too large for RAM, feed the same split to the streaming path via
    :func:`interleaved_channel_files` + :func:`folder_frame_source` instead.
    """
    gcamp_files, emo_files = interleaved_channel_files(
        folder, pattern, top_percent, channel_order, frame_order
    )
    gcamp = np.stack(
        [np.asarray(tifffile.imread(str(f)), dtype=np.float64) for f in gcamp_files],
        axis=-1,
    )
    emo = np.stack(
        [np.asarray(tifffile.imread(str(f)), dtype=np.float64) for f in emo_files],
        axis=-1,
    )
    return gcamp, emo


# ---------------------------------------------------------------------------
# Whole-recording channel selection with a single read
#
# ``channel_order="auto"`` above decides from TWO frames. The functions below
# instead read every frame of a block once, keep the two interleaved groups
# UNLABELLED (odd / even) in a buffer while collecting per-frame statistics, and
# only then decide which group is GCaMP -- from every frame of the recording.
# The labelled buffers are handed to the unchanged streaming math, so the results
# are bit-identical to the two-read path for the same channel order.
# ---------------------------------------------------------------------------

# Per-frame level statistics :func:`decide_channel_order` can decide on.
LEVEL_STATISTICS = ("top_percent", "median")
# Which pixels of each frame the per-frame statistics are taken over:
#   "frame"     -- the whole (rotated, half-resolution) frame;
#   "roi_union" -- only the pixels under the recording's ROI boxes (cortex).
LEVEL_REGIONS = ("frame", "roi_union")
# What :func:`decide_channel_order` can conclude about one block.
CHANNEL_VERDICTS = ("certain", "uncertain", "not_interleaved")
# Where a block's half-resolution frames are kept between the read and pass 2.
BUFFER_MODES = ("auto", "ram", "disk", "none")
# ``auto`` buffers in RAM only if this multiple of the estimate fits in the free
# RAM (the streaming math and the OS need room too), and on disk only if this
# multiple fits on the buffer drive.
RAM_SAFETY_FACTOR = 1.5
DISK_SAFETY_FACTOR = 1.1


@dataclass(frozen=True)
class ChannelDecisionConfig:
    """Thresholds of :func:`decide_channel_order`.

    The defaults were calibrated on sampled full-resolution frames; here they are
    applied to every half-resolution trimmed frame, so validate them against a
    checked channel assignment before trusting them on a new cohort.
    """

    # Per-frame level the order and the separation are decided on:
    #   "median"      -- the frame median (default; the more reliable of the two);
    #   "top_percent" -- mean of the brightest ``top_percent``% pixels. It can put
    #                    channels close in brightness in the wrong order, where the
    #                    median does not.
    level_statistic: str = "median"
    # Used by level_statistic "top_percent" (and always recorded in ChannelStats).
    top_percent: float = 10.0
    # Pixels the per-frame statistics (levels AND alternation) are taken over; see
    # LEVEL_REGIONS. "roi_union" needs the caller to supply the mask (the pipeline
    # builds it from the recording's ROI set: wfci.streaming.roi_union_level_mask).
    # Over cortex the two channels can be close or even reversed in brightness
    # (the GCaMP/emo ratio spreads wider over the ROI union than over the frame), so
    # dimmer-is-GCaMP is calibrated for "frame" only; validate before relying on it.
    level_region: str = "frame"
    # The brightness alternates frame to frame when the lag-1 autocorrelation of
    # the interleaved per-frame medians is at or below this (strict alternation ~-1).
    alternation_max: float = -0.9
    # The two groups are told apart reliably when |level difference| / mean
    # within-group scatter is at least this.
    separation_min: float = 10.0


def frame_median_and_top_mean(frame: np.ndarray, top_percent: float = 10.0) -> tuple[float, float]:
    """``(median, mean of the brightest top_percent% pixels)`` of one frame.

    One ``np.partition`` gives both, which matters because this runs on every
    frame of every recording. The median equals ``np.median`` (mean of the two
    middle values for an even pixel count); the top set is the ``k`` largest
    pixels, ``k = round(n * top_percent / 100)``.
    """
    flat = np.asarray(frame, dtype=np.float64).ravel()
    n = flat.size
    k = max(1, int(round(n * top_percent / 100.0)))
    mid = n // 2
    even_count = n % 2 == 0
    kth = sorted({mid - 1, mid, n - k} if even_count else {mid, n - k})
    part = np.partition(flat, kth)
    median = 0.5 * (part[mid - 1] + part[mid]) if even_count else part[mid]
    return float(median), float(part[n - k:].mean())


class ChannelStats:
    """Per-frame statistics of the two unlabelled interleaved groups.

    Filled one frame pair at a time while a block is read (``add``), so it never
    needs the frames themselves. ``odd`` = the files at 1-based odd positions
    (``files[0::2]``), ``even`` = ``files[1::2]``.
    """

    def __init__(self, top_percent: float = 10.0, level_mask: np.ndarray | None = None) -> None:
        self.top_percent = float(top_percent)
        # Optional boolean [y, x] mask at the frame's resolution: statistics are then
        # taken over frame[level_mask] only. None = the whole frame.
        if level_mask is not None:
            level_mask = np.asarray(level_mask, dtype=bool)
            if not level_mask.any():
                raise ValueError("level_mask selects no pixel.")
        self.level_mask = level_mask
        self._odd_medians: list[float] = []
        self._even_medians: list[float] = []
        self._odd_levels: list[float] = []
        self._even_levels: list[float] = []

    def add(self, odd_frame: np.ndarray, even_frame: np.ndarray) -> None:
        """Record one acquisition-ordered pair: the odd frame, then the even one."""
        if self.level_mask is not None:
            if odd_frame.shape != self.level_mask.shape or even_frame.shape != self.level_mask.shape:
                raise ValueError(f"level_mask {self.level_mask.shape} does not match the frames "
                                 f"{odd_frame.shape} / {even_frame.shape}.")
            odd_frame, even_frame = odd_frame[self.level_mask], even_frame[self.level_mask]
        odd_median, odd_level = frame_median_and_top_mean(odd_frame, self.top_percent)
        even_median, even_level = frame_median_and_top_mean(even_frame, self.top_percent)
        self._odd_medians.append(odd_median)
        self._even_medians.append(even_median)
        self._odd_levels.append(odd_level)
        self._even_levels.append(even_level)

    @property
    def level_region(self) -> str:
        """"roi_union" when the statistics were taken under a mask, else "frame"."""
        return "frame" if self.level_mask is None else "roi_union"

    @property
    def n_frames(self) -> int:
        """Frame pairs seen so far (= frames per group)."""
        return len(self._odd_medians)

    @property
    def odd_medians(self) -> np.ndarray:
        return np.asarray(self._odd_medians, dtype=np.float64)

    @property
    def even_medians(self) -> np.ndarray:
        return np.asarray(self._even_medians, dtype=np.float64)

    @property
    def odd_levels(self) -> np.ndarray:
        return np.asarray(self._odd_levels, dtype=np.float64)

    @property
    def even_levels(self) -> np.ndarray:
        return np.asarray(self._even_levels, dtype=np.float64)


def channel_stats_from_stacks(odd: np.ndarray, even: np.ndarray, trim: int,
                              downsample: float = 0.5,
                              top_percent: float = 10.0,
                              level_mask_for: Callable[[tuple[int, int]], np.ndarray] | None = None,
                              ) -> ChannelStats:
    """:class:`ChannelStats` of two in-memory ``[y, x, time]`` groups (debug path).

    Each frame is trimmed and box-downsampled exactly as the streaming read does,
    so a debug run decides from the same kind of frames as a full one, just fewer.
    ``level_mask_for(half_shape)`` optionally returns the pixel mask (see
    :func:`read_interleaved_block`).
    """
    n = min(odd.shape[2], even.shape[2])
    level_mask = None
    if level_mask_for is not None and n > trim:
        level_mask = level_mask_for(imresize_box(odd[:, :, trim], downsample).shape)
    stats = ChannelStats(top_percent, level_mask)
    for idx in range(trim, n):
        stats.add(imresize_box(odd[:, :, idx], downsample),
                  imresize_box(even[:, :, idx], downsample))
    return stats


def _lag1_autocorrelation(values: np.ndarray) -> float:
    """Pearson r between ``values[:-1]`` and ``values[1:]``; NaN when undefined.

    Written out rather than via ``np.corrcoef`` to keep BLAS out of the path.
    """
    a = values[:-1] - values[:-1].mean()
    b = values[1:] - values[1:].mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def decide_channel_order(stats: ChannelStats,
                         config: ChannelDecisionConfig = ChannelDecisionConfig(),
                         ) -> tuple[str, str, dict[str, Any]]:
    """Which interleaved group is GCaMP, from every frame of a block.

    Returns ``(order, verdict, details)``. This is the ONE place the decision
    statistic lives, so it can be replaced without touching the buffering or the
    pipeline.

    ``alternation``
        Lag-1 autocorrelation of the interleaved per-frame medians (odd0, even0,
        odd1, even1, ...). Two interleaved channels give ~-1; a value near 0 means
        the brightness does not alternate (unsynchronised
        LED), and then NO assignment is correct.
    ``separation``
        ``|median(odd levels) - median(even levels)|`` over the mean within-group
        standard deviation of the levels.
    ``levels``
        Per-frame mean of the brightest ``top_percent``% pixels, or the per-frame
        median (``config.level_statistic``). The dimmer group
        (lower median level) is GCaMP on this rig: odd dimmer (or equal) ->
        ``"gcamp_first"``, else ``"emo_first"`` -- the same tie rule as
        :func:`auto_channel_order`.

    ``verdict``: ``"certain"`` (alternates and separates), ``"uncertain"``
    (alternates, too close to tell apart -- the order is still the dimmer-is-GCaMP
    guess) or ``"not_interleaved"`` (does not alternate; the order is meaningless).
    """
    n = stats.n_frames
    if n < 2:
        raise ValueError(f"Need at least 2 frame pairs to decide the channel order; got {n}.")
    if config.level_region not in LEVEL_REGIONS:
        raise ValueError(f"level_region={config.level_region!r} is not one of {LEVEL_REGIONS}.")
    if stats.level_region != config.level_region:
        # The statistics were collected over a different region than the config
        # asks for: deciding anyway would silently use the wrong pixels.
        raise ValueError(f"Channel statistics were taken over {stats.level_region!r} but "
                         f"level_region={config.level_region!r}.")
    # odd0, even0, odd1, even1, ... -- acquisition order.
    sequence = np.empty(2 * n, dtype=np.float64)
    sequence[0::2] = stats.odd_medians
    sequence[1::2] = stats.even_medians
    alternation = _lag1_autocorrelation(sequence)

    if config.level_statistic == "top_percent":
        odd_levels, even_levels = stats.odd_levels, stats.even_levels
    elif config.level_statistic == "median":
        odd_levels, even_levels = stats.odd_medians, stats.even_medians
    else:
        raise ValueError(f"level_statistic={config.level_statistic!r} is not one of "
                         f"{LEVEL_STATISTICS}.")
    level_odd = float(np.median(odd_levels))
    level_even = float(np.median(even_levels))
    within = 0.5 * (float(odd_levels.std()) + float(even_levels.std()))
    difference = abs(level_odd - level_even)
    if within > 0:
        separation = difference / within
    else:
        # Constant levels: infinitely separated if they differ, not at all if not.
        separation = float("inf") if difference > 0 else 0.0

    order = "gcamp_first" if level_odd <= level_even else "emo_first"
    # A NaN alternation (constant medians) fails the <= test: not interleaved.
    if not alternation <= config.alternation_max:
        verdict = "not_interleaved"
    elif separation >= config.separation_min:
        verdict = "certain"
    else:
        verdict = "uncertain"
    details = dict(level_odd=level_odd, level_even=level_even,
                   alternation=alternation, separation=separation, n_frames=n,
                   level_statistic=config.level_statistic, level_region=config.level_region)
    return order, verdict, details


def buffer_dtype_for(raw_dtype: Any, raw_shape: tuple[int, ...],
                     downsample: float = 0.5) -> np.dtype:
    """dtype that holds a block's half-resolution frames EXACTLY.

    An integer frame of 16 bits or fewer, box-downsampled by 0.5 on even
    dimensions, is a grid of 2x2 means: multiples of 0.25 below 65536, which
    float32 represents exactly (18 significant bits). Anything else -- a float or
    wider integer raw frame, an odd dimension (MATLAB's fractional box weights), a
    different scale -- is buffered in float64 so the buffer can never round.
    :func:`read_interleaved_block` additionally checks every float32 cast.
    """
    raw = np.dtype(raw_dtype)
    exact_float32 = (raw.kind in "ui" and raw.itemsize <= 2 and downsample == 0.5
                     and all(int(d) % 2 == 0 for d in raw_shape[:2]))
    return np.dtype(np.float32 if exact_float32 else np.float64)


def estimate_buffer_bytes(n_frames_per_group: int, half_shape: tuple[int, int],
                          dtype: Any) -> int:
    """Bytes needed to buffer BOTH groups of one block (``n`` frames each)."""
    return 2 * int(n_frames_per_group) * int(half_shape[0]) * int(half_shape[1]) \
        * np.dtype(dtype).itemsize


def choose_buffer_mode(requested: str, estimate_bytes: int, buffered_bytes: int = 0,
                       buffer_dir: str | os.PathLike | None = None) -> str:
    """Resolve ``channel_buffer`` for one block.

    An explicit ``"ram"`` / ``"disk"`` / ``"none"`` is returned as is (``"disk"``
    without a ``buffer_dir`` raises). ``"auto"`` picks, in order:

    1. ``"ram"`` if ``estimate * 1.5`` fits in the available RAM minus what this
       run has already buffered in RAM (``buffered_bytes``);
    2. ``"disk"`` if ``buffer_dir`` is set and has room for ``estimate * 1.1``;
    3. ``"none"`` (no buffer: the TIFFs are read twice).
    """
    if requested not in BUFFER_MODES:
        raise ValueError(f"channel_buffer={requested!r} is not one of {BUFFER_MODES}.")
    if requested == "disk" and not buffer_dir:
        raise ValueError("channel_buffer='disk' needs a buffer_dir.")
    if requested != "auto":
        return requested
    available = psutil.virtual_memory().available - int(buffered_bytes)
    if estimate_bytes * RAM_SAFETY_FACTOR <= available:
        return "ram"
    if buffer_dir:
        os.makedirs(buffer_dir, exist_ok=True)
        if estimate_bytes * DISK_SAFETY_FACTOR <= shutil.disk_usage(buffer_dir).free:
            return "disk"
    return "none"


# -- half-resolution frame sources --------------------------------------------
# A block's frames live in one of three places after the read; each segment class
# yields them back as float64 [y, x] frames, the type the streaming math expects.

class _ArraySegment:
    """Frames held in a ``[time, y, x]`` in-memory array (``ram`` mode)."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def frames(self) -> Iterator[np.ndarray]:
        for idx in range(len(self)):
            # float32 -> float64 is exact, so pass 2 sees the very values it would
            # have computed from the TIFF.
            yield self.array[idx].astype(np.float64)


class _NpyFileSegment:
    """Frames held in a ``[time, y, x]`` .npy file (``disk`` mode).

    Each frame is read with its own short ``open`` rather than through a memmap:
    nothing keeps the file mapped, so it can be deleted or moved as soon as the
    run ends -- including after an exception, when a memmap still referenced from
    the traceback would lock the file on Windows.
    """

    def __init__(self, path: str, n_frames: int, frame_shape: tuple[int, int],
                 dtype: np.dtype, data_offset: int) -> None:
        self.path = path
        self.n_frames = int(n_frames)
        self.frame_shape = tuple(frame_shape)
        self.dtype = np.dtype(dtype)
        self.data_offset = int(data_offset)

    def __len__(self) -> int:
        return self.n_frames

    def frames(self) -> Iterator[np.ndarray]:
        frame_bytes = self.frame_shape[0] * self.frame_shape[1] * self.dtype.itemsize
        for idx in range(self.n_frames):
            with open(self.path, "rb") as fh:
                fh.seek(self.data_offset + idx * frame_bytes)
                raw = fh.read(frame_bytes)
            yield np.frombuffer(raw, dtype=self.dtype).reshape(self.frame_shape).astype(np.float64)


class _TiffSegment:
    """Frames re-read from the TIFFs (``none`` mode): decode, rotate, downsample."""

    def __init__(self, files: list[Path], rot90_k: int, downsample: float) -> None:
        self.files = list(files)
        self.rot90_k = int(rot90_k)
        self.downsample = downsample

    def __len__(self) -> int:
        return len(self.files)

    def frames(self) -> Iterator[np.ndarray]:
        for frame in iter_folder_frames(self.files, self.rot90_k):
            yield imresize_box(frame, self.downsample)


@dataclass
class HalfResSource:
    """One channel as already trimmed, once-downsampled float64 frames.

    The buffered counterpart of :class:`FrameSource`. ``segments`` are the
    channel's blocks in time order (one block for a single recording, several for
    a merged run). ``trim`` is the ``trim`` the streaming pipeline will be called
    with, and ``count`` includes it, so ``count - trim`` is the number of frames
    actually held -- the same arithmetic the pipeline applies to a raw source.
    :func:`wfci.streaming.stream_trial_roi` checks that ``trim`` and
    ``downsample`` match its own, because these frames have both already applied.
    """

    segments: list
    trim: int
    downsample: float

    @property
    def n_frames(self) -> int:
        return sum(len(segment) for segment in self.segments)

    @property
    def count(self) -> int:
        return self.n_frames + self.trim

    def frames(self) -> Iterator[np.ndarray]:
        """A fresh pass over every held frame, in time order."""
        for segment in self.segments:
            yield from segment.frames()


def _half_res(raw: np.ndarray, rot90_k: int, downsample: float) -> np.ndarray:
    """Raw TIFF frame -> half-resolution float64 frame.

    The exact steps :func:`iter_folder_frames` + ``streaming._half_res_frames``
    apply, in the same order, so the result is the same array.
    """
    frame = np.asarray(raw, dtype=np.float64)
    if rot90_k % 4:
        frame = np.ascontiguousarray(np.rot90(frame, rot90_k))
    return imresize_box(frame, downsample)


@dataclass
class OddEvenBlock:
    """One ``t#`` block read once: its two unlabelled groups plus what pass 1 needs.

    ``odd`` / ``even`` are the trimmed half-resolution frames (segments, see
    :class:`HalfResSource`). ``sum_odd`` / ``sum_even`` are the running sums of
    the frames inside the baseline window -- pass 1 of the streaming pipeline,
    done during the read so no buffer mode needs a second look for it.
    """

    folder: str
    odd: Any
    even: Any
    n_frames: int
    mode: str
    stats: ChannelStats
    sum_odd: np.ndarray | None
    sum_even: np.ndarray | None
    n_in_window: int
    raw_dtype: str
    buffer_dtype: str | None
    # RAM held by this block's buffer (0 unless mode == "ram").
    ram_bytes: int = 0
    # Temporary directory of a ``disk`` buffer, removed by ``release``.
    disk_dir: str | None = None
    _released: bool = field(default=False, repr=False)

    def labelled(self, order: str):
        """``(gcamp_segment, emo_segment, gcamp_sum, emo_sum)`` for ``order``."""
        if order == "gcamp_first":
            return self.odd, self.even, self.sum_odd, self.sum_even
        if order == "emo_first":
            return self.even, self.odd, self.sum_even, self.sum_odd
        raise ValueError(f"order must be 'gcamp_first' or 'emo_first', got {order!r}.")

    def keep(self, odd_path: str, even_path: str) -> bool:
        """Save the unlabelled odd/even volumes as ``.npy``; False if unbuffered.

        ``ram`` buffers are written out; ``disk`` buffers are MOVED (they are
        already valid .npy files), so they are not removed by :meth:`release`.
        """
        if self.mode == "none":
            return False
        os.makedirs(os.path.dirname(os.path.abspath(odd_path)), exist_ok=True)
        if self.mode == "ram":
            np.save(odd_path, self.odd.array)
            np.save(even_path, self.even.array)
        else:
            shutil.move(self.odd.path, odd_path)
            shutil.move(self.even.path, even_path)
            self.odd.path, self.even.path = odd_path, even_path
        return True

    def release(self) -> None:
        """Drop the RAM buffer and delete a disk buffer's temporary directory."""
        if self._released:
            return
        self._released = True
        if self.mode == "ram":
            self.odd.array = self.even.array = None
        if self.disk_dir is not None and os.path.isdir(self.disk_dir):
            shutil.rmtree(self.disk_dir)


def split_odd_even(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Acquisition-ordered files -> ``(odd, even)`` groups of equal length.

    1-based odd positions (1st, 3rd, ...) are ``files[0::2]``. Truncated to equal
    length exactly as :func:`interleaved_channel_files` does.
    """
    odd, even = list(files[0::2]), list(files[1::2])
    n = min(len(odd), len(even))
    return odd[:n], even[:n]


def _open_npy_for_frames(path: str, n_frames: int, frame_shape: tuple[int, int],
                         dtype: np.dtype):
    """Open ``path`` for writing a ``[n, y, x]`` .npy frame by frame.

    Returns ``(file_handle, data_offset)``: the header is written up front so
    the finished file is a plain .npy, and frames are appended in order.
    """
    fh = open(path, "wb")
    header = {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
              "fortran_order": False, "shape": (int(n_frames), *frame_shape)}
    np.lib.format.write_array_header_1_0(fh, header)
    return fh, fh.tell()


def read_interleaved_block(
    files: list[Path],
    trim: int,
    downsample: float = 0.5,
    rot90_k: int = 0,
    window: tuple[int, int] = (0, 2**62),
    channel_buffer: str = "auto",
    buffer_dir: str | os.PathLike | None = None,
    buffered_bytes: int = 0,
    top_percent: float = 10.0,
    level_mask_for: Callable[[tuple[int, int]], np.ndarray] | None = None,
) -> OddEvenBlock:
    """Read ONE interleaved block from disk exactly once.

    ``level_mask_for(half_shape)``, when given, returns a boolean ``[y, x]`` mask at
    the half-resolution frame shape; the decision statistics are then taken over
    the masked pixels only (``ChannelDecisionConfig.level_region = "roi_union"``).
    It is called once, after the first frame fixes the shape.

    ``files`` are the block's images in acquisition order (:func:`sorted_frame_files`).
    They are read pairwise in that order (odd frame, then even frame), skipping the
    first ``trim`` frames of each group, which are never decoded. Each frame is
    rotated by ``rot90_k`` and box-downsampled by ``downsample`` -> the half-res
    frame the pipeline works on. For every pair this:

    * updates the decision statistics (:class:`ChannelStats`);
    * adds the frame to the group's baseline sum if its post-trim index is in
      ``window`` (``[start, stop)``, block-local) -- this is pass 1;
    * stores it in the buffer chosen by :func:`choose_buffer_mode`, which is
      sized from the frame count and the first frame's shape and dtype.

    In ``none`` mode nothing is stored and the returned segments re-read the
    TIFFs, so pass 2 is the second read. Any ``disk`` buffer is deleted if the
    read itself fails.
    """
    odd_files, even_files = split_odd_even(files)
    n = len(odd_files)
    if n <= trim:
        raise ValueError(
            f"{os.path.dirname(str(files[0])) if files else '<empty>'}: only {n} "
            f"frames/channel, at or below the {trim}-frame onset trim."
        )
    n_keep = n - trim
    start, stop = window

    # The first frame fixes the frame shape, the raw dtype and so the buffer size.
    first_raw = tifffile.imread(str(odd_files[trim]))
    raw_dtype = np.dtype(first_raw.dtype)
    first_half = _half_res(first_raw, rot90_k, downsample)
    half_shape = first_half.shape
    buf_dtype = buffer_dtype_for(raw_dtype, first_raw.shape, downsample)
    estimate = estimate_buffer_bytes(n_keep, half_shape, buf_dtype)
    mode = choose_buffer_mode(channel_buffer, estimate, buffered_bytes, buffer_dir)

    stats = ChannelStats(top_percent,
                         None if level_mask_for is None else level_mask_for(half_shape))
    sums = [np.zeros(half_shape, dtype=np.float64), np.zeros(half_shape, dtype=np.float64)]
    n_in_window = 0
    disk_dir = None
    handles: list = []
    offsets: list[int] = []
    arrays: list[np.ndarray] = []
    if mode == "ram":
        arrays = [np.empty((n_keep, *half_shape), dtype=buf_dtype) for _ in range(2)]
    elif mode == "disk":
        os.makedirs(buffer_dir, exist_ok=True)
        disk_dir = tempfile.mkdtemp(prefix="wfci_channel_buffer_", dir=str(buffer_dir))
        for name in ("odd", "even"):
            fh, offset = _open_npy_for_frames(os.path.join(disk_dir, f"{name}.npy"),
                                              n_keep, half_shape, buf_dtype)
            handles.append(fh)
            offsets.append(offset)

    finished = False
    try:
        for i in range(n_keep):
            odd_half = first_half if i == 0 else _half_res(
                tifffile.imread(str(odd_files[trim + i])), rot90_k, downsample)
            even_half = _half_res(tifffile.imread(str(even_files[trim + i])), rot90_k, downsample)
            for half in (odd_half, even_half):
                if half.shape != half_shape:
                    raise ValueError(f"Frame shape {half.shape} differs from the "
                                     f"block's first frame {half_shape}.")
            stats.add(odd_half, even_half)
            # Pass 1: the running sum, in frame order -- the same additions the
            # streaming pass 1 makes, so the baseline is the same array.
            if start <= i < stop:
                sums[0] += odd_half
                sums[1] += even_half
                n_in_window += 1
            if mode in ("ram", "disk"):
                for g, half in enumerate((odd_half, even_half)):
                    stored = half.astype(buf_dtype)
                    if buf_dtype == np.float32 and not np.array_equal(stored, half):
                        # Cannot happen for <=16-bit integer input (buffer_dtype_for);
                        # checked anyway because bit-identity rests on it.
                        raise ValueError("A half-resolution frame is not exactly "
                                         "representable in float32; refusing to round.")
                    if mode == "ram":
                        arrays[g][i] = stored
                    else:
                        handles[g].write(stored.tobytes())
        finished = True
    finally:
        for fh in handles:
            fh.close()
        if not finished and disk_dir is not None:
            shutil.rmtree(disk_dir, ignore_errors=True)

    if mode == "ram":
        segments = (_ArraySegment(arrays[0]), _ArraySegment(arrays[1]))
    elif mode == "disk":
        segments = tuple(_NpyFileSegment(os.path.join(disk_dir, f"{name}.npy"), n_keep,
                                         half_shape, buf_dtype, offsets[g])
                         for g, name in enumerate(("odd", "even")))
    else:
        segments = (_TiffSegment(odd_files[trim:], rot90_k, downsample),
                    _TiffSegment(even_files[trim:], rot90_k, downsample))
    has_window = n_in_window > 0
    return OddEvenBlock(
        folder=os.path.dirname(str(files[0])),
        odd=segments[0], even=segments[1], n_frames=n_keep, mode=mode, stats=stats,
        sum_odd=sums[0] if has_window else None, sum_even=sums[1] if has_window else None,
        n_in_window=n_in_window, raw_dtype=raw_dtype.name,
        buffer_dtype=None if mode == "none" else np.dtype(buf_dtype).name,
        ram_bytes=estimate if mode == "ram" else 0, disk_dir=disk_dir,
    )
