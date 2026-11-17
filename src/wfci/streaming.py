"""Constant-memory (streaming) variant of steps 1 + 3.

The in-memory path (:func:`wfci.pipeline.run_pipeline`) loads every trial's full
``[y, x, time]`` stack into RAM, exactly like the MATLAB script that reads the
whole multi-page TIFF into one ``double`` array before doing anything. At
512x512 x float64 = 2 MB/frame that is 10+ GB for a multi-GB acquisition, per
channel -- which is precisely the case the MATLAB comment
``ricorda che se pesa piu di 4gb non lo legge`` was warning about.

Nothing in the math actually needs the whole stack resident:

  * the baseline image is a *temporal mean*  -> a running sum is enough;
  * the DeltaF/F correction is *per-frame* once the two mean images are known;
  * both 0.5x box downsamples are *per-frame spatial* operations;
  * step 3 reduces each frame to four ROI ``nanmean`` values -> a tiny trace.

So we stream the file in TWO passes and keep only a couple of small images
resident at any time:

  pass 1: accumulate the baseline-window sum of the (half-resolution) frames
          -> mean images ``MIf`` / ``MIr``;
  pass 2: re-read frame by frame, correct -> DeltaF/F, downsample again, and
          reduce to the four ROI means -> one ``[time, 4]`` trace per trial.

The per-frame arithmetic is byte-for-byte the same operations as the in-memory
path (``imresize_box``, the correction formula, the ROI ``nanmean``), so results
match the MATLAB-validated pipeline up to floating-point summation order in the
baseline mean (~1e-13, negligible for DeltaF/F percentages). The trade-off is
two passes over the file instead of one, and no ``dff_stack`` is retained (it is
streamed away), so step-2 visualization needs the in-memory path instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ROIConfig
from .gsr import GSRConfig, spatial_mean
from .io import (
    ChannelDecisionConfig,
    FrameSource,
    HalfResSource,
    OddEvenBlock,
    decide_channel_order,
    read_interleaved_block,
    split_odd_even,
    tiff_frame_source,
)
from .mask import valid_from_mask
from .profiles import Profile
from .resize import imresize_box
from .roi import box_slices_for, functional_connectivity


@dataclass
class StreamingResult:
    """Outputs of the streaming pipeline.

    Mirrors :class:`wfci.pipeline.PipelineResult` but without ``dff_stack``: the
    corrected stack is never materialised (that is the whole point of streaming).
    """

    temp_roi: np.ndarray         # TEMP_ROI        [time, n_rois, trial]
    R: np.ndarray                # per-trial       [n_rois, n_rois, trial]
    R_mean: np.ndarray           # trial mean      [n_rois, n_rois]
    averaged_traces: np.ndarray  # trial mean      [time, n_rois]


def _as_frame_source(channel: FrameSource | HalfResSource | str | Path):
    """Coerce a channel argument to a :class:`FrameSource`.

    A :class:`FrameSource` (or an already-downsampled :class:`HalfResSource`)
    passes through unchanged; a path (``str``/``Path``) is treated as a single
    multi-page TIFF -- the original streaming input -- so the older
    ``stream_trial_roi(gcamp_path, emo_path, ...)`` call style keeps working.
    """
    if isinstance(channel, (FrameSource, HalfResSource)):
        return channel
    return tiff_frame_source(channel)


def _half_res_frames(source, trim: int, downsample: float):
    """Yield trimmed, once-downsampled ``[y, x]`` frames from a frame source.

    Reproduces the first two MATLAB steps per frame: drop the first ``trim``
    frames (``out(:,:,21:end)``), then ``imresize(...,0.5,'box')``. Resizing a
    single 2-D frame is identical to resizing the whole stack slice-by-slice,
    because ``imresize_box`` only touches the spatial axes. ``source.open()``
    gives a fresh iterator, so each call is one independent pass over the data.

    A :class:`~wfci.io.HalfResSource` already holds trimmed, downsampled frames
    (the single-read channel-selection buffer), so they are yielded as they are --
    after checking it was built for this very ``trim`` and ``downsample``.
    """
    if isinstance(source, HalfResSource):
        if source.trim != trim or source.downsample != downsample:
            raise ValueError(
                f"HalfResSource was built for trim={source.trim}, downsample="
                f"{source.downsample}, but is streamed with trim={trim}, "
                f"downsample={downsample}."
            )
        yield from source.frames()
        return
    gen = source.open()
    for _ in range(trim):
        next(gen, None)  # skip the trimmed lead-in frames
    for frame in gen:
        yield imresize_box(frame, downsample)


def stream_trial_roi(
    gcamp: FrameSource | str | Path,
    emo: FrameSource | str | Path,
    cfg: ROIConfig,
    baseline_slice: slice = slice(None),
    trim: int = 20,
    downsample: float = 0.5,
    mask: np.ndarray | None = None,
    gsr: GSRConfig | None = None,
    mask_threshold: float = 0.0,
    expected_grid: tuple[int, int] | None = None,
    pixel_dump=None,
    baselines: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Stream one ``(gcamp, emo)`` trial from disk to its ``[time, n_rois]`` trace.

    Constant memory: only the two running-sum images (pass 1) and the current
    frame plus the two mean images (pass 2) are ever resident.

    Parameters
    ----------
    gcamp, emo:
        The two channels as :class:`~wfci.io.FrameSource` objects, i.e. any
        re-iterable frame sequence (single multi-page TIFF, folder of single-page
        TIFFs, or one channel of an interleaved folder). A bare path is also
        accepted and treated as a multi-page TIFF, for the original call style.
        The storage format is thus decoupled from the streaming math here.
    cfg:
        ROI geometry (Bregma reference + the boxes), as for the in-memory path.
        Box order defines the column order of the returned trace.
    baseline_slice:
        Temporal window (over the trimmed frames) for the mean baseline image.
        ``slice(None)`` for resting-state, ``slice(0, 278)`` for stimulated.
    trim, downsample:
        Same meaning as :func:`wfci.correction.build_dff_stack`.
    mask:
        ``[y, x]`` brain mask at the final (twice-downsampled) resolution. Applied
        per frame, which is identical to masking the whole stack because the
        decision is per pixel and constant in time.
    gsr:
        Global signal regression config, or None.
    mask_threshold:
        See :func:`wfci.mask.valid_from_mask`.
    expected_grid:
        The final ``(rows, cols)`` the ROI atlas was drawn for; see
        :func:`wfci.roi.box_slices_for`. None skips the check.
    pixel_dump:
        Optional :class:`wfci.dump.PixelDump` (or anything with the same
        ``baselines``/``frame``/``close`` methods). When given, every frame's raw
        and corrected pixels are written to disk from *inside* pass 2 -- no extra
        read, no extra resident array, so the two-pass and one-frame invariants
        (two passes, one frame in memory) are unaffected. None (the default) is the pixel-free path.
    baselines:
        ``(mean_f, mean_r)`` already computed -- pass 1 done elsewhere. The
        single-read channel selection (:func:`prepare_interleaved_trial`) sums the
        baseline window while it reads the block, so only pass 2 runs here. None
        (the default) runs pass 1 as always.
    """
    gcamp_src = _as_frame_source(gcamp)
    emo_src = _as_frame_source(emo)
    if pixel_dump is not None and gsr is not None:
        # A post-GSR per-pixel stack does not exist in this path by construction:
        # _stream_pass2_gsr accumulates only the sufficient statistics (a, b,
        # g_vec) precisely so it never has to hold one, and materialising it would
        # need a third pass over the file -- breaking the two-pass invariant. Dumping the
        # PRE-GSR pixels under a name that implies otherwise would be worse than
        # refusing, so refuse.
        raise ValueError(
            "pixel_dump is not supported together with GSR: the streaming GSR "
            "path never materialises a post-GSR per-pixel frame (it accumulates "
            "only the OLS statistics), and producing one would require a third "
            "pass over the data. Dump without GSR, or use the in-memory path."
        )
    invalid = None if mask is None else ~valid_from_mask(mask, mask_threshold)
    # Frame count after trimming; both channels are assumed equal length, so we
    # take the shorter one to stay in lockstep.
    n_total = min(gcamp_src.count, emo_src.count)
    n_time = n_total - trim
    # Which trimmed-frame indices fall inside the baseline window.
    base_lo, base_hi, _ = baseline_slice.indices(n_time)
    base_start, base_stop = base_lo, base_hi

    if baselines is not None:
        mean_f, mean_r = baselines
    else:
        # --- pass 1: baseline mean over half-resolution frames ----------------
        sum_f = sum_r = None
        count = 0
        g_stream = _half_res_frames(gcamp_src, trim, downsample)
        e_stream = _half_res_frames(emo_src, trim, downsample)
        for idx, (g_half, e_half) in enumerate(zip(g_stream, e_stream)):
            if idx >= n_time:
                break
            if base_start <= idx < base_stop:
                if sum_f is None:
                    sum_f = np.zeros_like(g_half)
                    sum_r = np.zeros_like(e_half)
                sum_f += g_half  # running temporal sum, not the whole stack
                sum_r += e_half
                count += 1
        mean_f = sum_f / count  # MIf
        mean_r = sum_r / count  # MIr

    # Precompute + validate the ROI box slices once, against the final
    # quarter-res grid they will be applied to. The frame shape is known now
    # (mean_f was built at that resolution) rather than only inside the loop, so a
    # geometry error surfaces before pass 2 reads the file a second time.
    final_shape = imresize_box(mean_f, downsample).shape
    resolved = box_slices_for(cfg, final_shape, expected_grid)
    names = [name for name, _, _ in resolved]
    box_slices = [(rs, cs) for _, rs, cs in resolved]

    if gsr is not None:
        return _stream_pass2_gsr(
            gcamp_src, emo_src, mean_f, mean_r, n_time, trim, downsample,
            box_slices, invalid, gsr,
        )

    # --- pass 2: per-frame correction -> ROI means ---------------------------
    temp_roi = np.empty((n_time, len(names)), dtype=np.float64)
    g_stream = _half_res_frames(gcamp_src, trim, downsample)
    e_stream = _half_res_frames(emo_src, trim, downsample)
    if pixel_dump is not None:
        pixel_dump.baselines(mean_f, mean_r)
    try:
        for idx, (g_half, e_half) in enumerate(zip(g_stream, e_stream)):
            if idx >= n_time:
                break
            # DeltaF/F (%) for this frame, exactly as hemodynamic_correction does it.
            dff_half = ((g_half / mean_f) / (e_half / mean_r) - 1.0) * 100.0
            # Second 0.5x box downsample (per-frame == whole-stack slice).
            dff_q = imresize_box(dff_half, downsample)
            if invalid is not None:
                _check_mask_shape(invalid, dff_q)
                dff_q = np.where(invalid, np.nan, dff_q)
            if pixel_dump is not None:
                # Written from the loop's own locals, after masking, so the dump
                # holds exactly the pixels the ROI means below are taken from.
                pixel_dump.frame(idx, g_half, e_half, dff_q)
            for j, (rs, cs) in enumerate(box_slices):
                temp_roi[idx, j] = np.nanmean(dff_q[rs, cs])
    finally:
        if pixel_dump is not None:
            pixel_dump.close()
    return temp_roi


def _check_mask_shape(invalid: np.ndarray, frame: np.ndarray) -> None:
    if invalid.shape != frame.shape:
        raise ValueError(
            f"Mask shape {invalid.shape} does not match the corrected frame's "
            f"{frame.shape}. The mask must be at the FINAL (twice-downsampled) "
            f"resolution -- see wfci.pipeline.prepare_mask."
        )


def _stream_pass2_gsr(
    gcamp_src: FrameSource,
    emo_src: FrameSource,
    mean_f: np.ndarray,
    mean_r: np.ndarray,
    n_time: int,
    trim: int,
    downsample: float,
    box_slices: list[tuple[slice, slice]],
    invalid: np.ndarray | None,
    gsr: GSRConfig,
) -> np.ndarray:
    """Pass 2 with global signal regression, in constant memory.

    GSR looks fundamentally anti-streaming: it regresses **each pixel's whole
    time-series** against the global signal, which naively means holding
    ``[y, x, time]`` resident -- destroying the property that makes this module
    worth having. It doesn't, for two reasons.

    **1. Per-pixel OLS needs only sufficient statistics**, all accumulable one
    frame at a time: ``N``, ``sum(g)``, ``sum(g^2)`` (scalars) and ``sum(p)``,
    ``sum(g*p)`` (two ``[y, x]`` images). The global signal ``g(t)`` is a
    *spatial* mean of frame ``t``, so it is known at frame ``t`` -- no lookahead.

    **2. The ROI mean is linear and ``g(t)`` is one scalar per frame**, so the
    regressed ROI trace never needs the regressed *stack*::

        T_B(t) = mean_B( p(t) - a*g(t) - b )
               = mean_B(p(t)) - g(t)*mean_B(a) - mean_B(b)

    Every term is either accumulated per frame (``mean_B(p(t))``, ``g(t)``) or
    computed once at the end from the two stat images (``mean_B(a)``,
    ``mean_B(b)``).

    So GSR costs two extra ``[y, x]`` images and two ``[time]`` vectors, and the
    pass count stays at TWO.

    The precondition is that the invalid-pixel set is **static over time**; it is
    checked here rather than assumed. See :func:`_check_static_nans`.
    """
    n_rois = len(box_slices)
    sum_p = sum_gp = nan_count = None
    g_vec = np.empty(n_time, dtype=np.float64)
    raw_roi = np.empty((n_time, n_rois), dtype=np.float64)

    g_stream = _half_res_frames(gcamp_src, trim, downsample)
    e_stream = _half_res_frames(emo_src, trim, downsample)
    for idx, (g_half, e_half) in enumerate(zip(g_stream, e_stream)):
        if idx >= n_time:
            break
        dff_half = ((g_half / mean_f) / (e_half / mean_r) - 1.0) * 100.0
        dff_q = imresize_box(dff_half, downsample)
        if invalid is not None:
            _check_mask_shape(invalid, dff_q)
            dff_q = np.where(invalid, np.nan, dff_q)

        if sum_p is None:
            sum_p = np.zeros_like(dff_q)
            sum_gp = np.zeros_like(dff_q)
            nan_count = np.zeros(dff_q.shape, dtype=np.int64)

        bad = ~np.isfinite(dff_q)
        nan_count += bad

        # Two views of the same frame, so every reduction below excludes exactly
        # the non-finite pixels -- NaN *and* inf -- matching what the in-memory
        # path does (wfci.gsr drops any pixel that is not finite, and its global
        # signal averages only finite pixels). Plain np.nanmean would not: it
        # ignores NaN but propagates inf.
        clean = np.where(bad, 0.0, dff_q)      # for the running sums: finite
        masked = np.where(bad, np.nan, dff_q)  # for the nanmeans: inf -> NaN

        n_finite = dff_q.size - int(bad.sum())
        if n_finite == 0:
            raise ValueError(
                f"The global signal is undefined at frame {idx}: it has no valid "
                f"pixels at all. Check the brain mask and the input data."
            )
        # The global signal of frame t, averaged as gsr.global_mean says
        # ("column" = MATLAB's nested nanmean), exactly as wfci.gsr.global_signal.
        g_t = float(spatial_mean(dff_q, ~bad, gsr.global_mean))
        g_vec[idx] = g_t

        with warnings.catch_warnings():
            # A fully-masked ROI box is all-NaN -- a legitimate NaN column, just as
            # in the in-memory path.
            warnings.simplefilter("ignore", RuntimeWarning)
            for j, (rs, cs) in enumerate(box_slices):
                raw_roi[idx, j] = np.nanmean(masked[rs, cs])

        # Zero-filled at the invalid pixels so the sums stay finite; their a/b are
        # overwritten with NaN below, so the zeros never reach a result.
        sum_p += clean
        sum_gp += g_t * clean

    _check_static_nans(nan_count, n_time)
    valid = nan_count == 0

    # The OLS fit, from the accumulated statistics. g(t) was kept in full (it is
    # only [time] floats), so the mean and variance of g are computed from it
    # directly rather than from sum(g)/sum(g^2): the centred form is what the
    # in-memory path uses, and reproducing it keeps the two paths agreeing to
    # roundoff instead of to catastrophic-cancellation error.
    g_mean = g_vec.mean()
    g_centred = g_vec - g_mean
    g_var = float(g_centred @ g_centred)
    if g_var <= gsr.min_variance:
        raise ValueError(
            f"The global signal has no variance (sum of squares {g_var:.3g}), so "
            f"no slope is identifiable. This usually means a constant or "
            f"single-frame recording."
        )

    # sum_t (g(t) - gbar) * p(t) == sum(g*p) - gbar*sum(p), i.e. N*cov(g, p).
    cov = sum_gp - g_mean * sum_p
    a = cov / g_var
    b = sum_p / n_time - a * g_mean
    a[~valid] = np.nan
    b[~valid] = np.nan

    # T_B(t) = raw_B(t) - g(t)*mean_B(a) - mean_B(b)
    temp_roi = np.empty((n_time, n_rois), dtype=np.float64)
    with warnings.catch_warnings():
        # A fully-masked box is all-NaN; that is a legitimate NaN column, exactly
        # as the in-memory path produces.
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, (rs, cs) in enumerate(box_slices):
            mean_a = np.nanmean(a[rs, cs])
            mean_b = np.nanmean(b[rs, cs])
            temp_roi[:, j] = raw_roi[:, j] - g_vec * mean_a - mean_b
    return temp_roi


def _check_static_nans(nan_count: np.ndarray, n_time: int) -> None:
    """Refuse to stream GSR when a pixel is NaN for only SOME frames.

    The two-pass identity computes each frame's ROI mean *during* pass 2, but
    ``nan_policy="drop_pixel"`` decides validity from the pixel's WHOLE
    time-series. If a pixel is valid in some frames and NaN in others, streaming
    would include it wherever it happens to be valid, while the in-memory path
    excludes it from every frame. The two would then disagree -- subtly, and only
    on real data.

    Under mask-only NaNs the invalid set is static and the identity is exact. A
    partially-NaN pixel means something else produced NaNs mid-recording (e.g. a
    zero in ``emo(t)`` making the ratio non-finite).

    The in-memory result is the reference, so this
    raises rather than silently returning different numbers.
    """
    partial = (nan_count > 0) & (nan_count < n_time)
    if not partial.any():
        return
    n_partial = int(partial.sum())
    ys, xs = np.nonzero(partial)
    example = f"(y={ys[0]}, x={xs[0]}) is NaN in {int(nan_count[ys[0], xs[0]])}/{n_time} frames"
    raise ValueError(
        f"Cannot stream GSR: {n_partial} pixel(s) are NaN in some frames but not "
        f"all -- e.g. {example}. Streaming GSR requires the invalid-pixel set to "
        f"be static over time (it holds for mask-only NaNs); a partially-NaN pixel "
        f"would be treated differently here than by the in-memory path, which "
        f"drops such pixels entirely. Re-run with streaming disabled "
        f"(--no-streaming) to get the correct, in-memory result."
    )


def run_streaming(
    trial_sources: list[tuple[FrameSource | str | Path, FrameSource | str | Path]],
    cfg: ROIConfig,
    baseline_slice: slice = slice(None),
    corr_window: slice = slice(None),
    trim: int = 20,
    downsample: float = 0.5,
    mask: np.ndarray | None = None,
    gsr: GSRConfig | None = None,
    mask_threshold: float = 0.0,
    expected_grid: tuple[int, int] | None = None,
    pixel_dump_factory=None,
    trial_baselines: list | None = None,
) -> StreamingResult:
    """Streaming pipeline over a list of ``(gcamp, emo)`` trials.

    Each trial is a pair of :class:`~wfci.io.FrameSource` objects (or bare TIFF
    paths, treated as multi-page files). It is streamed to its ``[time, n_rois]``
    ROI trace (constant memory); only the tiny per-trial traces are stacked, then
    the usual per-trial correlation and trial averaging run on them.

    ``mask`` / ``gsr`` mirror :func:`wfci.pipeline.run_pipeline`'s optional stages;
    the mask must already be at the final resolution (see
    :func:`wfci.pipeline.prepare_mask`).

    ``pixel_dump_factory`` is an optional ``(trial_index) -> PixelDump | None``.
    It is a factory rather than a single sink because each trial needs its own
    files; returning None for a trial dumps nothing for it. See
    :mod:`wfci.dump`.

    ``trial_baselines`` is an optional list aligned with ``trial_sources``: an
    entry ``(mean_f, mean_r)`` skips that trial's pass 1 (see
    :func:`prepare_interleaved_trial`), None runs it.
    """
    traces = [
        stream_trial_roi(
            g, e, cfg, baseline_slice, trim, downsample,
            mask=mask, gsr=gsr, mask_threshold=mask_threshold,
            expected_grid=expected_grid,
            pixel_dump=None if pixel_dump_factory is None else pixel_dump_factory(i),
            baselines=None if trial_baselines is None else trial_baselines[i],
        )
        for i, (g, e) in enumerate(trial_sources)
    ]
    # [time, n_rois, trial]
    temp_roi = np.stack(traces, axis=-1)
    R, R_mean, averaged_traces = functional_connectivity(temp_roi, window=corr_window)
    return StreamingResult(temp_roi, R, R_mean, averaged_traces)


def run_streaming_profile(
    trial_sources,
    cfg: ROIConfig,
    profile: Profile,
    mask: np.ndarray | None = None,
    pixel_dump_factory=None,
    trial_baselines: list | None = None,
) -> StreamingResult:
    """Streaming counterpart of :func:`wfci.pipeline.run_profile`.

    Same profile, same numbers, constant memory -- the storage x memory x profile
    axes stay independent, so any profile runs either way. ``pixel_dump_factory``
    and ``trial_baselines`` are passed straight through to :func:`run_streaming`.
    """
    from .pipeline import prepare_mask

    cfg = ROIConfig(y_1=cfg.y_1, x_2=cfg.x_2, boxes=dict(profile.atlas))
    return run_streaming(
        trial_sources,
        cfg,
        baseline_slice=profile.baseline,
        corr_window=profile.corr_window,
        trim=profile.trim,
        downsample=profile.downsample,
        mask=prepare_mask(mask, profile),
        gsr=profile.gsr,
        expected_grid=profile.atlas.grid,
        pixel_dump_factory=pixel_dump_factory,
        trial_baselines=trial_baselines,
    )


def run_streaming_resting_state(trial_sources, cfg, **kwargs) -> StreamingResult:
    """Resting-state streaming: full-recording baseline and correlation."""
    return run_streaming(
        trial_sources, cfg, baseline_slice=slice(None), corr_window=slice(None), **kwargs
    )


def run_streaming_stimulated(
    trial_sources,
    cfg,
    baseline_slice: slice = slice(0, 278),
    corr_window: slice = slice(279, 300),
    **kwargs,
) -> StreamingResult:
    """Stimulated streaming: pre-stimulus baseline (MATLAB 1:278) and
    stimulus-window correlation (MATLAB 280:300)."""
    return run_streaming(
        trial_sources, cfg, baseline_slice=baseline_slice, corr_window=corr_window,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Single-read channel selection
# ---------------------------------------------------------------------------

# channel_order recorded when a not_interleaved block is stored by position only.
UNASSIGNED_ORDER = "odd_even_unassigned"


@dataclass
class BlockChannelChoice:
    """What was decided for one block, and from what."""

    order: str                # applied: "gcamp_first" | "emo_first" | UNASSIGNED_ORDER
    configured: str           # "auto" | "gcamp_first" | "emo_first"
    verdict: str              # decide_channel_order's verdict, or "explicit"
    details: dict             # level_odd, level_even, alternation, separation, n_frames
    buffer_mode: str          # "ram" | "disk" | "none"


@dataclass
class PreparedTrial:
    """One trial read once and labelled: ready for pass 2.

    ``gcamp`` / ``emo`` are :class:`~wfci.io.HalfResSource` objects over every
    block, ``baselines`` the ``(mean_f, mean_r)`` pass 1 produced during the read.
    ``skipped`` is True when an ``auto`` block was ``not_interleaved``: then no
    sources are built and the trial must not be analysed. ``unassigned`` is True
    (with ``skipped`` also True) when ``keep_not_interleaved`` kept that block's
    buffer instead: read ``blocks[0].odd`` / ``.even``. Call :meth:`release`
    when done (also after an error) to free RAM buffers and delete disk buffers.
    """

    blocks: list[OddEvenBlock]
    choices: list[BlockChannelChoice]
    gcamp: HalfResSource | None
    emo: HalfResSource | None
    baselines: tuple[np.ndarray, np.ndarray] | None
    skipped: bool
    unassigned: bool = False

    @property
    def n_time(self) -> int:
        """Frames per channel pass 2 will produce."""
        return sum(block.n_frames for block in self.blocks)

    def meta(self) -> dict:
        """Provenance fields for the .npz / pixel-dump metadata.

        One value per block; ``;``-joined when there are several (merged runs),
        matching the existing ``channel_order`` field.
        """
        return channel_choice_meta(self.choices)

    def keep_buffers(self, paths: list[tuple[str, str]]) -> list[bool]:
        """Save each block's unlabelled odd/even volumes to ``paths[i]``."""
        return [block.keep(odd, even)
                for block, (odd, even) in zip(self.blocks, paths, strict=True)]

    def release(self) -> None:
        for block in self.blocks:
            block.release()


def channel_choice_meta(choices: list[BlockChannelChoice]) -> dict:
    """The channel-choice metadata fields for a list of block choices (see PreparedTrial.meta)."""
    def joined(values, fmt=str):
        values = list(values)
        return values[0] if len(values) == 1 else ";".join(fmt(v) for v in values)

    def num(v):
        return f"{v:.6g}"

    return dict(
        channel_order=";".join(c.order for c in choices),
        channel_order_configured=";".join(c.configured for c in choices),
        channel_verdict=";".join(c.verdict for c in choices),
        channel_level_odd=joined((c.details["level_odd"] for c in choices), num),
        channel_level_even=joined((c.details["level_even"] for c in choices), num),
        channel_alternation=joined((c.details["alternation"] for c in choices), num),
        channel_separation=joined((c.details["separation"] for c in choices), num),
        channel_decision_n_frames=joined(c.details["n_frames"] for c in choices),
        channel_buffer_mode=";".join(c.buffer_mode for c in choices),
        # Which per-frame level the order/separation were decided on (one per run).
        channel_level_statistic=choices[0].details.get("level_statistic", ""),
        # Which pixels those levels were taken over: "frame" or "roi_union".
        channel_level_region=choices[0].details.get("level_region", ""),
    )


def describe_channel_choice(name: str, choice: BlockChannelChoice) -> str:
    """One log line summarising a block's channel decision."""
    d = choice.details
    return (f"{name}: {choice.order} ({choice.verdict}; configured {choice.configured}) "
            f"levels odd {d['level_odd']:.1f} / even {d['level_even']:.1f}, "
            f"alternation {d['alternation']:+.3f}, separation {d['separation']:.1f}, "
            f"{d['n_frames']} frames ({d.get('level_region', 'frame')}), "
            f"buffer {choice.buffer_mode}")


def roi_union_level_mask(cfg: ROIConfig, expected_grid: tuple[int, int] | None = None,
                         downsample: float = 0.5):
    """``half_shape -> bool mask``: the ROI-box union lifted to the half-res frames.

    The boxes live on the FINAL grid (the half-res frame box-downsampled once more,
    as pass 2 does), so they are resolved there with :func:`box_slices_for` -- which
    raises on any box off the grid -- and each final pixel is expanded back to the
    block of half-res pixels it averages. That block correspondence is exact only
    for an integer reduction (0.5 on even dimensions); anything else raises.
    For :class:`~wfci.io.ChannelDecisionConfig` ``level_region = "roi_union"``.
    """
    def level_mask_for(half_shape: tuple[int, int]) -> np.ndarray:
        final_shape = imresize_box(np.zeros(half_shape), downsample).shape
        factors = {half_shape[0] / final_shape[0], half_shape[1] / final_shape[1]}
        if len(factors) != 1 or not float(next(iter(factors))).is_integer():
            raise ValueError(f"Cannot lift the ROI union from the final grid {final_shape} "
                             f"to half-res frames {half_shape}: not an integer reduction.")
        k = int(next(iter(factors)))
        union = np.zeros(final_shape, dtype=bool)
        for _, rows, cols in box_slices_for(cfg, final_shape, expected_grid):
            union[rows, cols] = True
        # Each final pixel = mean of a k x k half-res block: repeat along both axes.
        return np.repeat(np.repeat(union, k, axis=0), k, axis=1)

    return level_mask_for


def choose_block_channels(stats, configured: str, buffer_mode: str, name: str,
                          decision: ChannelDecisionConfig = ChannelDecisionConfig(),
                          decide=decide_channel_order, log=print,
                          keep_not_interleaved: bool = False) -> BlockChannelChoice:
    """Decide (``auto``) or accept (explicit) one block's order, and log it.

    Shared by the single-read streaming path and the in-memory debug paths, so
    both apply the same rule and print the same warnings. With
    ``keep_not_interleaved`` a ``not_interleaved`` block gets the order
    :data:`UNASSIGNED_ORDER` rather than the meaningless brightness guess.
    """
    order, verdict, details = decide(stats, decision)
    if configured != "auto":
        order, verdict = configured, "explicit"
    if verdict == "not_interleaved" and keep_not_interleaved:
        order = UNASSIGNED_ORDER
    choice = BlockChannelChoice(order, configured, verdict, details, buffer_mode)
    log("Channel: " + describe_channel_choice(name, choice))
    if verdict == "uncertain":
        log(f"WARNING: {name}: the two channels are too close to tell apart "
            f"(separation {details['separation']:.1f} < {decision.separation_min}); "
            f"proceeding with the dimmer group as GCaMP.")
    if verdict == "not_interleaved":
        action = ("keeping the odd and even frames unlabelled" if keep_not_interleaved
                  else "no channel assignment is correct, so it is skipped")
        log(f"{'UNASSIGNED' if keep_not_interleaved else 'SKIP   '}: {name}: brightness does "
            f"not alternate frame to frame (alternation {details['alternation']:+.3f} > "
            f"{decision.alternation_max}); {action}.")
    return choice


def prepare_interleaved_trial(
    block_files: list[list[Path]],
    channel_orders: list[str],
    trim: int,
    pipeline_trim: int,
    downsample: float = 0.5,
    baseline_slice: slice = slice(None),
    rot90_k: int = 0,
    channel_buffer: str = "auto",
    buffer_dir: str | Path | None = None,
    decision: ChannelDecisionConfig = ChannelDecisionConfig(),
    decide=decide_channel_order,
    block_names: list[str] | None = None,
    log=print,
    level_mask_for=None,
    keep_not_interleaved: bool = False,
) -> PreparedTrial:
    """Read every block of one trial once, decide each block's channels, label them.

    ``block_files`` holds each block's images in acquisition order (one block =
    one ``t#`` folder; several for a merged run). Each block is read once by
    :func:`~wfci.io.read_interleaved_block` (``trim`` onset frames dropped per
    channel), its buffer mode chosen as it is read against the RAM already used
    by earlier blocks. Then, per block, ``channel_orders[i]``:

    * ``"auto"`` -> ``decide(stats, decision)`` picks the order from every frame.
      ``uncertain`` proceeds with a warning; ``not_interleaved`` stops reading and
      returns a ``skipped`` trial (a merged unit is skipped whole). With
      ``keep_not_interleaved`` a SINGLE-block trial instead returns ``unassigned``
      with its buffer intact, for the caller to store by position; a merged unit
      is still skipped, because one concatenated odd/even pair cannot mix blocks
      whose groups mean different things;
    * ``"gcamp_first"`` / ``"emo_first"`` -> used as given (verdict ``"explicit"``;
      the statistics are still recorded).

    ``decision.level_region = "roi_union"`` needs ``level_mask_for`` (see
    :func:`roi_union_level_mask`); the statistics are then taken over those pixels.

    The labelled blocks are concatenated in order into one ``(gcamp, emo)`` pair
    of :class:`~wfci.io.HalfResSource` built for ``pipeline_trim`` -- the ``trim``
    the streaming pipeline is then called with (``trim`` for one block, 0 for a
    merged run whose blocks are already trimmed). ``baseline_slice`` is resolved
    over the concatenated length, as :func:`stream_trial_roi` would.

    **Exactness.** For one block the baseline sums are accumulated in exactly the
    order pass 1 uses, so the result is bit-identical. For several blocks the
    per-block sums are added afterwards; that is still bit-identical for integer
    raw data of <=16 bits (every partial sum is an exact multiple of 0.25 far
    below 2**51), and equal to roundoff otherwise.
    """
    if len(block_files) != len(channel_orders):
        raise ValueError(f"{len(block_files)} blocks but {len(channel_orders)} channel orders.")
    names = block_names or [str(i) for i in range(len(block_files))]
    if (decision.level_region == "roi_union") != (level_mask_for is not None):
        raise ValueError(f"level_region={decision.level_region!r} but level_mask_for is "
                         f"{'missing' if level_mask_for is None else 'given'}; pass it exactly "
                         f"when level_region is 'roi_union'.")

    # Total post-trim length, from the file lists alone, so the baseline window
    # can be placed before any frame is decoded.
    lengths = [len(split_odd_even(files)[0]) - trim for files in block_files]
    total = sum(lengths)
    base_lo, base_hi, _ = baseline_slice.indices(max(total, 0))

    blocks: list[OddEvenBlock] = []
    choices: list[BlockChannelChoice] = []
    finished = False
    try:
        offset = 0
        for files, configured, name, length in zip(block_files, channel_orders, names, lengths):
            window = (base_lo - offset, base_hi - offset)
            block = read_interleaved_block(
                files, trim, downsample, rot90_k, window,
                channel_buffer=channel_buffer, buffer_dir=buffer_dir,
                buffered_bytes=sum(b.ram_bytes for b in blocks),
                top_percent=decision.top_percent,
                level_mask_for=level_mask_for,
            )
            blocks.append(block)
            offset += length
            keep_block = keep_not_interleaved and len(block_files) == 1
            choice = choose_block_channels(block.stats, configured, block.mode, name,
                                           decision, decide, log, keep_block)
            choices.append(choice)
            if choice.verdict == "not_interleaved" and keep_block:
                # The buffer stays alive: the caller reads it, then calls release().
                finished = True
                return PreparedTrial(blocks, choices, None, None, None,
                                     skipped=True, unassigned=True)
            if choice.verdict == "not_interleaved":
                # A not_interleaved block skips the whole trial (merged unit too):
                # stop reading, free what was buffered.
                for b in blocks:
                    b.release()
                finished = True
                return PreparedTrial(blocks, choices, None, None, None, skipped=True)

        gcamp_segments, emo_segments = [], []
        sum_f = sum_r = None
        count = 0
        for block, choice in zip(blocks, choices):
            g_seg, e_seg, g_sum, e_sum = block.labelled(choice.order)
            gcamp_segments.append(g_seg)
            emo_segments.append(e_seg)
            if g_sum is None:
                continue
            if sum_f is None:
                # One block: these ARE pass 1's running sums, no further addition.
                sum_f, sum_r = g_sum, e_sum
            else:
                sum_f = sum_f + g_sum
                sum_r = sum_r + e_sum
            count += block.n_in_window
        if count == 0:
            raise ValueError(f"The baseline window {baseline_slice} holds no frame of "
                             f"the {total} post-trim frames.")
        baselines = (sum_f / count, sum_r / count)  # MIf, MIr
        gcamp = HalfResSource(gcamp_segments, pipeline_trim, downsample)
        emo = HalfResSource(emo_segments, pipeline_trim, downsample)
        finished = True
        return PreparedTrial(blocks, choices, gcamp, emo, baselines, skipped=False)
    finally:
        # Only on an exception: free RAM and delete disk buffers before it propagates.
        if not finished:
            for b in blocks:
                b.release()
