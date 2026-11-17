"""Single-read, whole-recording channel selection (channel_selection_spec.md).

Covers the spec's test list:

* exactness -- ``ram``, ``disk`` and ``none`` give ROI traces, correlation matrices
  and pixel dumps bit-identical to today's two-read path run with the same explicit
  order, for single and merged runs, with and without GSR;
* decision -- well separated / close / non-alternating recordings give ``certain`` /
  ``uncertain`` / ``not_interleaved`` and the right order for both parities;
* buffer selection -- with ``psutil`` mocked, ``auto`` picks ram, then disk, then
  none, and a merged run switches mode per block once the budget runs out;
* cleanup -- disk buffers go after success and after an exception, and are kept
  with ``keep_channel_buffer``;
* dtype guard -- non-integer or >16-bit raw frames are buffered in float64.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import wfci.io as wio  # noqa: E402
from wfci import PixelDump, ROIConfig, get_profile, imresize_box, run_streaming  # noqa: E402
from wfci.config import Box  # noqa: E402
from wfci.gsr import GSRConfig  # noqa: E402
from wfci.io import (  # noqa: E402
    ChannelDecisionConfig,
    ChannelStats,
    buffer_dtype_for,
    choose_buffer_mode,
    decide_channel_order,
    folder_frame_source,
    interleaved_channel_files,
    read_interleaved_block,
    sorted_frame_files,
)
from wfci.streaming import prepare_interleaved_trial  # noqa: E402

import extract_cohort as batch  # noqa: E402

FRAME_SHAPE = (32, 32)          # raw -> 16x16 half-res -> 8x8 final grid
TRIM = 3
DS = 0.5
Y_1, X_2 = 2, 2
BOXES = {"a": Box(1, 2, 1, 2), "b": Box(3, 4, 3, 4), "c": Box(1, 3, 4, 5)}
GCAMP_LEVEL, EMO_LEVEL = 3000, 9000


def _leftovers(buffer_dir: Path) -> list[str]:
    """What is left in a buffer folder (nothing if it was never created)."""
    return os.listdir(buffer_dir) if buffer_dir.exists() else []


def _cfg() -> ROIConfig:
    return ROIConfig(y_1=Y_1, x_2=X_2, boxes=dict(BOXES))


def write_recording(folder: Path, n_pairs: int = 24, gcamp_odd: bool = True,
                    gcamp_level: float = GCAMP_LEVEL, emo_level: float = EMO_LEVEL,
                    jitter: float = 30.0, seed: int = 0, dtype=np.uint16,
                    shape=FRAME_SHAPE, interleaved: bool = True) -> Path:
    """An interleaved recording of uint16 TIFFs, zero-padded names, acquisition order.

    Each frame = channel level * a smooth spatial pattern + a shared slow signal
    + per-frame jitter + pixel noise, so ROI traces correlate and nothing is
    constant. ``interleaved=False`` gives every frame the same level (no
    alternation at all).
    """
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    pattern = 1.0 + 0.3 * np.sin(yy / 5.0) * np.cos(xx / 7.0)
    for i in range(2 * n_pairs):
        pair = i // 2
        odd_position = i % 2 == 0          # 1-based odd = files[0::2]
        is_gcamp = odd_position == gcamp_odd
        level = gcamp_level if (is_gcamp or not interleaved) else emo_level
        if not interleaved:
            level = gcamp_level
        signal = 1.0 + 0.02 * np.sin(pair / 3.0) * (1.0 if is_gcamp else 0.5)
        frame = (level * pattern * signal + rng.normal(0, jitter)
                 + rng.normal(0, 20.0, shape))
        tifffile.imwrite(folder / f"f_{i + 1:05d}.tif",
                         np.clip(frame, 1, None).astype(dtype))
    return folder


# ---------------------------------------------------------------------------
# today's two-read path, reconstructed from the library primitives
# ---------------------------------------------------------------------------
def old_single_sources(folder: Path, order: str):
    gcamp_files, emo_files = interleaved_channel_files(folder, channel_order=order)
    return folder_frame_source(gcamp_files), folder_frame_source(emo_files)


def old_merged_sources(folders: list[Path], orders: list[str]):
    """What extract_cohort's _build_concat_streaming_source did: split, trim, concat."""
    all_g, all_e = [], []
    for folder, order in zip(folders, orders):
        g, e = interleaved_channel_files(folder, channel_order=order)
        n = min(len(g), len(e))
        all_g += list(g[TRIM:n])
        all_e += list(e[TRIM:n])
    return folder_frame_source(all_g), folder_frame_source(all_e)


def _dump(out_dir: Path, n_time: int) -> PixelDump:
    return PixelDump(out_dir, suffix="full", n_time=n_time, downsample=DS,
                     dff_dtype=np.float64, f_dtype=np.float64)


def _assert_same_result(a, b):
    for name in ("temp_roi", "R", "R_mean", "averaged_traces"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def _assert_same_dump(dir_a: Path, dir_b: Path):
    for name in ("dff", "f_gcamp", "f_emo"):
        np.testing.assert_array_equal(np.load(dir_a / f"pixels_{name}_full.npy"),
                                      np.load(dir_b / f"pixels_{name}_full.npy"), err_msg=name)
    meta_a, meta_b = np.load(dir_a / "pixels_meta_full.npz"), np.load(dir_b / "pixels_meta_full.npz")
    np.testing.assert_array_equal(meta_a["mean_f"], meta_b["mean_f"])
    np.testing.assert_array_equal(meta_a["mean_r"], meta_b["mean_r"])


# ---------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["ram", "disk", "none"])
@pytest.mark.parametrize("gcamp_odd", [True, False])
@pytest.mark.parametrize("gsr", [False, True])
def test_single_recording_is_bit_identical_to_the_two_read_path(tmp_path, mode, gcamp_odd, gsr):
    folder = write_recording(tmp_path / "t1", gcamp_odd=gcamp_odd)
    order = "gcamp_first" if gcamp_odd else "emo_first"
    gsr_cfg = GSRConfig() if gsr else None

    old_dump = None if gsr else _dump(tmp_path / "old", 24 - TRIM)
    old = run_streaming([old_single_sources(folder, order)], _cfg(), trim=TRIM,
                        downsample=DS, gsr=gsr_cfg,
                        pixel_dump_factory=None if old_dump is None else (lambda i: old_dump))

    # Explicit order, and "auto" (well separated, so it must find the same order).
    for configured in (order, "auto"):
        prepared = prepare_interleaved_trial(
            [sorted_frame_files(folder)], [configured], trim=TRIM, pipeline_trim=TRIM,
            downsample=DS, channel_buffer=mode, buffer_dir=tmp_path / "buf", log=lambda m: None)
        assert prepared.choices[0].order == order
        assert prepared.choices[0].buffer_mode == mode
        new_dump = None if gsr else _dump(tmp_path / f"new_{configured}", prepared.n_time)
        new = run_streaming([(prepared.gcamp, prepared.emo)], _cfg(), trim=TRIM,
                            downsample=DS, gsr=gsr_cfg,
                            pixel_dump_factory=None if new_dump is None else (lambda i: new_dump),
                            trial_baselines=[prepared.baselines])
        prepared.release()
        _assert_same_result(old, new)
        if not gsr:
            _assert_same_dump(tmp_path / "old", tmp_path / f"new_{configured}")


@pytest.mark.parametrize("modes", [("ram", "ram"), ("disk", "disk"), ("none", "none"),
                                   ("ram", "none"), ("none", "disk")])
@pytest.mark.parametrize("gsr", [False, True])
def test_merged_run_is_bit_identical_to_the_two_read_path(tmp_path, monkeypatch, modes, gsr):
    # Two blocks on DIFFERENT parities: each block is decided on its own.
    folders = [write_recording(tmp_path / "t1", gcamp_odd=True, seed=1),
               write_recording(tmp_path / "t2", n_pairs=20, gcamp_odd=False, seed=2)]
    orders = ["gcamp_first", "emo_first"]
    gsr_cfg = GSRConfig() if gsr else None
    n_time = (24 - TRIM) + (20 - TRIM)

    old_dump = None if gsr else _dump(tmp_path / "old", n_time)
    old = run_streaming([old_merged_sources(folders, orders)], _cfg(), trim=0,
                        downsample=DS, gsr=gsr_cfg,
                        pixel_dump_factory=None if old_dump is None else (lambda i: old_dump))

    # Force a per-block mode sequence through the mode chooser.
    sequence = iter(modes)
    monkeypatch.setattr(wio, "choose_buffer_mode", lambda *a, **k: next(sequence))
    prepared = prepare_interleaved_trial(
        [sorted_frame_files(f) for f in folders], ["auto", "auto"], trim=TRIM,
        pipeline_trim=0, downsample=DS, buffer_dir=tmp_path / "buf", log=lambda m: None)
    assert [c.order for c in prepared.choices] == orders
    assert [c.buffer_mode for c in prepared.choices] == list(modes)
    new_dump = None if gsr else _dump(tmp_path / "new", prepared.n_time)
    new = run_streaming([(prepared.gcamp, prepared.emo)], _cfg(), trim=0, downsample=DS,
                        gsr=gsr_cfg,
                        pixel_dump_factory=None if new_dump is None else (lambda i: new_dump),
                        trial_baselines=[prepared.baselines])
    prepared.release()
    _assert_same_result(old, new)
    if not gsr:
        _assert_same_dump(tmp_path / "old", tmp_path / "new")


def test_stimulated_baseline_window_is_placed_across_blocks(tmp_path):
    """A baseline window that ends inside the second block of a merged run."""
    folders = [write_recording(tmp_path / "t1", seed=1), write_recording(tmp_path / "t2", seed=2)]
    window = slice(5, 30)            # 21 post-trim frames in block 1, so it spans both
    old = run_streaming([old_merged_sources(folders, ["gcamp_first"] * 2)], _cfg(),
                        baseline_slice=window, trim=0, downsample=DS)
    prepared = prepare_interleaved_trial(
        [sorted_frame_files(f) for f in folders], ["gcamp_first"] * 2, trim=TRIM,
        pipeline_trim=0, downsample=DS, baseline_slice=window, channel_buffer="ram",
        log=lambda m: None)
    new = run_streaming([(prepared.gcamp, prepared.emo)], _cfg(), baseline_slice=window,
                        trim=0, downsample=DS, trial_baselines=[prepared.baselines])
    _assert_same_result(old, new)


def test_each_raw_file_is_decoded_once_with_a_buffer(tmp_path, monkeypatch):
    folder = write_recording(tmp_path / "t1")
    reads = {"n": 0}
    original = tifffile.imread

    def counting(*a, **k):
        reads["n"] += 1
        return original(*a, **k)

    monkeypatch.setattr(tifffile, "imread", counting)
    prepared = prepare_interleaved_trial([sorted_frame_files(folder)], ["auto"], trim=TRIM,
                                         pipeline_trim=TRIM, downsample=DS,
                                         channel_buffer="ram", log=lambda m: None)
    run_streaming([(prepared.gcamp, prepared.emo)], _cfg(), trim=TRIM, downsample=DS,
                  trial_baselines=[prepared.baselines])
    # Trimmed frames are never decoded; every kept frame exactly once.
    assert reads["n"] == 2 * (24 - TRIM)


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------
def _stats_from_levels(odd_levels, even_levels) -> ChannelStats:
    stats = ChannelStats(top_percent=10.0)
    for o, e in zip(odd_levels, even_levels):
        stats.add(np.full((8, 8), o), np.full((8, 8), e))
    return stats


@pytest.mark.parametrize("gcamp_odd, expected", [(True, "gcamp_first"), (False, "emo_first")])
def test_well_separated_recording_is_certain_for_both_parities(tmp_path, gcamp_odd, expected):
    folder = write_recording(tmp_path / "t1", gcamp_odd=gcamp_odd)
    block = read_interleaved_block(sorted_frame_files(folder), TRIM, channel_buffer="none")
    order, verdict, details = decide_channel_order(block.stats)
    assert (order, verdict) == (expected, "certain")
    assert details["alternation"] < -0.99
    assert details["n_frames"] == 24 - TRIM


@pytest.mark.parametrize("gcamp_odd, expected", [(True, "gcamp_first"), (False, "emo_first")])
def test_close_levels_are_uncertain_but_still_ordered(gcamp_odd, expected):
    # Alternating by 8 with a per-frame scatter of 1: lag-1 r ~ -0.94, separation ~8.
    rng = np.random.default_rng(3)
    dim = 1000 + rng.normal(0, 1, 400)
    bright = 1008 + rng.normal(0, 1, 400)
    odd, even = (dim, bright) if gcamp_odd else (bright, dim)
    order, verdict, details = decide_channel_order(_stats_from_levels(odd, even))
    assert verdict == "uncertain", details
    assert order == expected


def test_non_alternating_recording_is_not_interleaved(tmp_path):
    folder = write_recording(tmp_path / "t1", interleaved=False, jitter=60.0)
    block = read_interleaved_block(sorted_frame_files(folder), TRIM, channel_buffer="none")
    _, verdict, details = decide_channel_order(block.stats)
    assert verdict == "not_interleaved", details


def test_constant_frames_are_not_interleaved():
    _, verdict, details = decide_channel_order(_stats_from_levels([5.0] * 10, [5.0] * 10))
    assert verdict == "not_interleaved"
    assert np.isnan(details["alternation"])


def test_thresholds_come_from_the_config():
    rng = np.random.default_rng(4)
    stats = _stats_from_levels(1000 + rng.normal(0, 1, 400), 1008 + rng.normal(0, 1, 400))
    _, verdict, _ = decide_channel_order(stats, ChannelDecisionConfig(separation_min=2.0))
    assert verdict == "certain"


def test_not_interleaved_auto_block_skips_the_whole_merged_trial(tmp_path):
    folders = [write_recording(tmp_path / "t1"),
               write_recording(tmp_path / "t2", interleaved=False, jitter=60.0),
               write_recording(tmp_path / "t3")]
    prepared = prepare_interleaved_trial(
        [sorted_frame_files(f) for f in folders], ["auto"] * 3, trim=TRIM, pipeline_trim=0,
        downsample=DS, channel_buffer="disk", buffer_dir=tmp_path / "buf", log=lambda m: None)
    assert prepared.skipped
    # It stopped at the bad block: t3 was never read.
    assert [c.verdict for c in prepared.choices] == ["certain", "not_interleaved"]
    assert os.listdir(tmp_path / "buf") == []


def test_explicit_order_is_used_as_given(tmp_path):
    folder = write_recording(tmp_path / "t1", interleaved=False, jitter=60.0)
    prepared = prepare_interleaved_trial([sorted_frame_files(folder)], ["emo_first"],
                                         trim=TRIM, pipeline_trim=TRIM, downsample=DS,
                                         channel_buffer="ram", log=lambda m: None)
    assert not prepared.skipped
    assert (prepared.choices[0].order, prepared.choices[0].verdict) == ("emo_first", "explicit")


# ---------------------------------------------------------------------------
# buffer selection
# ---------------------------------------------------------------------------
def _mock_ram(monkeypatch, available: int):
    monkeypatch.setattr(wio.psutil, "virtual_memory", lambda: SimpleNamespace(available=available))


def _mock_disk(monkeypatch, free: int):
    monkeypatch.setattr(wio.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))


def test_auto_prefers_ram_then_disk_then_none(tmp_path, monkeypatch):
    estimate = 1000
    _mock_ram(monkeypatch, 1500)
    assert choose_buffer_mode("auto", estimate) == "ram"
    _mock_ram(monkeypatch, 1499)
    _mock_disk(monkeypatch, 1100)
    assert choose_buffer_mode("auto", estimate, buffer_dir=tmp_path) == "disk"
    assert choose_buffer_mode("auto", estimate, buffer_dir=None) == "none"
    _mock_disk(monkeypatch, 1099)
    assert choose_buffer_mode("auto", estimate, buffer_dir=tmp_path) == "none"


def test_auto_subtracts_what_is_already_buffered(monkeypatch):
    _mock_ram(monkeypatch, 3000)
    assert choose_buffer_mode("auto", 1000, buffered_bytes=1500) == "ram"
    assert choose_buffer_mode("auto", 1000, buffered_bytes=1501) == "none"


def test_explicit_modes_and_bad_values(tmp_path):
    assert choose_buffer_mode("none", 10**15) == "none"
    assert choose_buffer_mode("ram", 10**15) == "ram"
    with pytest.raises(ValueError, match="buffer_dir"):
        choose_buffer_mode("disk", 10)
    with pytest.raises(ValueError, match="not one of"):
        choose_buffer_mode("ssd", 10)


@pytest.mark.parametrize("disk_free, second", [(10**12, "disk"), (0, "none")])
def test_merged_run_switches_mode_once_the_ram_budget_runs_out(tmp_path, monkeypatch,
                                                                disk_free, second):
    folders = [write_recording(tmp_path / f"t{i}", seed=i) for i in (1, 2, 3)]
    # One block's estimate: 2 groups x 21 frames x 16x16 float32.
    estimate = 2 * (24 - TRIM) * 16 * 16 * 4
    # Room for exactly one block at the 1.5x safety factor (but not two).
    _mock_ram(monkeypatch, int(2 * estimate))
    _mock_disk(monkeypatch, disk_free)
    prepared = prepare_interleaved_trial(
        [sorted_frame_files(f) for f in folders], ["auto"] * 3, trim=TRIM, pipeline_trim=0,
        downsample=DS, channel_buffer="auto", buffer_dir=tmp_path / "buf", log=lambda m: None)
    assert [c.buffer_mode for c in prepared.choices] == ["ram", second, second]
    new = run_streaming([(prepared.gcamp, prepared.emo)], _cfg(), trim=0, downsample=DS,
                        trial_baselines=[prepared.baselines])
    prepared.release()
    old = run_streaming([old_merged_sources(folders, ["gcamp_first"] * 3)], _cfg(),
                        trim=0, downsample=DS)
    _assert_same_result(old, new)


# ---------------------------------------------------------------------------
# dtype guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, shape, expected", [
    (np.uint16, (32, 32), np.float32),
    (np.uint8, (32, 32), np.float32),
    (np.int16, (32, 32), np.float32),
    (np.uint32, (32, 32), np.float64),
    (np.float32, (32, 32), np.float64),
    (np.uint16, (31, 32), np.float64),     # odd side: fractional MATLAB box weights
])
def test_buffer_dtype_guard(raw, shape, expected):
    assert buffer_dtype_for(raw, shape, 0.5) == np.dtype(expected)


@pytest.mark.parametrize("dtype, expected", [(np.float32, "float64"), (np.uint32, "float64"),
                                             (np.uint16, "float32")])
def test_non_16_bit_raw_frames_are_buffered_in_float64(tmp_path, dtype, expected):
    folder = write_recording(tmp_path / "t1", dtype=dtype)
    for mode in ("ram", "disk"):
        block = read_interleaved_block(sorted_frame_files(folder), TRIM, channel_buffer=mode,
                                       buffer_dir=tmp_path / "buf")
        assert block.buffer_dtype == expected
        first = next(block.odd.frames())
        assert first.dtype == np.float64
        block.release()
    # And the result is still identical to the two-read path.
    prepared = prepare_interleaved_trial([sorted_frame_files(folder)], ["gcamp_first"],
                                         trim=TRIM, pipeline_trim=TRIM, downsample=DS,
                                         channel_buffer="ram", log=lambda m: None)
    new = run_streaming([(prepared.gcamp, prepared.emo)], _cfg(), trim=TRIM, downsample=DS,
                        trial_baselines=[prepared.baselines])
    old = run_streaming([old_single_sources(folder, "gcamp_first")], _cfg(), trim=TRIM,
                        downsample=DS)
    _assert_same_result(old, new)


# ---------------------------------------------------------------------------
# extract_cohort: per-job flow, meta, summary, cleanup
# ---------------------------------------------------------------------------
def _batch_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = dict(
        manifest="", output_root=str(tmp_path / "out"), select=[], max_rows=None,
        roi_set_dir=None, roi_set=None, debug=False, debug_frames=10,
        max_recordings=None, merge_recordings=False, workers=1, skip_existing=True,
        save_data=True, save_data_root=None, save_data_window=None,
        save_data_dff_dtype="float64", save_data_f_dtype="float64",
        pattern="*.tif", rot90_k=0, frame_order="name", trim_frames=None,
        channel_buffer="ram", buffer_dir=str(tmp_path / "buf"), keep_channel_buffer=False,
        alternation_max=-0.9, separation_min=10.0, top_percent=10.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def small_geometry(monkeypatch):
    """Run the batch on the small synthetic grid instead of a 128x128 atlas."""
    profile = replace(get_profile("cerebellar_rs"), atlas=dict(BOXES), trim=TRIM)

    def geometry(row, args, recording):
        return profile, _cfg(), None, 2 * Y_1, 2 * X_2

    monkeypatch.setattr(batch, "_geometry_for", geometry)
    return profile


def _job(tmp_path: Path, folders: list[Path], args, orders: list[str], merged=False):
    row = dict(day="240101", animal="A1", group="P", profile="cerebellar_rs",
               bregma_row=str(2 * Y_1), bregma_col=str(2 * X_2),
               recording_paths=";".join(str(f) for f in folders),
               channel_order=";".join(orders), total_frames="0", n_recordings=str(len(folders)))
    args.merge_recordings = merged
    return batch.build_jobs([row], args)


def test_batch_buffer_modes_give_identical_outputs(tmp_path, small_geometry):
    folder = write_recording(tmp_path / "raw" / "t1", gcamp_odd=False)
    outputs = {}
    for mode in ("ram", "disk", "none"):
        args = _batch_args(tmp_path / mode, channel_buffer=mode)
        job = _job(tmp_path, [folder], args, ["auto"])[0]
        summary = batch.execute_job(job)
        assert summary["status"] == "done" and summary["channel_verdict"] == "certain"
        with np.load(summary["npz_path"]) as npz:
            outputs[mode] = {k: npz[k] for k in ("temp_roi", "R_mean")}
            assert str(npz["channel_order"]) == "emo_first"
            assert str(npz["channel_order_configured"]) == "auto"
            assert str(npz["channel_buffer_mode"]) == mode
            assert int(npz["channel_decision_n_frames"]) == 24 - TRIM
            assert float(npz["channel_level_odd"]) > float(npz["channel_level_even"])
        dump_meta = np.load(os.path.join(job["out_dir"], "pixels_meta_full.npz"))
        assert str(dump_meta["channel_verdict"]) == "certain"
        assert _leftovers(tmp_path / mode / "buf") == []
    for mode in ("disk", "none"):
        for key in ("temp_roi", "R_mean"):
            np.testing.assert_array_equal(outputs["ram"][key], outputs[mode][key])


def test_batch_merged_meta_is_joined_per_block(tmp_path, small_geometry):
    folders = [write_recording(tmp_path / "raw" / "t1", gcamp_odd=True, seed=1),
               write_recording(tmp_path / "raw" / "t2", gcamp_odd=False, seed=2)]
    args = _batch_args(tmp_path, channel_buffer="auto")
    job = _job(tmp_path, folders, args, ["auto", "auto"], merged=True)[0]
    summary = batch.execute_job(job)
    assert summary["channel_verdict"] == "certain;certain"
    with np.load(summary["npz_path"]) as npz:
        assert str(npz["channel_order"]) == "gcamp_first;emo_first"
        assert len(str(npz["channel_level_odd"]).split(";")) == 2


def _write_manifest(path: Path, folders: list[Path]) -> None:
    """One manifest row per recording folder, named t1, t2, ... through the folder."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["day", "animal", "group", "profile", "bregma_row",
                                           "bregma_col", "recording_paths", "channel_order",
                                           "total_frames", "n_recordings"])
        w.writeheader()
        for i, folder in enumerate(folders):
            w.writerow(dict(day="240101", animal=f"A{i}", group="P", profile="cerebellar_rs",
                            bregma_row="4", bregma_col="4", recording_paths=str(folder),
                            channel_order="auto", total_frames="48", n_recordings="1"))


def _summary_rows(args) -> list[dict[str, str]]:
    import csv
    with open(os.path.join(args.output_root, "batch_summary.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_not_interleaved_is_stored_as_odd_even_by_default(tmp_path, small_geometry):
    folder = write_recording(tmp_path / "raw" / "t1", interleaved=False, jitter=60.0)
    args = _batch_args(tmp_path, channel_buffer="disk")
    job = _job(tmp_path, [folder], args, ["auto"])[0]
    summary = batch.execute_job(job)

    assert summary["status"] == batch.NEEDS_CHANNEL_DECISION
    assert summary["channel_verdict"] == "not_interleaved"
    assert summary["npz_path"] == "" and summary["mean_offdiag_R"] == ""
    # Position-named files only: nothing a downstream GCaMP/emo reader can pick up.
    assert sorted(os.listdir(job["out_dir"])) == [
        "pixels_f_even_full.npy", "pixels_f_odd_full.npy", "pixels_meta_odd_even_full.npz"]
    assert _leftovers(tmp_path / "buf") == []
    meta = np.load(os.path.join(job["out_dir"], "pixels_meta_odd_even_full.npz"))
    assert str(meta["channel_order"]) == "odd_even_unassigned"
    assert str(meta["channel_verdict"]) == "not_interleaved"

    # The volumes are exactly the odd and even raw frames on the final grid.
    files = sorted_frame_files(folder)
    n_time = len(files) // 2 - TRIM
    for name, group in (("odd", files[0::2]), ("even", files[1::2])):
        expected = np.stack([imresize_box(imresize_box(
            tifffile.imread(f).astype(np.float64), DS), DS) for f in group[TRIM:]])
        volume = np.load(os.path.join(job["out_dir"], f"pixels_f_{name}_full.npy"))
        assert volume.shape == (n_time, 8, 8)
        np.testing.assert_array_equal(volume, expected)


def test_stored_odd_even_matches_across_buffer_modes(tmp_path, small_geometry):
    folder = write_recording(tmp_path / "raw" / "t1", interleaved=False, jitter=60.0)
    volumes = {}
    for mode in ("ram", "disk", "none"):
        args = _batch_args(tmp_path / mode, channel_buffer=mode)
        job = _job(tmp_path, [folder], args, ["auto"])[0]
        batch.execute_job(job)
        volumes[mode] = np.load(os.path.join(job["out_dir"], "pixels_f_odd_full.npy"))
    np.testing.assert_array_equal(volumes["ram"], volumes["disk"])
    np.testing.assert_array_equal(volumes["ram"], volumes["none"])


def test_storing_odd_even_replaces_labelled_outputs_but_not_pedestal(tmp_path, small_geometry):
    folder = write_recording(tmp_path / "raw" / "t1", interleaved=False, jitter=60.0)
    args = _batch_args(tmp_path, channel_buffer="ram")
    job = _job(tmp_path, [folder], args, ["auto"])[0]
    os.makedirs(job["out_dir"])
    stale = ["connectivity_full.npz", "roi_fluorescence_full.csv", "roi_traces_full.png",
             "pixels_dff_full.npy", "pixels_f_gcamp_full.npy", "pixels_f_emo_full.npy",
             "pixels_meta_full.npz", "pixels_median_dff_20s_full.npy",
             "pixels_median_dff_20s_meta.json"]
    kept = ["pixels_median_dff_20s_blpedestal_full.npy", "unrelated.txt"]
    for name in stale + kept:
        Path(job["out_dir"], name).write_bytes(b"old")
    batch.execute_job(job)
    left = set(os.listdir(job["out_dir"]))
    assert not left & set(stale)
    assert set(kept) <= left


def test_explicit_order_on_not_interleaved_still_runs(tmp_path, small_geometry):
    folder = write_recording(tmp_path / "raw" / "t1", interleaved=False, jitter=60.0)
    args = _batch_args(tmp_path, channel_buffer="ram")
    job = _job(tmp_path, [folder], args, ["emo_first"])[0]
    assert batch.execute_job(job)["status"] == "done"


def test_merged_run_with_not_interleaved_block_is_still_skipped(tmp_path, small_geometry):
    folders = [write_recording(tmp_path / "raw" / "t1", seed=1),
               write_recording(tmp_path / "raw" / "t2", interleaved=False, jitter=60.0)]
    args = _batch_args(tmp_path, channel_buffer="ram")
    job = _job(tmp_path, folders, args, ["auto", "auto"], merged=True)[0]
    assert batch.execute_job(job)["status"] == "skipped"


def test_debug_mode_stores_odd_even(tmp_path, small_geometry):
    folder = write_recording(tmp_path / "raw" / "t1", interleaved=False, jitter=60.0)
    args = _batch_args(tmp_path, channel_buffer="ram", debug=True, debug_frames=20)
    job = _job(tmp_path, [folder], args, ["auto"])[0]
    assert batch.execute_job(job)["status"] == batch.NEEDS_CHANNEL_DECISION
    assert np.load(os.path.join(job["out_dir"], "pixels_f_even_debug.npy")).shape == (20 - TRIM, 8, 8)


def test_rerun_redoes_only_skipped_rows_and_replaces_them(tmp_path, small_geometry, monkeypatch):
    """The 2024 rerun: a skipped row with a stale .npz is redone, a done row is untouched."""
    good = write_recording(tmp_path / "raw" / "good" / "t1", seed=1)
    bad = write_recording(tmp_path / "raw" / "bad" / "t1", interleaved=False, jitter=60.0)
    args = _batch_args(tmp_path, channel_buffer="ram", skip_not_interleaved=True)
    args.manifest = str(tmp_path / "manifest.csv")
    _write_manifest(Path(args.manifest), [good, bad])
    monkeypatch.setattr(batch, "build_runtime_args", lambda: args)
    batch.main()
    assert [r["status"] for r in _summary_rows(args)] == ["done", "skipped"]

    # What October left behind: outputs from before the recording was flagged, which
    # must NOT count as complete.
    stale_dir = Path(args.output_root, "240101_A1", "t1")
    stale_dir.mkdir(parents=True)
    (stale_dir / "connectivity_full.npz").write_bytes(b"old")
    (stale_dir / "pixels_meta_full.npz").write_bytes(b"old")

    good_files = sorted(Path(args.output_root, "240101_A0", "t1").iterdir())
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in good_files}
    args.skip_not_interleaved = False
    batch.main()
    rows = _summary_rows(args)
    assert [(r["animal"], r["status"]) for r in rows] == [
        ("A0", "done"), ("A1", batch.NEEDS_CHANNEL_DECISION)]
    assert before == {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in good_files}
    assert not (stale_dir / "connectivity_full.npz").exists()
    assert (stale_dir / "pixels_f_odd_full.npy").exists()

    # A third run has nothing left to do, and the two rows stay as they are.
    monkeypatch.setattr(batch, "execute_job", lambda job: pytest.fail(f"re-ran {job['label']}"))
    batch.main()
    assert [r["status"] for r in _summary_rows(args)] == ["done", batch.NEEDS_CHANNEL_DECISION]


def test_batch_skips_a_not_interleaved_recording_with_no_outputs(tmp_path, small_geometry):
    folder = write_recording(tmp_path / "raw" / "t1", interleaved=False, jitter=60.0)
    args = _batch_args(tmp_path, channel_buffer="disk", skip_not_interleaved=True)
    job = _job(tmp_path, [folder], args, ["auto"])[0]
    summary = batch.execute_job(job)
    assert summary["status"] == "skipped"
    assert summary["channel_verdict"] == "not_interleaved"
    assert summary["npz_path"] == "" and summary["mean_offdiag_R"] == ""
    assert not os.path.exists(job["out_dir"])
    assert os.listdir(tmp_path / "buf") == []


def test_resume_does_not_reread_a_skipped_recording(tmp_path, small_geometry, monkeypatch):
    bad = write_recording(tmp_path / "raw" / "t1", interleaved=False, jitter=60.0)
    args = _batch_args(tmp_path, channel_buffer="ram", save_data=False, skip_not_interleaved=True)
    manifest = tmp_path / "manifest.csv"
    import csv
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["day", "animal", "group", "profile", "bregma_row",
                                           "bregma_col", "recording_paths", "channel_order",
                                           "total_frames", "n_recordings"])
        w.writeheader()
        w.writerow(dict(day="240101", animal="A1", group="P", profile="cerebellar_rs",
                        bregma_row="4", bregma_col="4", recording_paths=str(bad),
                        channel_order="auto", total_frames="48", n_recordings="1"))
    args.manifest = str(manifest)
    monkeypatch.setattr(batch, "build_runtime_args", lambda: args)
    batch.main()
    with open(os.path.join(args.output_root, "batch_summary.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["status"], r["channel_verdict"]) for r in rows] == [("skipped", "not_interleaved")]

    calls = {"n": 0}
    monkeypatch.setattr(batch, "execute_job", lambda job: calls.__setitem__("n", calls["n"] + 1))
    batch.main()
    assert calls["n"] == 0
    with open(os.path.join(args.output_root, "batch_summary.csv"), encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 1


def test_disk_buffer_is_removed_after_an_exception(tmp_path, small_geometry, monkeypatch):
    folder = write_recording(tmp_path / "raw" / "t1")
    args = _batch_args(tmp_path, channel_buffer="disk")
    job = _job(tmp_path, [folder], args, ["auto"])[0]

    def boom(*a, **k):
        assert len(os.listdir(tmp_path / "buf")) == 1     # the buffer exists mid-run
        raise RuntimeError("pass 2 failed")

    monkeypatch.setattr(batch, "run_streaming_profile", boom)
    with pytest.raises(RuntimeError, match="pass 2 failed"):
        batch.execute_job(job)
    assert os.listdir(tmp_path / "buf") == []


@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_keep_channel_buffer_saves_the_unlabelled_volumes(tmp_path, small_geometry, mode):
    folder = write_recording(tmp_path / "raw" / "t1", gcamp_odd=False)
    args = _batch_args(tmp_path, channel_buffer=mode, keep_channel_buffer=True)
    job = _job(tmp_path, [folder], args, ["auto"])[0]
    batch.execute_job(job)
    odd = np.load(os.path.join(job["out_dir"], "pixels_f_odd_full.npy"))
    even = np.load(os.path.join(job["out_dir"], "pixels_f_even_full.npy"))
    assert odd.shape == even.shape == (24 - TRIM, 16, 16)
    # Unlabelled: odd is still the odd group (here the bright emo channel).
    assert odd.mean() > even.mean()
    # Same values the dump's raw F was computed from (half-res, before the 2nd resize).
    files = sorted_frame_files(folder)
    first_odd = wio._half_res(tifffile.imread(str(files[2 * TRIM])), 0, DS)
    np.testing.assert_array_equal(odd[0], first_odd)
    assert _leftovers(tmp_path / "buf") == []


def test_debug_mode_decides_from_the_loaded_frames(tmp_path, small_geometry):
    folder = write_recording(tmp_path / "raw" / "t1", gcamp_odd=False)
    args = _batch_args(tmp_path, debug=True, debug_frames=12, save_data=False)
    job = _job(tmp_path, [folder], args, ["auto"])[0]
    summary = batch.execute_job(job)
    with np.load(summary["npz_path"]) as npz:
        assert str(npz["channel_order"]) == "emo_first"
        assert int(npz["channel_decision_n_frames"]) == 12 - TRIM
        assert str(npz["channel_buffer_mode"]) == "debug"


def test_level_statistic_median_vs_top_percent():
    """A group can be dimmer by its median yet brighter in its top 10% (the 2024
    misassignments): the two level statistics then pick opposite orders."""
    rng = np.random.default_rng(5)
    stats = ChannelStats(top_percent=10.0)
    for _ in range(50):
        odd = np.full((10, 10), 100.0) + rng.normal(0, 1)
        odd[:2, :] = 1000.0                       # 20% bright spots, low median
        even = np.full((10, 10), 300.0) + rng.normal(0, 1)
        stats.add(odd, even)
    top_order, _, top = decide_channel_order(stats, ChannelDecisionConfig(level_statistic="top_percent"))
    med_order, _, med = decide_channel_order(stats, ChannelDecisionConfig(level_statistic="median"))
    assert (top_order, med_order) == ("emo_first", "gcamp_first")
    assert (top["level_statistic"], med["level_statistic"]) == ("top_percent", "median")
    with pytest.raises(ValueError, match="level_statistic"):
        decide_channel_order(stats, ChannelDecisionConfig(level_statistic="mean"))


# ---------------------------------------------------------------------------
# level_region = "roi_union": decide from the pixels under the ROI boxes only
# ---------------------------------------------------------------------------
def test_level_region_roi_union_uses_only_the_masked_pixels():
    """Odd is dimmer over the whole frame but brighter inside the mask: the two
    regions pick opposite orders, and stats/config regions must agree."""
    rng = np.random.default_rng(7)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:3, :3] = True
    frame_stats, roi_stats = ChannelStats(), ChannelStats(level_mask=mask)
    for _ in range(40):
        odd = np.full((10, 10), 100.0) + rng.normal(0, 1)
        even = np.full((10, 10), 300.0) + rng.normal(0, 1)
        odd[mask], even[mask] = 500.0 + rng.normal(0, 1), 200.0 + rng.normal(0, 1)
        frame_stats.add(odd, even)
        roi_stats.add(odd, even)
    frame_order, _, frame_d = decide_channel_order(frame_stats, ChannelDecisionConfig())
    roi_order, _, roi_d = decide_channel_order(roi_stats, ChannelDecisionConfig(level_region="roi_union"))
    assert (frame_order, roi_order) == ("gcamp_first", "emo_first")
    assert (frame_d["level_region"], roi_d["level_region"]) == ("frame", "roi_union")
    assert roi_d["level_odd"] == pytest.approx(500.0, abs=5)
    with pytest.raises(ValueError, match="taken over 'roi_union'"):
        decide_channel_order(roi_stats, ChannelDecisionConfig())
    with pytest.raises(ValueError, match="taken over 'frame'"):
        decide_channel_order(frame_stats, ChannelDecisionConfig(level_region="roi_union"))
    with pytest.raises(ValueError, match="does not match"):
        roi_stats.add(np.zeros((4, 4)), np.zeros((4, 4)))


def test_roi_union_level_mask_is_the_final_grid_union_lifted_exactly():
    """Mean over the half-res mask == mean over the box union of the final frame,
    because each final pixel is the mean of its 2x2 half-res block."""
    from wfci import imresize_box
    from wfci.roi import box_slices_for
    from wfci.streaming import roi_union_level_mask

    half_shape = (FRAME_SHAPE[0] // 2, FRAME_SHAPE[1] // 2)
    mask = roi_union_level_mask(_cfg(), None, DS)(half_shape)
    assert mask.shape == half_shape and mask.dtype == bool
    half = np.random.default_rng(3).normal(1000, 50, half_shape)
    final = imresize_box(half, DS)
    union = np.zeros(final.shape, dtype=bool)
    for _, rows, cols in box_slices_for(_cfg(), final.shape):
        union[rows, cols] = True
    assert mask.sum() == 4 * union.sum()
    assert half[mask].mean() == pytest.approx(final[union].mean(), rel=1e-12)
    with pytest.raises(ValueError, match="integer reduction"):
        roi_union_level_mask(_cfg(), None, DS)((15, 16))


def test_prepare_needs_the_mask_exactly_when_level_region_is_roi_union(tmp_path):
    folder = write_recording(tmp_path / "t1")
    files = [sorted_frame_files(folder)]
    with pytest.raises(ValueError, match="level_mask_for is missing"):
        prepare_interleaved_trial(files, ["auto"], trim=TRIM, pipeline_trim=TRIM,
                                  decision=ChannelDecisionConfig(level_region="roi_union"),
                                  log=lambda m: None)
    with pytest.raises(ValueError, match="level_mask_for is given"):
        prepare_interleaved_trial(files, ["auto"], trim=TRIM, pipeline_trim=TRIM,
                                  level_mask_for=lambda shape: np.ones(shape, dtype=bool),
                                  log=lambda m: None)


@pytest.mark.parametrize("debug", [False, True])
def test_batch_records_the_level_region(tmp_path, small_geometry, debug):
    """roi_union keeps the outputs of a clean recording and records the region; its
    levels are the masked levels, not the whole-frame ones."""
    folder = write_recording(tmp_path / "raw" / "t1", gcamp_odd=False)
    levels = {}
    for region in ("frame", "roi_union"):
        args = _batch_args(tmp_path / region, channel_level_region=region, debug=debug,
                           debug_frames=12, save_data=False)
        summary = batch.execute_job(_job(tmp_path, [folder], args, ["auto"])[0])
        assert summary["status"] == "done"
        with np.load(summary["npz_path"]) as npz:
            assert str(npz["channel_level_region"]) == region
            assert str(npz["channel_order"]) == "emo_first"
            levels[region] = float(npz["channel_level_odd"])
    assert levels["frame"] != levels["roi_union"]
