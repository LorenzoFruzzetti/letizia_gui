"""plot_channel_traces.py on a session holding an odd/even (needs_channel_decision) recording."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_channel_traces as plot  # noqa: E402


def _recording(folder: Path, status: str, order: str, verdict: str) -> None:
    """A tiny recording whose sidecar and volumes follow the batch's naming for `status`."""
    folder.mkdir(parents=True)
    n_time, shape = 60, (4, 4)
    for channel in plot.CHANNELS[status]:
        np.save(folder / f"pixels_f_{channel}_full.npy",
                np.full((n_time, *shape), 1000.0, dtype=np.float32) + np.arange(n_time).reshape(-1, 1, 1))
    np.savez(folder / plot.META_NAME[status], channel_order=order, channel_verdict=verdict,
             channel_alternation="-0.5", channel_separation="2.0", axis_order="time,y,x",
             n_time=n_time, n_written=n_time, region=np.array([0, 4, 0, 4]), y_1=0, x_2=0)


@pytest.fixture
def unit(tmp_path, monkeypatch):
    """One unit: t1 labelled and done, t2 stored by position."""
    folder = tmp_path / "240101_A1"
    _recording(folder / "t1", "done", "gcamp_first", "certain")
    _recording(folder / "t2", plot.NEEDS_DECISION, "odd_even_unassigned", "not_interleaved")
    monkeypatch.setattr(plot, "recording_boxes", lambda meta: (None, {"a": None}))
    monkeypatch.setattr(plot, "crop_slices", lambda *a: (slice(0, 2), slice(0, 2)))
    summary = tmp_path / "summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["day", "animal", "recording", "status", "channel_verdict"])
        writer.writeheader()
        writer.writerow(dict(day="240101", animal="A1", recording="t1", status="done",
                             channel_verdict="certain"))
        writer.writerow(dict(day="240101", animal="A1", recording="t2",
                             status=plot.NEEDS_DECISION, channel_verdict="not_interleaved"))
    return tmp_path, folder, summary


def _job(folder: Path, records, level_region="roi_union") -> dict:
    return dict(folder=str(folder), records=records, check_disk=True, level_region=level_region,
                plots_dir="plots_roi_union", skipped_gap_frames=60,
                sampling_rate_hz=25.0, smooth_s=1.0)


def test_mixed_unit_draws_odd_even_and_labelled_recordings(unit):
    root, folder, summary = unit
    args = SimpleNamespace(channel_assignment_csv="none", old_manifest_csv="none",
                           batch_summary_csv=str(summary))
    records = plot.build_records(root, args)["240101_A1"]
    assert [r["order"] for r in records] == ["gcamp_first", "odd_even_unassigned"]

    plot.plot_folder(_job(folder, records))
    out = folder / "plots_roi_union"
    assert {p.name for p in out.iterdir()} == {
        "channel_traces.png", "channel_traces_t1.png", "channel_traces_t2.png", "channel_means.csv"}
    with open(out / "channel_means.csv", newline="", encoding="utf-8") as fh:
        rows = {row["recording"]: row for row in csv.DictReader(fh)}
    assert rows["t1"]["mean_gcamp"] and not rows["t1"]["mean_odd"]
    assert rows["t2"]["mean_odd"] and rows["t2"]["mean_even"] and not rows["t2"]["mean_gcamp"]
    assert rows["t2"]["status"] == plot.NEEDS_DECISION


def test_disk_and_summary_must_agree(unit):
    _, folder, _ = unit
    records = [dict(recording="t1", status="done"), dict(recording="t2", status="done")]
    with pytest.raises(RuntimeError, match="batch_summary"):
        plot.check_disk_against_records(folder, records)
