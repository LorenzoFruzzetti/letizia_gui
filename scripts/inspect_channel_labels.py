"""Browse the pixel dumps side by side as `channel_order: auto` labelled them.

`auto` (wfci.io.interleaved_channel_files) calls the DIMMER of the first two frames
GCaMP. When the two channels are close in brightness that guess can be wrong, or the
"GCaMP" frames may not be fluorescence at all.
This viewer shows, for one recording at a time, exactly what the batch saved:

    top row     full-field mean image of the auto-GCaMP group (meta `mean_f`),
                of the auto-reflectance group (`mean_r`), and the band-passed
                difference of the two (vessels show where they disagree)
    bottom row  one frame of each channel from the dumped crop
                (pixels_f_gcamp_full.npy / pixels_f_emo_full.npy) and the crop-mean
                trace of both channels, each relative to its own mean, with the
                current frame marked

What to look for: GCaMP fluorescence is a fairly smooth image of the cortex; green
reflectance shows the vessels dark and sharp. If both panels look like reflectance,
or the two traces overlap, the recording has no usable GCaMP channel. If the panels
look right but swapped, `auto` mis-assigned it.

Keys:
    n / p            next / previous recording
    right / left     +-1 frame          up / down   +-1 s
    pageup/pagedown  +-10 s
    c                toggle shared colour limits for the two channels (compare levels)
    q                close

Run (in the wfci_gui environment):
    python inspect_channel_labels.py --cohort <cohort folder>/cohort.yaml
    python inspect_channel_labels.py --cohort <...>/cohort.yaml --session 200101 --unit A1

The dumps and the frame rate come from the cohort file.

Edit RUN_CONFIG below to run directly from the editor without CLI arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("QtAgg")  # interactive window; pyside6 is in the wfci_gui env
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Edit this section to run without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "cohort_file": None,          # <cohort folder>/cohort.yaml (or pass --cohort)
    "session": None,              # <yymmdd> prefix of the unit folders; None = all
    "unit": None,                 # e.g. "A1" to start on one animal; None = first
    "color_percentiles": (1.0, 99.0),
    "prefer_cli_args": True,
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cohort", dest="cohort_file", default=defaults["cohort_file"],
                   help="The cohort file, <cohort folder>/cohort.yaml.")
    p.add_argument("--session", default=defaults["session"])
    p.add_argument("--unit", default=defaults["unit"])
    p.add_argument("--color-percentiles", type=float, nargs=2,
                   default=defaults["color_percentiles"])
    return p.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        args = parse_args(defaults=config)
    else:
        args = argparse.Namespace(
            cohort_file=config["cohort_file"],
            session=config["session"],
            unit=config["unit"],
            color_percentiles=config["color_percentiles"],
        )
    # The dumps and the frame rate come from the cohort file.
    from wfci_gui.logic.cohort import acquisition_value, load_script_cohort
    cohort = load_script_cohort(args.cohort_file)
    args.pixel_root = cohort.layout.pixel_data
    args.sampling_rate_hz = float(acquisition_value(cohort, "sampling_rate_hz"))
    return args


def find_recordings(pixel_root: Path, session: str | None) -> list[Path]:
    """Every <unit>/<t#> folder with a dump, optionally only units starting with `session`."""
    recordings = []
    for unit in sorted(p for p in pixel_root.iterdir() if p.is_dir()):
        if unit.name.startswith("_") or (session and not unit.name.startswith(session)):
            continue
        recordings += sorted(r for r in unit.iterdir() if (r / "pixels_meta_full.npz").exists())
    if not recordings:
        raise FileNotFoundError(f"No dumps under {pixel_root} for session {session!r}.")
    return recordings


def bandpass(image: np.ndarray) -> np.ndarray:
    """Vessel-scale structure: Gaussian sigma 1 minus sigma 8 (it isolates vessel-sized structure)."""
    image = np.asarray(image, dtype=np.float64)
    return gaussian_filter(image, 1) - gaussian_filter(image, 8)


def image_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


class Viewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.recordings = find_recordings(args.pixel_root, args.session)
        self.i_rec = 0
        if args.unit:
            matches = [i for i, r in enumerate(self.recordings) if args.unit in r.parent.name]
            if not matches:
                raise ValueError(f"No recording of unit {args.unit!r} in this session.")
            self.i_rec = matches[0]
        self.i_frame = 0
        self.shared_limits = False

        # matplotlib binds left/right (back/forward), p (pan), c (back) and q (quit) to
        # its toolbar; release them so they only drive this viewer.
        for keymap, keys in (("keymap.back", ("left", "c", "backspace")),
                             ("keymap.forward", ("right", "v")),
                             ("keymap.pan", ("p",)), ("keymap.quit", ("q", "ctrl+w"))):
            plt.rcParams[keymap] = [k for k in plt.rcParams[keymap] if k not in keys]

        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True,
                                           squeeze=False)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load_recording()
        plt.show()

    def load_recording(self) -> None:
        """Read one recording's meta and both raw-F volumes into memory."""
        rec = self.recordings[self.i_rec]
        meta = np.load(rec / "pixels_meta_full.npz", allow_pickle=True)
        self.mean_f = np.asarray(meta["mean_f"], dtype=np.float64)   # (256, 256) auto-GCaMP
        self.mean_r = np.asarray(meta["mean_r"], dtype=np.float64)   # (256, 256) auto-reflectance
        self.region = meta["region"]   # crop [row0, row1, col0, col1] on the 128 grid
        # volumes: (n_time, crop_rows, crop_cols), raw counts
        self.vol_f = np.load(rec / "pixels_f_gcamp_full.npy")
        self.vol_r = np.load(rec / "pixels_f_emo_full.npy")
        trace_f = self.vol_f.mean(axis=(1, 2), dtype=np.float64)
        trace_r = self.vol_r.mean(axis=(1, 2), dtype=np.float64)
        self.rel_f = (trace_f / trace_f.mean() - 1) * 100   # % of own mean
        self.rel_r = (trace_r / trace_r.mean() - 1) * 100
        self.trace_corr = float(np.corrcoef(trace_f, trace_r)[0, 1])
        self.bp_f, self.bp_r = bandpass(self.mean_f), bandpass(self.mean_r)
        self.image_corr = image_corr(self.bp_f, self.bp_r)
        self.i_frame = min(self.i_frame, self.vol_f.shape[0] - 1)
        self.draw()

    def limits(self, *images: np.ndarray) -> tuple[float, float]:
        low, high = self.args.color_percentiles
        values = np.concatenate([im.ravel() for im in images])
        return float(np.percentile(values, low)), float(np.percentile(values, high))

    def draw(self) -> None:
        rec = self.recordings[self.i_rec]
        rate = self.args.sampling_rate_hz
        for ax in self.axes.ravel():
            ax.clear()

        # --- top row: full-field mean images, as auto labelled them -------------------
        if self.shared_limits:
            lim_f = lim_r = self.limits(self.mean_f, self.mean_r)
        else:
            lim_f, lim_r = self.limits(self.mean_f), self.limits(self.mean_r)
        for ax, image, lim, label in (
                (self.axes[0, 0], self.mean_f, lim_f, "auto: GCaMP"),
                (self.axes[0, 1], self.mean_r, lim_r, "auto: reflectance")):
            ax.imshow(image, cmap="gray", vmin=lim[0], vmax=lim[1])
            ax.set_title(f"{label}  mean (median {np.median(image):.0f})")
            # crop that was dumped, on the 256 half-res field (region is on the 128 grid)
            row0, row1, col0, col1 = (2 * np.asarray(self.region)).tolist()
            ax.add_patch(plt.Rectangle((col0, row0), col1 - col0, row1 - row0,
                                       fill=False, edgecolor="yellow", linewidth=0.8))
        # difference of the normalised band-passed images: vessels present in one only
        diff = self.bp_r / self.bp_r.std() - self.bp_f / self.bp_f.std()
        lim = np.percentile(np.abs(diff), 99)
        self.axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
        self.axes[0, 2].set_title(f"band-passed refl - GCaMP (r = {self.image_corr:.2f})")

        # --- bottom row: one frame of each dumped channel, and the traces -------------
        frame_f, frame_r = self.vol_f[self.i_frame], self.vol_r[self.i_frame]
        if self.shared_limits:
            lim_f = lim_r = self.limits(frame_f, frame_r)
        else:
            lim_f, lim_r = self.limits(frame_f), self.limits(frame_r)
        for ax, frame, lim, label in ((self.axes[1, 0], frame_f, lim_f, "auto: GCaMP"),
                                      (self.axes[1, 1], frame_r, lim_r, "auto: reflectance")):
            ax.imshow(frame, cmap="gray", vmin=lim[0], vmax=lim[1])
            ax.set_title(f"{label}  frame {self.i_frame} ({self.i_frame / rate:.2f} s)")

        time_s = np.arange(self.rel_f.size) / rate
        ax = self.axes[1, 2]
        ax.plot(time_s, self.rel_f, lw=0.6, color="tab:green", label="auto: GCaMP")
        ax.plot(time_s, self.rel_r, lw=0.6, color="tab:red", alpha=0.8, label="auto: reflectance")
        ax.axvline(self.i_frame / rate, color="k", lw=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("crop mean, % of own mean")
        ax.set_title(f"crop-mean traces (r = {self.trace_corr:.2f})")
        ax.legend(loc="upper right", fontsize=8)

        for a in self.axes.ravel()[[0, 1, 2, 3, 4]]:
            a.set_xticks([]); a.set_yticks([])
        shared = "shared" if self.shared_limits else "independent"
        self.fig.suptitle(f"{rec.parent.name}/{rec.name}   [{self.i_rec + 1}/{len(self.recordings)}]"
                          f"   colour limits: {shared}   (n/p rec, arrows frames, c limits)")
        self.fig.canvas.draw_idle()

    def on_key(self, event) -> None:
        rate = int(round(self.args.sampling_rate_hz))
        steps = {"right": 1, "left": -1, "up": rate, "down": -rate,
                 "pageup": 10 * rate, "pagedown": -10 * rate}
        if event.key in steps:
            self.i_frame = int(np.clip(self.i_frame + steps[event.key], 0, self.vol_f.shape[0] - 1))
            self.draw()
        elif event.key in ("n", "p"):
            self.i_rec = (self.i_rec + (1 if event.key == "n" else -1)) % len(self.recordings)
            self.load_recording()
        elif event.key == "c":
            self.shared_limits = not self.shared_limits
            self.draw()
        elif event.key == "q":
            plt.close(self.fig)


if __name__ == "__main__":
    Viewer(build_runtime_args())
