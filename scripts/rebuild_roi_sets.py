"""Regenerate every per-recording ROI set as a pure rescale of the baseline atlas.

Why
---
Sets drawn page by page in the ROI editor can carry per-box nudges on top of the global
Lambda rescale. The atlas's structural relations survive such editing everywhere
*except* bilateral mirror symmetry, which two distinct causes break:

1. A ROUNDING tie. Round-half-to-even is symmetric about zero, but symmetry about zero
   is not the property the mirror needs -- translation invariance is.  A box of span 5
   puts its edge at ``centre*f - 2.5``, so mirroring requires
   ``round(u + 2.5) == round(u - 2.5) + 5``.  Half-to-even fails that whenever
   ``u = |centre| * f`` lands on an integer: round(5.5) = 6 but round(0.5) = 0, a gap
   of 6, so the pair comes out 1 px off-centre.

   No rounding rule fixes this.  The mirror needs oddness AND translation invariance,
   and at a tie the two contradict each other (oddness gives f(-0.5) = -f(0.5),
   translation gives f(-0.5) = f(0.5) - 1, so f(0.5) would have to be 0.5).  Ties-away
   merely moves the failure to pairs that straddle zero.  ``roi_editor.scale_boxes``
   therefore REFLECTS: a box left of Bregma is computed as the negated mirror of the
   same box on the right, so the pair is symmetric by construction at every factor.

2. A one-hemisphere hand nudge.  Because ``c`` copies the previous page's layout
   forward, a nudge propagates to every later session and accumulates, so a
   left-vs-right contrast becomes confounded with acquisition date (and with group,
   if groups were recorded in date blocks).  The editor's mirror lock
   (``roi_editor.ROIEditor._mirror_to_twin``, on by default) stops this one: an edit
   to either box of a pair rewrites the other as its exact mirror.

This script throws the nudges away and rebuilds each set as the baseline constellation
scaled by that recording's own Bregma->Lambda distance, which is exactly the geometry
the editor would have produced with the rescale knob alone.  Mirror symmetry is then
exact by construction.

The reference Lambda is 55, not the 30 written in cortex22_roi_set.yaml
--------------------------------------------------------------------
The editor hands ``args.lambda_offset`` (55) to a page seeded from the profile atlas
-- it never reads the atlas file's own ``lambda_row_offset``.  ROI sets are therefore
built treating CORTEX_22 as if drawn at 55 rows, and the rebuild must use the same
reference or it would change the ROI size.  The `30` in the baseline file is not the
scale reference; ``wfci.load_atlas`` ignores it.

Header normalisation
--------------------
t1..t5 of one <day>_<animal> are one animal in one session, so their geometry must be
identical.  A slip on one page (a jumped Bregma, a missed rescale) would break that, so
each animal is rebuilt from the Bregma and Lambda held by the majority of its own
recordings.

Run it:
    conda run --no-capture-output -n wfci_gui python rebuild_roi_sets.py --cohort <cohort folder>/cohort.yaml
    conda run --no-capture-output -n wfci_gui python rebuild_roi_sets.py --cohort <...>/cohort.yaml --dry-run
    conda run --no-capture-output -n wfci_gui python rebuild_roi_sets.py --roi-set-dir <drawn> --out-dir <rebuilt>
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from channel_rectangle import (
    append_roi_set_channel_order,
    append_roi_set_rectangle,
    read_roi_set_channel_order,
    read_roi_set_rectangle,
)
from roi_editor import (
    Box,
    flip_rows_about_bregma,
    mirror_twin,
    normalise_angle,
    rotate_boxes,
    save_roi_set,
    scale_boxes,
)
from wfci import load_atlas

# ---------------------------------------------------------------------------
# Edit this section to run the rebuild without passing CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # The cohort file, <cohort folder>/cohort.yaml (or pass --cohort). It gives the two
    # folders below when they are None: the hand-drawn sets (roi_sets/reviewed) and
    # where the regenerated ones go (roi_sets/rebuilt, which extract_cohort.py reads).
    "cohort_file": None,
    # Directory holding the hand-drawn sets to rebuild; None = the cohort's roi_sets/reviewed.
    # The GUI and the ROI editor pass a copy of only the sessions that changed.
    "roi_set_dir": None,
    # The constellation every set is regenerated from.
    "baseline": r"atlas/cortex22_roi_set.yaml",
    # Where the regenerated sets are written (same <key>.yaml names as the input);
    # None = the cohort's roi_sets/rebuilt. A folder of sets outside any cohort can be
    # rebuilt by giving both folders and no cohort.
    "out_dir": None,
    # Bregma->Lambda distance the BASELINE boxes are treated as being drawn at.
    # 55 = the ROI editor's lambda_offset, the scale reference ROI sets are built with.
    # See the module docstring before changing this.
    "reference_lambda": 55,
    # True: give every recording of one animal the Bregma/Lambda held by the majority
    # of that animal's own files, so t1..t5 come out identical.
    "normalise_headers": True,
    # False (matches the ROI editor): rescaling moves box CENTRES but keeps each
    # box's pixel area fixed, so ROI noise stays comparable across animals.
    "lambda_scales_box_size": False,
    # True: report what would be written without writing anything.
    "dry_run": False,
    "prefer_cli_args": True,
}

# t1..t5 of one <day>_<animal> are the same animal in the same session. The animal is
# everything between the day and the LAST "_t#": animal names may contain underscores
# (200101_A1_000_t1) or no digits (200101_Nan_t1); a <letters><digits> pattern would
# leave both ungrouped, i.e. without t1..t5 header normalisation.
KEY_RE = re.compile(r"(?P<animal>\d{6}_.+)_t(?P<t>\d+)")

BOX_FIELDS = ("row_start", "row_end", "col_start", "col_end")


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild ROI sets as a pure Lambda rescale of the baseline atlas.")
    p.add_argument("--cohort", dest="cohort_file", default=defaults["cohort_file"],
                   help="The cohort file; gives --roi-set-dir and --out-dir when they are not set.")
    p.add_argument("--roi-set-dir", default=defaults["roi_set_dir"])
    p.add_argument("--baseline", default=defaults["baseline"])
    p.add_argument("--out-dir", default=defaults["out_dir"])
    p.add_argument("--reference-lambda", type=int, default=defaults["reference_lambda"])
    p.add_argument("--no-normalise-headers", dest="normalise_headers",
                   action="store_false", default=defaults["normalise_headers"],
                   help="Rebuild each file from its own Bregma/Lambda instead of the "
                        "majority held by its animal.")
    p.add_argument("--dry-run", action="store_true", default=defaults["dry_run"])
    ns = p.parse_args()
    ns.lambda_scales_box_size = bool(defaults["lambda_scales_box_size"])
    return ns


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        cohort_file=config["cohort_file"],
        roi_set_dir=config["roi_set_dir"],
        baseline=config["baseline"],
        out_dir=config["out_dir"],
        reference_lambda=config["reference_lambda"],
        normalise_headers=bool(config["normalise_headers"]),
        lambda_scales_box_size=bool(config["lambda_scales_box_size"]),
        dry_run=bool(config["dry_run"]),
    )


def scale_boxes_mirrored(boxes: dict[str, Box], factor: float,
                         scale_size: bool = False) -> dict[str, Box]:
    """The editor's rescale, kept under the name this script's guarantee is stated in.

    It used to be a corrected COPY of ``roi_editor.scale_boxes``, which rounded
    half-to-even and so could put a mirrored pair 1 px off-centre.  The editor now
    rounds with :func:`roi_editor.round_half_away` itself, so the two are the same
    function and this only forwards -- the per-file assertion in :func:`main` still
    checks the result, so a regression there cannot pass unnoticed.
    """
    return scale_boxes(boxes, factor, scale_size)


def read_header(path: Path) -> dict[str, Any]:
    """Bregma / Lambda / grid / downsample of one ROI set, as written."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "bregma_row": int(data["bregma_row"]),
        "bregma_col": int(data["bregma_col"]),
        "lambda_row_offset": int(data["lambda_row_offset"]),
        "grid": tuple(data["grid"]),
        "downsample": float(data["downsample"]),
        "boxes": data["boxes"],
        # Drawn with Lambda ABOVE Bregma (roi_editor 'u'): the rebuild must come out
        # in the same orientation, or it silently un-flips the layout.
        "lambda_above_bregma": bool(data.get("lambda_above_bregma", False)),
        # Rotation of Lambda + constellation around Bregma (roi_editor z/x or a Lambda
        # drag). The rebuild is done unrotated and then turned by this, for the same
        # reason as the flip: dropping it would silently un-rotate the layout.
        "lambda_angle_deg": normalise_angle(data.get("lambda_angle_deg", 0.0)),
        # Per-recording keys the ROI editor appends: the
        # recording's own channel rectangle and the user's channel_order. Copied to the
        # rebuilt file unchanged -- no majority vote, they belong to this recording.
        "channel_rectangle": read_roi_set_rectangle(path),
        "channel_order": read_roi_set_channel_order(path),
    }


def unrotated_box_dicts(boxes: dict[str, Any], degrees: float) -> dict[str, Any]:
    """A saved (rotated) layout's boxes turned back to 0 degrees, as plain dicts, so
    the mirror error is measured about the layout's own midline."""
    if not degrees:
        return boxes
    as_boxes = {label: Box(**{f: int(b[f]) for f in BOX_FIELDS}) for label, b in boxes.items()}
    return {label: {f: int(getattr(b, f)) for f in BOX_FIELDS}
            for label, b in rotate_boxes(as_boxes, -degrees).items()}


def mirror_error(boxes: dict[str, Any], pairs: list[tuple[str, str, str]]) -> int:
    """Total absolute bilateral-mirror error of a layout, in pixels.

    Zero means every L box is the exact mirror of its R twin about Bregma. Used only
    to report what the rebuild removed.
    """
    total = 0
    for _, left, right in pairs:
        lc = (boxes[left]["col_start"] + boxes[left]["col_end"]) / 2
        rc = (boxes[right]["col_start"] + boxes[right]["col_end"]) / 2
        total += abs(lc + rc)
    return int(total)


def bilateral_pairs(labels: list[str]) -> list[tuple[str, str, str]]:
    """Pair each ``*L*`` label with its ``*R*`` twin: ('M2_alta', 'M2L_alta', 'M2R_alta').

    ``roi_editor.mirror_twin`` finds the side letter; this only keeps the L-side
    listing so each pair is reported once, under a side-less region name.
    """
    pairs = []
    for label in labels:
        twin = mirror_twin(label, labels)
        if twin is None:
            continue
        i = next(k for k, (a, b) in enumerate(zip(label, twin)) if a != b)
        if label[i] == "L":
            pairs.append((label[:i] + label[i + 1:], label, twin))
    return pairs


def resolve_folders(args: argparse.Namespace) -> tuple[Path, Path]:
    """(sets to rebuild, where the rebuilt sets go): the given folders, else the cohort's
    roi_sets/reviewed and roi_sets/rebuilt. Without both folders it needs the cohort file
    ("set cohort_file or pass --cohort")."""
    if args.roi_set_dir is not None and args.out_dir is not None:
        return Path(args.roi_set_dir), Path(args.out_dir)
    from wfci_gui.logic.cohort import load_script_cohort
    layout = load_script_cohort(args.cohort_file).layout
    return (Path(args.roi_set_dir) if args.roi_set_dir is not None else layout.reviewed_roi_sets,
            Path(args.out_dir) if args.out_dir is not None else layout.rebuilt_roi_sets)


def main() -> None:
    args = build_runtime_args()

    roi_dir, out_dir = resolve_folders(args)
    # A relative baseline (the default) is relative to the code folder, not to the
    # folder the job runs in.
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = Path(__file__).resolve().parents[1] / baseline_path

    # load_atlas gives the baseline the library's own validation (field names, types,
    # the grid check) instead of trusting the YAML blindly.
    base_atlas = load_atlas(baseline_path)
    base_boxes = dict(base_atlas)
    labels = list(base_boxes)
    pairs = bilateral_pairs(labels)

    base_raw = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    grid = tuple(base_raw["grid"])
    downsample = float(base_raw["downsample"])

    if mirror_error(base_raw["boxes"], pairs) != 0:
        raise ValueError(
            f"{baseline_path} is not bilaterally symmetric -- rebuilding from it "
            f"cannot restore mirror symmetry. Fix the baseline first."
        )

    # ---- collect the inputs, grouped by animal ----------------------------
    sources = sorted(p for p in roi_dir.glob("*.yaml")
                     if p.resolve() != baseline_path.resolve())
    if not sources:
        raise FileNotFoundError(f"No ROI sets to rebuild in {roi_dir}.")

    headers = {p.stem: read_header(p) for p in sources}
    animals: dict[str, list[str]] = defaultdict(list)
    ungrouped: list[str] = []
    for key in headers:
        m = KEY_RE.fullmatch(key)
        (animals[m.group("animal")].append(key) if m else ungrouped.append(key))
    for keys in animals.values():
        keys.sort(key=lambda s: int(s.rsplit("_t", 1)[1]))

    print(f"Baseline  : {baseline_path}  ({len(labels)} boxes, grid {grid[0]}x{grid[1]})")
    print(f"Reference : Bregma->Lambda = {args.reference_lambda} rows "
          f"(the scale the baseline boxes are treated as being drawn at)")
    print(f"Input     : {len(sources)} ROI sets in {roi_dir} "
          f"({len(animals)} animal-sessions"
          + (f" + {len(ungrouped)} ungrouped" if ungrouped else "") + ")")
    print(f"Output    : {out_dir}"
          + ("   [DRY RUN -- nothing will be written]" if args.dry_run else ""))
    print(f"Box size  : {'scales with Lambda' if args.lambda_scales_box_size else 'fixed'}")

    # ---- decide the Bregma/Lambda each recording is rebuilt at -------------
    # One animal is one session: its five recordings must share a geometry. Where they
    # disagree the majority wins -- a lone dissenting file is an editing slip, not a
    # measurement, and the majority is what the other four recordings agree on.
    # The orientation is part of the header tuple, so a lone file saved the other way
    # up is outvoted like a lone Bregma slip.
    # The rotation angle is voted with the rest, so one page left at a stray angle is
    # outvoted too.
    chosen: dict[str, tuple[int, int, int, bool, float]] = {}
    normalised: list[tuple[str, str, tuple, tuple]] = []
    for animal, keys in sorted(animals.items()):
        own = {k: (headers[k]["bregma_row"], headers[k]["bregma_col"],
                   headers[k]["lambda_row_offset"], headers[k]["lambda_above_bregma"],
                   headers[k]["lambda_angle_deg"])
               for k in keys}
        if args.normalise_headers:
            winner = Counter(own.values()).most_common(1)[0][0]
            for k in keys:
                if own[k] != winner:
                    normalised.append((animal, k, own[k], winner))
                chosen[k] = winner
        else:
            chosen.update(own)
    for key in ungrouped:
        h = headers[key]
        chosen[key] = (h["bregma_row"], h["bregma_col"], h["lambda_row_offset"],
                       h["lambda_above_bregma"], h["lambda_angle_deg"])

    if normalised:
        print(f"\nHeader normalisation -- {len(normalised)} file(s) moved onto their "
              f"session's majority:")
        for animal, key, was, now in normalised:
            bits = []
            if was[:2] != now[:2]:
                bits.append(f"bregma ({was[0]},{was[1]}) -> ({now[0]},{now[1]})")
            if was[2] != now[2]:
                bits.append(f"lambda {was[2]} -> {now[2]}")
            if was[3] != now[3]:
                bits.append("Lambda side -> " + ("ABOVE" if now[3] else "below") + " Bregma")
            if was[4] != now[4]:
                bits.append(f"rotation {was[4]:+g} -> {now[4]:+g} deg")
            print(f"  {key:<22} {'; '.join(bits)}")

    # ---- rebuild ----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_mirror = 0
    total_mirror_px = 0
    changed = 0
    written: list[Path] = []
    carried = {"channel_rectangle": 0, "channel_order": 0}
    per_animal_scale: dict[str, float] = {}

    for key in sorted(chosen):
        bregma_row, bregma_col, lam, above, angle = chosen[key]
        factor = lam / args.reference_lambda
        boxes = scale_boxes_mirrored(base_boxes, factor, args.lambda_scales_box_size)
        if above:
            # The baseline is drawn with Lambda below Bregma; mirror it about Bregma's
            # row exactly as the editor's 'u' did. Columns untouched, so still mirrored L/R.
            boxes = flip_rows_about_bregma(boxes)
        # The mirror check runs on this unrotated layout, where it is exact; rotation
        # comes last and can put a pair up to 1 px off the rotated midline (rounding).
        unrotated = {l: {f: int(getattr(b, f)) for f in BOX_FIELDS}
                     for l, b in boxes.items()}
        if angle:
            boxes = rotate_boxes(boxes, angle)

        # Everything below is reporting on what the rebuild changed.
        as_dict = {l: {f: int(getattr(b, f)) for f in BOX_FIELDS}
                   for l, b in boxes.items()}
        err = mirror_error(unrotated_box_dicts(headers[key]["boxes"],
                                               headers[key]["lambda_angle_deg"]), pairs)
        if err:
            fixed_mirror += 1
            total_mirror_px += err
        if as_dict != {l: {f: int(headers[key]["boxes"][l][f]) for f in BOX_FIELDS}
                       for l in labels}:
            changed += 1
        m = KEY_RE.fullmatch(key)
        per_animal_scale[m.group("animal") if m else key] = factor

        if mirror_error(unrotated, pairs) != 0:
            raise AssertionError(f"{key}: rebuilt layout is still not mirrored.")

        for name in carried:
            carried[name] += headers[key][name] is not None
        if args.dry_run:
            continue
        written.append(save_roi_set(
            out_dir / f"{key}.yaml",
            boxes,
            bregma_row,
            bregma_col,
            # The grid the set was DRAWN on, not the baseline's: rotated frames can
            # be 152x128. Identical for every 128x128 cohort.
            headers[key]["grid"],
            key,
            f"rebuilt by rebuild_roi_sets.py: {baseline_path.name} scaled by "
            f"lambda {lam}/{args.reference_lambda} = {factor:.4f}; bilaterally "
            f"symmetric by construction"
            + ("; mirrored up-down (Lambda above Bregma)" if above else "")
            + (f"; rotated {angle:+g} deg around Bregma" if angle else ""),
            lambda_offset=lam,
            downsample=downsample,
            lambda_above_bregma=above,
            lambda_angle_deg=angle,
        ))
        # Appended after save_roi_set, as the ROI editor does: every set without the keys
        # is byte-identical to what this script wrote before they existed.
        if headers[key]["channel_rectangle"] is not None:
            append_roi_set_rectangle(written[-1], headers[key]["channel_rectangle"])
        if headers[key]["channel_order"] is not None:
            append_roi_set_channel_order(written[-1], headers[key]["channel_order"])

    scales = sorted(per_animal_scale.values())
    print(f"\nRebuilt   : {len(chosen)} ROI sets")
    print(f"  scale vs baseline : {scales[0]:.3f}x .. {scales[-1]:.3f}x "
          f"(median {scales[len(scales)//2]:.3f}x)")
    print(f"  geometry changed  : {changed}/{len(chosen)} files")
    n_above = sum(1 for v in chosen.values() if v[3])
    if n_above:
        print(f"  orientation       : {n_above}/{len(chosen)} with Lambda ABOVE Bregma "
              f"(rebuilt mirrored up-down)")
    print(f"  mirror symmetry   : repaired in {fixed_mirror} file(s), "
          f"{total_mirror_px} px of asymmetry removed in total")
    print(f"  every rebuilt set is bilaterally symmetric (verified per file)")
    print(f"  carried per file  : {carried['channel_rectangle']} with their own channel_rectangle, "
          f"{carried['channel_order']} with a chosen channel_order")
    if args.dry_run:
        print("\nDRY RUN -- no files written.")
    else:
        print(f"\nWrote {len(written)} files to {out_dir}")
        print(f"\nUse them:\n"
              f'  conda run -n wfci_gui python extract_cohort.py --roi-set-dir {out_dir} --full')


if __name__ == "__main__":
    main()
