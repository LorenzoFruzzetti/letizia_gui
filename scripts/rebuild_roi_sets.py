"""Regenerate every per-recording ROI set as a pure rescale of the baseline atlas.

Why
---
The sets in ``roi_sets/`` were drawn page by page in roi_editor.py, and the editor's
per-box nudge was used on top of the global Lambda rescale.  Auditing them against the
baseline showed the atlas's structural relations survive everywhere *except* bilateral
mirror symmetry, which is broken in three regions:

    HL       294/356 files, always +1 px
    V1       289/356 files, -2 .. +6 px, drifting monotonically with acquisition date
    M2_alta  230/356 files, always +1 px

There are two distinct causes, and they need different fixes.

1. The uniform +1 px in HL and M2_alta is a ROUNDING BUG in ``scale_boxes``.  Its
   docstring claims round-half-to-even keeps a mirrored pair mirrored because it is
   symmetric about zero.  Symmetry about zero is not the property the mirror needs --
   translation invariance is.  A box of span 5 puts its edge at ``centre*f - 2.5``, so
   mirroring requires ``round(u + 2.5) == round(u - 2.5) + 5``.  Half-to-even fails
   that whenever ``u = |centre| * f`` lands on an integer: round(5.5) = 6 but
   round(0.5) = 0, a gap of 6, so the pair comes out 1 px off-centre.

   No rounding rule fixes this.  The mirror needs oddness AND translation invariance,
   and at a tie the two contradict each other (oddness gives f(-0.5) = -f(0.5),
   translation gives f(-0.5) = f(0.5) - 1, so f(0.5) would have to be 0.5).  Ties-away
   merely moves the failure to pairs that straddle zero.  ``roi_editor.scale_boxes``
   therefore REFLECTS: a box left of Bregma is computed as the negated mirror of the
   same box on the right, so the pair is symmetric by construction at every factor.
   Newly drawn sets no longer acquire this error, and the rebuild only repairs old ones.

2. V1's -2..+6 px error is too large for that and drifts monotonically with acquisition
   date, so it is a genuine one-hemisphere hand nudge.  Because ``c`` copies the
   previous page's layout forward, a nudge propagates to every later session and
   accumulates.  A left-vs-right V1 contrast on the current sets is confounded, and
   since the drift tracks date it is confounded with group if groups were run in date
   blocks.  The editor's mirror lock (``roi_editor.ROIEditor._mirror_to_twin``, on by
   default) is what stops this one recurring: an edit to either box of a pair rewrites
   the other as its exact mirror.

Bilateral row alignment, box sizes, antero-posterior and medio-lateral ordering,
co-alignment equalities, disjointness and hemisphere sidedness are clean in all 356
files, so nothing else needs repairing.

This script throws the nudges away and rebuilds each set as the baseline constellation
scaled by that recording's own Bregma->Lambda distance, which is exactly the geometry
the editor would have produced with the rescale knob alone.  Mirror symmetry is then
exact by construction.

The reference Lambda is 55, not the 30 written in cortex22_roi_set.yaml
--------------------------------------------------------------------
``_seed_layout`` hands the editor ``args.lambda_offset`` (RUN_CONFIG["lambda_offset"],
55) when a page is seeded from the profile atlas -- it never reads the atlas file's own
``lambda_row_offset``.  Every existing set was therefore built treating CORTEX_22 as if
drawn at 55 rows.  Measured: at reference 55 the pure rescale reproduces the drawn
boxes to a median 1.5 px and reproduces 2 files exactly; at 30 it is off by 23 px.
Using 55 keeps the rebuilt ROIs the same size as the ones behind existing results, so
this is a cleanup and not a change of scale.  The `30` in the baseline file is
inconsistent with the pipeline and is flagged in CLAUDE.md section 9.

Header normalisation
--------------------
t1..t5 of one <day>_<animal> are one animal in one session, so their geometry must be
identical.  Four sessions disagreed (260611_PV5 t5 jumped 30 Bregma rows; 260611_T4 t1
missed a rescale; 260708_PV5 and 260716_PV5 t4/t5 sat 2 Bregma rows off).  Each animal
is rebuilt from the Bregma and Lambda held by the majority of its own recordings.

Run it:
    conda run --no-capture-output -n wfci_gui python rebuild_roi_sets.py
    conda run --no-capture-output -n wfci_gui python rebuild_roi_sets.py --out-dir roi_sets/rebuilt
    conda run --no-capture-output -n wfci_gui python rebuild_roi_sets.py --dry-run
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
    # Directory holding the hand-drawn sets to rebuild.
    "roi_set_dir": r"roi_sets",
    # The constellation every set is regenerated from.
    "baseline": r"atlas/cortex22_roi_set.yaml",
    # Where the regenerated sets are written. Same <key>.yaml names as the input, so
    # extract_cohort.py can be pointed here with --roi-set-dir.
    "out_dir": r"roi_sets/rebuilt",
    # Bregma->Lambda distance the BASELINE boxes are treated as being drawn at.
    # 55 = RUN_CONFIG["lambda_offset"] in batch_roi_select.py, the value the existing
    # sets were actually built with. See the module docstring before changing this.
    "reference_lambda": 55,
    # True: give every recording of one animal the Bregma/Lambda held by the majority
    # of that animal's own files, so t1..t5 come out identical.
    "normalise_headers": True,
    # False (matches batch_roi_select.py): rescaling moves box CENTRES but keeps each
    # box's pixel area fixed, so ROI noise stays comparable across animals.
    "lambda_scales_box_size": False,
    # True: report what would be written without writing anything.
    "dry_run": False,
    "prefer_cli_args": True,
}

# t1..t5 of one <day>_<animal> are the same animal in the same session. The animal is
# everything between the day and the LAST "_t#": the 2024 names contain underscores
# (240108_SP20_000_t1) and some have no digits (261012_Nan_t1); the old pattern
# <letters><digits> left both ungrouped, i.e. without t1..t5 header normalisation.
KEY_RE = re.compile(r"(?P<animal>\d{6}_.+)_t(?P<t>\d+)")

BOX_FIELDS = ("row_start", "row_end", "col_start", "col_end")


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild ROI sets as a pure Lambda rescale of the baseline atlas.")
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
        # Per-recording keys the ROI editor appends (roi_editor_spec.md D15/D17): the
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


def main() -> None:
    args = build_runtime_args()

    roi_dir = Path(args.roi_set_dir)
    baseline_path = Path(args.baseline)
    out_dir = Path(args.out_dir)

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
            # The grid the set was DRAWN on, not the baseline's: the 2022 cohort's
            # rotated frames are 152x128. Identical for every 128x128 cohort.
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
