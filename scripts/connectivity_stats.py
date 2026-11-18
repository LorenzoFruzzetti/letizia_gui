"""One set of animals: does connectivity change after the baseline? Statistics on the detrended variant.

Input: per-recording Pearson R of `variant` (default the 20 s running-median baseline
dF/F) from the cohort's outputs/roi_pixel_connectivity, and its design.csv.

What is analysed is chosen per run, never by editing this script (any cohort works as it
is): the baseline timepoint, the included timepoints, the animals, the variant and the
region focus come from flags, else from CONNECTIVITY_STATS_* environment variables, else
from RUN_CONFIG (see `choice`). The folders come from the cohort file (--cohort).
Excluded recordings are the cohort file's `exclusions:`.

Time axis (`time_axis`, env CONNECTIVITY_STATS_TIME_AXIS):
    "timepoint" (default): sessions are grouped by the `timepoint` class of
        experimental_design.csv (days since the animal's day 1). The first class in
        `timepoints` is the baseline; a class that not every animal has is left out of
        `timepoints`. Contrasts are named after the class ("3D", "15D", ...).
    "day_index": the within-animal session number, days `baseline_day_index` ..
        `last_day_index`.

Everything is tested on Fisher z = arctanh(r). Session value = mean z of its recordings.
The animal is the unit of inference; recordings only enter through the mixed model,
where they are nested in session (they are not independent replicates of a day).

Tests, each for time point d vs baseline and for "post" = mean of the later points vs baseline:
    summary metrics (global, intra-left, intra-right, homotopic, heterotopic inter):
        - linear mixed model on recordings: z ~ C(day_index), random intercept per
          animal + variance component per session (Wald tests; omnibus over all days)
        - paired t-test on the animal session means (n = animals)
        - Friedman test across all days (animal session means)
    node strength (each ROI's mean z with the other ROIs): same mixed model and paired
        t-test, Benjamini-Hochberg across ROIs
    edges: paired t-test on animal session means, Benjamini-Hochberg across edges; and NBS
        (one-sample t on each animal's change, exact sign flips of whole animals, increase
        and decrease separately) -> nbs_components.csv, all_to_all_change_nbs.png/.svg and
        <focus>_to_all_change_nbs.png/.svg (dot = edge of a significant component)

Region focus (--focus / env CONNECTIVITY_STATS_FOCUS, "SSp" default or "motor"): the same tests on the region's
own edges and its edges to every other region (motor also split by its hemisphere),
written as <focus>_*.csv/.png next to the files below.

Output (<cohort folder>/outputs/connectivity_stats/<variant>_<animals tag>_<range tag>/,
e.g. _no_A3_BL-30D; the tag is all_animals when every animal of the design is included):
    session_values.csv, metric_tests.csv, node_tests.csv, edge_tests.csv,
    metrics_by_day.png, node_strength_change.png, edge_tests.png/.svg (z, t and r change),
    mean_matrices.png/.svg and mean_matrices_uncentered.png/.svg (parula, row min/max),
    <focus>_metrics_by_day_relative_r.png/.svg (paired/LMM p and uncorrected stars),
    SSp_x_motor_combined_change_r.png/.svg; motor focus also writes
    motor_x_regions_combined_change_r.png/.svg and motor_x_regions_combined_tests.csv,
    metrics_by_day_relative_r.png/.svg (mean +- SD across animals, relative and absolute r)

Run (C is <cohort folder>/cohort.yaml):
    conda run --no-capture-output -n wfci_gui python connectivity_stats.py --cohort C
    ... connectivity_stats.py --cohort C --baseline BL --timepoints BL 3D 15D --animals A1 A2 A3 \
        --variant pixels_median_dff_20s_full_gsr --focus motor
Environment variables instead of flags: CONNECTIVITY_STATS_COHORT_FILE, _BASELINE, _TIMEPOINTS
and _ANIMALS (comma-separated lists), _VARIANT, _FOCUS. The GUI runs it through
run_with_overrides.py (module-level parameters as --set name=value, flags after --).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from nbs import nbs
from parula_colormap import parula
from wfci_gui.logic.cohort import load_script_cohort

# ---------------------------------------------------------------- what to analyse
# Edit this section to run from the editor without flags or environment variables.
RUN_CONFIG = {
    "cohort_file": None,      # <cohort folder>/cohort.yaml (or pass --cohort)
    "baseline": None,         # baseline timepoint; None = the first included timepoint
    "timepoints": None,       # included timepoint classes; None = every timepoint of the cohort
    "animals": None,          # included animals; None = every animal of the design
    # 20 s running-median baseline dF/F, no GSR; any other roi_pixel_connectivity folder
    # (e.g. pixels_median_dff_20s_full_gsr) reruns everything on that variant.
    "variant": "pixels_median_dff_20s_full",
    "focus": "SSp",           # region focus: "SSp" or "motor"
}
_parser = argparse.ArgumentParser(description="Connectivity statistics of one cohort over time.")
_parser.add_argument("--cohort", dest="cohort_file", default=None,
                     help="The cohort file, <cohort folder>/cohort.yaml.")
_parser.add_argument("--baseline", default=None, help="Baseline timepoint (default: the first included).")
_parser.add_argument("--timepoints", nargs="*", default=None, help="Included timepoints (default: all).")
_parser.add_argument("--animals", nargs="*", default=None, help="Included animals (default: all).")
_parser.add_argument("--variant", default=None, help="roi_pixel_connectivity variant folder.")
_parser.add_argument("--focus", default=None, help="Region focus: SSp or motor.")
_cli = _parser.parse_args()


def choice(name, is_list=False):
    """One run choice: its flag when given, else env CONNECTIVITY_STATS_<NAME> (a list is
    comma-separated), else RUN_CONFIG."""
    value = getattr(_cli, name)
    if value is not None:
        return value
    env_value = os.environ.get(f"CONNECTIVITY_STATS_{name.upper()}", "").strip()
    if env_value:
        return [item.strip() for item in env_value.split(",") if item.strip()] if is_list else env_value
    return RUN_CONFIG[name]


# Every folder comes from the cohort file; a run without one stops here.
cohort = load_script_cohort(choice("cohort_file"))
design_path = cohort.layout.design_csv
connectivity_root = cohort.layout.connectivity
output_root = cohort.layout.connectivity_stats
exclude_recordings = sorted(cohort.excluded_keys())  # "<day>_<animal>/<t#>" keys
included_timepoints = list(choice("timepoints", is_list=True) or cohort.timepoints)
baseline_timepoint = choice("baseline") or included_timepoints[0]
if baseline_timepoint not in included_timepoints:
    raise ValueError(f"the baseline {baseline_timepoint!r} is not one of the included timepoints "
                     f"{included_timepoints}")
timepoints = [baseline_timepoint] + [t for t in included_timepoints if t != baseline_timepoint]  # first = baseline
include_animals = choice("animals", is_list=True) or None   # None = every animal of the design
variant = choice("variant")
focus_prefix = choice("focus")

# ---------------------------------------------------------------- parameters
connectivity_file = "roi_pixel_connectivity.npz"   # per recording: R_roi, roi_labels
exclude_days = []                        # session days dropped before anything else
default_time_axis = "timepoint"
default_relative_mode = None             # None: ratio without GSR, difference with GSR
# "timepoint": compare the timepoint classes (BL, 3D, ...); "day_index": the session number.
time_axis = os.environ.get("CONNECTIVITY_STATS_TIME_AXIS", default_time_axis)
baseline_day_index = 1                   # time_axis "day_index" only
# time_axis "day_index" only; the last session number that every animal has.
last_day_index = 4
# How the relative-r figures show a later day (env CONNECTIVITY_STATS_RELATIVE_MODE): "ratio" = r / day 1,
# "difference" = r - day 1. None: ratio without GSR, difference with GSR (day-1 r ~ 0 after GSR).
relative_mode = os.environ.get("CONNECTIVITY_STATS_RELATIVE_MODE", default_relative_mode)
fdr_alpha = 0.05                         # Benjamini-Hochberg level
uncorrected_alpha = 0.05                 # marks edges in the figure
color_limit_percentile = 99.0            # centred colour scale
fixed_color_limits = (-1.0, 1.0)         # mean_matrices_fixed.*: both rows (r and difference in r)
# Colormap of mean_matrices_uncentered.* (each row its own [min, max]): RdBu_r
# (blue = row min, red = row max). Not centred, so white is the row's
# mid-range, not 0.
uncentered_colormap = "RdBu_r"
nbs_primary_threshold = 2.5              # NBS edge cut-off on the paired t (df = n_animals - 1)
nbs_alpha = 0.05                         # NBS component FWER level
nbs_permutations = 5000                  # upper bound; the 2^n_animals sign flips are enumerated when fewer
# Our ROI label -> the anatomical label shown in tables and figures (side is the _L/_R suffix).
display_label_of = {
    "M2L_alta": "MOs-a_L", "M2L_bassa": "MOs-p_L", "M1L_alta": "MOp-a_L", "M1L_bassa": "MOp-p_L",
    "BFDL": "SSp-bfd_L", "TrL": "SSp-tr_L", "FLL": "SSp-fL_L", "HLL": "SSp-hl_L",
    "RSL_alta": "RSP_L", "V1aL": "VISa_L", "V1L": "VISp_L",
    "M2R_alta": "MOs-a_R", "M2R_bassa": "MOs-p_R", "M1R_alta": "MOp-a_R", "M1R_bassa": "MOp-p_R",
    "BFDR": "SSp-bfd_R", "TrR": "SSp-tr_R", "FLR": "SSp-fL_R", "HLR": "SSp-hl_R",
    "RSR_alta": "RSP_R", "V1aR": "VISa_R", "V1R": "VISp_R",
}

# Every session gets an integer time index (`days`, first = baseline) that the tests and
# plots use; day_label names it in tables and figures ("BL", "3D", ... or "day1", ...).
if time_axis == "timepoint":
    days = list(range(1, len(timepoints) + 1))
    day_label = dict(zip(days, timepoints))
    range_tag = f"{timepoints[0]}-{timepoints[-1]}"
    time_axis_label = "timepoint (days since day 1)"
elif time_axis == "day_index":
    days = list(range(baseline_day_index, last_day_index + 1))
    day_label = {day: f"day{day}" for day in days}
    range_tag = f"days{days[0]}-{days[-1]}"
    time_axis_label = "day_index"
else:
    raise ValueError(f"time_axis must be 'timepoint' or 'day_index', not {time_axis!r}")
baseline_day_index = days[0]
baseline_label = day_label[baseline_day_index]
later_days = days[1:]
contrast_of = {day: day_label[day] for day in later_days}   # contrast name of a later point
day_of_contrast = {name: day for day, name in contrast_of.items()}
contrasts = list(contrast_of.values()) + ["post"]   # each vs baseline

# ---------------------------------------------------------------- loading
# keep_default_na=False: a group may be the literal string "NA" (no group
# assigned yet), which pandas would otherwise read as NaN. Nothing here depends on a CSV
# cell being NaN rather than "" (the two notna() calls downstream are on computed columns).
design = pd.read_csv(design_path, dtype={"day": str}, keep_default_na=False)
design["session"] = design["day"] + "_" + design["animal"]
design["recording_key"] = design["session"] + "/" + design["recording"]
if time_axis == "timepoint":
    # day_index becomes the timepoint's position in `timepoints`; other classes (and R/T
    # rows, whose timepoint is blank) map to NaN and are dropped by the filter below.
    design["day_index"] = design["timepoint"].map({label: day for day, label in day_label.items()})
# Animals left out of the run name the output folder ("no_<animals>"; "all_animals" when none).
left_out_animals = (sorted(set(design["animal"]) - set(include_animals))
                    if include_animals is not None else [])
if include_animals is not None:
    unknown_animals = sorted(set(include_animals) - set(design["animal"]))
    if unknown_animals:
        raise ValueError(f"animals not in {design_path.name}: {unknown_animals}")
    design = design[design["animal"].isin(include_animals)]
design = design[~design["day"].isin(exclude_days)
                & ~design["recording_key"].isin(exclude_recordings)
                & design["day_index"].isin(days)].reset_index(drop=True)
design["day_index"] = design["day_index"].astype(int)
# One session per animal and time point, or the session means would mix two days.
sessions_per_point = design.groupby(["animal", "day_index"])["day"].nunique()
if (sessions_per_point > 1).any():
    raise ValueError(f"several sessions share an animal and time point: "
                     f"{sessions_per_point[sessions_per_point > 1].to_dict()}")
animals = sorted(design["animal"].unique())
missing = [(a, d) for a in animals for d in days
           if not ((design["animal"] == a) & (design["day_index"] == d)).any()]
if missing:
    raise ValueError(f"incomplete design (animal, day) missing: {missing}")
animals_tag = f"no_{'_'.join(left_out_animals)}" if left_out_animals else "all_animals"
output_dir = output_root / f"{variant}_{animals_tag}_{range_tag}"
output_dir.mkdir(parents=True, exist_ok=True)

# recording_z: (n_recordings, n_rois, n_rois) Fisher z of R, diagonal set to 0.
roi_labels = None
recording_z = []
for row in design.itertuples():
    data = np.load(connectivity_root / variant / row.session / row.recording / connectivity_file)
    labels = list(data["roi_labels"])
    if roi_labels is None:
        roi_labels = labels
    if labels != roi_labels:
        raise ValueError(f"ROI labels differ in {row.recording_key}")
    r = data["R_roi"].astype(np.float64)
    np.fill_diagonal(r, 0.0)
    recording_z.append(np.arctanh(r))
recording_z = np.array(recording_z)
n_rois = len(roi_labels)
tick_labels = [display_label_of[label] for label in roi_labels]
print(f"{variant}: {len(design)} recordings, animals {animals}, time points {[day_label[d] for d in days]}")

# ---------------------------------------------------------------- summary metrics
# Edge masks over the upper triangle; side from the display label suffix.
side = np.array([label[-1] for label in tick_labels])
twin_index = np.array([tick_labels.index(label[:-1] + ("R" if label[-1] == "L" else "L"))
                       for label in tick_labels])
row_index, col_index = np.triu_indices(n_rois, k=1)
same_side = side[row_index] == side[col_index]
homotopic = twin_index[row_index] == col_index
metric_masks = {
    "global": np.ones(len(row_index), dtype=bool),
    "intra_left": same_side & (side[row_index] == "L"),
    "intra_right": same_side & (side[row_index] == "R"),
    "homotopic": homotopic,
    "heterotopic_inter": ~same_side & ~homotopic,
}
# edge_z: (n_recordings, n_edges) upper-triangle z per recording.
edge_z = recording_z[:, row_index, col_index]
# node_z: (n_recordings, n_rois) mean z of each ROI with the other ROIs.
node_z = recording_z.sum(axis=2) / (n_rois - 1)

recording_values = design[["animal", "day_index", "session", "recording"]].copy()
for metric, mask in metric_masks.items():
    recording_values[metric] = edge_z[:, mask].mean(axis=1)
for i_roi, label in enumerate(tick_labels):
    recording_values[f"node_{label}"] = node_z[:, i_roi]

# Session values: mean over recordings, then (animal, day) tables for paired tests.
value_columns = [c for c in recording_values.columns if c not in ("animal", "day_index", "session", "recording")]
session_values = recording_values.groupby(["animal", "day_index"])[value_columns].mean().reset_index()
session_values.insert(2, "time_label", session_values["day_index"].map(day_label))
session_values.to_csv(output_dir / "session_values.csv", index=False)
# session_edge_z: (n_animals, n_days, n_edges)
session_edge_z = np.array([[edge_z[((design["animal"] == a) & (design["day_index"] == d)).to_numpy()].mean(axis=0)
                            for d in days] for a in animals])


def contrast_difference(table):
    """{contrast: (n_animals, ...) later day (or mean of later days) minus baseline}; table is (n_animals, n_days, ...)."""
    differences = {contrast_of[day]: table[:, i_day] - table[:, 0] for i_day, day in enumerate(days) if i_day > 0}
    differences["post"] = table[:, 1:].mean(axis=1) - table[:, 0]
    return differences


def p_with_stars(p_value):
    """p formatted for a figure with its significance: * < 0.05, ** < 0.01, *** < 0.001."""
    stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    return text + stars


def friedman_pvalue(table):
    """Friedman test across the timepoints (columns) of an animals x timepoints table.

    It needs at least three timepoints; with two it is NaN (the paired t-test is the
    test of a single later timepoint against the baseline).
    """
    if table.shape[1] < 3:
        return float("nan")
    return stats.friedmanchisquare(*table.T).pvalue


def paired_tests(table):
    """{contrast: (mean difference, t, p)} of a paired t-test across animals; table is (n_animals, n_days, ...)."""
    results = {}
    for contrast, difference in contrast_difference(table).items():
        t_value, p_value = stats.ttest_1samp(difference, 0.0, axis=0)
        results[contrast] = (difference.mean(axis=0), t_value, p_value)
    return results


def mixed_model_tests(column):
    """Recording-level z ~ C(day_index), animal intercept + session component; Wald per contrast and omnibus.

    Fit with lbfgs, then powell, then nm (one optimizer alone can stop at a false optimum),
    and an error if none converges.
    """
    data = recording_values[["animal", "session", "day_index", column]].rename(columns={column: "value"})
    model = smf.mixedlm("value ~ C(day_index)", data, groups="animal", re_formula="1",
                        vc_formula={"session": "0 + C(session)"}).fit(reml=True, method=["lbfgs", "powell", "nm"])
    if not model.converged:
        raise RuntimeError(f"mixed model did not converge: {column}")
    names = list(model.fe_params.index)   # t_test/wald_test take fixed-effect columns only
    day_terms = [f"C(day_index)[T.{day}]" for day in later_days]
    results = {}
    for contrast in contrasts:
        weights = np.zeros(len(names))
        terms = day_terms if contrast == "post" else [f"C(day_index)[T.{day_of_contrast[contrast]}]"]
        for term in terms:
            weights[names.index(term)] = 1.0 / len(terms)
        test = model.t_test(weights[None, :])
        results[contrast] = (float(np.ravel(test.effect)[0]), float(np.ravel(test.tvalue)[0]), float(np.ravel(test.pvalue)[0]))
    # Omnibus Wald chi2 that every later day equals the baseline, from the fixed-effect covariance.
    estimates = model.fe_params[day_terms].to_numpy()
    covariance = model.cov_params().loc[day_terms, day_terms].to_numpy()
    chi2 = float(estimates @ np.linalg.solve(covariance, estimates))
    return results, float(stats.chi2.sf(chi2, len(day_terms)))


# metric_tests: one row per metric x contrast.
metric_rows = []
for metric in metric_masks:
    table = session_values.pivot(index="animal", columns="day_index", values=metric).loc[animals, days].to_numpy()
    paired = paired_tests(table)
    mixed, omnibus_p = mixed_model_tests(metric)
    friedman_p = friedman_pvalue(table)
    for contrast in contrasts:
        metric_rows.append({
            "metric": metric, "contrast": contrast,
            "baseline_mean_z": table[:, 0].mean(),
            "mean_difference_z": paired[contrast][0],
            "n_animals_increase": int((contrast_difference(table)[contrast] > 0).sum()),
            "paired_t": paired[contrast][1], "paired_p": paired[contrast][2],
            "lmm_estimate_z": mixed[contrast][0], "lmm_t": mixed[contrast][1], "lmm_p": mixed[contrast][2],
            "lmm_omnibus_p": omnibus_p, "friedman_p": friedman_p,
        })
metric_tests = pd.DataFrame(metric_rows)
metric_tests.to_csv(output_dir / "metric_tests.csv", index=False)

# ---------------------------------------------------------------- node strength
node_rows = []
for label in tick_labels:
    column = f"node_{label}"
    table = session_values.pivot(index="animal", columns="day_index", values=column).loc[animals, days].to_numpy()
    paired = paired_tests(table)
    mixed, omnibus_p = mixed_model_tests(column)
    for contrast in contrasts:
        node_rows.append({"roi": label, "contrast": contrast,
                          "mean_difference_z": paired[contrast][0],
                          "paired_t": paired[contrast][1], "paired_p": paired[contrast][2],
                          "lmm_estimate_z": mixed[contrast][0], "lmm_t": mixed[contrast][1],
                          "lmm_p": mixed[contrast][2], "lmm_omnibus_p": omnibus_p})
node_tests = pd.DataFrame(node_rows)
for contrast in contrasts:
    rows = node_tests["contrast"] == contrast
    for test in ["paired", "lmm"]:
        node_tests.loc[rows, f"{test}_q"] = multipletests(node_tests.loc[rows, f"{test}_p"], method="fdr_bh")[1]
node_tests.to_csv(output_dir / "node_tests.csv", index=False)

# ---------------------------------------------------------------- edges
edge_results = paired_tests(session_edge_z)
# session_edge_r: (n_animals, n_days, n_edges) plain mean of Pearson r over the session's
# recordings (as the reference MATLAB analysis does). Display only: the tests stay on z.
session_edge_r = np.array([[np.tanh(edge_z[((design["animal"] == a) & (design["day_index"] == d)).to_numpy()]).mean(axis=0)
                            for d in days] for a in animals])
edge_difference_r = {contrast: difference.mean(axis=0)
                     for contrast, difference in contrast_difference(session_edge_r).items()}
edge_rows = []
for contrast, (difference, t_value, p_value) in edge_results.items():
    q_value = multipletests(p_value, method="fdr_bh")[1]
    for i_edge in range(len(row_index)):
        edge_rows.append({"contrast": contrast, "roi_a": tick_labels[row_index[i_edge]],
                          "roi_b": tick_labels[col_index[i_edge]],
                          "mean_difference_z": difference[i_edge],
                          "mean_difference_r": edge_difference_r[contrast][i_edge],
                          "paired_t": t_value[i_edge],
                          "paired_p": p_value[i_edge], "paired_q": q_value[i_edge]})
edge_tests = pd.DataFrame(edge_rows)
edge_tests.to_csv(output_dir / "edge_tests.csv", index=False)

# ---------------------------------------------------------------- NBS on edges
# Network-Based Statistic per contrast, as the multiple-comparison correction for the
# edge maps: observations are each animal's change from baseline (n_rois, n_rois,
# n_animals), edge statistic the one-sample t (= the paired t above), null from flipping
# the sign of whole animals -- 2^n_animals relabellings, enumerated exactly (p floor
# 1/2^n). Increases and decreases are separate one-sided runs. An edge is "NBS
# significant" when it belongs to a component with FWER p < nbs_alpha.
# nbs_edges: {contrast: (n_rois, n_rois) int, +1 increase / -1 decrease / 0 not significant}
nbs_edges = {}
nbs_rows = []
for contrast, difference in contrast_difference(session_edge_z).items():
    # difference: (n_animals, n_edges) -> difference_matrices: (n_rois, n_rois, n_animals)
    difference_matrices = np.zeros((n_rois, n_rois, len(animals)))
    difference_matrices[row_index, col_index, :] = difference.T
    difference_matrices[col_index, row_index, :] = difference.T
    marks = np.zeros((n_rois, n_rois), dtype=int)
    for direction, sign in [("increase", 1.0), ("decrease", -1.0)]:
        result = nbs(sign * difference_matrices, test="one_sample", primary_threshold=nbs_primary_threshold,
                     n_permutations=nbs_permutations, alpha=nbs_alpha, seed=0)
        for component in result.components:
            nbs_rows.append({"contrast": contrast, "direction": direction,
                             "primary_threshold": nbs_primary_threshold, "n_edges": len(component.edges),
                             "size": component.size, "p_value": component.p_value,
                             "significant": component.p_value < nbs_alpha, "exact": result.exact,
                             "edges": "; ".join(f"{tick_labels[a]}-{tick_labels[b]}" for a, b in component.edges)})
        for component in result.significant:
            marks[component.adjacency] = int(sign)
    nbs_edges[contrast] = marks
nbs_components = pd.DataFrame(nbs_rows, columns=["contrast", "direction", "primary_threshold", "n_edges", "size",
                                                 "p_value", "significant", "exact", "edges"])
nbs_components.to_csv(output_dir / "nbs_components.csv", index=False)
for contrast in contrasts:
    hits = nbs_components[(nbs_components["contrast"] == contrast) & nbs_components["significant"]]
    print(f"NBS {contrast}: " + (", ".join(f"{r.direction} {r.n_edges} edges p={r.p_value:.4f}" for r in hits.itertuples())
                                 or "no significant component"))


def plot_nbs_change(row_rois, file_name, title):
    """One panel per contrast: mean change in z, rows `row_rois` (ROI indices) x every ROI; dot = NBS-significant edge."""
    fig, axes = plt.subplots(len(contrasts), 1, figsize=(10, max(3.4, 0.21 * len(row_rois) + 1.7) * len(contrasts)),
                             constrained_layout=True, squeeze=False)
    # One colour scale over every plotted change, centred on 0 (as the other change maps).
    changes = {contrast: edge_matrix(edge_tests.loc[edge_tests["contrast"] == contrast, "mean_difference_z"].to_numpy(),
                                     np.nan)[row_rois] for contrast in contrasts}
    limit = np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in changes.values()])), color_limit_percentile)
    for i_contrast, contrast in enumerate(contrasts):
        ax = axes[i_contrast, 0]
        image = ax.imshow(changes[contrast], cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        # marks: (n_row_rois, n_rois); dot on every edge of a significant component.
        marks = nbs_edges[contrast][row_rois]
        hit_rows, hit_cols = np.nonzero(marks)
        ax.plot(hit_cols, hit_rows, ls="none", marker=".", color="black", ms=5)
        hits = nbs_components[(nbs_components["contrast"] == contrast) & nbs_components["significant"]]
        summary = ", ".join(f"{r.direction} {r.n_edges} edges p={r.p_value:.3f}" for r in hits.itertuples()) or "none"
        ax.set_title(f"{contrast} - {baseline_label}   (NBS: {summary})", fontsize=9)
        ax.set_yticks(range(len(row_rois)), [tick_labels[i] for i in row_rois], fontsize=7)
        ax.set_xticks(range(n_rois), tick_labels, rotation=90, fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.9, extend="both", label="change in z")
    fig.suptitle(f"{variant}, {len(animals)} animals: {title}\n(dot = edge in an NBS component with FWER p<{nbs_alpha}; "
                 f"paired t > {nbs_primary_threshold:g}, {2 ** len(animals)} sign flips of whole animals)", fontsize=10)
    fig.savefig(output_dir / f"{file_name}.png", dpi=200)
    fig.savefig(output_dir / f"{file_name}.svg")
    plt.close(fig)

# ---------------------------------------------------------------- console summary
pd.set_option("display.width", 200)
print(metric_tests[["metric", "contrast", "baseline_mean_z", "mean_difference_z", "n_animals_increase",
                    "paired_p", "lmm_estimate_z", "lmm_p", "lmm_omnibus_p", "friedman_p"]]
      .round(4).to_string(index=False))
for contrast in contrasts:
    rows = edge_tests[edge_tests["contrast"] == contrast]
    nodes = node_tests[node_tests["contrast"] == contrast]
    print(f"{contrast}: edges up {(rows['mean_difference_z'] > 0).sum()}/{len(rows)}, "
          f"p<{uncorrected_alpha} {(rows['paired_p'] < uncorrected_alpha).sum()} "
          f"(up {((rows['paired_p'] < uncorrected_alpha) & (rows['mean_difference_z'] > 0)).sum()}), "
          f"q<{fdr_alpha} {(rows['paired_q'] < fdr_alpha).sum()}; "
          f"nodes lmm q<{fdr_alpha}: {list(nodes.loc[nodes['lmm_q'] < fdr_alpha, 'roi'])}, "
          f"paired q<{fdr_alpha}: {list(nodes.loc[nodes['paired_q'] < fdr_alpha, 'roi'])}")

# ---------------------------------------------------------------- figure: metrics by day
fig, axes = plt.subplots(1, len(metric_masks), figsize=(3.6 * len(metric_masks), 3.8),
                         constrained_layout=True, squeeze=False)
for i_metric, metric in enumerate(metric_masks):
    ax = axes[0, i_metric]
    table = session_values.pivot(index="animal", columns="day_index", values=metric).loc[animals, days]
    for animal in animals:
        ax.plot(days, table.loc[animal], marker="o", lw=1, alpha=0.6, label=animal)
    ax.plot(days, table.mean(axis=0), color="black", lw=2.5, marker="s", label="mean")
    tests = metric_tests[metric_tests["metric"] == metric].set_index("contrast")
    for day in later_days:
        ax.annotate(f"p={tests.loc[contrast_of[day], 'lmm_p']:.3f}", (day, table.mean(axis=0)[day]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.set_title(f"{metric}\npost p={tests.loc['post', 'lmm_p']:.3f}, omnibus p={tests['lmm_omnibus_p'].iloc[0]:.3f}",
                 fontsize=9)
    ax.set_xticks(days, [day_label[d] for d in days])
    ax.set_xlabel(time_axis_label)
    ax.set_ylabel("mean Fisher z")
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"{variant}, {'+'.join(animals)}: session mean z by day (p = mixed model vs {baseline_label})")
fig.savefig(output_dir / "metrics_by_day.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure: node strength change
fig, axes = plt.subplots(len(contrasts), 1, figsize=(10, 2.6 * len(contrasts)),
                         constrained_layout=True, squeeze=False, sharex=True)
for i_contrast, contrast in enumerate(contrasts):
    ax = axes[i_contrast, 0]
    rows = node_tests[node_tests["contrast"] == contrast].set_index("roi").loc[tick_labels]
    colors = np.where(rows["lmm_q"] < fdr_alpha, "tab:red",
                      np.where(rows["lmm_p"] < uncorrected_alpha, "tab:orange", "tab:gray"))
    ax.bar(range(n_rois), rows["lmm_estimate_z"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("change in z")
    ax.set_title(f"{contrast} - {baseline_label} (red: q<{fdr_alpha}, orange: p<{uncorrected_alpha}, mixed model)",
                 fontsize=9)
axes[-1, 0].set_xticks(range(n_rois), tick_labels, rotation=90, fontsize=7)
fig.suptitle(f"{variant}, {'+'.join(animals)}: node strength change")
fig.savefig(output_dir / "node_strength_change.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure: edges
# Row 1: mean difference in z; row 2: paired t; row 3: mean difference in Pearson r (session
# mean r, as the reference MATLAB analysis does). Dot: p < uncorrected_alpha; star: q < fdr_alpha (both from z).
row_specs = [("mean_difference_z", "mean change in Fisher z"), ("paired_t", "paired t (df = %d)" % (len(animals) - 1)),
             ("mean_difference_r", "mean change in Pearson r")]
fig, axes = plt.subplots(len(row_specs), len(contrasts), figsize=(4.6 * len(contrasts), 4.7 * len(row_specs)),
                         constrained_layout=True, squeeze=False)
for i_row, (column, label) in enumerate(row_specs):
    values = edge_tests[column].to_numpy()
    limit = np.percentile(np.abs(values), color_limit_percentile)
    for i_contrast, contrast in enumerate(contrasts):
        ax = axes[i_row, i_contrast]
        rows = edge_tests[edge_tests["contrast"] == contrast]
        matrix = np.zeros((n_rois, n_rois))
        matrix[row_index, col_index] = rows[column].to_numpy()
        matrix = matrix + matrix.T
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        for p_column, alpha, marker, size in [("paired_p", uncorrected_alpha, ".", 3), ("paired_q", fdr_alpha, "*", 7)]:
            hit = (rows[p_column] < alpha).to_numpy()
            ax.plot(np.r_[col_index[hit], row_index[hit]], np.r_[row_index[hit], col_index[hit]],
                    ls="none", marker=marker, color="black", ms=size)
        ax.set_title(f"{contrast} - {baseline_label}", fontsize=9)
        ax.set_xticks(range(n_rois), tick_labels, rotation=90, fontsize=5)
        ax.set_yticks(range(n_rois), tick_labels, fontsize=5)
    fig.colorbar(image, ax=axes[i_row, :], shrink=0.8, extend="both", label=label)
fig.suptitle(f"{variant}, {'+'.join(animals)}: edge change vs {baseline_label} "
             f"(dot p<{uncorrected_alpha}, star BH q<{fdr_alpha}, paired across {len(animals)} animals)")
fig.savefig(output_dir / "edge_tests.png", dpi=200)
fig.savefig(output_dir / "edge_tests.svg")
plt.close(fig)

# ---------------------------------------------------------------- figure: mean matrices by day
# Top row: the group mean Pearson r per
# day (equal-weight mean of the animals' session mean r), bottom row day N - day 1. One
# colour scale per row, centred on 0 at the color_limit_percentile of |off-diagonal value|.
# group_r: (n_days, n_edges); group_difference_r: (n_later_days, n_edges)
group_r = session_edge_r.mean(axis=0)
group_difference_r = group_r[1:] - group_r[:1]


def edge_matrix(edge_values, diagonal):
    """(n_rois, n_rois) symmetric matrix from upper-triangle edge values."""
    matrix = np.full((n_rois, n_rois), float(diagonal))
    matrix[row_index, col_index] = edge_values
    matrix[col_index, row_index] = edge_values
    return matrix


# Two versions: "centered" (RdBu_r, each row centred on 0 at the color_limit_percentile of
# |value|; mean_matrices.*) and "uncentered" (`uncentered_colormap`, each row its own [min, max] of the
# off-diagonal values;
# mean_matrices_uncentered.*), plus "fixed" (RdBu_r, both rows at `fixed_color_limits`, so
# figures are comparable across variants and cohorts; mean_matrices_fixed.*) and
# "per_panel" (parula, every heatmap its own [min, max] and colorbar; mean_matrices_per_panel.*).
for scale in ("centered", "uncentered", "fixed", "per_panel"):
    per_panel = scale == "per_panel"
    if per_panel:
        colormap, extend, file_tag = parula, "neither", "_per_panel"
        scale_rule = "[min, max] of each panel"
    elif scale == "fixed":
        matrix_limits = difference_limits = fixed_color_limits
        colormap, extend, file_tag = "RdBu_r", "neither", "_fixed"
        scale_rule = f"fixed {fixed_color_limits[0]:g}..{fixed_color_limits[1]:g}"
    elif scale == "centered":
        matrix_limits = (-np.percentile(np.abs(group_r), color_limit_percentile),
                         np.percentile(np.abs(group_r), color_limit_percentile))
        difference_limits = (-np.percentile(np.abs(group_difference_r), color_limit_percentile),
                             np.percentile(np.abs(group_difference_r), color_limit_percentile))
        colormap, extend, file_tag = "RdBu_r", "both", ""
        scale_rule = f"centred on 0 at the {color_limit_percentile:g}th percentile of |value|"
    else:
        matrix_limits = (group_r.min(), group_r.max())
        difference_limits = (group_difference_r.min(), group_difference_r.max())
        colormap, extend, file_tag = uncentered_colormap, "neither", "_uncentered"
        scale_rule = "[min, max] of each row"
    fig, axes = plt.subplots(2, len(days), figsize=(4.6 * len(days), 9.4),
                             constrained_layout=True, squeeze=False)
    # Per-panel colorbars take width, so those titles name the animal count, not the animals.
    # Long animal names overlap the neighbouring panel; 40 characters fits.
    joined_animals = "+".join(animals)
    animals_title = (f"mean of {len(animals)} animals" if per_panel or len(joined_animals) > 40
                     else f"mean {joined_animals}")
    for i_day, day in enumerate(days):
        ax = axes[0, i_day]
        if per_panel:
            # The edge values are exactly the off-diagonal entries, so the diagonal (r = 1) is ignored.
            matrix_limits = (group_r[i_day].min(), group_r[i_day].max())
        image_matrix = ax.imshow(edge_matrix(group_r[i_day], 1.0), cmap=colormap,
                                 vmin=matrix_limits[0], vmax=matrix_limits[1], interpolation="nearest")
        ax.set_title(f"{day_label[day]}: {animals_title}", fontsize=9)
        if per_panel:
            fig.colorbar(image_matrix, ax=ax, shrink=0.8, label="mean Pearson r")
        ax = axes[1, i_day]
        if i_day == 0:
            ax.set_axis_off()
            continue
        if per_panel:
            difference_limits = (group_difference_r[i_day - 1].min(), group_difference_r[i_day - 1].max())
        image_difference = ax.imshow(edge_matrix(group_difference_r[i_day - 1], 0.0), cmap=colormap,
                                     vmin=difference_limits[0], vmax=difference_limits[1], interpolation="nearest")
        ax.set_title(f"{day_label[day]} - {baseline_label}: {animals_title}", fontsize=9)
        if per_panel:
            fig.colorbar(image_difference, ax=ax, shrink=0.8, label="difference in Pearson r")
    for ax in axes.flat:
        if ax.axison:
            ax.set_xticks(range(n_rois), tick_labels, rotation=90, fontsize=5)
            ax.set_yticks(range(n_rois), tick_labels, fontsize=5)
    if per_panel:
        fig.suptitle(f"{variant}: group mean of {'+'.join(animals)} (plain mean of Pearson r). "
                     f"Each panel its own scale ({scale_rule}, diagonal ignored)")
    else:
        fig.colorbar(image_matrix, ax=axes[0, :], shrink=0.8, extend=extend, label="mean Pearson r")
        fig.colorbar(image_difference, ax=axes[1, 1:], shrink=0.8, extend=extend, label="difference in Pearson r")
        fig.suptitle(f"{variant}: group mean of {'+'.join(animals)} (plain mean of Pearson r).\n"
                     f"Shared scales ({scale_rule}): r {matrix_limits[0]:.2f}..{matrix_limits[1]:.2f}, "
                     f"difference {difference_limits[0]:.2f}..{difference_limits[1]:.2f} (diagonal ignored)")
    fig.savefig(output_dir / f"mean_matrices{file_tag}.png", dpi=200)
    fig.savefig(output_dir / f"mean_matrices{file_tag}.svg")
    plt.close(fig)
print(f"saved {output_dir}")

# ================================================================ region focus
# Same tests restricted to one region's ROIs (`focus_prefix`, --focus / CONNECTIVITY_STATS_FOCUS: "SSp"
# somatosensory, default, or "motor" MOs + MOp). Edge families are smaller here, so
# Benjamini-Hochberg runs over the focus edges only:
#     within: edges between two focus ROIs; to_all: every edge with a focus ROI.
# Output files are prefixed with the focus name, so both foci share one output folder.
region_prefixes = {"SSp": ("SSp",), "motor": ("MOs", "MOp"), "RSP": ("RSP",), "visual": ("VIS",)}
if focus_prefix not in ("SSp", "motor"):
    raise ValueError(f"the region focus must be 'SSp' or 'motor', not {focus_prefix!r}")
other_region_prefixes = {region: prefixes for region, prefixes in region_prefixes.items() if region != focus_prefix}
# True: also split each focus-to-region metric by the focus ROI's hemisphere (_L_to_ / _R_to_).
focus_split_by_side = focus_prefix == "motor"
focus_index = np.array([i for i, label in enumerate(tick_labels) if label.startswith(region_prefixes[focus_prefix])])
focus_labels = [tick_labels[i] for i in focus_index]
is_focus = np.isin(np.arange(n_rois), focus_index)
both_focus = is_focus[row_index] & is_focus[col_index]
one_focus = is_focus[row_index] ^ is_focus[col_index]
# other_end: index of the non-focus ROI of a one_focus edge.
other_end = np.where(is_focus[row_index], col_index, row_index)
focus_metric_masks = {
    f"{focus_prefix}_within_left": both_focus & same_side & (side[row_index] == "L"),
    f"{focus_prefix}_within_right": both_focus & same_side & (side[row_index] == "R"),
    f"{focus_prefix}_homotopic": both_focus & homotopic,
    f"{focus_prefix}_heterotopic_inter": both_focus & ~same_side & ~homotopic,
}
for region, prefixes in other_region_prefixes.items():
    focus_metric_masks[f"{focus_prefix}_to_{region}"] = one_focus & np.array(
        [tick_labels[i].startswith(prefixes) for i in other_end])
if focus_split_by_side:
    # focus_end: index of the focus ROI of a one_focus edge; its side picks the hemisphere.
    focus_end = np.where(is_focus[row_index], row_index, col_index)
    for region in other_region_prefixes:
        for focus_side in ("L", "R"):
            focus_metric_masks[f"{focus_prefix}_{focus_side}_to_{region}"] = (
                focus_metric_masks[f"{focus_prefix}_to_{region}"] & (side[focus_end] == focus_side))
for metric, mask in focus_metric_masks.items():
    recording_values[metric] = edge_z[:, mask].mean(axis=1)
focus_session_values = recording_values.groupby(["animal", "day_index"])[list(focus_metric_masks)].mean().reset_index()
focus_session_values.insert(2, "time_label", focus_session_values["day_index"].map(day_label))
focus_session_values.to_csv(output_dir / f"{focus_prefix}_session_values.csv", index=False)

focus_rows = []
for metric in focus_metric_masks:
    table = focus_session_values.pivot(index="animal", columns="day_index", values=metric).loc[animals, days].to_numpy()
    paired = paired_tests(table)
    mixed, omnibus_p = mixed_model_tests(metric)
    friedman_p = friedman_pvalue(table)
    for contrast in contrasts:
        focus_rows.append({
            "metric": metric, "contrast": contrast, "n_edges": int(focus_metric_masks[metric].sum()),
            "baseline_mean_z": table[:, 0].mean(), "mean_difference_z": paired[contrast][0],
            "n_animals_increase": int((contrast_difference(table)[contrast] > 0).sum()),
            "paired_t": paired[contrast][1], "paired_p": paired[contrast][2],
            "lmm_estimate_z": mixed[contrast][0], "lmm_p": mixed[contrast][2],
            "lmm_omnibus_p": omnibus_p, "friedman_p": friedman_p})
focus_tests = pd.DataFrame(focus_rows)
focus_tests.to_csv(output_dir / f"{focus_prefix}_metric_tests.csv", index=False)

# Edge tests with BH inside each focus family, per contrast.
focus_edge_tests = edge_tests.copy()
focus_edge_tests["edge"] = np.tile(np.arange(len(row_index)), len(contrasts))
for family, mask in [("within", both_focus), ("to_all", both_focus | one_focus)]:
    column = f"q_{family}"
    focus_edge_tests[column] = np.nan
    for contrast in contrasts:
        rows = (focus_edge_tests["contrast"] == contrast) & mask[focus_edge_tests["edge"].to_numpy()]
        focus_edge_tests.loc[rows, column] = multipletests(focus_edge_tests.loc[rows, "paired_p"], method="fdr_bh")[1]
focus_edge_tests = focus_edge_tests[(both_focus | one_focus)[focus_edge_tests["edge"].to_numpy()]].drop(columns="edge")
focus_edge_tests.to_csv(output_dir / f"{focus_prefix}_edge_tests.csv", index=False)

print(focus_tests[["metric", "contrast", "n_edges", "baseline_mean_z", "mean_difference_z", "n_animals_increase",
                   "paired_p", "lmm_p", "lmm_omnibus_p", "friedman_p"]].round(4).to_string(index=False))
for contrast in contrasts:
    rows = focus_edge_tests[focus_edge_tests["contrast"] == contrast]
    within = rows["q_within"].notna()
    print(f"{focus_prefix} {contrast}: within {within.sum()} edges, p<{uncorrected_alpha} "
          f"{(rows.loc[within, 'paired_p'] < uncorrected_alpha).sum()}, q<{fdr_alpha} {(rows['q_within'] < fdr_alpha).sum()}; "
          f"to_all {len(rows)} edges, p<{uncorrected_alpha} {(rows['paired_p'] < uncorrected_alpha).sum()}, "
          f"q<{fdr_alpha} {(rows['q_to_all'] < fdr_alpha).sum()}")

# ---------------------------------------------------------------- figure: focus metrics by day
n_metrics = len(focus_metric_masks)
n_cols = 4
n_rows = int(np.ceil(n_metrics / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.6 * n_rows), constrained_layout=True, squeeze=False)
for ax in axes.flat[n_metrics:]:
    ax.set_axis_off()
for ax, metric in zip(axes.flat, focus_metric_masks):
    table = focus_session_values.pivot(index="animal", columns="day_index", values=metric).loc[animals, days]
    for animal in animals:
        ax.plot(days, table.loc[animal], marker="o", lw=1, alpha=0.6, label=animal)
    ax.plot(days, table.mean(axis=0), color="black", lw=2.5, marker="s", label="mean")
    tests = focus_tests[focus_tests["metric"] == metric].set_index("contrast")
    for day in later_days:
        ax.annotate(f"{tests.loc[contrast_of[day], 'paired_p']:.3f}\n{tests.loc[contrast_of[day], 'lmm_p']:.3f}",
                    (day, table.mean(axis=0)[day]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=6)
    ax.set_title(f"{metric} ({int(focus_metric_masks[metric].sum())} edges)\n"
                 f"post p paired {tests.loc['post', 'paired_p']:.3f}, mixed {tests.loc['post', 'lmm_p']:.3f}", fontsize=8)
    ax.set_xticks(days, [day_label[d] for d in days])
    ax.set_xlabel(time_axis_label)
    ax.set_ylabel("mean Fisher z")
axes[0, 0].legend(fontsize=6)
fig.suptitle(f"{variant}, {'+'.join(animals)}: {focus_prefix} connectivity by day "
             f"(numbers above the mean: paired p / mixed-model p vs {baseline_label})")
fig.savefig(output_dir / f"{focus_prefix}_metrics_by_day.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure: focus matrices
# Top: group mean r between focus ROIs per day (plain mean of r over recordings, then
# animals, as the reference MATLAB analysis does). Bottom: mean change in z vs day 1; dot p < 0.05,
# star BH q < fdr_alpha within the focus family.
session_r = {(a, d): np.tanh(recording_z[((design["animal"] == a) & (design["day_index"] == d)).to_numpy()]).mean(axis=0)
             for a in animals for d in days}
group_focus_r = {d: np.mean([session_r[(a, d)] for a in animals], axis=0)[np.ix_(focus_index, focus_index)]
                 for d in days}
for d in days:
    np.fill_diagonal(group_focus_r[d], np.nan)
r_values = np.concatenate([group_focus_r[d][np.isfinite(group_focus_r[d])] for d in days])
fig, axes = plt.subplots(2, len(days) + 1, figsize=(4.0 * (len(days) + 1), 8.2), constrained_layout=True, squeeze=False)
for i_col, day in enumerate(days):
    image_r = axes[0, i_col].imshow(group_focus_r[day], cmap="viridis", vmin=r_values.min(), vmax=r_values.max(),
                                    interpolation="nearest")
    axes[0, i_col].set_title(f"{day_label[day]}: mean r", fontsize=9)
axes[0, -1].set_axis_off()
fig.colorbar(image_r, ax=axes[0, :-1], shrink=0.8, label="mean Pearson r")
within_rows = focus_edge_tests[focus_edge_tests["q_within"].notna()]
difference_limit = np.percentile(np.abs(within_rows["mean_difference_z"]), color_limit_percentile)
position_of = {label: i for i, label in enumerate(focus_labels)}
axes[1, 0].set_axis_off()
for i_contrast, contrast in enumerate(contrasts):
    ax = axes[1, i_contrast + 1]
    rows = within_rows[within_rows["contrast"] == contrast]
    matrix = np.full((len(focus_labels), len(focus_labels)), np.nan)
    for row in rows.itertuples():
        matrix[position_of[row.roi_a], position_of[row.roi_b]] = row.mean_difference_z
        matrix[position_of[row.roi_b], position_of[row.roi_a]] = row.mean_difference_z
        marker = "*" if row.q_within < fdr_alpha else ("." if row.paired_p < uncorrected_alpha else None)
        if marker:
            for y_pos, x_pos in [(position_of[row.roi_a], position_of[row.roi_b]),
                                 (position_of[row.roi_b], position_of[row.roi_a])]:
                ax.plot(x_pos, y_pos, ls="none", marker=marker, color="black", ms=8 if marker == "*" else 5)
    image_difference = ax.imshow(matrix, cmap="RdBu_r", vmin=-difference_limit, vmax=difference_limit,
                                 interpolation="nearest")
    ax.set_title(f"{contrast} - {baseline_label}: change in z", fontsize=9)
fig.colorbar(image_difference, ax=axes[1, 1:], shrink=0.8, extend="both", label="mean change in Fisher z")
for ax in axes.flat:
    if ax.axison:
        ax.set_xticks(range(len(focus_labels)), focus_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(focus_labels)), focus_labels, fontsize=7)
fig.suptitle(f"{variant}, {'+'.join(animals)}: {focus_prefix} ROIs "
             f"(dot p<{uncorrected_alpha}, star BH q<{fdr_alpha} over the {int(both_focus.sum())} {focus_prefix} edges)")
fig.savefig(output_dir / f"{focus_prefix}_matrices.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure: focus ROI to every ROI
# Rows: focus ROIs; columns: every ROI. Mean change in z; dot p < 0.05, star BH over
# all edges touching a focus ROI.
fig, axes = plt.subplots(len(contrasts), 1, figsize=(10, 3.4 * len(contrasts)), constrained_layout=True, squeeze=False)
to_all_limit = np.percentile(np.abs(focus_edge_tests["mean_difference_z"]), color_limit_percentile)
for i_contrast, contrast in enumerate(contrasts):
    ax = axes[i_contrast, 0]
    rows = focus_edge_tests[focus_edge_tests["contrast"] == contrast]
    matrix = np.full((len(focus_labels), n_rois), np.nan)
    for row in rows.itertuples():
        for focus_end, other in [(row.roi_a, row.roi_b), (row.roi_b, row.roi_a)]:
            if focus_end not in position_of:
                continue
            matrix[position_of[focus_end], tick_labels.index(other)] = row.mean_difference_z
            marker = "*" if row.q_to_all < fdr_alpha else ("." if row.paired_p < uncorrected_alpha else None)
            if marker:
                ax.plot(tick_labels.index(other), position_of[focus_end], ls="none", marker=marker,
                        color="black", ms=8 if marker == "*" else 4)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-to_all_limit, vmax=to_all_limit, interpolation="nearest")
    ax.set_yticks(range(len(focus_labels)), focus_labels, fontsize=7)
    ax.set_xticks(range(n_rois), tick_labels, rotation=90, fontsize=7)
    ax.set_title(f"{contrast} - {baseline_label}", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.9, extend="both", label="change in z")
fig.suptitle(f"{variant}, {'+'.join(animals)}: {focus_prefix} ROI to every ROI "
             f"(dot p<{uncorrected_alpha}, star BH q<{fdr_alpha} over {int((both_focus | one_focus).sum())} edges)")
fig.savefig(output_dir / f"{focus_prefix}_to_all_change.png", dpi=200)
plt.close(fig)

# The same maps with NBS as the correction (see "NBS on edges"): the focus rows, and the
# whole network (written by both focus passes, identically).
plot_nbs_change(list(focus_index), f"{focus_prefix}_to_all_change_nbs", f"{focus_prefix} ROI to every ROI, NBS")
plot_nbs_change(list(range(n_rois)), "all_to_all_change_nbs", "every ROI to every ROI, NBS")
print(f"saved {focus_prefix} figures in {output_dir}")

# ================================================================ relative to baseline, in Pearson r
# Display only: each animal's session mean r (plain mean of r over the metric's edges and
# the session's recordings, as the reference MATLAB analysis does) divided by its own day-1 value, so day 1
# is 1 for every animal. The black line is the mean of the animals' ratios. p-values are
# still the Fisher-z tests above (paired / mixed model): r is bounded at 1 and its
# variance shrinks near it, so it is the right scale to look at but not to test on.
recording_edge_r = np.tanh(edge_z)   # (n_recordings, n_edges)


def plot_relative_to_baseline(masks, tests, file_name, title):
    """Per-animal baseline ratios (no GSR) or differences (GSR), with z-test p-values."""
    n_panels = len(masks)
    n_panel_cols = min(n_panels, 4)
    n_panel_rows = int(np.ceil(n_panels / n_panel_cols))
    fig, axes = plt.subplots(n_panel_rows, n_panel_cols, figsize=(3.8 * n_panel_cols, 3.6 * n_panel_rows),
                             constrained_layout=True, squeeze=False)
    for ax in axes.flat[n_panels:]:
        ax.set_axis_off()
    for ax, (metric, mask) in zip(axes.flat, masks.items()):
        values = design[["animal", "day_index"]].assign(r=recording_edge_r[:, mask].mean(axis=1))
        # table: (animals, days) session mean r; relative: divided by each animal's day 1.
        table = values.groupby(["animal", "day_index"])["r"].mean().unstack().loc[animals, days]
        # GSR can leave baseline r near zero or negative: use differences, as in the
        # whole-network figure, so the direction and size remain interpretable.
        if relative_mode == "ratio":
            relative = table.div(table[baseline_day_index], axis=0)
            relative_label, reference_line = f"mean r / {baseline_label}", 1.0
        else:
            relative = table.sub(table[baseline_day_index], axis=0)
            relative_label, reference_line = f"mean r - {baseline_label}", 0.0
        for animal in animals:
            ax.plot(days, relative.loc[animal], marker="o", lw=1, alpha=0.6, label=animal)
        ax.plot(days, relative.mean(axis=0), color="black", lw=2.5, marker="s", label="mean")
        ax.axhline(reference_line, color="gray", lw=0.8, ls="--")
        metric_tests_here = tests[tests["metric"] == metric].set_index("contrast")
        for day in later_days:
            # Above the highest animal, so the label never sits on a line.
            ax.annotate(f"{p_with_stars(metric_tests_here.loc[contrast_of[day], 'paired_p'])}\n"
                        f"{p_with_stars(metric_tests_here.loc[contrast_of[day], 'lmm_p'])}",
                        (day, relative[day].max()), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=6)
        ax.set_title(f"{metric} ({int(mask.sum())} edges), {baseline_label} mean r {table[baseline_day_index].mean():.2f}\n"
                     f"post p paired {p_with_stars(metric_tests_here.loc['post', 'paired_p'])}, "
                     f"mixed {p_with_stars(metric_tests_here.loc['post', 'lmm_p'])}", fontsize=8)
        ax.set_xticks(days, [day_label[d] for d in days])
        ax.set_xlabel(time_axis_label)
        ax.set_ylabel(relative_label)
        ax.margins(y=0.22)  # space for the two p-value lines above the highest animal
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{title}\n(numbers above each time point: paired p / mixed-model p vs {baseline_label}, "
                 f"on Fisher z; * p<0.05, ** p<0.01, *** p<0.001, uncorrected)")
    fig.savefig(output_dir / file_name, dpi=200)
    fig.savefig((output_dir / file_name).with_suffix(".svg"))
    plt.close(fig)


def plot_relative_and_absolute_r_sd(masks, tests, file_stem, title, show_animals=False):
    """One column per metric. Top: session mean r / day-1 r; bottom: session mean r.

    Both rows show the mean over animals +- the SD across animals (ddof = 1), not the
    individual animals. Saved as PNG and SVG. After GSR the day-1 mean r is near 0 or
    negative, so a ratio is meaningless (and flips sign); the top row is then
    session r - day-1 r instead (`relative_mode`). show_animals=True draws one thin line
    per animal under the black mean (no SD bars).
    """
    fig, axes = plt.subplots(2, len(masks), figsize=(3.6 * len(masks), 7.0),
                             constrained_layout=True, squeeze=False)
    for i_metric, (metric, mask) in enumerate(masks.items()):
        values = design[["animal", "day_index"]].assign(r=recording_edge_r[:, mask].mean(axis=1))
        # table: (animals, days) session mean r; relative: divided by each animal's day 1.
        table = values.groupby(["animal", "day_index"])["r"].mean().unstack().loc[animals, days]
        if relative_mode == "ratio":
            relative = table.div(table[baseline_day_index], axis=0)
            relative_label, reference_line = f"mean r / {baseline_label}", 1.0
        else:
            relative = table.sub(table[baseline_day_index], axis=0)
            relative_label, reference_line = f"mean r - {baseline_label}", 0.0
        metric_tests_here = tests[tests["metric"] == metric].set_index("contrast")
        for i_row, (data, ylabel) in enumerate([(relative, relative_label),
                                                (table, "mean Pearson r")]):
            ax = axes[i_row, i_metric]
            mean = data.mean(axis=0)
            sd = data.std(axis=0, ddof=1)
            if show_animals:
                for animal in animals:
                    ax.plot(days, data.loc[animal], marker="o", lw=1, alpha=0.6, label=animal)
                ax.plot(days, mean, color="black", lw=2.5, marker="s", label="mean")
                label_top = data.max(axis=0)   # p-values sit above the highest animal
            else:
                ax.errorbar(days, mean, yerr=sd, color="black", lw=2, marker="s", capsize=4,
                            label=f"mean +- SD (n = {len(animals)})")
                label_top = mean + sd
            ax.set_xticks(days, [day_label[d] for d in days])
            ax.set_xlabel(time_axis_label)
            ax.set_ylabel(ylabel)
            if i_row == 0:
                ax.axhline(reference_line, color="gray", lw=0.8, ls="--")
                for day in later_days:
                    ax.annotate(f"{metric_tests_here.loc[contrast_of[day], 'paired_p']:.3f}\n"
                                f"{metric_tests_here.loc[contrast_of[day], 'lmm_p']:.3f}",
                                (day, label_top[day]), textcoords="offset points", xytext=(0, 4),
                                ha="center", fontsize=6)
                ax.set_title(f"{metric} ({int(mask.sum())} edges), {baseline_label} mean r {table[baseline_day_index].mean():.2f}\n"
                             f"post p paired {metric_tests_here.loc['post', 'paired_p']:.3f}, "
                             f"mixed {metric_tests_here.loc['post', 'lmm_p']:.3f}", fontsize=8)
            else:
                ax.set_title(f"{metric}: Pearson r", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{title}\n(numbers above each day: paired p / mixed-model p vs {baseline_label}, on Fisher z)")
    fig.savefig(output_dir / f"{file_stem}.png", dpi=200)
    fig.savefig(output_dir / f"{file_stem}.svg")
    plt.close(fig)


# Ratio to day 1 without GSR; difference from day 1 with GSR (day-1 r ~ 0 or negative).
if relative_mode is None:
    relative_mode = "difference" if variant.removesuffix("_roi_rebuilt").endswith("_gsr") else "ratio"
if relative_mode not in ("ratio", "difference"):
    raise ValueError(f"CONNECTIVITY_STATS_RELATIVE_MODE must be 'ratio' or 'difference', not {relative_mode!r}")
plot_relative_and_absolute_r_sd(metric_masks, metric_tests, "metrics_by_day_relative_r",
                                f"{variant}, {'+'.join(animals)}: mean Pearson r "
                                f"{'relative to' if relative_mode == 'ratio' else 'change from'} {baseline_label} "
                                f"(top) and absolute (bottom), mean +- SD across animals")
plot_relative_and_absolute_r_sd(metric_masks, metric_tests, "metrics_by_day_relative_r_animals",
                                f"{variant}, {'+'.join(animals)}: mean Pearson r "
                                f"{'relative to' if relative_mode == 'ratio' else 'change from'} {baseline_label} "
                                f"(top) and absolute (bottom), each animal and their mean", show_animals=True)
plot_relative_to_baseline(focus_metric_masks, focus_tests, f"{focus_prefix}_metrics_by_day_relative_r.png",
                          f"{variant}, {'+'.join(animals)}: {focus_prefix} mean Pearson r "
                          f"{'relative to' if relative_mode == 'ratio' else 'change from'} {baseline_label}")
print(f"saved relative-r figures in {output_dir}")

# ================================================================ SSp x motor, change in r by day
# Every SSp ROI paired with the motor ROIs of one hemisphere (`motor_groups`), one panel
# per hemisphere. y = session mean r minus the same animal's day-1 mean r, averaged over
# animals. "ipsi" = SSp ROI on the motor ROIs' side, "contra" = the other side.
# Statistics run on the COMBINED edges of a group: per recording the mean Fisher z over
# the group's edges, then the same paired t / mixed model / Friedman as above.
# The SSp ROIs, independent of which region the focus section above used.
ssp_labels = [label for label in tick_labels if label.startswith(region_prefixes["SSp"])]
motor_groups = {
    "left_motor": ["MOs-a_L", "MOs-p_L", "MOp-a_L", "MOp-p_L"],
    "right_motor": ["MOs-a_R", "MOs-p_R", "MOp-a_R", "MOp-p_R"],
}
motor_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]   # one per motor ROI, same order in both groups
recording_r = np.tanh(recording_z)   # (n_recordings, n_rois, n_rois), diagonal 0
# session_r_matrix: (n_animals, n_days, n_rois, n_rois) plain mean r over the session's recordings.
session_r_matrix = np.array([[recording_r[((design["animal"] == a) & (design["day_index"] == d)).to_numpy()].mean(axis=0)
                              for d in days] for a in animals])
# change_r: (n_animals, n_days, n_rois, n_rois) minus each animal's day 1.
change_r = session_r_matrix - session_r_matrix[:, :1]

# Edge sets per group: (ssp index, motor index) pairs, all / ipsi / contra.
group_edges = {}
for group_name, motor_labels in motor_groups.items():
    motor_side = motor_labels[0][-1]
    pairs = [(tick_labels.index(ssp), tick_labels.index(motor)) for motor in motor_labels for ssp in ssp_labels]
    group_edges[group_name] = {
        "all": pairs,
        "ipsi": [(s, m) for s, m in pairs if tick_labels[s][-1] == motor_side],
        "contra": [(s, m) for s, m in pairs if tick_labels[s][-1] != motor_side],
    }

# Combined-edge tests on Fisher z (recording level -> session -> animal).
combined_rows = []
for group_name, subsets in group_edges.items():
    for subset, pairs in subsets.items():
        column = f"SSp_x_{group_name}_{subset}"
        ssp_index, motor_index = np.array(pairs).T
        recording_values[column] = recording_z[:, ssp_index, motor_index].mean(axis=1)
        table = (recording_values.groupby(["animal", "day_index"])[column].mean().unstack()
                 .loc[animals, days].to_numpy())
        paired = paired_tests(table)
        mixed, omnibus_p = mixed_model_tests(column)
        friedman_p = friedman_pvalue(table)
        # Display value in r: mean r over the subset's edges, minus day 1, per animal.
        change_r_subset = change_r[:, :, ssp_index, motor_index].mean(axis=2)   # (n_animals, n_days)
        for contrast in contrasts:
            day_columns = [days.index(day_of_contrast[contrast])] if contrast != "post" else list(range(1, len(days)))
            combined_rows.append({
                "group": group_name, "subset": subset, "n_edges": len(pairs), "contrast": contrast,
                "baseline_mean_r": session_r_matrix[:, 0, ssp_index, motor_index].mean(),
                "mean_change_r": change_r_subset[:, day_columns].mean(),
                "n_animals_increase": int((contrast_difference(table)[contrast] > 0).sum()),
                "mean_difference_z": paired[contrast][0], "paired_t": paired[contrast][1],
                "paired_p": paired[contrast][2], "lmm_estimate_z": mixed[contrast][0],
                "lmm_p": mixed[contrast][2], "lmm_omnibus_p": omnibus_p, "friedman_p": friedman_p})
combined_tests = pd.DataFrame(combined_rows)
combined_tests.to_csv(output_dir / "SSp_x_motor_combined_tests.csv", index=False)
print(combined_tests[["group", "subset", "n_edges", "contrast", "baseline_mean_r", "mean_change_r",
                      "n_animals_increase", "paired_p", "lmm_p", "lmm_omnibus_p", "friedman_p"]]
      .round(4).to_string(index=False))

# Per-edge group-mean change in r, for the illustrative figure and as a table.
edge_change_rows = []
for group_name, subsets in group_edges.items():
    for s, m in subsets["all"]:
        for i_day, day in enumerate(days):
            edge_change_rows.append({"group": group_name, "ssp": tick_labels[s], "motor": tick_labels[m],
                                     "side": "ipsi" if (s, m) in subsets["ipsi"] else "contra", "day_index": day, "time_label": day_label[day],
                                     "mean_change_r": change_r[:, i_day, s, m].mean(),
                                     "sem_change_r": change_r[:, i_day, s, m].std(ddof=1) / np.sqrt(len(animals))})
edge_change = pd.DataFrame(edge_change_rows)
edge_change.to_csv(output_dir / "SSp_x_motor_edge_change_r.csv", index=False)

# ---------------------------------------------------------------- figure: every edge
# One line per SSp-motor edge: colour = motor ROI, solid = ipsilateral SSp, dashed = contralateral.
fig, axes = plt.subplots(1, len(motor_groups), figsize=(7 * len(motor_groups), 5.2), constrained_layout=True,
                         squeeze=False, sharey=True)
for i_group, (group_name, motor_labels) in enumerate(motor_groups.items()):
    ax = axes[0, i_group]
    for motor, color in zip(motor_labels, motor_colors):
        for ssp in ssp_labels:
            rows = edge_change[(edge_change["group"] == group_name) & (edge_change["motor"] == motor)
                               & (edge_change["ssp"] == ssp)]
            is_ipsi = rows["side"].iloc[0] == "ipsi"
            ax.plot(days, rows["mean_change_r"], color=color, lw=1.2, alpha=0.8, ls="-" if is_ipsi else "--")
            ax.annotate(ssp, (days[-1], rows["mean_change_r"].iloc[-1]), textcoords="offset points",
                        xytext=(3, 0), fontsize=5, color=color, va="center")
    for motor, color in zip(motor_labels, motor_colors):
        ax.plot([], [], color=color, lw=2, label=motor)
    ax.plot([], [], color="gray", ls="-", label="ipsilateral SSp")
    ax.plot([], [], color="gray", ls="--", label="contralateral SSp")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(days, [day_label[d] for d in days])
    ax.set_xlim(days[0] - 0.2, days[-1] + 0.6)
    ax.set_xlabel(time_axis_label)
    ax.set_ylabel(f"change in Pearson r vs {baseline_label}")
    ax.set_title(f"SSp (8 ROIs) x {group_name.replace('_', ' ')}: {len(group_edges[group_name]['all'])} edges, "
                 f"mean of {len(animals)} animals", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
fig.suptitle(f"{variant}, {'+'.join(animals)}: every SSp-motor edge, change in r from {baseline_label}")
fig.savefig(output_dir / "SSp_x_motor_edges_change_r.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure: combined edges
# Thin lines: each animal's combined (all edges) change in r. Black: mean +- SEM.
# Coloured: ipsilateral / contralateral subsets' mean. p-values (combined edges, on z):
# paired t / mixed model per day.
subset_colors = {"ipsi": "tab:purple", "contra": "tab:olive"}
fig, axes = plt.subplots(1, len(motor_groups), figsize=(6 * len(motor_groups), 4.8), constrained_layout=True,
                         squeeze=False, sharey=True)
for i_group, (group_name, subsets) in enumerate(group_edges.items()):
    ax = axes[0, i_group]
    ssp_index, motor_index = np.array(subsets["all"]).T
    animal_change = change_r[:, :, ssp_index, motor_index].mean(axis=2)   # (n_animals, n_days)
    for i_animal, animal in enumerate(animals):
        ax.plot(days, animal_change[i_animal], color="gray", lw=0.8, alpha=0.6, marker="o", ms=3)
        ax.annotate(animal, (days[-1], animal_change[i_animal, -1]), textcoords="offset points", xytext=(3, 0),
                    fontsize=6, color="gray", va="center")
    mean_change = animal_change.mean(axis=0)
    sem_change = animal_change.std(axis=0, ddof=1) / np.sqrt(len(animals))
    ax.errorbar(days, mean_change, yerr=sem_change, color="black", lw=2.5, marker="s", capsize=3,
                label=f"all {len(subsets['all'])} edges (mean +- SEM)")
    for subset in ["ipsi", "contra"]:
        s_index, m_index = np.array(subsets[subset]).T
        ax.plot(days, change_r[:, :, s_index, m_index].mean(axis=2).mean(axis=0), color=subset_colors[subset],
                lw=1.5, marker="^", label=f"{subset}lateral SSp ({len(subsets[subset])} edges)")
    tests = combined_tests[(combined_tests["group"] == group_name) & (combined_tests["subset"] == "all")].set_index("contrast")
    for i_day, day in enumerate(later_days, start=1):
        ax.annotate(f"{p_with_stars(tests.loc[contrast_of[day], 'paired_p'])}\n"
                    f"{p_with_stars(tests.loc[contrast_of[day], 'lmm_p'])}",
                    (day, max(animal_change[:, i_day].max(), mean_change[i_day] + sem_change[i_day])),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=7)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xticks(days, [day_label[d] for d in days])
    ax.set_xlim(days[0] - 0.2, days[-1] + 0.5)
    ax.set_xlabel(time_axis_label)
    ax.set_ylabel(f"change in Pearson r vs {baseline_label}")
    ax.set_title(f"SSp x {group_name.replace('_', ' ')} (combined), {baseline_label} mean r "
                 f"{tests['baseline_mean_r'].iloc[0]:.2f}\npost p paired {p_with_stars(tests.loc['post', 'paired_p'])}, "
                 f"mixed {p_with_stars(tests.loc['post', 'lmm_p'])}; "
                 f"omnibus {p_with_stars(tests['lmm_omnibus_p'].iloc[0])}", fontsize=9)
    ax.margins(y=0.22)
    ax.legend(fontsize=7, loc="upper left")
fig.suptitle(f"{variant}, {'+'.join(animals)}: SSp-motor combined edges\n"
             f"(numbers: paired p / mixed-model p vs {baseline_label}, tests on Fisher z; "
             f"* p<0.05, ** p<0.01, *** p<0.001, uncorrected)", fontsize=10)
fig.savefig(output_dir / "SSp_x_motor_combined_change_r.png", dpi=200)
fig.savefig(output_dir / "SSp_x_motor_combined_change_r.svg")
plt.close(fig)
print(f"saved SSp x motor figures in {output_dir}")

# ================================================================ motor x every other region
# Motor focus only: the SSp x motor combined figure for every region the motor ROIs connect
# to. One panel per motor group (rows) and partner region (columns); a panel's combined
# edges are every (partner ROI, motor ROI) pair, split ipsi / contra by the partner ROI's
# side. Tests on the combined edges' Fisher z, as for SSp x motor above.
if focus_prefix == "motor":
    partner_regions = list(other_region_prefixes)   # SSp, RSP, visual
    panel_edges = {}
    region_rows = []
    for group_name, motor_labels in motor_groups.items():
        motor_side = motor_labels[0][-1]
        for region in partner_regions:
            partner_labels = [label for label in tick_labels if label.startswith(region_prefixes[region])]
            pairs = [(tick_labels.index(partner), tick_labels.index(motor))
                     for motor in motor_labels for partner in partner_labels]
            subsets = {"all": pairs,
                       "ipsi": [(s, m) for s, m in pairs if tick_labels[s][-1] == motor_side],
                       "contra": [(s, m) for s, m in pairs if tick_labels[s][-1] != motor_side]}
            panel_edges[(group_name, region)] = subsets
            for subset, subset_pairs in subsets.items():
                column = f"motor_x_{region}_{group_name}_{subset}"
                partner_index, motor_index = np.array(subset_pairs).T
                recording_values[column] = recording_z[:, partner_index, motor_index].mean(axis=1)
                table = (recording_values.groupby(["animal", "day_index"])[column].mean().unstack()
                         .loc[animals, days].to_numpy())
                paired = paired_tests(table)
                mixed, omnibus_p = mixed_model_tests(column)
                friedman_p = friedman_pvalue(table)
                # Display value in r: mean r over the subset's edges, minus the baseline, per animal.
                change_r_subset = change_r[:, :, partner_index, motor_index].mean(axis=2)   # (n_animals, n_days)
                for contrast in contrasts:
                    day_columns = [days.index(day_of_contrast[contrast])] if contrast != "post" else list(range(1, len(days)))
                    region_rows.append({
                        "group": group_name, "region": region, "subset": subset, "n_edges": len(subset_pairs),
                        "contrast": contrast,
                        "baseline_mean_r": session_r_matrix[:, 0, partner_index, motor_index].mean(),
                        "mean_change_r": change_r_subset[:, day_columns].mean(),
                        "n_animals_increase": int((contrast_difference(table)[contrast] > 0).sum()),
                        "mean_difference_z": paired[contrast][0], "paired_t": paired[contrast][1],
                        "paired_p": paired[contrast][2], "lmm_estimate_z": mixed[contrast][0],
                        "lmm_p": mixed[contrast][2], "lmm_omnibus_p": omnibus_p, "friedman_p": friedman_p})
    region_tests = pd.DataFrame(region_rows)
    region_tests.to_csv(output_dir / "motor_x_regions_combined_tests.csv", index=False)
    print(region_tests[region_tests["subset"] == "all"][
        ["group", "region", "n_edges", "contrast", "baseline_mean_r", "mean_change_r", "n_animals_increase",
         "paired_p", "lmm_p", "lmm_omnibus_p"]].round(4).to_string(index=False))

    # Thin grey lines: each animal's combined change in r; black: mean +- SEM; coloured:
    # ipsi / contra subsets' mean. Numbers: paired p / mixed-model p per time point, with stars.
    fig, axes = plt.subplots(len(motor_groups), len(partner_regions),
                             figsize=(5.2 * len(partner_regions), 4.6 * len(motor_groups)),
                             constrained_layout=True, squeeze=False, sharey=True)
    for i_group, group_name in enumerate(motor_groups):
        for i_region, region in enumerate(partner_regions):
            ax = axes[i_group, i_region]
            subsets = panel_edges[(group_name, region)]
            partner_index, motor_index = np.array(subsets["all"]).T
            animal_change = change_r[:, :, partner_index, motor_index].mean(axis=2)   # (n_animals, n_days)
            for i_animal, animal in enumerate(animals):
                ax.plot(days, animal_change[i_animal], color="gray", lw=0.8, alpha=0.6, marker="o", ms=3)
                ax.annotate(animal, (days[-1], animal_change[i_animal, -1]), textcoords="offset points",
                            xytext=(3, 0), fontsize=6, color="gray", va="center")
            mean_change = animal_change.mean(axis=0)
            sem_change = animal_change.std(axis=0, ddof=1) / np.sqrt(len(animals))
            ax.errorbar(days, mean_change, yerr=sem_change, color="black", lw=2.5, marker="s", capsize=3,
                        label=f"all {len(subsets['all'])} edges (mean +- SEM)")
            for subset in ["ipsi", "contra"]:
                s_index, m_index = np.array(subsets[subset]).T
                ax.plot(days, change_r[:, :, s_index, m_index].mean(axis=2).mean(axis=0),
                        color=subset_colors[subset], lw=1.5, marker="^",
                        label=f"{subset}lateral {region} ({len(subsets[subset])} edges)")
            tests = region_tests[(region_tests["group"] == group_name) & (region_tests["region"] == region)
                                 & (region_tests["subset"] == "all")].set_index("contrast")
            for i_day, day in enumerate(later_days, start=1):
                # Above the highest animal, so the label never sits on a line.
                ax.annotate(f"{p_with_stars(tests.loc[contrast_of[day], 'paired_p'])}\n"
                            f"{p_with_stars(tests.loc[contrast_of[day], 'lmm_p'])}",
                            (day, max(animal_change[:, i_day].max(), mean_change[i_day] + sem_change[i_day])),
                            textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
            ax.axhline(0, color="black", lw=0.8, ls=":")
            ax.set_xticks(days, [day_label[d] for d in days])
            ax.set_xlim(days[0] - 0.2, days[-1] + 0.5)
            ax.set_xlabel(time_axis_label)
            ax.set_ylabel(f"change in Pearson r vs {baseline_label}")
            ax.margins(y=0.22)
            ax.set_title(f"{group_name.replace('_', ' ')} x {region} (combined), {baseline_label} mean r "
                         f"{tests['baseline_mean_r'].iloc[0]:.2f}\npost p paired {p_with_stars(tests.loc['post', 'paired_p'])}, "
                         f"mixed {p_with_stars(tests.loc['post', 'lmm_p'])}; omnibus {p_with_stars(tests['lmm_omnibus_p'].iloc[0])}",
                         fontsize=9)
            ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(f"{variant}, {'+'.join(animals)}: motor to every other region, combined edges\n"
                 f"(numbers: paired p / mixed-model p vs {baseline_label}, tests on Fisher z; "
                 f"* p<0.05, ** p<0.01, *** p<0.001, uncorrected)", fontsize=10)
    fig.savefig(output_dir / "motor_x_regions_combined_change_r.png", dpi=200)
    fig.savefig(output_dir / "motor_x_regions_combined_change_r.svg")
    plt.close(fig)
    print(f"saved motor x regions figures in {output_dir}")
