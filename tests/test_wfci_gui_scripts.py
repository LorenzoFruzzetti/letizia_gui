"""Acceptance checks of the new scripts against the real cohorts (gui_spec.md 5.1).

Skipped where the real files are not on this machine. The raw disks are not needed:
the manifests are rebuilt from the cohorts' recordings tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from wfci_gui.logic.cohort import load_cohort, load_tables, used_recordings
from wfci_gui.logic.manifest import manifest_rows, manifest_text

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _cohort(name: str):
    path = REPO_ROOT / "cohorts" / f"{name}.yaml"
    if not path.exists():
        pytest.skip(f"{path} not present")
    return load_cohort(path)


@pytest.mark.parametrize("name", ["botox_2023", "botox_2024", "botox_2022"])
def test_manifest_reproduced_byte_for_byte(name):
    cohort = _cohort(name)
    on_disk = cohort.path_of("manifest")
    if not on_disk.exists():
        pytest.skip(f"{on_disk} not present")
    tables = load_tables(cohort)
    text = manifest_text(manifest_rows(cohort, tables["recordings"], tables["animals"]))
    assert text.encode("utf-8") == on_disk.read_bytes()


@pytest.mark.parametrize("name", ["botox_2023", "botox_2024"])
def test_roi_preflight_has_no_violations_on_existing_sets(name):
    from roi_preflight import check_set
    cohort = _cohort(name)
    used = used_recordings(load_tables(cohort))
    folder = cohort.path_of("roi_set_dir")
    names = [f"{r.day}_{r.animal}_{r.recording}.yaml" for r in used.itertuples()]
    if not all((folder / n).exists() for n in names):
        pytest.skip("rebuilt ROI sets not present")
    for name_ in names:
        data = yaml.safe_load((folder / name_).read_text(encoding="utf-8"))
        assert check_set(data, [-29, 47, -44, 43], [128, 128]) == [], name_


def test_preflight_catches_a_box_outside_the_window():
    from roi_preflight import check_set
    template = yaml.safe_load((REPO_ROOT / "atlas" / "cortex22_roi_set.yaml").read_text(encoding="utf-8"))
    data = dict(template, bregma_row=120, bregma_col=134, grid=[128, 128])
    data["boxes"] = dict(data["boxes"], V1L=dict(row_start=45, row_end=50, col_start=-25, col_end=-20))
    problems = check_set(data, [-29, 47, -44, 43], [128, 128])
    assert any("V1L outside the dump window" in p for p in problems)
    near_edge = dict(template, bregma_row=10, bregma_col=134, grid=[128, 128])
    assert any("does not fit" in p for p in check_set(near_edge, [-29, 47, -44, 43], [128, 128]))


def test_design_2023_reproduced():
    from build_cohort_design import build_design
    cohort = _cohort("botox_2023")
    existing = cohort.path_of("design_csv")
    if not existing.exists():
        pytest.skip("2023 design table not present")
    design = build_design(cohort, load_tables(cohort))
    text = design.to_csv(index=False, lineterminator="\r\n")
    if cohort.exclusions:
        pytest.skip("the 2023 cohort file now lists exclusions; the reference table has none")
    assert text.encode("utf-8") == existing.read_bytes()
