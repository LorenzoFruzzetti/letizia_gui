"""The example cohort through every step, with the commands the GUI builds, and M8:
nothing is written inside the code folder.

Each job runs as the GUI runs it (``wfci_gui.logic.params.build_job``: the same
arguments, environment and working folder), with this Python instead of ``conda run``.
Step 3's hand drawing is ``make_example_cohort.py --place-roi-sets``; the two
confirmations the user gives on the pages (acquisition facts, exclusions) are set in the
cohort file as the pages do.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from wfci_gui.logic.cohort import load_cohort, load_tables, save_cohort
from wfci_gui.logic.conda import job_environment
from wfci_gui.logic.params import build_job
from wfci_gui.logic.steps import cache_windows, chosen_variants, compute_statuses

REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".pytest_cache"}


def code_folder_state() -> dict[str, tuple[int, int]]:
    """(size, modification time) of every file of the code folder, by relative path."""
    state = {}
    for path in REPO_ROOT.rglob("*"):
        relative = path.relative_to(REPO_ROOT)
        if relative.parts[0] in IGNORED or not path.is_file():
            continue
        stat = path.stat()
        state[relative.as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return state


def run(arguments: list[str], cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run([sys.executable, "-u", *arguments], cwd=cwd, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, f"{arguments[:2]} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return result.stdout


def run_job(key: str, cohort_file: Path, base_env: dict[str, str], variant=None) -> str:
    """One GUI job: built by the GUI's command builder, run with this Python."""
    cohort = load_cohort(cohort_file)
    job = build_job(key, cohort, load_tables(cohort), Path("conda.bat"), variant)
    if job.before is not None:
        job.before()
    python = job.command.index("python")
    assert job.command[python + 1] == "-u"
    env = job_environment(REPO_ROOT, job.env, base=base_env)
    return run(job.command[python + 2:], job.cwd, env)


@pytest.fixture
def base_env(tmp_path, monkeypatch):
    import os
    env = dict(os.environ, LOCALAPPDATA=str(tmp_path / "localappdata"), MPLBACKEND="Agg")
    return env


def test_example_cohort_runs_every_step_and_leaves_the_code_folder_unchanged(tmp_path, base_env):
    before = code_folder_state()
    workspace = tmp_path / "workspace"
    env = job_environment(REPO_ROOT, base=base_env)
    run([str(REPO_ROOT / "scripts" / "make_example_cohort.py"), "--workspace", str(workspace),
         "--seconds", "12"], tmp_path, env)
    cohort_file = workspace / "example" / "cohort.yaml"
    statuses = lambda: compute_statuses(load_cohort(cohort_file), load_tables(load_cohort(cohort_file)), None, [])

    run_job("scan", cohort_file, base_env)
    run_job("facts", cohort_file, base_env)
    cohort = load_cohort(cohort_file)
    cohort.data["acquisition"]["confirmed"] = True          # the Acquisition page's Confirm
    save_cohort(cohort)
    assert statuses()["acquisition"].done

    run_job("bake", cohort_file, base_env)
    run([str(REPO_ROOT / "scripts" / "make_example_cohort.py"), "--place-roi-sets",
         "--cohort", str(cohort_file)], cohort_file.parent, env)
    run_job("rebuild", cohort_file, base_env)
    run_job("preflight", cohort_file, base_env)
    assert statuses()["rois"].done, statuses()["rois"].detail

    plan = run_job("extract", cohort_file, base_env)
    assert "(from manifest)" in plan
    assert statuses()["extract"].done, statuses()["extract"].detail

    run_job("qc", cohort_file, base_env)
    import pandas as pd
    qc = pd.read_csv(load_cohort(cohort_file).layout.outputs / "qc_table.csv", dtype=str, keep_default_na=False)
    assert set(qc["rect_agrees"]) == {"True"}, qc[["recording", "rect_channel_order", "rect_ratio"]]
    assert set(qc["suggest"]) == {"ok"}, qc[["recording", "reason"]]
    cohort = load_cohort(cohort_file)
    cohort.data["exclusions_confirmed"] = True               # the QC page's Confirm
    save_cohort(cohort)
    assert statuses()["qc"].done

    for window in cache_windows(cohort):
        run_job("cache_median", cohort_file, base_env, window)
    assert statuses()["caches"].done, statuses()["caches"].detail
    with pytest.raises(ValueError, match="run step 7"):     # the optional cache needs the design
        build_job("cache_blpedestal", load_cohort(cohort_file), load_tables(cohort), Path("conda.bat"))
    run_job("design", cohort_file, base_env)
    assert statuses()["design"].done, statuses()["design"].detail
    run_job("cache_blpedestal", cohort_file, base_env)
    for variant in chosen_variants(cohort):
        run_job("connectivity", cohort_file, base_env, variant)
        run_job("stats", cohort_file, base_env, variant)
    final = statuses()
    assert final["analysis_a"].done, final["analysis_a"].detail

    after = code_folder_state()
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    assert changed == [], f"written inside the code folder: {changed[:20]}"
