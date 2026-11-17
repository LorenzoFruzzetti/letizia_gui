"""wfci_gui.logic: plain pytest, no Qt (gui_spec.md 12)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest
import yaml

from gui_cohort_tree import make_cohort, recording_keys, write_extraction, write_rebuilt_sets
from wfci_gui.logic.cohort import (
    CohortError, TableError, dump_cohort, import_table, load_cohort, load_tables, pending_diff,
    save_cohort, validate_table,
)
from wfci_gui.logic.conda import (
    CondaNotFound, check_arguments, find_conda, job_environment, python_command,
)
from wfci_gui.logic.defaults import (
    OverrideError, apply_assignment_overrides, format_literal, read_script_parameters,
)
from wfci_gui.logic.design import check_design
from wfci_gui.logic.jobs import Job, JobQueue, append_history, read_history
from wfci_gui.logic.manifest import manifest_rows, manifest_text
from wfci_gui.logic.params import Parameter, flag_arguments
from wfci_gui.logic.presets import (
    diff_overrides, hash_warning, list_presets, load_preset, save_preset,
)
from wfci_gui.logic.steps import compute_statuses

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- cohort file
def test_cohort_round_trip_and_diff(tmp_path):
    cohort = load_cohort(make_cohort(tmp_path))
    assert cohort.name == "fake"
    assert pending_diff(cohort) == "" or "---" in pending_diff(cohort)  # canonical order may differ
    save_cohort(cohort)
    assert pending_diff(cohort) == ""
    cohort.data["timepoints"] = ["BL", "3D", "15D"]
    diff = pending_diff(cohort)
    assert "-timepoints: [BL, 3D]" in diff and "+timepoints: [BL, 3D, 15D]" in diff
    save_cohort(cohort)
    assert load_cohort(cohort.path).timepoints == ["BL", "3D", "15D"]
    assert not list(cohort.path.parent.glob(".*.tmp"))  # atomic save left nothing behind


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(typo_key=1), "unknown key"),
    (lambda d: d.pop("manifest"), "missing required"),
    (lambda d: d["acquisition"].update(frame_order="random"), "frame_order"),
    (lambda d: d["acquisition"].update(colour="blue"), "acquisition: unknown"),
    (lambda d: d.update(exclusions=[{"recording": "230101_M1/t1", "reason": " "}]), "needs a reason"),
    (lambda d: d.update(exclusions=[{"recording": "M1/t1", "reason": "x"}]), "not <yymmdd>"),
    (lambda d: d.update(overrides={"nonsense": {}}), "unknown job"),
])
def test_cohort_validation_rejects(tmp_path, mutate, message):
    path = make_cohort(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(CohortError, match=message):
        load_cohort(path)


def test_relative_paths_are_relative_to_repo(tmp_path):
    cohort = load_cohort(make_cohort(tmp_path))
    cohort.data["pixel_root"] = "pixel_data_x"
    assert cohort.path_of("pixel_root") == REPO_ROOT / "pixel_data_x"
    assert cohort.path_of("design_csv") == REPO_ROOT / "pixel_data_x" / "experimental_design.csv"


# ----------------------------------------------------------------------------- tables
def test_table_import_accepts_and_normalises(tmp_path):
    source = tmp_path / "s.csv"
    source.write_text("session_folder,day,timepoint,day_zero,include\nA,230101,BL,0,YES\n", encoding="utf-8")
    frame = import_table(source, "sessions", ["BL", "3D"])
    assert frame.loc[0, "day_zero"] == "false" and frame.loc[0, "include"] == "true"


def test_table_import_refuses_with_row_and_reason(tmp_path):
    source = tmp_path / "s.csv"
    source.write_text("session_folder,day,timepoint,day_zero,include\n"
                      "A,2301,BL,false,true\nB,230102,99D,maybe,true\n", encoding="utf-8")
    with pytest.raises(TableError) as error:
        import_table(source, "sessions", ["BL", "3D"])
    text = str(error.value)
    assert "row 1: day = '2301'" in text
    assert "row 2: day_zero = 'maybe'" in text
    assert "row 2: timepoint '99D'" in text


def test_table_columns_and_cross_checks():
    frame = pd.DataFrame({"day": ["230101"], "animal": ["M9"], "recording": ["t1"], "path": ["x"],
                          "n_files": ["6"], "status": ["ok"], "use": ["true"], "channel_order": ["sideways"]})
    reasons = [str(i) for i in validate_table("recordings", frame, known_animals={"M1"})]
    assert any("channel_order" in r for r in reasons)
    assert any("not in animals.csv" in r for r in reasons)
    extra = frame.assign(extra="1")
    assert "unknown column" in str(validate_table("recordings", extra)[0])
    animals = pd.DataFrame({"raw_name": ["000"], "animal": ["000"], "group": ["NA"], "mouse_line": ["x"],
                            "include": ["true"]})
    assert "all digits" in str(validate_table("animals", animals)[0])


# ----------------------------------------------------------------------------- conda
def _fake_conda(home: Path, folder: str) -> Path:
    path = home / folder / "condabin" / "conda.bat"
    path.parent.mkdir(parents=True)
    path.write_text("@echo off\n", encoding="utf-8")
    return path


def test_conda_discovery_order(tmp_path):
    environ = {"USERPROFILE": str(tmp_path)}
    with pytest.raises(CondaNotFound):
        find_conda(environ)
    exe_conda = _fake_conda(tmp_path / "custom", "conda")
    environ["CONDA_EXE"] = str(exe_conda.parent.parent / "Scripts" / "conda.exe")
    assert find_conda(environ) == exe_conda
    miniconda = _fake_conda(tmp_path, "miniconda3")
    assert find_conda(environ) == miniconda
    anaconda = _fake_conda(tmp_path, "anaconda3")
    assert find_conda(environ) == anaconda


def test_command_building():
    command = python_command(Path("C:/c/conda.bat"), ["script.py", "--x", "1"])
    assert command[1:7] == ["run", "--no-capture-output", "-n", "wfci_gui", "python", "-u"]
    assert command[7:] == ["script.py", "--x", "1"]
    environment = job_environment(Path("C:/repo"), {"CONNECTIVITY_STATS_COHORT": "2023"}, base={})
    assert environment["PYTHONPATH"] == f"{Path('C:/repo/src')};{Path('C:/repo/scripts')}"
    assert environment["CONNECTIVITY_STATS_COHORT"] == "2023"
    with pytest.raises(ValueError, match="cmd.exe"):
        check_arguments(["--label", "dF/F (%)"])


# ----------------------------------------------------------------------------- ast defaults
SCRIPT = textwrap.dedent('''
    import argparse, os
    from pathlib import Path
    MODES = ("a", "b")
    RUN_CONFIG = {"rate": 10.0, "root": Path("x"), "flag": True, "items": ["p", "q"], "mode": "a"}
    def parse(defaults):
        p = argparse.ArgumentParser()
        p.add_argument("--rate", type=float, default=defaults["rate"])
        p.add_argument("--root", type=Path, default=defaults["root"])
        p.add_argument("--flag", action=argparse.BooleanOptionalAction, default=defaults["flag"])
        p.add_argument("--items", nargs="*", default=defaults["items"])
        p.add_argument("--mode", choices=MODES, default=defaults["mode"])
        mode = p.add_mutually_exclusive_group()
        mode.add_argument("--full", dest="debug", action="store_false", default=False)
        mode.add_argument("--debug", dest="debug", action="store_true")
    cohort = os.environ.get("MY_COHORT", "2026")
    alpha = 0.05
    if cohort == "x":
        timepoints = ["BL", "3D"]
    else:
        timepoints = ["BL"]
    root = Path(__file__).parent / "out"
''')


def test_defaults_read_with_ast(tmp_path):
    path = tmp_path / "script.py"
    path.write_text(SCRIPT, encoding="utf-8")
    parameters = read_script_parameters(path)
    flags = {p.name: p for p in parameters.flags}
    assert flags["rate"].default == 10.0 and flags["rate"].literal
    assert not flags["root"].literal and "Path" in flags["root"].source
    assert flags["flag"].action == "boolean_optional"
    assert flags["mode"].choices == ("a", "b")
    assert flags["items"].nargs == "*" and flags["items"].default == ["p", "q"]
    assert flags["debug"].action == "bool_pair"
    env = {p.name: p for p in parameters.env}
    assert env["MY_COHORT"].default == "2026"
    assigns = {p.name: p for p in parameters.assigns}
    assert assigns["alpha"].default == 0.05 and assigns["alpha"].overridable
    assert assigns["timepoints"].conditional and assigns["timepoints"].overridable
    assert not assigns["root"].overridable          # computed: read-only in the GUI
    assert not assigns["cohort"].overridable


def test_flag_arguments():
    pair = Parameter("debug", "flag", flags={"--full": False, "--debug": True}, action="bool_pair")
    assert flag_arguments(pair, True) == ["--debug"] and flag_arguments(pair, False) == ["--full"]
    optional = Parameter("gsr", "flag", flags={"--gsr": None}, action="boolean_optional")
    assert flag_arguments(optional, False) == ["--no-gsr"]
    listed = Parameter("exclude", "flag", flags={"--exclude": None}, nargs="*")
    assert flag_arguments(listed, []) == ["--exclude"]
    assert flag_arguments(listed, ["a/t1", "b/t2"]) == ["--exclude", "a/t1", "b/t2"]
    custom = Parameter("csv", "flag", flags={"--csv": None}, type_name="custom")
    assert flag_arguments(custom, None) == ["--csv", "none"]


def test_overrides_replace_every_branch_and_refuse_computed():
    tree = apply_assignment_overrides(SCRIPT, {"timepoints": ["BL", "30D"], "alpha": 0.1})
    namespace = {"__file__": "x.py"}
    exec(compile(tree, "x.py", "exec"), namespace)
    assert namespace["timepoints"] == ["BL", "30D"] and namespace["alpha"] == 0.1
    with pytest.raises(OverrideError, match="computed"):
        apply_assignment_overrides(SCRIPT, {"root": "elsewhere"})
    with pytest.raises(OverrideError, match="no module-level assignment"):
        apply_assignment_overrides(SCRIPT, {"alpah": 0.1})
    assert format_literal(["BL", "3D"]) == "['BL','3D']"


def test_run_with_overrides_matches_direct_run(tmp_path):
    script = tmp_path / "flat.py"
    script.write_text("import sys\nvalues = [1, 2]\nname = 'a'\n"
                      "print(__name__, values, name, sys.argv[1:], __file__.endswith('flat.py'))\n",
                      encoding="utf-8")
    direct = subprocess.run([sys.executable, str(script), "arg"], capture_output=True, text=True, check=True)
    wrapped = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_with_overrides.py"), str(script), "--", "arg"],
                             capture_output=True, text=True, check=True)
    assert wrapped.stdout.splitlines()[-1] == direct.stdout.strip()
    changed = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_with_overrides.py"), str(script),
                              "--set", "values=[3,4]", "--set", "name='two words'"],
                             capture_output=True, text=True, check=True)
    assert "override values = [3, 4]" in changed.stdout
    assert changed.stdout.splitlines()[-1].startswith("__main__ [3, 4] two words")


# ----------------------------------------------------------------------------- presets, jobs
def test_presets_save_load_diff_and_hash(tmp_path):
    save_preset("extract", "fast", {"workers": 2}, "hash1", root=tmp_path)
    assert list_presets("extract", tmp_path) == ["fast"]
    payload = load_preset("extract", "fast", tmp_path)
    assert payload["overrides"] == {"workers": 2}
    assert hash_warning(payload, "hash1") == ""
    assert "different version" in hash_warning(payload, "hash2")
    assert diff_overrides({"workers": 1, "x": 5}, {"workers": 2}) == [("workers", 1, 2), ("x", 5, "<default>")]


def test_job_queue_and_history(tmp_path):
    queue = JobQueue()
    first, second = Job("scan", "a", command=["x"]), Job("qc", "b", command=["y"])
    queue.add(first)
    queue.add(second)
    assert queue.next() is first and queue.next() is None   # one at a time
    finished = queue.finish(1)
    append_history(tmp_path / "h.jsonl", finished)
    assert queue.next() is second
    record = read_history(tmp_path / "h.jsonl")[0]
    assert record["step"] == "scan" and record["exit_code"] == 1


# ----------------------------------------------------------------------------- design check
def test_design_completeness():
    design = pd.DataFrame({"animal": ["A", "A", "B", "C", "C"], "timepoint": ["BL", "3D", "BL", "BL", "3D"]})
    assert check_design(design, "BL", ["BL", "3D"], ["A", "C"]).complete
    result = check_design(design, "BL", ["BL", "3D"], ["A", "B", "C"])
    assert not result.complete and result.missing == [("B", "3D")]
    assert result.drop_animals == ["B"] and result.drop_timepoints == ["3D"]
    no_baseline = check_design(design.iloc[1:], "BL", ["BL", "3D"], ["A", "B", "C"])
    assert no_baseline.baseline_missing == ["A"]
    assert not check_design(design, "15D", ["BL", "3D"], ["A", "C"]).complete


# ----------------------------------------------------------------------------- status from disk
def _scan(cohort_path, *flags):
    import scan_cohort
    argv = sys.argv
    sys.argv = ["scan_cohort.py", "--cohort", str(cohort_path), *flags]
    scan_cohort.main()
    sys.argv = argv


def test_status_follows_the_files_on_disk(tmp_path):
    path = make_cohort(tmp_path)
    cohort = load_cohort(path)
    statuses = compute_statuses(cohort, load_tables(cohort), None, [])
    assert not statuses["scan"].done and not statuses["rois"].done

    _scan(path, "--no-make-links")
    cohort = load_cohort(path)
    tables = load_tables(cohort)
    statuses = compute_statuses(cohort, tables, None, [])
    assert statuses["scan"].done, statuses["scan"].detail
    assert statuses["acquisition"].done
    assert statuses["rois"].detail.startswith("0 of 8 drawn")

    keys = recording_keys()
    write_rebuilt_sets(cohort.path_of("roi_set_dir"), keys)
    import roi_preflight
    sys_argv = sys.argv
    sys.argv = ["roi_preflight.py", "--cohort", str(path)]
    with pytest.raises(SystemExit) as exit_info:
        roi_preflight.main()
    sys.argv = sys_argv
    assert exit_info.value.code == 0
    assert compute_statuses(cohort, tables, None, [])["rois"].done

    write_extraction(cohort.path_of("output_root"), cohort.path_of("pixel_root"), keys,
                     needs_decision=(keys[0],))
    statuses = compute_statuses(cohort, tables, None, [])
    assert statuses["extract"].done and "1 need a channel decision" in statuses["extract"].detail
    assert not statuses["qc"].done

    (cohort.path_of("output_root") / "qc_table.csv").write_text("recording\n", encoding="utf-8")
    cohort.data["exclusions"] = [{"recording": keys[0], "reason": "no alternation"}]
    cohort.data["exclusions_confirmed"] = True
    statuses = compute_statuses(cohort, tables, None, [])
    assert statuses["qc"].done
    assert not statuses["caches"].done                       # no cache markers yet
    for key in keys[1:]:
        (cohort.path_of("pixel_root") / key / "pixels_median_dff_20s_meta.json").write_text("{}")
    assert compute_statuses(cohort, tables, None, [])["caches"].done

    import build_cohort_design
    save_cohort(cohort)
    sys.argv = ["build_cohort_design.py", "--cohort", str(path)]
    build_cohort_design.main()
    sys.argv = sys_argv
    statuses = compute_statuses(load_cohort(path), tables, None, [])
    assert statuses["design"].done, statuses["design"].detail
    design = pd.read_csv(cohort.path_of("design_csv"), dtype=str)
    assert len(design) == 7 and set(design["timepoint"]) == {"BL", "3D"}   # excluded one is out

    # Analysis A: present and fresh -> done; an input rewritten later -> stale, not done.
    root = cohort.path_of("connectivity_root")
    for variant in ("pixels_dff_full", "pixels_dff_full_gsr", "pixels_median_dff_20s_full",
                    "pixels_median_dff_20s_full_gsr"):
        for key in keys[1:]:
            (root / variant / key).mkdir(parents=True)
            (root / variant / key / "roi_pixel_connectivity.npz").write_bytes(b"x")
    assert compute_statuses(cohort, tables, None, [])["analysis_a"].done
    volume = cohort.path_of("pixel_root") / keys[1] / "pixels_dff_full.npy"
    later = os.stat(volume).st_mtime + 100
    os.utime(volume, (later, later))
    status = compute_statuses(cohort, tables, None, [])["analysis_a"]
    assert not status.done and "stale" in status.warnings[0]


def test_scan_statuses_aliases_and_user_columns(tmp_path):
    path = make_cohort(tmp_path)
    (tmp_path / "raw" / "SESSION_A" / "m1_a" / "t2" / "frame_6.tif").unlink()  # frame 6 missing
    (tmp_path / "raw" / "SESSION_A" / "m2_a" / "notes").mkdir()
    _scan(path, "--no-make-links")
    cohort = load_cohort(path)
    recordings = load_tables(cohort)["recordings"]
    status = dict(zip(recordings["day"] + "_" + recordings["animal"] + "/" + recordings["recording"],
                      recordings["status"]))
    assert status["230101_M1/t2"] == "wrong_count"
    assert status["230101_M2/notes"] == "not_a_recording"
    assert set(recordings["animal"]) == {"M1", "M2"}        # m1_a and m1_b are aliases of M1
    # A user edit survives a rescan.
    recordings.loc[recordings["recording"] == "t1", "channel_order"] = "gcamp_first"
    recordings.to_csv(cohort.table_path("recordings"), index=False)
    _scan(path, "--no-make-links")
    again = load_tables(load_cohort(path))["recordings"]
    assert (again.loc[again["recording"] == "t1", "channel_order"] == "gcamp_first").all()
    # Recordings needing different explicit orders are split one row per recording.
    rows = manifest_rows(cohort, again, load_tables(cohort)["animals"])
    assert all(r["n_recordings"] == 1 for r in rows if r["animal"] == "M2")


def test_scan_refuses_unknown_animal_folder(tmp_path):
    path = make_cohort(tmp_path)
    (tmp_path / "raw" / "SESSION_B" / "mystery").mkdir()
    with pytest.raises(ValueError, match="mystery"):
        _scan(path, "--no-make-links")


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only (spec D6)")
def test_scan_creates_junction_view(tmp_path):
    path = make_cohort(tmp_path, link_root=True)
    _scan(path)
    link = tmp_path / "links" / "230104" / "M2" / "t1"
    assert link.is_dir() and len(list(link.glob("*.tif"))) == 6
    _scan(path)                     # second run: existing junctions are accepted, nothing raises
    manifest = (tmp_path / "manifests" / "fake_manifest.csv").read_text(encoding="utf-8")
    assert str(link) in manifest


def test_manifest_text_is_crlf(tmp_path):
    cohort = load_cohort(make_cohort(tmp_path))
    _scan(cohort.path, "--no-make-links")
    text = manifest_text(manifest_rows(cohort, load_tables(cohort)["recordings"], load_tables(cohort)["animals"]))
    assert text.count("\r\n") == text.count("\n") == 5   # header + 4 units


def test_dump_cohort_rejects_invalid(tmp_path):
    cohort = load_cohort(make_cohort(tmp_path))
    data = dict(cohort.data, name="bad name")
    with pytest.raises(CohortError):
        dump_cohort(data)
