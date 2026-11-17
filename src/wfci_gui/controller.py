"""The only code that changes application state (GUI_DESIGN_GUIDELINES.md 2.3).

Views call one method per user intent and redraw when ``changed`` is emitted. Jobs run
one at a time as ``QProcess`` subprocesses (gui_spec.md D1, D10, section 7); their output
is streamed to the log dock and to ``<output_root>/logs/<job>_<timestamp>.log``.
"""

from __future__ import annotations

import copy
import ctypes
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qtkit.qt import QtCore

from .logic import REPO_ROOT
from .logic.cohort import (
    TABLE_KINDS, Cohort, dump_cohort, import_table, load_cohort, load_tables, new_cohort_data,
    pending_diff, save_cohort, used_recordings, validate_cohort, validate_tables, write_table,
)
from .logic.conda import (
    CondaNotFound, conda_candidates, display_command, env_prefix, job_environment,
    powershell_command, python_command,
)
from .logic.jobs import Job, JobQueue, append_history, log_file_name, read_history
from .logic.params import JOBS, build_job
from .logic.presets import import_overrides
from .logic.steps import (
    blocking_steps, compute_statuses, history_notes, quick_checks, read_setup_result, setup_check_path,
)


@dataclass
class AppState:
    cohort: Cohort | None = None
    saved_text: str = ""                                  # cohort file text as last loaded/saved
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    dirty_tables: set[str] = field(default_factory=set)
    statuses: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    setup_result: dict | None = None
    quick_checks: list[dict] = field(default_factory=list)
    conda: Path | None = None
    env_path: Path | None = None
    advanced: bool = False
    busy: bool = False                                    # PipelineGuideView reads this

    @property
    def cohort_dirty(self) -> bool:
        return self.cohort is not None and dump_cohort(self.cohort.data) != self.saved_text


class Controller(QtCore.QObject):
    changed = QtCore.Signal()          # any state change: views redraw
    log_line = QtCore.Signal(str)      # one line of job output (or a GUI message)
    queue_changed = QtCore.Signal()

    def __init__(self, config_dir: Path, repo: Path = REPO_ROOT, environ: dict | None = None):
        super().__init__()
        self.config_dir = Path(config_dir)
        self.repo = Path(repo)
        self.state = AppState()
        self.queue = JobQueue()
        self.process: QtCore.QProcess | None = None
        self.log_handle = None
        self._partial = ""
        self.state.conda = self._find_conda(environ)
        self.state.env_path = env_prefix(self.state.conda) if self.state.conda else None
        self.refresh_status()

    @staticmethod
    def _find_conda(environ) -> Path | None:
        return next((c for c in conda_candidates(environ) if c.is_file()), None)

    def conda(self) -> Path:
        if self.state.conda is None:
            raise CondaNotFound("conda was not found: install it, then run setup_env.ps1 by hand")
        return self.state.conda

    # ----- cohort -------------------------------------------------------------------
    def open_cohort(self, path: str | Path) -> None:
        cohort = load_cohort(path)
        self.state.cohort = cohort
        self.state.saved_text = dump_cohort(cohort.data)
        self.state.tables = load_tables(cohort)
        self.state.dirty_tables = set()
        self.refresh_status()

    def close_cohort(self) -> None:
        self.state.cohort = None
        self.state.tables = {}
        self.state.dirty_tables = set()
        self.refresh_status()

    def create_cohort(self, data: dict[str, Any], directory: Path) -> Path:
        """Write a new cohort file with empty tables, then open it."""
        validate_cohort(data, data["name"])
        path = Path(directory) / f"{data['name']}.yaml"
        if path.exists():
            raise FileExistsError(f"{path} already exists")
        cohort = Cohort(data, path)
        save_cohort(cohort)
        for kind in TABLE_KINDS:
            write_table(cohort.table_path(kind), kind, load_tables(cohort)[kind])
        self.open_cohort(path)
        return path

    def new_cohort_data(self, **values) -> dict[str, Any]:
        return new_cohort_data(**values)

    def pending_diff(self) -> str:
        return pending_diff(self.state.cohort)

    def save_cohort(self) -> None:
        """Save the cohort file and every edited table (the view showed the diff first)."""
        cohort = self._require_cohort()
        save_cohort(cohort)
        self.state.saved_text = dump_cohort(cohort.data)
        for kind in sorted(self.state.dirty_tables):
            write_table(cohort.table_path(kind), kind, self.state.tables[kind])
        self.state.dirty_tables = set()
        self.refresh_status()

    def revert_cohort(self) -> None:
        self.open_cohort(self._require_cohort().path)

    def _require_cohort(self) -> Cohort:
        if self.state.cohort is None:
            raise RuntimeError("no cohort is open")
        return self.state.cohort

    def _edit(self, mutate) -> None:
        """Apply ``mutate`` to a copy of the cohort data, validate, then keep it."""
        cohort = self._require_cohort()
        data = copy.deepcopy(cohort.data)
        mutate(data)
        validate_cohort(data, str(cohort.path))
        cohort.data = data
        self.refresh_status()

    def set_cohort_value(self, key: str, value: Any) -> None:
        def mutate(data):
            if value is None and key not in ("link_root",):
                data.pop(key, None)
            else:
                data[key] = value
        self._edit(mutate)

    def set_acquisition(self, values: dict[str, Any], confirmed: bool) -> None:
        def mutate(data):
            data["acquisition"].update(values)
            data["acquisition"]["confirmed"] = confirmed
        self._edit(mutate)

    def set_override(self, job: str, name: str, value: Any) -> None:
        self._edit(lambda data: data.setdefault("overrides", {}).setdefault(job, {}).__setitem__(name, value))

    def reset_override(self, job: str, name: str) -> None:
        def mutate(data):
            group = data.get("overrides", {}).get(job, {})
            group.pop(name, None)
            if not group:
                data.get("overrides", {}).pop(job, None)
        self._edit(mutate)

    def replace_overrides(self, job: str, values: dict[str, Any]) -> None:
        def mutate(data):
            overrides = data.setdefault("overrides", {})
            if values:
                overrides[job] = dict(values)
            else:
                overrides.pop(job, None)
        self._edit(mutate)

    def import_all_overrides(self, path: Path) -> dict[str, dict[str, Any]]:
        overrides = import_overrides(path)
        self._edit(lambda data: data.__setitem__("overrides", overrides))
        return overrides

    def set_exclusions(self, exclusions: list[dict[str, str]], confirmed: bool) -> None:
        def mutate(data):
            data["exclusions"] = exclusions
            data["exclusions_confirmed"] = confirmed
        self._edit(mutate)

    # ----- tables -------------------------------------------------------------------
    def set_table(self, kind: str, frame: pd.DataFrame) -> None:
        self.state.tables[kind] = frame.reset_index(drop=True)
        self.state.dirty_tables.add(kind)
        self.refresh_status()

    def import_table(self, kind: str, path: Path) -> None:
        """Import a CSV; an invalid one is refused (TableError lists every row and reason)."""
        cohort = self._require_cohort()
        animals = set(self.state.tables["animals"]["animal"]) if kind == "recordings" else None
        frame = import_table(path, kind, cohort.timepoints, animals)
        self.set_table(kind, frame)

    def table_issues(self) -> dict:
        return validate_tables(self.state.cohort, self.state.tables) if self.state.cohort else {}

    # ----- status -------------------------------------------------------------------
    def refresh_status(self) -> None:
        state = self.state
        state.setup_result = read_setup_result(self.config_dir)
        n_recordings = len(used_recordings(state.tables)) if state.tables else 0
        state.quick_checks = quick_checks(state.cohort, state.conda, state.env_path, n_recordings)
        if state.cohort is not None:
            state.statuses = compute_statuses(state.cohort, state.tables, state.setup_result,
                                              state.quick_checks)
            state.history = read_history(self.history_path())
        else:
            state.statuses = {}
            state.history = read_history(self.history_path())
        state.busy = self.queue.busy
        self.changed.emit()

    def history_path(self) -> Path:
        cohort = self.state.cohort
        return (cohort.path_of("output_root") if cohort else self.config_dir) / "gui_jobs.jsonl"

    def logs_dir(self) -> Path:
        cohort = self.state.cohort
        return (cohort.path_of("output_root") if cohort else self.config_dir) / "logs"

    def history_notes(self) -> dict[str, str]:
        return history_notes(self.state.history)

    def blockers(self, step: str) -> list[str]:
        return blocking_steps(step, self.state.statuses) if self.state.statuses else []

    # ----- jobs ---------------------------------------------------------------------
    def make_job(self, job: str, variant: Any = None, extra: dict | None = None,
                 label: str | None = None) -> Job:
        cohort = self._require_cohort()
        return build_job(job, cohort, self.state.tables, self.conda(), variant, extra, label, self.repo)

    def setup_check_job(self, tests: bool) -> Job:
        arguments = ["-m", "wfci_gui.setup_check", "--out", str(setup_check_path(self.config_dir))]
        if tests:
            arguments.append("--tests")
        return self._plain_job("setup", "setup_check", "Setup checks" + (" + tests" if tests else ""),
                               python_command(self.conda(), arguments))

    def repair_job(self) -> Job:
        return self._plain_job("setup", "repair", "Repair environment (setup_env.ps1)",
                               powershell_command(self.repo / "env" / "setup_env.ps1",["-SkipTests"]))

    def new_roi_editor_job(self) -> Job:
        """Step 3 with the new editor (python -m wfci_gui.roi_editor), settings from the cohort file."""
        cohort = self._require_cohort()
        return self._plain_job("rois", NEW_ROI_EDITOR_KEY, "New ROI editor", python_command(
            self.conda(), ["-m", "wfci_gui.roi_editor", "--cohort", str(cohort.path)]))

    def viewer_job(self, script: str, arguments: list[str], label: str, step: str = "qc") -> Job:
        """A study script run as is (QC viewers): its own flags, no overrides."""
        return self._plain_job(step, Path(script).stem, label,
                               python_command(self.conda(), [script, *arguments]))

    def _plain_job(self, step: str, key: str, label: str, command: list[str]) -> Job:
        return Job(step=step, key=key, label=label, command=command, cwd=self.repo,
                   log_path=self.logs_dir() / log_file_name(key), history_path=self.history_path())

    def enqueue(self, job: Job) -> None:
        self._check_saved()
        self.queue.add(job)
        self.queue_changed.emit()
        self._start_next()

    def _check_saved(self) -> None:
        """Jobs read the cohort file and tables from disk, so they must be saved first."""
        if self.state.cohort_dirty or self.state.dirty_tables:
            raise RuntimeError("The cohort has unsaved changes: save it before running a job "
                               "(jobs read the cohort file from disk).")

    def remove_pending(self, index: int) -> None:
        self.queue.remove(index)
        self.queue_changed.emit()

    def stop(self) -> None:
        """Kill the running job's whole process tree; queued jobs are kept."""
        if self.process is None:
            return
        pid = self.process.processId()
        self.log_line.emit(f"[gui] stopping {self.queue.running.label} (process tree {pid})")
        if os.name == "nt":
            # conda.bat -> conda -> python: QProcess.kill would leave python running.
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            self.process.kill()

    def launch_detached(self, script: str, arguments: list[str]) -> None:
        """Start an interactive viewer outside the queue (it can stay open while jobs run)."""
        process = QtCore.QProcess(self)
        command = python_command(self.conda(), [script, *arguments])
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(self.repo))
        process.setProcessEnvironment(self._process_environment({}))
        process.startDetached()
        self.log_line.emit(f"[gui] opened {script} {' '.join(arguments)}")

    def _process_environment(self, extra: dict[str, str]) -> QtCore.QProcessEnvironment:
        environment = QtCore.QProcessEnvironment()
        for key, value in job_environment(self.repo, extra).items():
            environment.insert(key, value)
        return environment

    def _start_next(self) -> None:
        job = self.queue.next()
        if job is None:
            return
        if job.before is not None:
            job.before()
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(job.log_path, "w", encoding="utf-8", newline="")
        header = [f"# job: {job.label}", f"# started: {job.start}",
                  f"# command: {display_command(job.command, job.env)}"]
        header += [f"# override {k} = {v!r}" for k, v in job.overrides.items()]
        for line in header:
            self._emit(line)
        self.process = QtCore.QProcess(self)
        self.process.setProgram(job.command[0])
        self.process.setArguments(job.command[1:])
        self.process.setWorkingDirectory(str(job.cwd))
        self.process.setProcessEnvironment(self._process_environment(job.env))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self._partial = ""
        keep_awake(True)
        self.process.start()
        self.state.busy = True
        self.queue_changed.emit()
        self.changed.emit()

    def _emit(self, line: str) -> None:
        if self.log_handle is not None:
            self.log_handle.write(line + "\r\n")
            self.log_handle.flush()
        self.log_line.emit(line)

    def _read_output(self) -> None:
        text = self._partial + bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        lines = text.replace("\r\n", "\n").split("\n")
        self._partial = lines.pop()
        for line in lines:
            self._emit(line)

    def _process_error(self, error) -> None:
        # FailedToStart never emits finished: report it and finish the job by hand.
        if error == QtCore.QProcess.FailedToStart:
            self._emit(f"[gui] could not start {self.queue.running.command[0]}: {self.process.errorString()}")
            self._finished(-1, None)

    def _finished(self, exit_code: int, _status) -> None:
        if self.queue.running is None:
            return
        if self._partial:
            self._emit(self._partial)
            self._partial = ""
        job = self.queue.finish(exit_code)
        self._emit(f"# finished: {job.end}, exit code {exit_code}")
        self.log_handle.close()
        self.log_handle = None
        append_history(job.history_path, job)
        self.process.deleteLater()
        self.process = None
        if not self.queue.pending:
            keep_awake(False)
        self.state.busy = False
        self._after(job)
        self.refresh_status()
        self.queue_changed.emit()
        self._start_next()

    def _after(self, job: Job) -> None:
        """Step 3: when the ROI editor closes, rebuild this cohort's new sets, then pre-flight."""
        if job.key not in ("roi_select", NEW_ROI_EDITOR_KEY) or job.exit_code != 0 \
                or self.state.cohort is None:
            return
        if self.has_new_reviewed_sets():
            self.queue.add(self.make_job("rebuild"))
            self.log_line.emit("[gui] ROI editor closed: queued rebuild of this cohort's new sets")
        else:
            self.log_line.emit("[gui] ROI editor closed: no new or changed sets, nothing to rebuild")
        self.queue.add(self.make_job("preflight"))

    def has_new_reviewed_sets(self) -> bool:
        """True when step 3 has reviewed sets newer than their rebuilt copies."""
        cohort = self.state.cohort
        if cohort is None:
            return False
        reviewed = cohort.path_of("reviewed_roi_set_dir")
        rebuilt = cohort.path_of("roi_set_dir")
        for row in used_recordings(self.state.tables).itertuples():
            name = f"{row.day}_{row.animal}_{row.recording}.yaml"
            source, target = reviewed / name, rebuilt / name
            if source.exists() and (not target.exists() or target.stat().st_mtime < source.stat().st_mtime):
                return True
        return False

    def job_spec(self, job: str):
        return JOBS[job]


# Job key of the new ROI editor; when it closes, step 3 rebuilds and pre-flights as for the old one.
NEW_ROI_EDITOR_KEY = "roi_editor_new"


def keep_awake(on: bool) -> None:
    """Keep Windows from sleeping while a job runs (as run_botox_2024_overnight.ps1 does)."""
    if os.name != "nt":
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    flags = es_continuous | es_system_required if on else es_continuous
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
