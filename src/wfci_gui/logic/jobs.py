"""Job queue model and job history. No Qt: the controller runs them.

One job runs at a time; the rest wait in order. Every finished job appends one JSON line
to ``<cohort folder>/outputs/gui_jobs.jsonl``: step, command, overrides, start, end, exit code, log.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Job:
    step: str                           # step key the job belongs to (status/history)
    label: str                          # shown in the queue, e.g. "Extract: smoke test"
    key: str = ""                       # job key (logic.params.JOBS) or "setup_check" / "repair"
    command: list[str] = field(default_factory=list)  # full argv, conda.bat first
    cwd: Path = Path(".")
    env: dict[str, str] = field(default_factory=dict)        # job-specific variables
    overrides: dict[str, Any] = field(default_factory=dict)  # recorded in the history
    log_path: Path | None = None
    history_path: Path | None = None
    # Python callable run in the GUI process before the command starts (e.g. step 3's
    # copy of the cohort's reviewed sets). It must be quick; None = nothing.
    before: Any = None
    start: str = ""
    end: str = ""
    exit_code: int | None = None

    def record(self) -> dict[str, Any]:
        return dict(step=self.step, key=self.key, label=self.label, command=self.command,
                    env=self.env, overrides=self.overrides, start=self.start, end=self.end,
                    exit_code=self.exit_code, log_path=str(self.log_path or ""))


class JobQueue:
    """FIFO queue with at most one running job."""

    def __init__(self):
        self.pending: list[Job] = []
        self.running: Job | None = None

    @property
    def busy(self) -> bool:
        return self.running is not None

    def add(self, job: Job) -> None:
        self.pending.append(job)

    def next(self) -> Job | None:
        """Move the first pending job to running (None when busy or empty)."""
        if self.running is not None or not self.pending:
            return None
        self.running = self.pending.pop(0)
        self.running.start = timestamp()
        return self.running

    def finish(self, exit_code: int) -> Job:
        job = self.running
        if job is None:
            raise RuntimeError("no job is running")
        job.exit_code = exit_code
        job.end = timestamp()
        self.running = None
        return job

    def clear_pending(self) -> list[Job]:
        dropped, self.pending = self.pending, []
        return dropped

    def remove(self, index: int) -> Job:
        return self.pending.pop(index)


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_file_name(step: str, when: datetime | None = None) -> str:
    """``<step>_<timestamp>.log``."""
    when = when or datetime.now()
    return f"{step}_{when.strftime('%Y%m%d_%H%M%S')}.log"


def append_history(path: Path, job: Job) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(job.record(), default=str) + "\n")


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
