"""Find conda and build the exact subprocess commands every job runs (gui_spec.md D1, 7).

Every job is ``conda.bat run --no-capture-output -n wfci_gui python -u <script> ...``:
  * through ``conda run``, because calling the env's python.exe directly crashes numpy's
    BLAS (CLAUDE.md 9.17);
  * with ``--no-capture-output``, because otherwise errors are hidden (9.25);
  * with ``PYTHONPATH=<repo>\\src;<repo>\\scripts``, because the env's editable install may point at
    another checkout (9.13 / 9.26).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ENV_NAME = "wfci_gui"
# conda.bat is a batch file: cmd.exe re-parses its arguments after the quotes are gone,
# so these characters would be redirections or variable expansions (CLAUDE.md 9.18, 9.39).
CMD_METACHARACTERS = set('%^&|<>"')


class CondaNotFound(FileNotFoundError):
    pass


def conda_candidates(environ: dict[str, str] | None = None) -> list[Path]:
    """Where conda is looked for, in order (spec 9.1)."""
    environ = dict(os.environ if environ is None else environ)
    home = Path(environ.get("USERPROFILE", str(Path.home())))
    candidates = [home / "anaconda3" / "condabin" / "conda.bat",
                  home / "miniconda3" / "condabin" / "conda.bat"]
    if environ.get("CONDA_EXE"):
        exe = Path(environ["CONDA_EXE"])
        # CONDA_EXE is ...\Scripts\conda.exe; prefer the condabin\conda.bat next to it.
        candidates += [exe.parent.parent / "condabin" / "conda.bat", exe]
    return candidates


def find_conda(environ: dict[str, str] | None = None) -> Path:
    for candidate in conda_candidates(environ):
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(c) for c in conda_candidates(environ))
    raise CondaNotFound(f"conda was not found. Tried:\n  {tried}\n"
                        "Install Anaconda/Miniconda or set CONDA_EXE.")


def conda_root(conda: Path) -> Path:
    """``...\\anaconda3\\condabin\\conda.bat`` -> ``...\\anaconda3``."""
    conda = Path(conda)
    return conda.parent.parent


def env_prefix(conda: Path, env_name: str = ENV_NAME) -> Path:
    return conda_root(conda) / "envs" / env_name


def check_arguments(arguments: list[str]) -> None:
    """Refuse an argument cmd.exe would mangle, instead of running a different command."""
    for argument in arguments:
        bad = sorted(set(argument) & CMD_METACHARACTERS)
        if bad:
            raise ValueError(f"argument {argument!r} contains {bad}, which cmd.exe re-parses when "
                             "conda.bat runs (CLAUDE.md 9.18). Use a value without them.")


def python_command(conda: Path, arguments: list[str], env_name: str = ENV_NAME) -> list[str]:
    """``conda.bat run --no-capture-output -n <env> python -u <arguments...>``."""
    arguments = [str(a) for a in arguments]
    check_arguments(arguments)
    return [str(conda), "run", "--no-capture-output", "-n", env_name, "python", "-u", *arguments]


def powershell_command(script: Path, arguments: list[str] | None = None) -> list[str]:
    """Run a .ps1 file (setup_env.ps1) without depending on the user's execution policy."""
    return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
            *[str(a) for a in arguments or []]]


def job_environment(repo_root: Path, extra: dict[str, str] | None = None,
                    base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment of a job: the GUI's own, plus PYTHONPATH and the job's variables."""
    environment = dict(os.environ if base is None else base)
    environment["PYTHONPATH"] = f"{repo_root / 'src'};{repo_root / 'scripts'}"
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update({key: str(value) for key, value in (extra or {}).items()})
    return environment


def display_command(command: list[str], extra_env: dict[str, str] | None = None) -> str:
    """The command as the user could paste it in PowerShell (shown in the advanced view)."""
    prefix = "".join(f"$env:{key} = '{value}'; " for key, value in (extra_env or {}).items())
    return prefix + "& " + subprocess.list2cmdline(command)
