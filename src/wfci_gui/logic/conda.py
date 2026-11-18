"""Find conda and build the exact subprocess commands every job runs.

Every job is ``conda.bat run --no-capture-output -n wfci_gui python -u <script> ...``:
  * through ``conda run``, because calling the env's python.exe directly crashes numpy's
    BLAS;
  * with ``--no-capture-output``, because otherwise errors are hidden;
  * with ``PYTHONPATH=<code folder>\\src;<code folder>\\scripts``: the env holds third-party
    packages only, and this folder's code comes from the folder itself.

Which conda: the one Install.bat recorded in ``%APPDATA%\\wfci_gui\\install.json``. The
search of the usual places is only a fallback for when that file is missing or names a
conda that is gone; the GUI then tells the user to run Install.bat.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ENV_NAME = "wfci_gui"
# conda.bat is a batch file: cmd.exe re-parses its arguments after the quotes are gone,
# so these characters would be redirections or variable expansions.
CMD_METACHARACTERS = set('%^&|<>"')


class CondaNotFound(FileNotFoundError):
    pass


INSTALL_HINT = "Run Install.bat in the program folder to install or repair it."


def install_record_path(environ: dict[str, str] | None = None) -> Path:
    """``%APPDATA%\\wfci_gui\\install.json``, written by the installer."""
    environ = dict(os.environ if environ is None else environ)
    return Path(environ.get("APPDATA") or str(Path.home() / ".config")) / "wfci_gui" / "install.json"


def installed_conda(environ: dict[str, str] | None = None) -> Path | None:
    """The conda the installer recorded, or None when there is no record or it is gone."""
    path = install_record_path(environ)
    if not path.is_file():
        return None
    conda = Path(json.loads(path.read_text(encoding="utf-8"))["conda"])
    return conda if conda.is_file() else None


def conda_candidates(environ: dict[str, str] | None = None) -> list[Path]:
    """Where conda is looked for when the installer left no record, in order."""
    environ = dict(os.environ if environ is None else environ)
    home = Path(environ.get("USERPROFILE", str(Path.home())))
    candidates = [home / name / "condabin" / "conda.bat" for name in ("anaconda3", "miniconda3", "miniforge3")]
    if environ.get("CONDA_EXE"):
        exe = Path(environ["CONDA_EXE"])
        # CONDA_EXE is ...\Scripts\conda.exe; prefer the condabin\conda.bat next to it.
        candidates += [exe.parent.parent / "condabin" / "conda.bat", exe]
    return candidates


def find_conda(environ: dict[str, str] | None = None) -> Path:
    """The installer's conda, else the first one found in the usual places."""
    recorded = installed_conda(environ)
    if recorded is not None:
        return recorded
    for candidate in conda_candidates(environ):
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(c) for c in conda_candidates(environ))
    raise CondaNotFound(f"conda was not found. Tried:\n  {tried}\n{INSTALL_HINT}")


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
                             "conda.bat runs. Use a value without them.")


def python_command(conda: Path, arguments: list[str], env_name: str = ENV_NAME) -> list[str]:
    """``conda.bat run --no-capture-output -n <env> python -u <arguments...>``."""
    arguments = [str(a) for a in arguments]
    check_arguments(arguments)
    return [str(conda), "run", "--no-capture-output", "-n", env_name, "python", "-u", *arguments]


def powershell_command(script: Path, arguments: list[str] | None = None) -> list[str]:
    """Run a .ps1 file (setup_env.ps1) without depending on the user's execution policy."""
    return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
            *[str(a) for a in arguments or []]]


def pycache_prefix(environ: dict[str, str] | None = None) -> Path:
    r"""``%LOCALAPPDATA%\wfci_gui\pycache``: where Python writes bytecode for the GUI and its jobs."""
    environ = os.environ if environ is None else environ
    base = environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(base) / "wfci_gui" / "pycache"


def job_environment(repo_root: Path, extra: dict[str, str] | None = None,
                    base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment of a job: the GUI's own, plus PYTHONPATH and the job's variables."""
    environment = dict(os.environ if base is None else base)
    environment["PYTHONPATH"] = f"{repo_root / 'src'};{repo_root / 'scripts'}"
    # Bytecode goes to the user's local folder, never into the code folder (M8).
    environment["PYTHONPYCACHEPREFIX"] = str(pycache_prefix(environment))
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update({key: str(value) for key, value in (extra or {}).items()})
    return environment


def display_command(command: list[str], extra_env: dict[str, str] | None = None) -> str:
    """The command as the user could paste it in PowerShell (shown in the advanced view)."""
    prefix = "".join(f"$env:{key} = '{value}'; " for key, value in (extra_env or {}).items())
    return prefix + "& " + subprocess.list2cmdline(command)
