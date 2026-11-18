"""The workspace: the folder that holds one folder per cohort. No Qt.

The user chooses it at first launch (default ``Documents\\WFCI``); its path is kept in the
Windows user settings (``QSettings("wfci_gui", "wfci_gui")``, key ``workspace``), which
the GUI and the ROI editor both read. Two rules a workspace must meet:

* it is not inside the code folder (nothing writes there at run time);
* it is on an NTFS drive, because step 1's raw-folder view (``links/``) is made of
  directory junctions, which only NTFS has.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from . import REPO_ROOT

SETTINGS_ORGANISATION = "wfci_gui"
SETTINGS_APPLICATION = "wfci_gui"
WORKSPACE_KEY = "workspace"


def default_workspace(environ: dict[str, str] | None = None) -> Path:
    """``<user profile>\\Documents\\WFCI``."""
    environ = os.environ if environ is None else environ
    home = environ.get("USERPROFILE") or str(Path.home())
    return Path(home) / "Documents" / "WFCI"


def filesystem_of(path: str | Path) -> str:
    """File system name of the drive holding ``path`` (``NTFS``, ``FAT32``, ``exFAT``...).

    The nearest existing parent is asked, so a folder still to be created works too.
    """
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    if os.name != "nt":
        return "not Windows"
    name = ctypes.create_unicode_buffer(64)
    root = path.anchor
    if not ctypes.windll.kernel32.GetVolumeInformationW(root, None, 0, None, None, None, name, len(name)):
        raise OSError(f"cannot read the file system of {root} (Windows error {ctypes.GetLastError()})")
    return name.value


def workspace_problem(path: str | Path, code_folder: Path = REPO_ROOT) -> str | None:
    """Why ``path`` cannot be the workspace, or None when it can."""
    if not str(path).strip():
        return "Choose a folder."
    path = Path(path)
    if not path.is_absolute():
        return f"{path} is not a full path (it must start with a drive, e.g. D:\\)."
    path = path.resolve()
    code_folder = Path(code_folder).resolve()
    if path == code_folder or code_folder in path.parents:
        return (f"{path} is inside the program folder {code_folder}. Keep your data elsewhere: "
                "the program folder is replaced on every update.")
    if path.exists() and not path.is_dir():
        return f"{path} is a file, not a folder."
    filesystem = filesystem_of(path)
    if os.name == "nt" and filesystem != "NTFS":
        return (f"{path.anchor} is formatted {filesystem}. The workspace must be on an NTFS drive "
                "(the raw-folder view uses junctions, which only NTFS has).")
    return None


def workspace_from_setting(value, environ: dict[str, str] | None = None) -> Path:
    """The workspace a saved setting names; the default when nothing is saved yet."""
    return Path(str(value)) if value else default_workspace(environ)
