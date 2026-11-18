"""Parameter presets.

  * Per job: ``<root>/<job>/<name>.json`` holds the overrides only, plus the sha256 of the
    script they were made for. Loading one made for a different script version warns.
  * Per cohort: export / import of every job's overrides in one JSON file.
Loading always shows a diff against the current values before it is applied.

``root`` is a presets folder of the workspace: the cohort's own (``CohortLayout.presets``)
or the one shared by every cohort (``CohortLayout.workspace_presets``). Never the code folder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .cohort import OVERRIDE_GROUPS, write_text_atomic
from .jobs import timestamp

_MISSING = object()


class PresetError(ValueError):
    pass


def preset_path(job: str, name: str, root: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\- ]+", name):
        raise PresetError(f"preset name {name!r}: use letters, digits, spaces, '_' or '-'")
    return Path(root) / job / f"{name}.json"


def list_presets(job: str, root: Path) -> list[str]:
    folder = Path(root) / job
    return sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []


def save_preset(job: str, name: str, overrides: dict[str, Any], script_hash: str,
                root: Path) -> Path:
    path = preset_path(job, name, root)
    payload = dict(job=job, overrides=overrides, script_hash=script_hash, saved=timestamp())
    write_text_atomic(path, json.dumps(payload, indent=2, default=str) + "\n")
    return path


def load_preset(job: str, name: str, root: Path) -> dict[str, Any]:
    payload = json.loads(preset_path(job, name, root).read_text(encoding="utf-8"))
    if payload.get("job") != job or not isinstance(payload.get("overrides"), dict):
        raise PresetError(f"{name}: not a preset for {job}")
    return payload


def hash_warning(payload: dict[str, Any], current_hash: str) -> str:
    """Non-empty when the preset was saved against a different version of the script."""
    if payload.get("script_hash") and payload["script_hash"] != current_hash:
        return ("This preset was saved for a different version of the script "
                f"(saved {payload.get('saved', '?')}). Check every value before running.")
    return ""


def diff_overrides(current: dict[str, Any], new: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(name, current value, new value) for every parameter that changes; <default> = unset."""
    changes = []
    for name in sorted(set(current) | set(new)):
        old_value = current.get(name, _MISSING)
        new_value = new.get(name, _MISSING)
        if old_value != new_value:
            changes.append((name, "<default>" if old_value is _MISSING else old_value,
                            "<default>" if new_value is _MISSING else new_value))
    return changes


def export_overrides(path: Path, cohort_name: str, overrides: dict[str, dict[str, Any]]) -> None:
    payload = dict(cohort=cohort_name, exported=timestamp(), overrides=overrides)
    write_text_atomic(Path(path), json.dumps(payload, indent=2, default=str) + "\n")


def import_overrides(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = payload.get("overrides")
    if not isinstance(overrides, dict):
        raise PresetError(f"{path}: no 'overrides' mapping")
    unknown = set(overrides) - set(OVERRIDE_GROUPS)
    if unknown:
        raise PresetError(f"{path}: unknown job(s) {sorted(unknown)}")
    return overrides
