"""Run a flat script with some of its module-level assignments replaced (gui_spec.md 6).

    python run_with_overrides.py <script.py> --set name=value [--set name=value ...] [-- args]

``value`` is a Python literal (``0.05``, ``'BL'``, ``['BL','3D']``, ``None``); a value
that is not a literal is taken as plain text. The script on disk is NOT modified: its
source is parsed, the literal assigned to each name is replaced in the syntax tree, and
the result is compiled with the script's own file name and executed as ``__main__``.
So line numbers in tracebacks are the script's own, and ``__file__`` is the script.

Refused, before anything runs:
  * a name the script never assigns at module level (a typo would do nothing silently);
  * a name whose value is computed (``repo_root / "x"``) rather than a literal;
  * overrides on a script that uses multiprocessing/concurrent.futures: worker processes
    re-import the file from disk and would silently see the ORIGINAL values.

With no ``--set`` the script runs exactly as ``python <script.py>`` would.
The log starts with the overrides applied, so every job log records them.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import sys
import types
from pathlib import Path

# src/ holds wfci_gui (the parser shared with the GUI); a job's PYTHONPATH already has
# it, this makes `python run_with_overrides.py` work from the editor too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfci_gui.logic.defaults import apply_assignment_overrides, literal_value  # noqa: E402

# Edit this section to run without CLI flags (CLAUDE.md 5.2): a script and its overrides.
RUN_CONFIG = {
    "script": None,          # e.g. "connectivity_stats.py"
    "overrides": {},         # e.g. {"fdr_alpha": 0.1}
    "script_args": [],
}

PARALLEL_MODULES = ("multiprocessing", "concurrent.futures")


def parse_assignment(text: str) -> tuple[str, object]:
    """``name=value`` -> (name, Python value); a non-literal value is kept as text."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--set expects name=value, got {text!r}")
    name, raw = text.split("=", 1)
    name = name.strip()
    if not name.isidentifier():
        raise argparse.ArgumentTypeError(f"{name!r} is not a valid variable name")
    try:
        node = ast.parse(raw, mode="eval").body
    except SyntaxError:
        return name, raw
    ok, value = literal_value(node)
    return name, value if ok else raw


def uses_parallel_workers(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.startswith(PARALLEL_MODULES) for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PARALLEL_MODULES):
            return True
    return False


def run(script: Path, overrides: dict[str, object], script_args: list[str]) -> None:
    script = script.resolve()
    text = script.read_text(encoding="utf-8")
    tree = apply_assignment_overrides(text, overrides, filename=str(script))
    if overrides and uses_parallel_workers(ast.parse(text)):
        raise RuntimeError(
            f"{script.name} uses {' / '.join(PARALLEL_MODULES)}: worker processes re-import "
            "the file from disk and would run with the ORIGINAL values. Overrides refused.")

    print(f"run_with_overrides: {script}")
    if overrides:
        for name, value in overrides.items():
            print(f"  override {name} = {value!r}")
    else:
        print("  no overrides (the script runs unchanged)")
    sys.stdout.flush()

    code = compile(tree, str(script), "exec")
    # Execute as a real __main__ module so pickling, inspect and `if __name__ == "__main__"`
    # behave as when the script is run directly.
    module = types.ModuleType("__main__")
    module.__file__ = str(script)
    module.__builtins__ = builtins
    sys.modules["__main__"] = module
    sys.argv = [str(script), *script_args]
    sys.path[0] = str(script.parent)
    exec(code, module.__dict__)


def main() -> None:
    if len(sys.argv) == 1 and RUN_CONFIG["script"]:
        run(Path(RUN_CONFIG["script"]), dict(RUN_CONFIG["overrides"]), list(RUN_CONFIG["script_args"]))
        return
    argv = sys.argv[1:]
    script_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, script_args = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", type=Path)
    parser.add_argument("--set", dest="assignments", action="append", default=[],
                        type=parse_assignment, metavar="NAME=VALUE")
    args = parser.parse_args(argv)
    overrides = dict(args.assignments)
    if len(overrides) != len(args.assignments):
        raise SystemExit("run_with_overrides: a name is set twice")
    run(args.script, overrides, script_args)


if __name__ == "__main__":
    main()
