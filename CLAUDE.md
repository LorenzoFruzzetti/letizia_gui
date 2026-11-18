# Agent rules for this folder

Read [REFERENCE.md](REFERENCE.md) first: it is the technical map of this folder (layout,
entrypoints, steps, cohort schema, jobs, tool rules, pipeline rules, design decisions,
known gaps). [README.md](README.md) is for the lab member who uses the program.

## 1. Black box

- Nothing writes inside this folder at run time: no outputs, no logs, no bytecode, no
  caches, no settings. User data lives in the workspace (`<workspace>\<cohort>\...`); the
  program's own files in `%APPDATA%\wfci_gui` and `%LOCALAPPDATA%\wfci_gui`.
  `tests\test_example_cohort.py` checks this; keep it passing.
- No dataset content: no real paths, animal names, session dates, group labels, results
  or history of earlier analyses, in code, comments, tests or documents. Tests use
  synthetic data only (`tests\gui_cohort_tree.py`, `scripts\make_example_cohort.py`).
- No history sections in documents. State a rule and its reason in general terms.

## 2. Operating contract

- Preserve behaviour unless the user asks for a change; say so when a change alters what
  a step computes.
- Prefer small, verifiable edits over broad rewrites. Run the test suite after a change.
- Ask before risky deletes, irreversible moves, or archive operations.
- Keep REFERENCE.md and README.md in sync when entrypoints, inputs, outputs, the cohort
  schema or the layout change.

## 3. Environment (`wfci_gui`)

- The conda env is `wfci_gui`, built by `Install.bat` / `env\setup_env.ps1` from the exact
  pins in `env\packages.json`. Third-party packages only; never `pip install -e`.
- Adding a package: add its exact pin to the right stage of `env\packages.json`, add its
  import name to `PACKAGES` in `src\wfci_gui\setup_check.py`, run
  `env\setup_env.ps1 -Mode Repair`. statsmodels-like packages the conda solver chokes on
  go in the pip stage.
- Run Python only through `conda run --no-capture-output -n wfci_gui ...`, with
  `PYTHONPATH=<this folder>\src;<this folder>\scripts` and
  `PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\wfci_gui\pycache`. Never call the env's `python.exe`
  directly: numpy's BLAS DLLs fail to load.
- The conda path is in `%APPDATA%\wfci_gui\install.json` (`conda`). conda is not on PATH.
- Pass version specs with `>=` / `<` only from PowerShell, never through Git Bash (cmd.exe
  would read them as redirections).
- In Windows PowerShell 5.1, do not wrap `conda run` in `$ErrorActionPreference = "Stop"`:
  conda's stderr warnings become fatal. Check `$LASTEXITCODE`.
- Log long runs with cmd redirection (`cmd /c "... > log 2>&1"`), not `Tee-Object`.
- The GUI cannot be seen through a remote terminal. Check GUI work headless (pytest-qt with
  `QT_QPA_PLATFORM=offscreen`, `widget.grab().save(...)`); `env\install.ps1 -Screenshot
  <folder>` draws the install wizard's pages.

## 4. Code

- No try/except: errors surface. Add error handling only when the user asks, and then log
  the error so it is easy to understand.
- Comments explain purpose and non-obvious decisions, not what the code says.
- Every script runs from the editor without flags: a `RUN_CONFIG` block after the imports
  holds every parameter, `cohort_file: None` among them, and `build_runtime_args` prefers
  CLI flags when any are given:

```python
# Edit this section to run without CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "cohort_file": None,      # <workspace>/<cohort>/cohort.yaml (or pass --cohort)
    "window_s": 20.0,         # running-median window, seconds
    "prefer_cli_args": True,
}


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{k: v for k, v in config.items() if k != "prefer_cli_args"})
```

- A script gets every cohort location from its cohort file
  (`wfci_gui.logic.cohort.load_script_cohort`, `Cohort.layout`). No per-cohort tables,
  paths or exclusion lists in a script. Run without a cohort it stops with
  "set cohort_file or pass --cohort".
- The GUI reads script defaults with `ast` (`wfci_gui.logic.defaults`): keep defaults as
  literals in `RUN_CONFIG`, `add_argument(default=...)` or module-level assignments.
- `wfci_gui.logic` and `wfci_gui.roi_editor.model` / `sources` import no Qt. Scripts may
  import `wfci_gui.logic`; `wfci_gui.logic` never imports a script.
- Files are UTF-8. Python files may be LF or CRLF (git normalises them); `.bat` and `.ps1`
  are CRLF and ASCII (Windows PowerShell 5.1 reads a file without a byte-order mark as
  the ANSI code page). When editing a CRLF file from Python, read and write bytes and check
  with `git diff --stat`.

## 5. Test scripts (flat analysis scripts)

Flat analysis scripts (`connectivity_stats.py`) read one dataset, compute, and save plots.
They are read and edited by hand, so clarity beats abstraction:

- Order: imports; parameters (every value someone might change, one block); loading;
  computation (one commented block per step); plotting and saving.
- Parameters are module-level `lower_snake_case`, units in the name (`window_s`,
  `sampling_rate_hz`) or a trailing comment. Derive index values from physical ones.
- Annotate the shape of every non-trivial array where it is made:
  `# recording_z: (n_recordings, n_rois, n_rois)`.
- Spell names out; keep intermediate results in named variables.
- Figures: `constrained_layout=True`, `squeeze=False`; labels from the data; save with
  `fig.savefig(output_dir / "<name>.png", dpi=200)` then `plt.close(fig)`; titles state
  the parameters that produced them.
- The GUI runs them through `run_with_overrides.py` (module-level values as
  `--set name=value`, flags after `--`), so keep their parameters literal.

## 6. Tests

`conda run --no-capture-output -n wfci_gui python -m pytest -q` from this folder
(`pytest.ini` names the test folders and turns the cache off; `conftest.py` puts this
folder's `src` and `scripts` first on `sys.path`). Every window test fixture makes any
modal dialog raise, so a headless run cannot hang.
