# REFERENCE: WFCI Cohort, the technical map

For an agent or developer working in this folder. The user guide is
[README.md](README.md); coding rules are in [CLAUDE.md](CLAUDE.md).

## 1. What this folder is

A desktop program (user-visible name **WFCI Cohort**, package `wfci_gui`, conda env
`wfci_gui`) that takes a cohort of wide-field calcium-imaging recordings from the raw
TIFFs to functional-connectivity statistics, in steps 0–7 and analysis A. It holds the
GUI, the ROI editor, the analysis library `wfci`, the pipeline scripts the GUI runs, the
cortex22 atlas, the install wizard and the tests. Nothing else.

**Black-box rule.** The folder is replaced on update and nothing writes inside it at run
time:

- user data lives in a **workspace** outside it, one folder per cohort (section 5);
- the program's own files are in `%APPDATA%\wfci_gui` (`install.json`, settings checks,
  error logs, `start.log`, `install_<time>.log`) and `%LOCALAPPDATA%\wfci_gui\pycache`
  (bytecode, through `PYTHONPYCACHEPREFIX`);
- the workspace location, the Recent list and window layout are Windows user settings
  (`QSettings("wfci_gui", ...)`);
- jobs run with the cohort folder as working folder and full script paths;
- `tests\test_example_cohort.py` runs every step and asserts that the folder (ignoring
  `.git` and `.pytest_cache`) is unchanged.

The folder holds no dataset content: tests use synthetic data only.

## 2. Directory map

```text
gui_folder\
|-- Install.bat            starts env\install.ps1 (wizard), hidden console
|-- Start.bat              starts env\start.ps1, hidden console
|-- README.md  REFERENCE.md  CLAUDE.md
|-- conftest.py            puts src\ and scripts\ first on sys.path for the tests
|-- pytest.ini             test folders; -p no:cacheprovider
|-- atlas\cortex22_roi_set.yaml   the 22-box atlas (128 grid, offsets from Bregma)
|-- env\
|   |-- packages.json      exact pins, in four install stages
|   |-- setup_env.ps1      the install engine: Create / Repair / Rebuild, checks, install.json
|   |-- install.ps1        the Windows Forms wizard (runs setup_env.ps1)
|   `-- start.ps1          reads install.json, sets PYTHONPATH, runs python -m wfci_gui
|-- src\
|   |-- wfci\              analysis library: io (frame reading, channel decision),
|   |                      streaming (single-read pipeline), roi, gsr, dump (pixel volumes),
|   |                      atlases, profiles, config, resize
|   |-- nbs\               network-based statistic (used by connectivity_stats.py)
|   |-- qtkit\             reusable Qt shell (commands, docks, guide, parameter dialogs);
|   |                      its own README, GUI_DESIGN_GUIDELINES.md and tests
|   `-- wfci_gui\
|       |-- __init__.py    APP_VERSION
|       |-- __main__.py    python -m wfci_gui; first-launch workspace dialog; excepthook
|       |-- window.py      MainWindow (qtkit ShellWindow), NewCohortDialog, WorkspaceDialog
|       |-- controller.py  the only code that changes state; job queue on QProcess
|       |-- setup_check.py python -m wfci_gui.setup_check (step 0's checks)
|       |-- views\         pages.py (one page per step), widgets.py
|       |-- logic\         no Qt: cohort (file + tables), layout (fixed locations),
|       |                  workspace, params (jobs, forms, commands), defaults (ast reader),
|       |                  steps (definitions + status), jobs, presets, manifest, design, conda
|       `-- roi_editor\    python -m wfci_gui.roi_editor (section 11)
|-- scripts\               pipeline scripts (section 4) and helpers:
|                          roi_editor.py (geometry, ROI-set I/O, the reference ROIEditor),
|                          channel_rectangle.py, recording_helpers.py, pixel_helpers.py,
|                          design_helpers.py, parula_colormap.py, run_with_overrides.py
|-- tests\                 pytest suite; gui_cohort_tree.py builds a tiny fake cohort
`-- .vscode\               launch.json, settings.json
```

## 3. Entrypoints

All Python runs as `& <conda.bat> run --no-capture-output -n wfci_gui python ...`, with
`PYTHONPATH=<folder>\src;<folder>\scripts` and
`PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\wfci_gui\pycache`. `<conda.bat>` is `conda` in
`%APPDATA%\wfci_gui\install.json`.

| What | How |
|---|---|
| Install / repair / rebuild the env | `Install.bat` (wizard); or `powershell -NoProfile -ExecutionPolicy Bypass -File env\setup_env.ps1 [-Conda <conda.bat>] [-Mode Create\|Repair\|Rebuild] [-SkipChecks] [-RunTests] [-LogFile <path>]` |
| Start the GUI | `Start.bat [<cohort.yaml>]`; or `python -m wfci_gui [<cohort.yaml>] [--config-dir <dir>]` |
| ROI editor alone | `python -m wfci_gui.roi_editor [--cohort <cohort.yaml> \| --manifest <csv> \| --folder <dirs>]` |
| Setup checks | `python -m wfci_gui.setup_check --out <json> [--tests]` |
| Any pipeline script | `python -u scripts\<script>.py --cohort <cohort.yaml> [flags]`, or set `RUN_CONFIG["cohort_file"]` and run it from the editor |
| Example cohort | `python scripts\make_example_cohort.py --workspace <dir> [--name example] [--animals 3] [--recordings 1] [--seconds 12]`; then `... --place-roi-sets --cohort <cohort.yaml>` after the scan |
| Tests | `python -m pytest -q` in the folder (about 2.5 minutes; `test_example_cohort.py` is 1.5 of them) |
| Wizard layout check | `powershell -File env\install.ps1 -Screenshot <dir>` writes `install_page1..4.png` |

`.vscode\launch.json` has the GUI, the ROI editor (empty / on a cohort), "Current script
on a cohort", the example cohort and the tests.

## 4. Steps

Status is read from the files on disk every time (`logic\steps.py`), so it is right after
a crash or a run outside the GUI. A step can run only when the steps it needs are done.

| Step | Script(s) / job keys | Needs | Done when |
|---|---|---|---|
| 0 Setup | `python -m wfci_gui.setup_check` (`setup_check`), `env\setup_env.ps1 -Mode Repair` (`repair`) | | every check passes, the test suite included |
| 1 Scan | `scan_cohort.py` (`scan`) | 0 | sessions and animals tables filled; `manifest.csv` matches `recordings.csv` |
| 2 Acquisition | `acquisition_facts.py` (`facts`) | 1 | the eight acquisition fields set and `confirmed: true` (page's Confirm) |
| 3 ROIs | `bake_alignment_images.py` (`bake`), `python -m wfci_gui.roi_editor` (`roi_editor_new`), `rebuild_roi_sets.py` (`rebuild`), `roi_preflight.py` (`preflight`) | 2 | every used recording has a rebuilt set and passes the pre-flight run after the newest rebuild |
| 4 Extraction | `extract_cohort.py` (`extract`) | 3 | every used recording has `connectivity_full.npz` or is `needs_channel_decision`, and none is stale (its ROI set's `channel_order` differs from the order it was extracted with) |
| 5 QC | `qc_cohort.py` (`qc`); viewers: `plot_channel_traces.py`, `plot_channel_traces_by_animal.py`, `inspect_channel_labels.py`, `channel_rectangle_viewer.py` | 4 | `qc_table.csv` newer than `batch_summary.csv`, and `exclusions_confirmed: true` |
| 6 Caches | `cache_median_dff.py` (`cache_median`, one job per window), `cache_median_dff_baseline_pedestal.py` (`cache_blpedestal`, optional, needs `design.csv`) | 5 | a median cache per window for every used, non-excluded, extracted recording, none older than its dump |
| 7 Design | `build_cohort_design.py` (`design`) | 1, 5 | `design.csv` covers every used, non-excluded recording with a group and a timepoint |
| A Connectivity | `roi_pixel_connectivity.py` (`connectivity`, per variant), `connectivity_stats.py` (`stats`, per variant, flat) | 6, 7 | every chosen variant has results, none older than its input volume |

Step 3 in the GUI: "Open ROI editor" queues the editor; when it closes with exit code 0,
the controller queues `rebuild` (only when a reviewed set is newer than its rebuilt copy;
whole sessions are copied to `outputs\tmp\rebuild_input_<time>` and rebuilt, never the
whole folder) and then `preflight`.

Variants of analysis A (`overrides.analysis_a.variants`, default all four):
`pixels_dff_full`, `pixels_dff_full_gsr`, `pixels_median_dff_20s_full`,
`pixels_median_dff_20s_full_gsr`.

## 5. Workspace and cohort file

```text
<workspace>\                 chosen at first launch (default Documents\WFCI); NTFS; not in the code folder
|-- <cohort>\                one per cohort; the folder name is the cohort name
|   |-- cohort.yaml
|   |-- tables\              sessions.csv, animals.csv, recordings.csv
|   |-- manifest.csv         step 1
|   |-- design.csv           step 7
|   |-- links\<yymmdd>\<animal>\<t#>   junctions to the raw recording folders (step 1)
|   |-- roi_sets\            reviewed\ (editor saves), rebuilt\ (extraction reads),
|   |                        alignment_images\ (<key>_odd.tif, <key>_even.tif, <key>_odd_even.json)
|   |-- pixel_data\<day>_<animal>\<t#>\   unless pixel_root redirects it
|   |-- outputs\             <day>_<animal>\<t#>\ (extraction), batch_summary.csv,
|   |                        acquisition_facts.json, roi_preflight.csv, qc_table.csv,
|   |                        roi_pixel_connectivity\, connectivity_stats\, logs\, gui_jobs.jsonl, tmp\
|   |-- channel_rectangle_params.json   optional; missing = channel_rectangle.py defaults
|   `-- presets\<job>\<name>.json
`-- presets\                 shared by every cohort
```

Every location is fixed by the cohort folder (`logic\layout.py`, `CohortLayout`). A
script gets it from `--cohort <cohort.yaml>`.

**`cohort.yaml`** (`logic\cohort.py`; unknown keys and the older path keys are rejected
on load; saves are atomic, and the GUI shows a diff first):

| Key | Meaning |
|---|---|
| `app_version` | version that last saved the file; the GUI refuses a newer one |
| `name` | letters, digits, `_`, `-` |
| `raw_root` | the raw data folder; relative = relative to the cohort folder |
| `pixel_root` | optional; default `pixel_data` in the cohort folder |
| `timepoints` | list, e.g. `[BL, 3D, 15D, 30D]`; the first is the default baseline |
| `acquisition` | `sampling_rate_hz` (per channel), `frame_order` (`name` / `number`), `trim_s`, `rot90_k`, `grid` (`[128, 128]`), `file_glob`, `expected_files`, `confirmed` |
| `exclusions` | list of `{recording: <yymmdd>_<animal>/<t#>, reason}` (QC page) |
| `exclusions_confirmed` | step 5 is done only when true |
| `overrides` | per job key: only values that differ from the script defaults; `analysis_a`: `variants`, `baseline`, `timepoints`, `animals`, `focus` |

**Tables.** `sessions.csv`: `session_folder, day (yymmdd), timepoint, day_zero, include`.
`animals.csv`: `raw_name, animal, group, mouse_line, include` (several raw names may map
to one animal). `recordings.csv` (written by the scan, user edits of `use` and
`channel_order` kept): `day, animal, recording, path, n_files, status, use,
channel_order`.

**ROI-set files** (`roi_sets\reviewed\<key>.yaml`, `roi_sets\rebuilt\<key>.yaml`,
`<key> = <yymmdd>_<animal>_<t#>`): `name, grid, source, bregma_row, bregma_col,
downsample, lambda_row_offset, [lambda_above_bregma], [lambda_angle_deg], boxes`, then two
optional keys appended at the end and written only when set: `channel_rectangle:
{length_factor, width_factor, start_offset_factor}` (the rectangle moved for this
recording) and `channel_order: gcamp_first|emo_first` (the GCaMP parity chosen in the
editor). Every reader accepts a file without them. Box order is the column order of R:
never sort the boxes.

## 6. How parameters reach scripts

- `logic\defaults.py` reads each script's parameters with `ast`, without running it:
  argparse flags (defaults resolved through `RUN_CONFIG["key"]`), `os.environ.get(...)`
  variables, and module-level assignments of flat scripts. A computed module-level value
  is shown read-only.
- `logic\params.py` `JOBS`: per job, the `fixed` values (always the cohort file, plus the
  acquisition facts it needs, marked "not set" until step 2 fills them, plus per-run
  choices such as the variant), GUI `defaults`, and the user's `overrides` from
  `cohort.yaml`. A job with a "not set" value is refused with the reason.
- Argparse scripts get flags; only values that differ from the script default are passed.
  Environment variables are set on the job. A flat script (`connectivity_stats.py`) runs as
  `run_with_overrides.py <script> --set name=value ... -- <flags>`: the module-level
  literals are replaced in the parsed source (the file on disk is untouched) and refused
  when the script uses process pools.
- `connectivity_stats.py` takes `--cohort --baseline --timepoints --animals --variant
  --focus`, else `CONNECTIVITY_STATS_<NAME>` variables (lists comma-separated), else
  `RUN_CONFIG`. No script edit is needed for a new cohort.
- Presets: per job, `<cohort>\presets\<job>\<name>.json` (overrides plus the script's
  sha256; a different script version warns). File ▸ Export / Import all parameters
  writes or reads every job's overrides in one JSON file.

## 7. Jobs

- One job at a time (`logic\jobs.py` `JobQueue`, `controller.py` `QProcess`). Command:
  `<conda.bat> run --no-capture-output -n wfci_gui python -u <full script path> ...`,
  working folder = the cohort folder (the settings folder when no cohort is open),
  environment from `logic\conda.py` `job_environment`.
- Arguments containing `% ^ & | < > "` are refused: `conda.bat` passes them through cmd.exe.
- Log: `<cohort>\outputs\logs\<job>_<time>.log`, starting with the command and the
  overrides. History: one JSON line per finished job in `<cohort>\outputs\gui_jobs.jsonl`.
- Stop: `taskkill /T /F` on the process tree (conda.bat → conda → python); the
  extraction resumes from its `.npz` files next time.
- While a job runs, Windows is kept awake (`SetThreadExecutionState`).
- After every finished job the controller rereads the tables that have no unsaved edits
  (the scan writes `recordings.csv`), then the status.
- Viewers (`inspect_channel_labels.py`, `channel_rectangle_viewer.py`) start outside the
  queue.

## 8. Tool rules

- Always run Python through `conda run --no-capture-output -n wfci_gui`; never call the
  env's `python.exe` directly: numpy's BLAS DLLs fail to load (`0xc06d007f`). Without
  `--no-capture-output` errors are hidden.
- Pass version pins with `>=` only from PowerShell, never through Git Bash: cmd.exe reads
  `>` as a redirection when `conda.bat` runs.
- No `$ErrorActionPreference = "Stop"` around `conda run` in Windows PowerShell 5.1:
  conda's stderr warnings become terminating errors. Check `$LASTEXITCODE`.
- `conda run` adds blank lines to a command's output: parse a tagged line, not the last one.
- Log long runs with cmd redirection (`cmd /c "... > log 2>&1"`), not `Tee-Object`.
- Write CRLF files as bytes (read bytes, edit, write bytes) and check with `git diff --stat`:
  text-mode writes from Python can produce `\r\r\n`. `.bat` and `.ps1` files are CRLF and
  ASCII (PowerShell 5.1 reads BOM-less files as ANSI).
- pyqtgraph: never call `ImageItem.setRect` before `setImage`.
- Parallel workers: process pools are unreliable on Windows for the extraction (native DLL
  failures in spawned workers). `workers=1` (serial, in the main process) is the default;
  more is experimental.
- The GUI cannot be seen through a remote terminal: check it headless (pytest-qt,
  `QT_QPA_PLATFORM=offscreen`, `widget.grab().save(...)`). Every window-test fixture makes
  any modal dialog raise, so a headless test cannot hang. The wizard: `install.ps1
  -Screenshot`.

## 9. Pipeline rules a script author must not break

- Constants: the cortex22 atlas boxes; the atlas scale reference `reference_lambda = 55`
  (rows from Bregma to Lambda the atlas is drawn at); the Bregma-relative dump window
  `(-29, 47, -44, 43)` (rows, cols on the 128 grid); the channel-rectangle default
  parameters in `channel_rectangle.py`; the dump data types (dF/F float16, raw F float32).
- Raw frames are 512×512; the pipeline downsamples twice by 0.5 (box average) to the
  128 analysis grid. Bregma in manifests and ROI sets is in 256-grid units (2× the 128
  grid); box offsets are 128-grid rows/columns from Bregma.
- The ROI scaling keeps mirror symmetry: left/right boxes are scaled as mirror pairs about
  the midline, never independently. `rebuild_roi_sets.py` regenerates every set as a pure
  Lambda rescale of the atlas, so hand nudges do not accumulate.
- Convert a dump to float64 before reducing it (means, correlations): float16 sums lose
  precision.
- The mean-baseline dF/F dump (`pixels_dff_*`) is in percentage points; the median-
  baseline cache (`pixels_median_dff_<w>s_*`) is a ratio. Do not mix them without
  converting.
- The channel order of each recording: the ROI set's `channel_order` when set, else the
  manifest's (`auto` = decided from every frame of the recording; `gcamp_first` /
  `emo_first` = as given). `channel_order_source` (`roi_set` / `manifest`) is in the
  recording's meta and in `batch_summary.csv`. A `not_interleaved` recording is stored as
  odd/even frames with status `needs_channel_decision` and no dF/F.

## 10. Design decisions

**GUI** (from its design, as built):

| # | Decision |
|---|---|
| D1 | The GUI orchestrates: it runs the scripts as `conda run` subprocesses and never imports the analysis code into its own process. |
| D2 | Users are lab members who do not code: each step has help text and a check, and is blocked until its prerequisites pass. "Show commands (advanced)" shows the exact command. |
| D3 | Scope: steps 0–7 and analysis family A. |
| D4 | `qtkit` is in `src\qtkit`; the app is `src\wfci_gui`, run with `python -m wfci_gui`. |
| D6 | Windows first: `pathlib` paths; the junction view is Windows-only. |
| D7 | The environment is built by an install script, and the Setup page checks and repairs it. |
| D9 | Parameters: defaults read from the scripts, only overrides stored, every parameter editable, presets per job and per cohort. |
| D10 | Jobs run one at a time in a queue; status comes from files on disk; a history adds "last run / failed". |
| D11 | QC never excludes on its own: it suggests, the user decides and gives a reason. |
| D12 | The ROI editor is its own program on the qtkit shell. |
| D13 | Order of work was: main GUI, step pages and analysis A, ROI editor. |

**ROI editor** (as built):

| # | Decision |
|---|---|
| D1 | A Qt-free model (`model.py`) plus a pyqtgraph view that draws it on both panels. |
| D2 | Odd and even panels side by side, linked views, titled only "odd" / "even": no GCaMP label, to avoid bias. |
| D3 | Both panels show the 128 grid; the rectangle decision is computed on the 256 grid. |
| D4 | One setting, frames per parity (default 10), drives the images and the decision. |
| D5 | Boxes, Bregma, Lambda and the rectangle are drawn on both panels; a drag in either updates both. |
| D6 | Every feature of the reference editor is kept. |
| D7 | No arguments: empty window. `--cohort` / `RUN_CONFIG["cohort"]` loads at start. |
| D8 | File menu: Open cohort…, Open manifest…, Recent. A folder is opened only through the Settings dialog. |
| D9 | An opened manifest is the page list (its rows and `recording_paths`, no folder scan); `group` and `channel_order` are columns. |
| D10 | A manifest's settings come from the `cohort.yaml` in the manifest's folder; without one, the Settings dialog opens pre-filled and must be accepted. Open cohort… and Open manifest… start in the workspace. |
| D11 | Opening new data with unsaved pages asks Save all / Discard / Cancel. |
| D12 | Table columns `key, group, manifest, rectangle, choice, agree, status, bregma, lambda`, sortable and filterable. |
| D13 | Preview cache: `<key>_odd.tif` + `<key>_even.tif` (+ sidecar). An old single `<key>.tif` is shown on the odd panel only, with no decision; it is never written or deleted. |
| D14 | The rectangle can be moved per recording; untouched pages use the cohort's `channel_rectangle_params.json`. |
| D15 | Stored as `channel_rectangle: {length_factor, width_factor, start_offset_factor}`, only where moved; `statistic` stays global. |
| D16 | Rectangle motion: locked to the Bregma–Lambda axis, centred on it, handles for length and width, values clipped to the parameter limits. |
| D17 | `rebuild_roi_sets.py` copies `channel_rectangle` (and `channel_order`) per file unchanged. |
| D18 | `qc_cohort.py` uses the recording's own rectangle when its ROI set has one, else the cohort's parameters; `rectangle_source` (`roi_set` / `params`) says which. |
| D19 | A rectangle move is an edit: undoable, marks the page edited, saved with "Save modified"; copy-from-previous and apply-to-all carry it. |
| D20 | The verdict bar updates while dragging; the table cells on release. |
| D21 | Images are read when a page is first opened; "Cache all previews" fills the cache in the background. |
| D22 | `bake_alignment_images.py` writes the same odd/even pair, with the same write-then-rename. |
| D24 | Pages load in animal → day → t# order. |

The ROI editor's `RUN_CONFIG` is its only fallback when started without flags (D7); the
Settings dialog can still open recording folders directly (D8).

**This folder** (M1–M18):

| # | Decision |
|---|---|
| M1 | A standalone folder with its own git history. |
| M2 | No dataset content: files, code contents and names are neutral. |
| M3 | The copied scripts are edited freely. |
| M4 | Only what the GUI launches (steps 0–7, analysis A) and their imports. Families B–F come later, one per page. |
| M5 | Constants that came from data stay as plain defaults (section 9). |
| M6 | Tests run on synthetic cohorts only. |
| M7 | User data lives in a workspace, one folder per cohort. |
| M8 | Nothing writes inside the code folder at run time. |
| M9 | Every script takes `--cohort`; no script holds a per-cohort table or exclusion list. |
| M10 | Conda env `wfci_gui`: third-party packages only, exact pins, no editable install. |
| M11 | Install is a double-click wizard; update is replacing the folder. |
| M12 | No design spec or handoff in the folder: this file holds what is still valid. |
| M13 | Cohorts are created here; there is no import of other cohorts. |
| M14 | Names: "WFCI Cohort" (user), `gui_folder`, `wfci_gui`. |
| M15 | Helper functions live in neutral modules (`recording_helpers.py`, `pixel_helpers.py`, `design_helpers.py`). |
| M16 | Only the new ROI editor is offered. `scripts\roi_editor.py` stays whole: the editor imports its functions and the equivalence tests drive its `ROIEditor` class. |
| M17 | Channel-rectangle parameters are per cohort (`channel_rectangle_params.json`), written by the viewer's Save. |
| M18 | The extraction uses the ROI set's `channel_order` over the manifest's (section 9). |

## 11. The ROI editor

- `roi_editor\model.py` (pages, every edit, undo; no Qt) and `sources.py` (cohort /
  manifest / folder loading, seeding, the preview cache; no Qt). `controller.py` maps
  commands to model edits; `view.py` draws a page on a pyqtgraph panel; `window.py` is the
  qtkit window. `logic.py` holds settings, page status and rebuild inputs.
- `scripts\roi_editor.py` is the reference: geometry functions, `save_roi_set` /
  `load_roi_set`, `HELP`, and the `ROIEditor` class. `tests\test_roi_editor_model.py` and
  `tests\test_wfci_roi_editor.py` check that the new editor writes the same bytes as
  `ROIEditor` for the same edits; keep both.
- Preview cache in `roi_sets\alignment_images`: `<key>_odd.tif`, `<key>_even.tif` (float32,
  128 grid, mean of the first *frames per parity* odd / even frames after `rot90_k` and two
  0.5× box downsamples) and `<key>_odd_even.json`, the settings they were made with; a pair
  whose sidecar differs is remade. Writes go through `.partial` + rename.
- The two ROI-set keys (`channel_rectangle`, `channel_order`) are appended at the end of
  the file and written only when set (section 5).
- The editor does not decide channels from the cached 128 images: the downsampled
  decision disagreed with the full-grid decision too often. The rectangle decision uses
  the 256 grid.
- Settings: `EditorSettings` (folders, pattern, frame order, rotation, save / seed /
  rebuilt folders, image folder, `rectangle_params`, frames per parity). From the GUI all
  come from the cohort file. "Rebuild" in the editor runs `rebuild_roi_sets.py` on a temp
  copy of the changed sessions, with this folder's `PYTHONPATH`.

## 12. Not built yet, and known gaps

- Analysis families B–F.
- How each recording's channel order is chosen: the ROI set's `channel_order` when set,
  else the manifest's (M18, section 9).
- The ROI editor has not been used by a person. Never run on real raw data:
  `acquisition_facts.py`, the folder walk of `scan_cohort.py`, and the editor's rectangle
  decision.
- The channel-rectangle default parameters were tuned on one cohort and gave the
  opposite channel order on most recordings of another. Each cohort must check them in
  the viewer (step 5) and save its own `channel_rectangle_params.json`.
- The install wizard's window was checked only as rendered images (`-Screenshot`). Its
  engine (`setup_env.ps1`) has run for real in Create, Repair and Rebuild modes. Neither
  has been tried on a Windows account without conda (the Miniforge download path), and
  `Start.bat` has not been started by hand.
- The GUI and the ROI editor have been driven only headless (tests, and an acceptance
  run through `MainWindow` / `EditorWindow` with real `conda run` jobs), never by a person.
- `src\wfci` has no separate library document; its invariants are in section 9.

## 13. Tests

`& <conda.bat> run --no-capture-output -n wfci_gui python -m pytest -q` in the folder.

State: **485 passed, 10 skipped** in the `wfci_gui` env (about 2.5 minutes). The 10 skips
are reproduction checks that look for inputs this folder never holds (MATLAB reference
files under `matlab_cortical\` and `tests\matlab_reference\`, sample frames under `data\`,
a `roi_setsebuilt\` folder): `test_atlas_transcription.py` (3), `test_parity.py` (3),
`test_streaming.py` (3), `test_extract_cohort.py` (1). They are kept as they were and
always skip here; remove them or give them synthetic inputs.

- `tests\test_example_cohort.py`: the example cohort through steps 1–7 and analysis A with
  the commands the GUI builds; QC agrees with the example's channels; the code folder is
  unchanged afterwards (M8).
- `tests\test_wfci_gui_logic.py`, `test_wfci_gui_window.py`: cohort file, layout,
  workspace, commands, status (including step 4 stale on a changed `channel_order`),
  headless windows.
- `tests\test_channel_selection.py`, `test_extract_cohort.py`: the extraction, including
  the ROI set's `channel_order` (applied, source recorded, invalid value refused, dry run,
  re-extraction on a change).
- `tests\test_roi_editor_model.py`, `test_wfci_roi_editor.py`: the editor against the
  reference `ROIEditor`, byte for byte.
- `tests\test_dump.py::test_dump_peak_memory_does_not_grow_with_recording_length`: a
  warm-up run and a garbage collection before each measured run make it independent of the
  tests before it (ratio 1.88 against a limit of 2).
