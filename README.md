# WFCI Cohort

WFCI Cohort takes a cohort of wide-field calcium-imaging recordings from the raw TIFF
files to functional-connectivity results. It works in steps, and each step says when it
is done. You don't need to read or edit any code.

**Two folders, kept apart:**

- **The program folder** (this one, `gui_folder`). You don't change anything in it, and
  the program never writes in it. To update, you replace it.
- **Your workspace** (by default `Documents\WFCI`). Your cohorts live there, one folder
  per cohort, with everything the steps produce. The raw recordings stay where they are;
  the program only reads them.

## Install (once per Windows user)

1. Double-click **`Install.bat`** in this folder. A window opens. The first time it needs
   the internet.
2. **Conda.** The list shows the conda installations found on this computer. Pick one, or
   click **Install Miniforge for me** if the list is empty. *Next* is enabled once the chosen
   conda works.
3. **Environment.** Click **Install**. This creates the `wfci_gui` environment, a separate
   set of Python packages with exact versions (1–2 GB, 5–20 minutes the first time).
4. **Install.** One bar per stage and the live log. If something fails, click **Copy log**
   and send it (see *When something fails*).
5. **Finished.** Every check should say PASS. Leave **Create a desktop shortcut** ticked
   and click **Finish**.

## Start

Double-click **`Start.bat`** (or the **WFCI Cohort** desktop shortcut). No console window
opens.

The first time, the program asks where your **workspace** is. Accept `Documents\WFCI`
or choose another folder. It must be on an NTFS drive (any normal internal Windows disk),
and not inside the program folder. You can change it later with
**Cohort ▸ Choose workspace…**.

## Try it on the example cohort

**Cohort ▸ Create example cohort** writes a small made-up cohort into your workspace
(3 mice, 2 sessions, short recordings; about a minute) and opens it. Every step works
on it, so you can learn the program without real data. In step 3 you place the boxes
yourself, the same way as for real data.

## Create a cohort

**Cohort ▸ New cohort…** asks for:

- **Name**: letters, digits, `_` and `-`. It becomes the cohort's folder name.
- **Raw data folder**: the folder that holds the session folders.
- **Frame rate** and **frame order**: leave them blank if you don't know them; step 2 reads
  them from the files.
- **Timepoints**: for example `BL, 3D, 15D, 30D`. The first one is the baseline.
- **Start from the parameters of**: copies the parameter changes of another cohort.

Then work through the steps in the list on the left. Each page explains its step,
shows whether it is done, and has its buttons:

| Step | What you do |
|---|---|
| 0 Setup | Run checks. Repair if something is missing. |
| 1 Scan | Fill the *Tables* tab: which session folder is which day and timepoint, which folder name is which mouse. Then scan. |
| 2 Acquisition | Read facts from the files, check them, and click **Confirm**. |
| 3 ROIs | Bake previews. Open the ROI editor: place Bregma and Lambda on each recording, choose which frames are GCaMP if needed, save. The rebuild and pre-flight then run by themselves. |
| 4 Extraction | Run the smoke test first, then the full run (minutes per recording; it can be stopped and resumed). |
| 5 Quality control | Build the table. Tick the recordings to exclude, give each a reason, and press **Confirm exclusion list**. |
| 6 Caches | Build the median dF/F caches. |
| 7 Design | Build the experimental design. |
| A Connectivity | Choose baseline, timepoints and animals. Run. |

Jobs run one at a time in the queue at the bottom; **Stop** stops the running one.
**Run ▸ Open folder** opens the current step's results.

## Where things are

Inside the workspace, each cohort has its own folder:

```text
Documents\WFCI\
|-- <cohort>\
|   |-- cohort.yaml          the cohort's settings (the program edits it for you)
|   |-- tables\              sessions.csv, animals.csv, recordings.csv
|   |-- manifest.csv         the recording list (step 1)
|   |-- design.csv           the experimental design (step 7)
|   |-- links\               short-name shortcuts to the raw recording folders (step 1)
|   |-- roi_sets\            the ROI boxes: reviewed\ (yours), rebuilt\, alignment_images\
|   |-- pixel_data\          per-pixel volumes and caches (large; can be moved, see below)
|   |-- outputs\             results of every step; logs\ holds one log per job
|   `-- presets\             saved parameter sets of this cohort
`-- presets\                 parameter sets shared by all cohorts
```

The per-pixel data is large (about 200 MB per recording). To keep it on another disk, close
the program, open the cohort's `cohort.yaml` in Notepad and add a line
`pixel_root: E:\WFCI_pixels\<cohort>` (any folder on that disk), before step 4. The cohort's results are in
`outputs\`: connectivity under `outputs\roi_pixel_connectivity\`, statistics under
`outputs\connectivity_stats\`.

## Update

1. Close the program.
2. Replace the contents of the program folder with the new version. Your workspace is not
   touched.
3. If the new version says its package list changed, run **Install.bat** again and choose
   **Repair**.

A cohort saved by a newer version can't be opened by an older one. The program says so.

## When something fails

Send these to whoever maintains the program:

- **A step failed:** the job's log. The queue at the bottom shows its path; the logs are in
  `<cohort>\outputs\logs\`.
- **The program showed an error box:** `%APPDATA%\wfci_gui\wfci_gui.log` (Help ▸ Open
  settings folder).
- **The program did not start:** `%APPDATA%\wfci_gui\start.log`.
- **The install failed:** `%APPDATA%\wfci_gui\install_<date>_<time>.log` (or **Copy log**
  in the install window).

---

# For developers

The technical map of this folder is [REFERENCE.md](REFERENCE.md). Coding rules for an
agent are in [CLAUDE.md](CLAUDE.md).

## Entrypoints

Every command runs through `conda run` with this folder's `src` and `scripts` on
`PYTHONPATH`. From PowerShell in this folder, take the conda path from
`%APPDATA%\wfci_gui\install.json`:

```powershell
$conda = (Get-Content "$env:APPDATA\wfci_gui\install.json" | ConvertFrom-Json).conda
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
$env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\wfci_gui\pycache"

# the GUI, and the ROI editor on its own
& $conda run --no-capture-output -n wfci_gui python -m wfci_gui
& $conda run --no-capture-output -n wfci_gui python -m wfci_gui.roi_editor --cohort "$env:USERPROFILE\Documents\WFCI\example\cohort.yaml"

# any pipeline script on a cohort (scan_cohort.py shown)
& $conda run --no-capture-output -n wfci_gui python -u scripts\scan_cohort.py --cohort "$env:USERPROFILE\Documents\WFCI\example\cohort.yaml"

# the example cohort, and the tests
& $conda run --no-capture-output -n wfci_gui python scripts\make_example_cohort.py --workspace "$env:USERPROFILE\Documents\WFCI"
& $conda run --no-capture-output -n wfci_gui python -m pytest -q
```

From VS Code: `.vscode/launch.json` has configurations for the GUI, the ROI editor
(empty or on a cohort), the open script on a cohort, the example cohort and the tests.
Each script also runs from the editor without flags: set `cohort_file` in its
`RUN_CONFIG` block.

## Input

- **Raw recordings:** one folder per recording (`<raw_root>\<session folder>\<animal
  folder>\t<k>`), one TIFF per frame, two interleaved channels (GCaMP and reflectance).
  Frame numbers are in the file names. Each header holds the frame interval
  (`Time_From_Last`).
- **A cohort file**, `<workspace>\<cohort>\cohort.yaml`, and its three tables. The GUI
  writes them. REFERENCE.md section 5 has the schema.

## Output

All output goes to the cohort folder (layout above). The program's own settings, logs
and bytecode go to `%APPDATA%\wfci_gui` and `%LOCALAPPDATA%\wfci_gui`. Nothing is
written in this folder.

## Directory map

```text
gui_folder\
|-- Install.bat, Start.bat   install wizard; start without a console
|-- README.md, REFERENCE.md, CLAUDE.md
|-- src\wfci\                the analysis library (streaming pipeline, ROIs, dumps)
|-- src\wfci_gui\            the GUI (logic\ has no Qt; views\, controller.py, window.py)
|-- src\wfci_gui\roi_editor\ the ROI editor (model.py and sources.py have no Qt)
|-- src\qtkit\               the shared Qt window shell, with its own README and tests
|-- src\nbs\                 network-based statistic
|-- scripts\                 the pipeline scripts (one per step) and their helpers
|-- atlas\                   cortex22_roi_set.yaml, the 22-box atlas
|-- env\                     packages.json (pins), setup_env.ps1, install.ps1, start.ps1
|-- tests\                   the test suite (synthetic data only)
`-- .vscode\                 launch configurations
```

## Example

`scripts\make_example_cohort.py` writes the example cohort (the GUI's **Cohort ▸ Create
example cohort**). `tests\test_example_cohort.py` runs it through every step with the
commands the GUI builds, and checks that this folder is unchanged afterwards.
