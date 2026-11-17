# Build or repair the `letizia` conda environment (gui_spec.md 9.1).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1 [-SkipTests]
#
# Re-running on a complete environment changes nothing: every stage first checks what
# is already importable and installs only what is missing.
#
# Why it is built in stages and not with one `conda env create`: one big transaction
# failed four times on this kind of machine while unpacking mkl / qt6-main
# (CLAUDE.md 9.19); small transactions do not lose the whole env on a stall.
# Version pins are only ever passed from PowerShell, never through Git Bash, where
# cmd.exe would read `>=` as a redirection (9.18).

param(
    [string]$EnvName = "letizia",
    [switch]$SkipTests
)

# Native stderr lines (conda warnings) must not become fatal (CLAUDE.md 9.47), so the
# script checks $LASTEXITCODE itself instead of setting $ErrorActionPreference = "Stop".
$Repo = $PSScriptRoot

function Find-Conda {
    $candidates = @(
        (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
        (Join-Path $env:USERPROFILE "miniconda3\condabin\conda.bat")
    )
    if ($env:CONDA_EXE) {
        $candidates += (Join-Path (Split-Path (Split-Path $env:CONDA_EXE)) "condabin\conda.bat")
        $candidates += $env:CONDA_EXE
    }
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw "conda not found. Tried: $($candidates -join ', '). Install Miniconda first."
}

function Invoke-Checked([string]$What, [scriptblock]$Block) {
    Write-Host "==> $What"
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function Test-Import([string]$Module) {
    # Through conda run: calling the env's python.exe directly breaks numpy's BLAS (9.17).
    & $Conda run -n $EnvName python -c "import $Module" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-Missing([hashtable]$Packages) {
    $missing = @()
    foreach ($module in $Packages.Keys) {
        if (-not (Test-Import $module)) { $missing += $Packages[$module] }
    }
    return $missing
}

$Conda = Find-Conda
Write-Host "conda: $Conda"
$EnvDir = Join-Path (Split-Path (Split-Path $Conda)) "envs\$EnvName"

# Stage 1: the env itself, Python 3.11.
if (-not (Test-Path (Join-Path $EnvDir "python.exe"))) {
    Invoke-Checked "create env $EnvName (python 3.11)" {
        & $Conda create -n $EnvName -c conda-forge -y "python=3.11"
    }
} else {
    Write-Host "env $EnvName exists: $EnvDir"
}

# Stage 2: the scientific stack. Import name -> conda spec.
$Stage2 = @{
    "numpy" = "numpy>=1.24"; "scipy" = "scipy>=1.10"; "pandas" = "pandas>=2.0";
    "tifffile" = "tifffile>=2023.7.10"; "matplotlib" = "matplotlib>=3.7"; "yaml" = "pyyaml>=6.0";
    "pytest" = "pytest>=7.4"; "pip" = "pip"
}
$missing = Get-Missing $Stage2
if ($missing.Count) {
    Invoke-Checked "install $($missing -join ' ')" { & $Conda install -n $EnvName -c conda-forge -y @missing }
} else { Write-Host "stage 2 complete" }

# Stage 3: the GUI stack (only the GUIs need Qt; a failure here does not block the batch).
$Stage3 = @{ "pyqtgraph" = "pyqtgraph>=0.13"; "PySide6" = "pyside6>=6.5"; "pytestqt" = "pytest-qt>=4.2" }
$missing = Get-Missing $Stage3
if ($missing.Count) {
    Invoke-Checked "install $($missing -join ' ')" { & $Conda install -n $EnvName -c conda-forge -y @missing }
} else { Write-Host "stage 3 complete" }

# Stage 4: pip, because the conda solver hangs on statsmodels (CLAUDE.md 9.51).
$Stage4 = @{ "statsmodels" = "statsmodels>=0.14"; "psutil" = "psutil>=5.9" }
$missing = Get-Missing $Stage4
if ($missing.Count) {
    Invoke-Checked "pip install $($missing -join ' ')" { & $Conda run -n $EnvName python -m pip install @missing }
} else { Write-Host "stage 4 complete" }

# Stage 5: this repository in editable mode, but only if wfci does not already resolve
# to this checkout (9.13 / 9.26: an editable install may point at another checkout).
$Expected = (Resolve-Path (Join-Path $Repo "src\wfci")).Path
$WfciFile = (& $Conda run -n $EnvName python -c "import wfci, os; print(os.path.dirname(wfci.__file__))" 2>$null | Select-Object -Last 1)
if ($LASTEXITCODE -ne 0 -or $WfciFile -ne $Expected) {
    Push-Location $Repo
    Invoke-Checked "pip install -e . (this checkout)" { & $Conda run -n $EnvName python -m pip install -e . }
    Pop-Location
    $WfciFile = (& $Conda run -n $EnvName python -c "import wfci, os; print(os.path.dirname(wfci.__file__))" | Select-Object -Last 1)
}
if ($WfciFile -ne $Expected) {
    throw "wfci resolves to '$WfciFile', not '$Expected' (CLAUDE.md 9.13/9.26)."
}
Write-Host "wfci resolves to this checkout: $WfciFile"

# Last: the test suite, as the final check.
if (-not $SkipTests) {
    Push-Location $Repo
    $env:PYTHONPATH = "$Repo\src;$Repo"
    Invoke-Checked "test suite" { & $Conda run --no-capture-output -n $EnvName python -m pytest -q -p no:cacheprovider tests }
    Pop-Location
}
Write-Host "Environment $EnvName is complete."
