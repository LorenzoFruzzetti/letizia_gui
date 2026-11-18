# Build, repair or rebuild the `wfci_gui` conda environment from env\packages.json.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File env\setup_env.ps1 [-Conda <conda.bat>]
#       [-Mode Create|Repair|Rebuild] [-SkipChecks] [-RunTests] [-LogFile <path>]
#
# Install.bat's wizard runs this script and shows its progress; the GUI's Setup page runs
# it for "Repair". It can also be run on its own (no window).
#
#   Create   make the env when it does not exist, otherwise the same as Repair
#   Repair   install only the packages that are missing or at another version than the pin
#   Rebuild  delete the env and build it again from scratch
#
# The env is built in stages with exact versions (env\packages.json): one large
# transaction can stall while unpacking the biggest packages, and a stalled small one
# loses nothing. statsmodels and psutil come from pip because the conda solver hangs on
# statsmodels. Nothing is installed in editable mode: the env holds third-party packages
# only, and the GUI puts this folder's src\ and scripts\ on PYTHONPATH, so the env works
# with any copy of the folder.
#
# At the end it writes %APPDATA%\wfci_gui\install.json (conda path, env name, code folder),
# which Start.bat and the GUI read, and runs the setup checks.
#
# Progress lines for the wizard: "##STAGE <i> <n> <name>", "##STAGE_DONE <i>",
# "##CHECK PASS|FAIL <text>", "##DONE" and "##FAILED <message>".
#
# Rules kept here on purpose:
#   * no $ErrorActionPreference = "Stop": conda writes warnings to stderr, which would
#     become fatal in Windows PowerShell 5.1; $LASTEXITCODE is checked instead;
#   * python always runs through `conda run`: calling the env's python.exe directly
#     fails to load numpy's BLAS DLLs.

param(
    [string]$Conda = "",
    [ValidateSet("Create", "Repair", "Rebuild")]
    [string]$Mode = "Create",
    [switch]$SkipChecks,
    [switch]$RunTests,
    [string]$LogFile = ""
)

$CodeFolder = Split-Path -Parent $PSScriptRoot
$Packages = Get-Content -Raw -Path (Join-Path $PSScriptRoot "packages.json") | ConvertFrom-Json
$EnvName = $Packages.env_name
$Channel = $Packages.channel
$ConfigDir = Join-Path $env:APPDATA "wfci_gui"
$InstallRecord = Join-Path $ConfigDir "install.json"

function Write-Log([string]$Text) {
    Write-Output $Text
    if ($LogFile) { Add-Content -Path $LogFile -Value $Text -Encoding UTF8 }
}

function Stop-Install([string]$Message) {
    Write-Log "##FAILED $Message"
    exit 1
}

function Find-Conda {
    # 1. the installation chosen last time; 2. the usual places; 3. CONDA_EXE.
    $candidates = @()
    if (Test-Path $InstallRecord) {
        $record = Get-Content -Raw -Path $InstallRecord | ConvertFrom-Json
        $candidates += $record.conda
    }
    foreach ($name in "anaconda3", "miniconda3", "miniforge3") {
        $candidates += (Join-Path $env:USERPROFILE "$name\condabin\conda.bat")
    }
    if ($env:CONDA_EXE) {
        $candidates += (Join-Path (Split-Path (Split-Path $env:CONDA_EXE)) "condabin\conda.bat")
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return $candidate }
    }
    Stop-Install "conda not found (tried: $($candidates -join ', ')). Run Install.bat and choose or install it."
}

function Invoke-Native([string]$What, [string[]]$Arguments) {
    # Runs conda with its output in the log; stops the install when it fails.
    Write-Log "> conda $($Arguments -join ' ')"
    & $Conda @Arguments 2>&1 | ForEach-Object { Write-Log "$_" }
    if ($LASTEXITCODE -ne 0) { Stop-Install "$What failed (exit code $LASTEXITCODE). See the log." }
}

function Get-OutdatedPackages([object[]]$Stage) {
    # The packages of a stage that are missing or at another version than the pin, by
    # their installed metadata (stripped: some packages' metadata ends in a space). One
    # `conda run` for the whole stage. The answer is tagged:
    # conda run adds blank lines of its own around the output.
    $wanted = @($Stage | ForEach-Object { "$($_.dist)==$($_.version)" })
    $code = "import importlib.metadata as m, sys; " +
            "have = {d.metadata['Name'].lower().replace('_', '-'): d.version.strip() for d in m.distributions()}; " +
            "print('OUTDATED:' + ' '.join(a for a in sys.argv[1:] if have.get(a.split('==')[0]) != a.split('==')[1]))"
    $output = & $Conda run -n $EnvName python -c $code @wanted 2>$null
    $line = @($output | Where-Object { "$_".StartsWith("OUTDATED:") }) | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or -not $line) { return $Stage }
    $names = @($line.Substring(9).Trim().Split(" ") | Where-Object { $_ } | ForEach-Object { $_.Split("=")[0] })
    return @($Stage | Where-Object { $names -contains $_.dist })
}

if (-not $Conda) { $Conda = Find-Conda }
if (-not (Test-Path $Conda)) { Stop-Install "conda not found at $Conda" }
& $Conda --version 2>&1 | ForEach-Object { Write-Log "conda: $Conda ($_)" }
if ($LASTEXITCODE -ne 0) { Stop-Install "$Conda --version failed: this is not a working conda" }
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null

$EnvDir = Join-Path (Split-Path (Split-Path $Conda)) "envs\$EnvName"
$EnvExists = Test-Path (Join-Path $EnvDir "python.exe")
if ($Mode -eq "Rebuild" -and $EnvExists) {
    Write-Log "##STAGE 0 $($Packages.stages.Count) Removing the old environment"
    Invoke-Native "removing env $EnvName" @("env", "remove", "-n", $EnvName, "-y")
    $EnvExists = $false
}

$total = $Packages.stages.Count
$index = 0
foreach ($stage in $Packages.stages) {
    $index += 1
    Write-Log "##STAGE $index $total $($stage.name)"
    $specs = @($stage.packages | ForEach-Object { $_.spec })
    if ($stage.installer -eq "create") {
        if ($EnvExists) {
            Write-Log "env $EnvName exists: $EnvDir"
        } else {
            Invoke-Native "creating env $EnvName" (@("create", "-n", $EnvName, "-c", $Channel, "--override-channels", "-y") + $specs)
            $EnvExists = $true
        }
    } else {
        $outdated = Get-OutdatedPackages @($stage.packages)
        if ($outdated.Count -eq 0) {
            Write-Log "already installed at the pinned versions: $($specs -join ', ')"
        } else {
            $toInstall = @($outdated | ForEach-Object { $_.spec })
            if ($stage.installer -eq "pip") {
                Invoke-Native "pip install" (@("run", "--no-capture-output", "-n", $EnvName, "python", "-m", "pip", "install", "--no-input") + $toInstall)
            } else {
                Invoke-Native "installing $($toInstall -join ' ')" (@("install", "-n", $EnvName, "-c", $Channel, "--override-channels", "-y") + $toInstall)
            }
        }
    }
    Write-Log "##STAGE_DONE $index"
}

$record = [ordered]@{
    conda = $Conda; env_name = $EnvName; code_folder = $CodeFolder
    installed = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
}
# Written as UTF-8 without a byte-order mark, which Python's json reads.
[IO.File]::WriteAllText($InstallRecord, ($record | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))
Write-Log "written: $InstallRecord"

if (-not $SkipChecks) {
    $env:PYTHONPATH = "$CodeFolder\src;$CodeFolder\scripts"
    $env:PYTHONPYCACHEPREFIX = Join-Path $env:LOCALAPPDATA "wfci_gui\pycache"
    $checkFile = Join-Path $ConfigDir "setup_check.json"
    $arguments = @("run", "--no-capture-output", "-n", $EnvName, "python", "-m", "wfci_gui.setup_check", "--out", $checkFile)
    if ($RunTests) { $arguments += "--tests" }
    Push-Location $ConfigDir
    & $Conda @arguments 2>&1 | ForEach-Object {
        $line = "$_"
        if ($line -match "^(PASS|FAIL)\s+(.*)$") { Write-Log "##CHECK $($Matches[1]) $($Matches[2])" } else { Write-Log $line }
    }
    $checksFailed = $LASTEXITCODE -ne 0
    Pop-Location
    if ($checksFailed) { Stop-Install "some setup checks failed (listed above)" }
}
Write-Log "##DONE"
exit 0
