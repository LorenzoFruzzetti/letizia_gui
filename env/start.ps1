# Start WFCI Cohort (Start.bat runs this with a hidden window).
#
# Reads %APPDATA%\wfci_gui\install.json (written by Install.bat) for the conda to use, sets
# PYTHONPATH to this folder's src and scripts and PYTHONPYCACHEPREFIX to
# %LOCALAPPDATA%\wfci_gui\pycache (nothing is written inside the program folder), and runs
#   conda run --no-capture-output -n wfci_gui python -m wfci_gui [arguments]
# Its output goes to %APPDATA%\wfci_gui\start.log; when the program stops with an error
# before its window can show it, a message box names that log.

Add-Type -AssemblyName System.Windows.Forms

$CodeFolder = Split-Path -Parent $PSScriptRoot
$ConfigDir = Join-Path $env:APPDATA "wfci_gui"
$Record = Join-Path $ConfigDir "install.json"
$Log = Join-Path $ConfigDir "start.log"

function Show-Problem([string]$Text) {
    [void][System.Windows.Forms.MessageBox]::Show($Text, "WFCI Cohort", "OK", "Error")
    exit 1
}

if (-not (Test-Path $Record)) {
    Show-Problem "WFCI Cohort is not installed for this Windows user yet.`n`nDouble-click Install.bat in $CodeFolder."
}
$install = Get-Content -Raw -Path $Record | ConvertFrom-Json
if (-not (Test-Path $install.conda)) {
    Show-Problem "The conda chosen at install time is gone:`n$($install.conda)`n`nRun Install.bat in $CodeFolder again."
}

$env:PYTHONPATH = "$CodeFolder\src;$CodeFolder\scripts"
$env:PYTHONPYCACHEPREFIX = Join-Path $env:LOCALAPPDATA "wfci_gui\pycache"
$arguments = @($args | ForEach-Object { "`"$_`"" }) -join " "
# cmd redirection, not Tee-Object: the whole run goes to the log as it happens.
$command = "`"`"$($install.conda)`" run --no-capture-output -n $($install.env_name) python -m wfci_gui $arguments > `"$Log`" 2>&1`""
Push-Location $ConfigDir
& cmd.exe /c $command
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) {
    $tail = (Get-Content -Path $Log -Tail 12) -join "`n"
    Show-Problem "WFCI Cohort stopped with an error (exit code $code).`n`n$tail`n`nThe full output is in $Log. Send it to whoever maintains the program."
}
