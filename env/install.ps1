# Install wizard of WFCI Cohort (started by Install.bat). Windows Forms, four pages:
#
#   1 Conda        the conda installations found; Browse...; "Install Miniforge for me".
#                  The chosen one must answer `conda --version` before Next is enabled.
#   2 Environment  whether the wfci_gui env exists: Create / Repair / Rebuild from scratch.
#   3 Install      setup_env.ps1 runs in the background: one progress bar per stage, the
#                  live log, "Copy log". The log is %APPDATA%\wfci_gui\install_<time>.log.
#   4 Finish       the setup checks as pass / fail, "Create desktop shortcut", "Start now".
#
# Needs nothing installed beforehand; needs the internet the first time (packages, and
# Miniforge when chosen). -Screenshot <folder> draws each page to a PNG and exits (used
# to check the layout on a machine where the window cannot be seen).

param([string]$Screenshot = "")

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$CodeFolder = Split-Path -Parent $PSScriptRoot
$Packages = Get-Content -Raw -Path (Join-Path $PSScriptRoot "packages.json") | ConvertFrom-Json
$EnvName = $Packages.env_name
$ConfigDir = Join-Path $env:APPDATA "wfci_gui"
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
$MiniforgeUrl = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
$MiniforgeTarget = Join-Path $env:USERPROFILE "miniforge3"

$script:Conda = ""
$script:Process = $null
$script:LogPath = ""
$script:LogOffset = 0
$script:Failed = ""
$script:Checks = @()
$script:Busy = $false

# ----------------------------------------------------------------------------- helpers
function Find-CondaInstallations {
    $found = @()
    $record = Join-Path $ConfigDir "install.json"
    if (Test-Path $record) { $found += (Get-Content -Raw $record | ConvertFrom-Json).conda }
    foreach ($name in "anaconda3", "miniconda3", "miniforge3") {
        $found += (Join-Path $env:USERPROFILE "$name\condabin\conda.bat")
    }
    if ($env:CONDA_EXE) { $found += (Join-Path (Split-Path (Split-Path $env:CONDA_EXE)) "condabin\conda.bat") }
    return @($found | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)
}

function Resolve-CondaPath([string]$Path) {
    # A conda.bat / conda.exe, or an installation folder holding condabin\conda.bat.
    if (Test-Path $Path -PathType Container) {
        $candidate = Join-Path $Path "condabin\conda.bat"
        if (Test-Path $candidate) { return $candidate }
        return ""
    }
    return $Path
}

function Test-Conda([string]$Path) {
    if (-not $Path -or -not (Test-Path $Path)) { return "" }
    $version = & $Path --version 2>$null | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0) { return "" }
    return "$version"
}

function New-Label([string]$Text, [int]$Top, [int]$Height = 40) {
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Text
    $label.SetBounds(20, $Top, 600, $Height)
    return $label
}

# ----------------------------------------------------------------------------- window
$form = New-Object System.Windows.Forms.Form
$form.Text = "WFCI Cohort - Install"
$form.ClientSize = New-Object System.Drawing.Size(640, 480)
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.StartPosition = "CenterScreen"
$form.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$title = New-Object System.Windows.Forms.Label
$title.SetBounds(20, 12, 600, 28)
$title.Font = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Bold)
$form.Controls.Add($title)

$pages = @()
foreach ($i in 0..3) {
    $panel = New-Object System.Windows.Forms.Panel
    $panel.SetBounds(0, 44, 640, 380)
    $panel.Visible = $false
    $form.Controls.Add($panel)
    $pages += $panel
}
$titles = @("1 of 4 - Conda", "2 of 4 - Environment", "3 of 4 - Install", "4 of 4 - Finished")

$back = New-Object System.Windows.Forms.Button
$back.Text = "< Back"
$back.SetBounds(360, 436, 80, 28)
$next = New-Object System.Windows.Forms.Button
$next.Text = "Next >"
$next.SetBounds(446, 436, 80, 28)
$cancel = New-Object System.Windows.Forms.Button
$cancel.Text = "Cancel"
$cancel.SetBounds(540, 436, 80, 28)
$form.Controls.AddRange(@($back, $next, $cancel))

# --- page 1: conda -------------------------------------------------------------------
$p1 = $pages[0]
$p1.Controls.Add((New-Label ("The program runs Python through conda. Choose a conda installation, or let the " +
    "installer download Miniforge (a small conda, about 90 MB) into $MiniforgeTarget.") 4 40))
$condaList = New-Object System.Windows.Forms.ListBox
$condaList.SetBounds(20, 50, 600, 120)
foreach ($path in Find-CondaInstallations) { [void]$condaList.Items.Add($path) }
$p1.Controls.Add($condaList)
$browse = New-Object System.Windows.Forms.Button
$browse.Text = "Browse..."
$browse.SetBounds(20, 178, 100, 28)
$installMiniforge = New-Object System.Windows.Forms.Button
$installMiniforge.Text = "Install Miniforge for me"
$installMiniforge.SetBounds(128, 178, 180, 28)
$p1.Controls.AddRange(@($browse, $installMiniforge))
$condaStatus = New-Label "" 216 60
$p1.Controls.Add($condaStatus)

function Select-Conda([string]$Path) {
    $path = Resolve-CondaPath $Path
    $version = Test-Conda $path
    if ($version) {
        $script:Conda = $path
        $condaStatus.ForeColor = [System.Drawing.Color]::DarkGreen
        $condaStatus.Text = "Works: $version`n$path"
    } else {
        $script:Conda = ""
        $condaStatus.ForeColor = [System.Drawing.Color]::DarkRed
        $condaStatus.Text = "Not a working conda: $Path"
    }
    $next.Enabled = [bool]$script:Conda
}

$condaList.Add_SelectedIndexChanged({ if ($condaList.SelectedItem) { Select-Conda $condaList.SelectedItem } })
$browse.Add_Click({
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Choose conda.bat (in the condabin folder of a conda installation)"
    $dialog.Filter = "conda (conda.bat;conda.exe)|conda.bat;conda.exe"
    if ($dialog.ShowDialog($form) -eq "OK") {
        $index = $condaList.Items.Add($dialog.FileName)
        $condaList.SelectedIndex = $index
    }
})
$installMiniforge.Add_Click({
    if (Test-Path (Join-Path $MiniforgeTarget "condabin\conda.bat")) {
        $condaStatus.Text = "Miniforge is already installed in $MiniforgeTarget."
        $index = $condaList.Items.Add((Join-Path $MiniforgeTarget "condabin\conda.bat"))
        $condaList.SelectedIndex = $index
        return
    }
    $script:LogPath = Join-Path $ConfigDir ("install_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $script:LogOffset = 0
    $exe = Join-Path $env:TEMP "Miniforge3-Windows-x86_64.exe"
    # In a separate hidden process, so the window keeps responding during the download.
    $command = "`$ProgressPreference = 'SilentlyContinue'; " +
        "Add-Content -Path '$($script:LogPath)' -Value 'downloading $MiniforgeUrl'; " +
        "Invoke-WebRequest -UseBasicParsing -Uri '$MiniforgeUrl' -OutFile '$exe'; " +
        "Add-Content -Path '$($script:LogPath)' -Value 'installing Miniforge into $MiniforgeTarget'; " +
        "Start-Process -Wait -FilePath '$exe' -ArgumentList '/InstallationType=JustMe','/RegisterPython=0','/AddToPath=0','/S','/D=$MiniforgeTarget'; " +
        "Add-Content -Path '$($script:LogPath)' -Value 'Miniforge installed'"
    $script:Process = Start-Process -PassThru -WindowStyle Hidden -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command)
    $script:Busy = $true
    $installMiniforge.Enabled = $false
    $next.Enabled = $false
    $condaStatus.ForeColor = [System.Drawing.Color]::Black
    $condaStatus.Text = "Downloading and installing Miniforge (a few minutes)..."
    $miniforgeTimer.Start()
})
$miniforgeTimer = New-Object System.Windows.Forms.Timer
$miniforgeTimer.Interval = 1000
$miniforgeTimer.Add_Tick({
    if (-not $script:Process.HasExited) { return }
    $miniforgeTimer.Stop()
    $script:Busy = $false
    $installMiniforge.Enabled = $true
    $conda = Join-Path $MiniforgeTarget "condabin\conda.bat"
    if (Test-Path $conda) {
        $index = $condaList.Items.Add($conda)
        $condaList.SelectedIndex = $index
    } else {
        $condaStatus.ForeColor = [System.Drawing.Color]::DarkRed
        $condaStatus.Text = "Miniforge could not be installed. Check the internet connection. Log: $($script:LogPath)"
    }
})

# --- page 2: environment -------------------------------------------------------------
$p2 = $pages[1]
$envStatus = New-Label "" 4 40
$p2.Controls.Add($envStatus)
$modeCreate = New-Object System.Windows.Forms.RadioButton
$modeCreate.Text = "Create the environment"
$modeCreate.SetBounds(20, 56, 600, 24)
$modeRepair = New-Object System.Windows.Forms.RadioButton
$modeRepair.Text = "Repair: add what is missing (keeps what is there)"
$modeRepair.SetBounds(20, 84, 600, 24)
$modeRebuild = New-Object System.Windows.Forms.RadioButton
$modeRebuild.Text = "Rebuild from scratch: delete the environment and build it again"
$modeRebuild.SetBounds(20, 112, 600, 24)
$p2.Controls.AddRange(@($modeCreate, $modeRepair, $modeRebuild))
$p2.Controls.Add((New-Label ("The environment is a separate set of Python packages, '$EnvName', with exact " +
    "versions. It takes 1-2 GB and 5-20 minutes the first time. Your data is not touched.") 150 40))

function Update-EnvironmentPage {
    $envDir = Join-Path (Split-Path (Split-Path $script:Conda)) "envs\$EnvName"
    $exists = Test-Path (Join-Path $envDir "python.exe")
    if ($exists) {
        $envStatus.Text = "The environment '$EnvName' exists:`n$envDir"
        $modeCreate.Enabled = $false
        $modeRepair.Enabled = $true
        $modeRebuild.Enabled = $true
        $modeRepair.Checked = $true
    } else {
        $envStatus.Text = "The environment '$EnvName' does not exist yet. It will be created in`n$envDir"
        $modeCreate.Enabled = $true
        $modeRepair.Enabled = $false
        $modeRebuild.Enabled = $false
        $modeCreate.Checked = $true
    }
}

# --- page 3: install -----------------------------------------------------------------
$p3 = $pages[2]
$stageNames = @($Packages.stages | ForEach-Object { $_.name }) + @("Checks")
$bars = @()
$top = 4
foreach ($name in $stageNames) {
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $name
    $label.SetBounds(20, $top + 3, 160, 20)
    $bar = New-Object System.Windows.Forms.ProgressBar
    $bar.SetBounds(186, $top, 434, 20)
    $p3.Controls.AddRange(@($label, $bar))
    $bars += $bar
    $top += 26
}
$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Multiline = $true
$logBox.ScrollBars = "Vertical"
$logBox.ReadOnly = $true
$logBox.Font = New-Object System.Drawing.Font("Consolas", 8)
$logBox.SetBounds(20, $top + 6, 600, 336 - $top)
$copyLog = New-Object System.Windows.Forms.Button
$copyLog.Text = "Copy log"
$copyLog.SetBounds(20, 352, 100, 26)
$logLabel = New-Label "" 356 20
$logLabel.SetBounds(128, 356, 492, 20)
$p3.Controls.AddRange(@($logBox, $copyLog, $logLabel))
$copyLog.Add_Click({ if ($logBox.Text) { [System.Windows.Forms.Clipboard]::SetText($logBox.Text) } })

$installTimer = New-Object System.Windows.Forms.Timer
$installTimer.Interval = 400

function Read-NewLogLines {
    if (-not (Test-Path $script:LogPath)) { return @() }
    $stream = [System.IO.File]::Open($script:LogPath, "Open", "Read", "ReadWrite")
    [void]$stream.Seek($script:LogOffset, "Begin")
    $reader = New-Object System.IO.StreamReader($stream)
    $text = $reader.ReadToEnd()
    $script:LogOffset = $stream.Position
    $reader.Close()
    return @($text -split "`r?`n" | Where-Object { $_ -ne "" })
}

function Show-LogLine([string]$Line) {
    if ($Line -match "^##STAGE (\d+) (\d+) (.*)$") {
        $index = [int]$Matches[1]
        if ($index -ge 1) {
            $bars[$index - 1].Style = "Marquee"
            $bars[$index - 1].MarqueeAnimationSpeed = 30
        }
    } elseif ($Line -match "^##STAGE_DONE (\d+)$") {
        $bar = $bars[[int]$Matches[1] - 1]
        $bar.Style = "Continuous"
        $bar.Value = 100
        if ([int]$Matches[1] -eq $bars.Count - 1) {
            $bars[-1].Style = "Marquee"
            $bars[-1].MarqueeAnimationSpeed = 30
        }
    } elseif ($Line -match "^##CHECK (PASS|FAIL) (.*)$") {
        $script:Checks += , @($Matches[1], $Matches[2])
    } elseif ($Line -match "^##FAILED (.*)$") {
        $script:Failed = $Matches[1]
    } elseif ($Line -eq "##DONE") {
        $bars[-1].Style = "Continuous"
        $bars[-1].Value = 100
    }
    if (-not $Line.StartsWith("##")) { $logBox.AppendText($Line + "`r`n") }
}

function Start-Install {
    $mode = if ($modeRebuild.Checked) { "Rebuild" } elseif ($modeRepair.Checked) { "Repair" } else { "Create" }
    $script:LogPath = Join-Path $ConfigDir ("install_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    $script:LogOffset = 0
    $script:Failed = ""
    $script:Checks = @()
    foreach ($bar in $bars) { $bar.Style = "Continuous"; $bar.Value = 0 }
    $logBox.Clear()
    $logLabel.Text = "Log: $($script:LogPath)"
    $arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSScriptRoot\setup_env.ps1`"",
                   "-Conda", "`"$($script:Conda)`"", "-Mode", $mode, "-LogFile", "`"$($script:LogPath)`"")
    $script:Process = Start-Process -PassThru -WindowStyle Hidden -FilePath "powershell.exe" -ArgumentList $arguments
    $script:Busy = $true
    $back.Enabled = $false
    $next.Enabled = $false
    $installTimer.Start()
}

$installTimer.Add_Tick({
    foreach ($line in Read-NewLogLines) { Show-LogLine $line }
    if (-not $script:Process.HasExited) { return }
    foreach ($line in Read-NewLogLines) { Show-LogLine $line }
    $installTimer.Stop()
    $script:Busy = $false
    if ($script:Process.ExitCode -ne 0 -and -not $script:Failed) {
        $script:Failed = "the install stopped (exit code $($script:Process.ExitCode))"
    }
    if ($script:Failed) {
        $logBox.AppendText("`r`nFAILED: $($script:Failed)`r`nSend the log file to whoever maintains the program: $($script:LogPath)`r`n")
        $back.Enabled = $true
        $next.Text = "Retry"
        $next.Enabled = $true
    } else {
        Show-Page 3
    }
})

# --- page 4: finish ------------------------------------------------------------------
$p4 = $pages[3]
$finishText = New-Label "" 4 40
$p4.Controls.Add($finishText)
$checkList = New-Object System.Windows.Forms.ListView
$checkList.View = "Details"
$checkList.FullRowSelect = $true
$checkList.SetBounds(20, 50, 600, 230)
[void]$checkList.Columns.Add("Result", 70)
[void]$checkList.Columns.Add("Check", 510)
$p4.Controls.Add($checkList)
$shortcut = New-Object System.Windows.Forms.CheckBox
$shortcut.Text = "Create a desktop shortcut"
$shortcut.Checked = $true
$shortcut.SetBounds(20, 292, 300, 24)
$startNow = New-Object System.Windows.Forms.CheckBox
$startNow.Text = "Start WFCI Cohort now"
$startNow.Checked = $true
$startNow.SetBounds(20, 318, 300, 24)
$p4.Controls.AddRange(@($shortcut, $startNow))

function Update-FinishPage {
    $checkList.Items.Clear()
    foreach ($check in $script:Checks) {
        $item = New-Object System.Windows.Forms.ListViewItem($check[0])
        [void]$item.SubItems.Add($check[1])
        $item.ForeColor = if ($check[0] -eq "PASS") { [System.Drawing.Color]::DarkGreen } else { [System.Drawing.Color]::DarkRed }
        [void]$checkList.Items.Add($item)
    }
    $finishText.Text = "WFCI Cohort is installed. Start it with Start.bat in $CodeFolder (or the desktop shortcut). The log is $($script:LogPath)."
}

function Complete-Install {
    if ($shortcut.Checked) {
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "WFCI Cohort.lnk"))
        $link.TargetPath = Join-Path $CodeFolder "Start.bat"
        $link.WorkingDirectory = $CodeFolder
        $link.WindowStyle = 7
        $link.Description = "WFCI Cohort"
        $link.Save()
    }
    if ($startNow.Checked) {
        Start-Process -FilePath (Join-Path $CodeFolder "Start.bat") -WorkingDirectory $CodeFolder -WindowStyle Hidden
    }
}

# ----------------------------------------------------------------------------- navigation
$script:Page = 0
function Show-Page([int]$Index) {
    $script:Page = $Index
    for ($i = 0; $i -lt $pages.Count; $i++) { $pages[$i].Visible = ($i -eq $Index) }
    $title.Text = $titles[$Index]
    $back.Enabled = $Index -in 1, 2
    $next.Text = "Next >"
    switch ($Index) {
        0 { $next.Enabled = [bool]$script:Conda }
        1 { Update-EnvironmentPage; $next.Text = "Install"; $next.Enabled = $true }
        2 { $next.Enabled = $false }
        3 { Update-FinishPage; $back.Enabled = $false; $next.Text = "Finish"; $next.Enabled = $true; $cancel.Enabled = $false }
    }
}

$next.Add_Click({
    switch ($script:Page) {
        0 { Show-Page 1 }
        1 { Show-Page 2; Start-Install }
        2 { $next.Text = "Next >"; Start-Install }        # Retry after a failure
        3 { Complete-Install; $form.Close() }
    }
})
$back.Add_Click({ if (-not $script:Busy -and $script:Page -gt 0) { Show-Page ($script:Page - 1) } })
$cancel.Add_Click({ $form.Close() })
$form.Add_FormClosing({
    param($sender, $event)
    if ($script:Busy) {
        $answer = [System.Windows.Forms.MessageBox]::Show($form, "The install is still running. Stop it and close?",
            "WFCI Cohort - Install", "YesNo", "Warning")
        if ($answer -ne "Yes") { $event.Cancel = $true; return }
        Stop-Process -Id $script:Process.Id -Force
    }
})

if ($condaList.Items.Count -gt 0) { $condaList.SelectedIndex = 0 }
Show-Page 0

if ($Screenshot) {
    # Draw every page to <folder>\install_page<N>.png without showing the window.
    New-Item -ItemType Directory -Force -Path $Screenshot | Out-Null
    $script:Checks = @(@("PASS", "python: 3.11.16"), @("FAIL", "example of a failed check"))
    # Controls are drawn only once the window has been shown: show it off-screen.
    $form.StartPosition = "Manual"
    $form.Location = New-Object System.Drawing.Point(-3000, -3000)
    $form.ShowInTaskbar = $false
    $form.Show()
    foreach ($i in 0..3) {
        if ($i -eq 1 -and -not $script:Conda) { $script:Conda = "C:\conda\condabin\conda.bat" }
        Show-Page $i
        if ($i -eq 2) { $logBox.Text = "conda: example`r`n> conda install ..."; $bars[0].Value = 100 }
        [System.Windows.Forms.Application]::DoEvents()
        $bitmap = New-Object System.Drawing.Bitmap($form.Width, $form.Height)
        $form.DrawToBitmap($bitmap, (New-Object System.Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
        $bitmap.Save((Join-Path $Screenshot "install_page$($i + 1).png"))
        $bitmap.Dispose()
    }
    exit 0
}

[void]$form.ShowDialog()
