@echo off
rem Install WFCI Cohort for this Windows user: opens the install wizard.
rem Run it again after replacing this folder with a newer version.
start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0env\install.ps1"
