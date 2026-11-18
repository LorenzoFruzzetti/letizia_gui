@echo off
rem Start WFCI Cohort (no console window). Install.bat must have run once for this user.
start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0env\start.ps1" %*
