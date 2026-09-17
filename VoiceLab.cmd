@echo off
rem Abre VoiceLab AI (doble clic). Argumentos opcionales: -Port 8100 -NoBrowser -Rebuild
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
if errorlevel 1 pause
