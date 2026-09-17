@echo off
rem Instala VoiceLab AI (doble clic). Ej.: install.cmd -InstallPrereqs  /  install.cmd -Engines f5tts
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
pause
