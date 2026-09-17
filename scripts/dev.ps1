# Inicia backend (http://127.0.0.1:8000) y frontend (http://127.0.0.1:5173) en ventanas separadas.
$root = Split-Path -Parent $PSScriptRoot
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root'; .\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --reload-dir backend/app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root\frontend'; npm run dev"
Write-Host 'Abre http://127.0.0.1:5173'
