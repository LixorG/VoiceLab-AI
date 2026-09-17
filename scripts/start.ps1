# Inicia VoiceLab AI como aplicación (un solo proceso y puerto: API + interfaz compilada) y abre el navegador.
#
#   scripts\start.ps1                 # http://127.0.0.1:8000
#   scripts\start.ps1 -Port 8100
#   scripts\start.ps1 -NoBrowser
#   scripts\start.ps1 -Rebuild        # recompila la interfaz aunque parezca actualizada
#
# Para desarrollar con recarga en caliente usa scripts\dev.ps1.
param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$Rebuild
)

# Native tools (npm, pip, vite) write warnings to stderr; PowerShell 5.1 would turn them into terminating
# errors under 'Stop'. Every external call is checked through its exit code instead.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$py = "$root\backend\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host 'VoiceLab AI no está instalado. Ejecuta primero install.cmd (o scripts\setup.ps1).' -ForegroundColor Red
    exit 1
}

# FFmpeg de winget puede no estar en el PATH de esta ventana.
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $bin = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($bin) { $env:PATH = "$($bin.DirectoryName);$env:PATH" }
    else { Write-Host 'AVISO: FFmpeg no encontrado: no se podrán procesar audios.' -ForegroundColor Yellow }
}

# Recompilar la interfaz si no existe o si el código es más nuevo que la compilación.
$dist = "$root\frontend\dist\index.html"
$sources = Get-ChildItem "$root\frontend\src", "$root\frontend\index.html", "$root\frontend\package.json" -Recurse -File -ErrorAction SilentlyContinue
$stale = (-not (Test-Path $dist)) -or $Rebuild -or (($sources | Measure-Object LastWriteTime -Maximum).Maximum -gt (Get-Item $dist).LastWriteTime)
if ($stale) {
    if ((Get-Command npm -ErrorAction SilentlyContinue) -and (Test-Path "$root\frontend\node_modules")) {
        Write-Host 'Compilando la interfaz…' -ForegroundColor Cyan
        Push-Location "$root\frontend"
        npm run build
        $ok = $LASTEXITCODE -eq 0
        Pop-Location
        if (-not $ok) { Write-Host 'No se pudo compilar la interfaz.' -ForegroundColor Red; exit 1 }
    } elseif (-not (Test-Path $dist)) {
        Write-Host 'Falta la interfaz compilada y Node.js no está instalado. Ejecuta scripts\setup.ps1.' -ForegroundColor Red
        exit 1
    }
}

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "El puerto $Port ya está en uso (¿VoiceLab AI ya está abierto?). Usa -Port para elegir otro." -ForegroundColor Red
    exit 1
}

$url = "http://127.0.0.1:$Port"
if (-not $NoBrowser) {
    # Open the browser once the server answers (the first start imports PyTorch and can take a while).
    Start-Job -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 120; $i++) {
            try { Invoke-WebRequest "$u/api/system/health" -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $u; return } catch { Start-Sleep 1 }
        }
    } -ArgumentList $url | Out-Null
}

Write-Host "VoiceLab AI en $url  (Ctrl+C para cerrar)" -ForegroundColor Green
$env:HOST = '127.0.0.1'
$env:PORT = "$Port"
Set-Location $root
& $py -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port $Port --log-level warning
