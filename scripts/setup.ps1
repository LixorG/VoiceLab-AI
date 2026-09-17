# Instalación de VoiceLab AI en Windows (idempotente: se puede repetir para actualizar o reparar).
#
#   scripts\setup.ps1                              # GPU NVIDIA + F5/E2-TTS + Qwen3-TTS + transcripción
#   scripts\setup.ps1 -Engines f5tts               # solo algunos motores (f5tts incluye E2-TTS)
#   scripts\setup.ps1 -Engines none                # sin motores de voz (solo biblioteca, transcripción y UI)
#   scripts\setup.ps1 -Cpu                         # PyTorch sin CUDA (muy lento, para probar)
#   scripts\setup.ps1 -InstallPrereqs              # instala con winget lo que falte (Python 3.11, Node LTS, FFmpeg)
#   scripts\setup.ps1 -SkipFrontend                # no instala Node ni compila la interfaz (usa frontend\dist si existe)
#
# Los pesos de los modelos NO se descargan aquí: se descargan desde la app o con
#   backend\.venv\Scripts\python -m app.cli download f5tts
param(
    [string[]]$Engines = @('f5tts', 'qwen3tts'),
    [switch]$Cpu,
    [switch]$InstallPrereqs,
    [switch]$SkipFrontend
)

# Native tools (npm, pip, vite) write warnings to stderr; PowerShell 5.1 would turn them into terminating
# errors under 'Stop'. Every external call is checked through its exit code instead.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$TorchVersion = '2.8.0'
$TorchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
$TorchTag = if ($Cpu) { 'cpu' } else { 'cu128' }
# `install.cmd -Engines f5tts,qwen3tts` arrives as one string: split and validate here.
$Engines = @($Engines | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
$invalid = $Engines | Where-Object { $_ -notin @('f5tts', 'qwen3tts', 'none') }
if ($invalid) { Write-Host "Motor no válido: $($invalid -join ', '). Opciones: f5tts, qwen3tts, none." -ForegroundColor Red; exit 1 }
$withEngines = -not ($Engines -contains 'none')

function Info($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { Fail "Falló: $exe $($arguments -join ' ')" }
}
function Refresh-Path {
    $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH', 'User')
}
function Winget-Install($id) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { Fail "winget no está disponible; instala $id manualmente." }
    Info "Instalando $id con winget…"
    winget install --id $id -e --accept-source-agreements --accept-package-agreements
    Refresh-Path
}

# ---------------------------------------------------------------- requisitos
Info 'Comprobando requisitos'
$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $python = 'py311' }
}
if (-not $python) {
    if ($InstallPrereqs) { Winget-Install 'Python.Python.3.11' } else { Fail 'Falta Python 3.11. Instálalo (winget install Python.Python.3.11) o usa -InstallPrereqs.' }
}

if (-not $SkipFrontend -and -not (Get-Command node -ErrorAction SilentlyContinue)) {
    if ($InstallPrereqs) { Winget-Install 'OpenJS.NodeJS.LTS' } else { Fail 'Falta Node.js 20+. Instálalo (winget install OpenJS.NodeJS.LTS), usa -InstallPrereqs o -SkipFrontend.' }
}
if (-not $SkipFrontend) {
    $nodeMajor = [int]((node --version).TrimStart('v').Split('.')[0])
    if ($nodeMajor -lt 20) { Fail "Node.js $(node --version) es demasiado antiguo: se necesita 20 o superior." }
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $bin = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $bin -and $InstallPrereqs) { Winget-Install 'Gyan.FFmpeg' }
    elseif (-not $bin) { Write-Host 'AVISO: FFmpeg no está en el PATH. Es obligatorio para procesar audio (winget install Gyan.FFmpeg).' -ForegroundColor Yellow }
}

# ---------------------------------------------------------------- backend
Push-Location "$root\backend"
try {
    if (-not (Test-Path .venv\Scripts\python.exe)) {
        Info 'Creando entorno virtual (Python 3.11)'
        Run 'py' @('-3.11', '-m', 'venv', '.venv')
    }
    $py = (Resolve-Path .venv\Scripts\python.exe).Path
    Run $py @('-m', 'pip', 'install', '--upgrade', 'pip', '--quiet')

    $extras = @('dev', 'audio', 'asr')
    $pipArgs = @()
    if ($withEngines) {
        $expected = "$TorchVersion+$TorchTag"
        $current = & $py -c "import torch; print(torch.__version__)" 2>$null
        if ($current -ne $expected) {
            Info "Instalando PyTorch $expected (puede tardar: ~3 GB con CUDA)"
            Run $py @('-m', 'pip', 'install', "torch==$expected", "torchaudio==$expected", '--index-url', $TorchIndex)
        } else {
            Info "PyTorch $expected ya instalado"
        }
        # Keep the exact torch build while resolving the engines (PyPI would pull a CPU build).
        $constraints = Join-Path $env:TEMP 'voicelab-torch-constraints.txt'
        Set-Content -Path $constraints -Value @("torch==$expected", "torchaudio==$expected") -Encoding ascii
        $pipArgs = @('-c', $constraints, '--extra-index-url', $TorchIndex)
        $extras += $Engines
    }
    Info "Instalando backend: $($extras -join ', ')"
    Run $py (@('-m', 'pip', 'install', '-e', ".[$($extras -join ',')]") + $pipArgs)
} finally {
    Pop-Location
}

# ---------------------------------------------------------------- frontend
if (-not $SkipFrontend) {
    Push-Location "$root\frontend"
    try {
        Info 'Instalando dependencias de la interfaz'
        if (Test-Path package-lock.json) { Run 'npm' @('ci', '--no-audit', '--no-fund') } else { Run 'npm' @('install', '--no-audit', '--no-fund') }
        Info 'Compilando la interfaz'
        Run 'npm' @('run', 'build')
    } finally {
        Pop-Location
    }
} elseif (-not (Test-Path "$root\frontend\dist\index.html")) {
    Write-Host 'AVISO: -SkipFrontend sin interfaz compilada (frontend\dist). Solo estará disponible la API.' -ForegroundColor Yellow
}

# ---------------------------------------------------------------- configuración
if (-not (Test-Path "$root\.env")) {
    Copy-Item "$root\.env.example" "$root\.env"
    Info 'Creado .env a partir de .env.example'
}
foreach ($dir in 'data', 'models') { New-Item -ItemType Directory -Force -Path "$root\$dir" | Out-Null }

Info 'Diagnóstico'
& "$root\backend\.venv\Scripts\python.exe" -m app.cli doctor
Write-Host "`nInstalación terminada. Inicia VoiceLab AI con VoiceLab.cmd (o scripts\start.ps1)." -ForegroundColor Green
if ($withEngines) { Write-Host 'Descarga los pesos desde la app (panel del motor) o con: backend\.venv\Scripts\python -m app.cli download f5tts' }
