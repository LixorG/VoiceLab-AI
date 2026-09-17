# Verificación completa: lint + tests con cobertura (backend y frontend) + build.
#   scripts\test.ps1            # todo
#   scripts\test.ps1 -Quick     # sin cobertura ni build (más rápido)
# Los tests de GPU real son opcionales: $env:VOICELAB_GPU_TESTS = '1' (ver CLAUDE.md).
param([switch]$Quick)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$failed = @()

function Step($name, [scriptblock]$block) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $block
    if ($LASTEXITCODE -ne 0) { $script:failed += $name; Write-Host "FALLÓ: $name" -ForegroundColor Red }
}

# FFmpeg es necesario para los tests de audio; si no está en el PATH se busca la instalación de winget.
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $bin = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($bin) { $env:PATH = "$($bin.DirectoryName);$env:PATH" }
}
$hasFfmpeg = [bool](Get-Command ffmpeg -ErrorAction SilentlyContinue)
if (-not $hasFfmpeg) { Write-Host 'AVISO: FFmpeg no encontrado; los tests de audio se omitirán y no se exigirá cobertura mínima.' -ForegroundColor Yellow }

Push-Location "$root\backend"
Step 'Backend · ruff' { & .\.venv\Scripts\python.exe -m ruff check app tests }
if ($Quick) {
    Step 'Backend · pytest' { & .\.venv\Scripts\python.exe -m pytest }
} else {
    $gate = if ($hasFfmpeg) { '--cov-fail-under=93' } else { '--cov-fail-under=0' }
    Step 'Backend · pytest + cobertura' { & .\.venv\Scripts\python.exe -m pytest --cov --cov-report=term-missing:skip-covered $gate }
}
Pop-Location

Push-Location "$root\frontend"
Step 'Frontend · oxlint' { npm run lint --silent }
Step 'Frontend · tsc' { npm run typecheck --silent }
if ($Quick) {
    Step 'Frontend · vitest' { npm test --silent }
} else {
    Step 'Frontend · vitest + cobertura' { npm run test:coverage --silent }
    Step 'Frontend · build' { npx vite build --logLevel warn }
}
Pop-Location

if ($failed.Count) {
    Write-Host "`nFallaron: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nTodo correcto." -ForegroundColor Green
exit 0
