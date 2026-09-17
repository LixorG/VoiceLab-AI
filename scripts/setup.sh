#!/usr/bin/env bash
# Instalación de VoiceLab AI en Linux / macOS (idempotente).
#
#   scripts/setup.sh                          # GPU NVIDIA (CUDA 12.8) + F5/E2-TTS + Qwen3-TTS + transcripción
#   ENGINES=f5tts scripts/setup.sh            # solo algunos motores ("none" = sin motores)
#   TORCH=cpu scripts/setup.sh                # PyTorch sin CUDA (en macOS se usa siempre la build por defecto, con MPS)
#   SKIP_FRONTEND=1 scripts/setup.sh          # no compilar la interfaz
#
# Requisitos: Python 3.11, Node.js 20+, FFmpeg (con rubberband para tono/velocidad) y, para Qwen3-TTS, sox.
# Nota: este script no se ha probado en una máquina Linux/macOS real (solo la instalación de Windows).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINES="${ENGINES:-f5tts,qwen3tts}"
TORCH="${TORCH:-cu128}"
TORCH_VERSION="2.8.0"
PYTHON="${PYTHON:-python3.11}"

info() { printf '\n\033[36m>> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

info "Comprobando requisitos"
command -v "$PYTHON" >/dev/null || fail "Falta Python 3.11 ($PYTHON). Instálalo o indica otro con PYTHON=..."
if [[ -z "${SKIP_FRONTEND:-}" ]]; then
  command -v node >/dev/null || fail "Falta Node.js 20+ (o usa SKIP_FRONTEND=1)."
  node_major="$(node --version | sed 's/^v//; s/\..*//')"
  (( node_major >= 20 )) || fail "Node.js $(node --version) es demasiado antiguo: se necesita 20+."
fi
command -v ffmpeg >/dev/null || printf 'AVISO: FFmpeg no está instalado; es obligatorio para procesar audio.\n'

cd "$ROOT/backend"
[[ -x .venv/bin/python ]] || { info "Creando entorno virtual"; "$PYTHON" -m venv .venv; }
PY="$ROOT/backend/.venv/bin/python"
"$PY" -m pip install --upgrade pip --quiet

extras="dev,audio,asr"
pip_extra=()
if [[ "$ENGINES" != "none" ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    torch_spec="torch==$TORCH_VERSION torchaudio==$TORCH_VERSION"; index_args=()
  else
    torch_spec="torch==$TORCH_VERSION+$TORCH torchaudio==$TORCH_VERSION+$TORCH"
    index_args=(--index-url "https://download.pytorch.org/whl/$TORCH")
  fi
  info "Instalando PyTorch ($torch_spec)"
  # shellcheck disable=SC2086
  "$PY" -m pip install $torch_spec ${index_args[@]+"${index_args[@]}"}
  constraints="$(mktemp)"
  "$PY" -c "import torch, torchaudio; print(f'torch=={torch.__version__}'); print(f'torchaudio=={torchaudio.__version__}')" > "$constraints"
  pip_extra=(-c "$constraints")
  [[ "$(uname)" == "Darwin" ]] || pip_extra+=(--extra-index-url "https://download.pytorch.org/whl/$TORCH")
  extras="$extras,$ENGINES"
fi
info "Instalando backend: $extras"
"$PY" -m pip install -e ".[$extras]" ${pip_extra[@]+"${pip_extra[@]}"}  # empty-array safe on bash 3.2 (macOS)

if [[ -z "${SKIP_FRONTEND:-}" ]]; then
  cd "$ROOT/frontend"
  info "Instalando y compilando la interfaz"
  npm ci --no-audit --no-fund
  npm run build
fi

cd "$ROOT"
[[ -f .env ]] || { cp .env.example .env; info "Creado .env"; }
mkdir -p data models
info "Diagnóstico"
"$PY" -m app.cli doctor || true
printf '\nInstalación terminada. Inicia VoiceLab AI con scripts/start.sh\n'
