#!/usr/bin/env bash
# Inicia VoiceLab AI (API + interfaz compilada en un solo puerto). PORT=8100 scripts/start.sh para cambiar el puerto.
# Nota: no probado en una máquina Linux/macOS real.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
PY="$ROOT/backend/.venv/bin/python"
[[ -x "$PY" ]] || { echo "VoiceLab AI no está instalado: ejecuta scripts/setup.sh" >&2; exit 1; }

if [[ ! -f "$ROOT/frontend/dist/index.html" || -n "$(find "$ROOT/frontend/src" -newer "$ROOT/frontend/dist/index.html" -type f -print -quit 2>/dev/null)" ]]; then
  if command -v npm >/dev/null && [[ -d "$ROOT/frontend/node_modules" ]]; then
    echo "Compilando la interfaz…"
    (cd "$ROOT/frontend" && npm run build)
  elif [[ ! -f "$ROOT/frontend/dist/index.html" ]]; then
    echo "Falta la interfaz compilada: ejecuta scripts/setup.sh" >&2; exit 1
  fi
fi

URL="http://127.0.0.1:$PORT"
if [[ -z "${NO_BROWSER:-}" ]]; then
  (
    for _ in $(seq 120); do
      if curl -fs "$URL/api/system/health" >/dev/null 2>&1; then
        { command -v xdg-open >/dev/null && xdg-open "$URL"; } || { command -v open >/dev/null && open "$URL"; } || true
        break
      fi
      sleep 1
    done
  ) &
fi

echo "VoiceLab AI en $URL (Ctrl+C para cerrar)"
cd "$ROOT"
exec "$PY" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port "$PORT" --log-level warning
