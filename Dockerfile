# VoiceLab AI — una sola imagen: interfaz compilada + API (puerto 8000).
#
#   docker compose up --build                                            # CPU
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # GPU NVIDIA (Linux / WSL2)
#
# Build args:
#   TORCH_INDEX    https://download.pytorch.org/whl/cpu (por defecto) o .../whl/cu128 para GPU
#   ENGINE_EXTRAS  motores a instalar: "audio,asr,f5tts,qwen3tts" (por defecto), "audio,asr" para ninguno
# Los pesos NO van en la imagen: se guardan en el volumen /models (descarga desde la app o `python -m app.cli download`).

# ---------------------------------------------------------------- interfaz
FROM node:24-bookworm-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# The type check runs in scripts/test.ps1; the image only needs the bundle.
RUN npx vite build

# ---------------------------------------------------------------- backend
FROM python:3.11-slim-bookworm
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.8.0
ARG ENGINE_EXTRAS=audio,asr,f5tts,qwen3tts

# ffmpeg de Debian incluye rubberband (tono/velocidad) y afftdn; sox lo usa qwen-tts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg sox curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY backend/pyproject.toml ./
COPY backend/app ./app

RUN set -eux; \
    case ",${ENGINE_EXTRAS}," in \
      *,f5tts,*|*,qwen3tts,*) \
        pip install --no-cache-dir "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"; \
        python -c "import torch, torchaudio; print(f'torch=={torch.__version__}'); print(f'torchaudio=={torchaudio.__version__}')" > /tmp/torch-constraints.txt; \
        pip install --no-cache-dir ".[${ENGINE_EXTRAS}]" -c /tmp/torch-constraints.txt --extra-index-url "${TORCH_INDEX}" ;; \
      *) pip install --no-cache-dir ".[${ENGINE_EXTRAS}]" ;; \
    esac

COPY --from=ui /ui/dist /app/frontend/dist
COPY scripts/docker-entrypoint.sh /usr/local/bin/voicelab-entrypoint
RUN chmod +x /usr/local/bin/voicelab-entrypoint \
    && useradd --create-home --uid 1000 voicelab \
    && mkdir -p /data /models \
    && chown voicelab:voicelab /data /models
USER voicelab

ENV HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/data \
    MODEL_DIR=/models \
    HF_HOME=/models/huggingface \
    FRONTEND_DIST=/app/frontend/dist \
    PYTHONUNBUFFERED=1
EXPOSE 8000
VOLUME ["/data", "/models"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fs "http://127.0.0.1:${PORT}/api/system/health" || exit 1
ENTRYPOINT ["voicelab-entrypoint"]
CMD ["sh", "-c", "uvicorn app.main:app --host \"$HOST\" --port \"$PORT\""]
