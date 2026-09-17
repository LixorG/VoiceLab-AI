"""Process-wide ASR manager: lazy loading, serialised GPU access, background model download."""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import BaseModel

from app.asr.base import ASRBackend, TranscriptionResult
from app.asr.faster_whisper_backend import FasterWhisperBackend
from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode, classify_exception

logger = logging.getLogger("voicelab.asr.manager")

# Approximate download sizes (CTranslate2 float16 weights), shown to the user before downloading.
MODEL_SIZES_MB = {
    "tiny": 75, "base": 145, "small": 485, "medium": 1530, "large-v1": 3090, "large-v2": 3090,
    "large-v3": 3090, "large-v3-turbo": 1620, "turbo": 1620, "distil-large-v3": 1510,
}

DownloadState = Literal["idle", "downloading", "failed"]


class ASRStatus(BaseModel):
    engine: str = "faster-whisper"
    model: str
    package_available: bool
    installed: bool
    loaded: bool
    device: str | None
    compute_type: str | None
    download_state: DownloadState
    download_error: str | None
    approx_size_mb: int | None


class ASRManager:
    def __init__(self, backend: ASRBackend) -> None:
        self.backend = backend
        self._lock = threading.Lock()
        self._download_state: DownloadState = "idle"
        self._download_error: str | None = None
        self._download_thread: threading.Thread | None = None
        self.last_used: float | None = None

    def status(self) -> ASRStatus:
        available = self.backend.package_available()
        return ASRStatus(
            model=self.backend.model_name,
            package_available=available,
            installed=available and self._download_state != "downloading" and self.backend.model_installed(),
            loaded=self.backend.loaded,
            device=self.backend.device,
            compute_type=self.backend.compute_type,
            download_state=self._download_state,
            download_error=self._download_error,
            approx_size_mb=MODEL_SIZES_MB.get(self.backend.model_name),
        )

    def start_download(self) -> ASRStatus:
        if not self.backend.package_available():
            raise AppError(ErrorCode.ASR_NOT_AVAILABLE, status_code=503)
        if self._download_state != "downloading" and not self.backend.model_installed():
            self._download_state, self._download_error = "downloading", None
            self._download_thread = threading.Thread(target=self._download, name="asr-download", daemon=True)
            self._download_thread.start()
        return self.status()

    def _download(self) -> None:
        try:
            self.backend.download_model()
            self._download_state = "idle"
            logger.info("asr_model_downloaded", extra={"model": self.backend.model_name})
        except Exception:
            logger.exception("asr_model_download_failed", extra={"model": self.backend.model_name})
            self._download_state = "failed"
            self._download_error = "No se pudo descargar el modelo. Comprueba la conexión y el espacio en disco."

    def transcribe(self, audio_16k: np.ndarray, language: str | None,
                   initial_prompt: str | None = None) -> TranscriptionResult:
        if not self.backend.package_available():
            raise AppError(ErrorCode.ASR_NOT_AVAILABLE, status_code=503)
        if self._download_state == "downloading" or not self.backend.model_installed():
            raise AppError(ErrorCode.ASR_MODEL_NOT_INSTALLED, status_code=409,
                           details={"modelo": self.backend.model_name,
                                    "descargando": self._download_state == "downloading"})
        with self._lock:  # one inference at a time on the GPU
            try:
                return self.backend.transcribe(audio_16k, language, initial_prompt)
            except AppError:
                raise
            except Exception as exc:
                code = classify_exception(exc)
                logger.exception("asr_transcribe_failed", extra={"code": code.value})
                if code is ErrorCode.GPU_MEMORY_ERROR:
                    raise AppError(code, status_code=507) from exc
                raise AppError(ErrorCode.ASR_ERROR, status_code=500) from exc
            finally:
                self.last_used = time.time()

    def unload(self) -> None:
        with self._lock:
            self.backend.unload()

    def unload_if_idle(self, max_idle_s: float) -> bool:
        """Unload when loaded and unused for `max_idle_s`; never waits for a running transcription."""
        if not self.backend.loaded or self.last_used is None or time.time() - self.last_used < max_idle_s:
            return False
        if not self.try_unload():
            return False
        logger.info("asr_unloaded_idle", extra={"idle_s": round(time.time() - self.last_used)})
        return True

    def try_unload(self) -> bool:
        """Unload unless a transcription is running right now."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self.backend.unload()
        finally:
            self._lock.release()
        return True


@lru_cache
def get_asr_manager() -> ASRManager:
    settings = get_settings()
    backend = FasterWhisperBackend(settings.asr_model, device=settings.device if settings.device in
                                   ("auto", "cuda", "cpu") else "auto",
                                   compute_type=settings.asr_compute_type)
    return ASRManager(backend)
