"""ModelManager: lazy loading, one TTS engine in memory, VRAM checks and user-triggered weight downloads."""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal, TypeVar

from pydantic import BaseModel

from app.asr.manager import get_asr_manager
from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode, classify_exception
from app.core.gpu import GPUManager
from app.engines.base import TTSBackend
from app.engines.registry import EngineRegistry, get_engine_registry

logger = logging.getLogger("voicelab.models")

DownloadState = Literal["idle", "downloading", "failed"]
T = TypeVar("T")


class EngineRuntimeStatus(BaseModel):
    engine: str
    variant: str
    package_installed: bool
    missing_packages: list[str]
    weights_installed: bool
    download_state: DownloadState
    download_error: str | None
    loaded: bool
    device: str | None


class LoadedModel(BaseModel):
    engine: str
    variant: str
    device: str
    loaded_at: float
    last_used: float | None = None
    in_use: bool = False


class ModelManager:
    def __init__(self, registry: EngineRegistry, gpu: GPUManager, device_preference: str = "auto") -> None:
        self.registry = registry
        self.gpu = gpu
        self.device_preference = device_preference
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None
        self._downloads: dict[tuple[str, str], DownloadState] = {}
        self._download_errors: dict[tuple[str, str], str] = {}
        self._in_use = 0

    # ----- status -----
    @property
    def loaded(self) -> LoadedModel | None:
        if self._loaded is None:
            return None
        return self._loaded.model_copy(update={"in_use": self._in_use > 0})

    def status(self, engine_id: str, variant: str | None) -> EngineRuntimeStatus:
        engine = self.registry.get(engine_id)
        v = engine.variant(variant).id
        key = (engine.id, v)
        installed = engine.is_installed()
        downloading = self._downloads.get(key) == "downloading"
        return EngineRuntimeStatus(
            engine=engine.id, variant=v, package_installed=installed, missing_packages=engine.missing_packages(),
            weights_installed=installed and not downloading and engine.weights_installed(v),
            download_state=self._downloads.get(key, "idle"), download_error=self._download_errors.get(key),
            loaded=self._loaded is not None and (self._loaded.engine, self._loaded.variant) == key,
            device=self._loaded.device if self._loaded and (self._loaded.engine, self._loaded.variant) == key else None,
        )

    # ----- downloads -----
    def start_download(self, engine_id: str, variant: str | None) -> EngineRuntimeStatus:
        engine = self.registry.get(engine_id)
        v = engine.variant(variant).id
        if not engine.is_installed():
            raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                           message=f"Instala primero el paquete de {engine.display_name}.",
                           details={"paquetes": engine.missing_packages()})
        key = (engine.id, v)
        if self._downloads.get(key) != "downloading" and not engine.weights_installed(v):
            self._downloads[key] = "downloading"
            self._download_errors.pop(key, None)
            threading.Thread(target=self._download, args=(engine, v), name=f"download-{engine.id}", daemon=True).start()
        return self.status(engine.id, v)

    def _download(self, engine: TTSBackend, variant: str) -> None:
        key = (engine.id, variant)
        try:
            engine.download_weights(variant)
            self._downloads[key] = "idle"
            logger.info("weights_downloaded", extra={"engine": engine.id, "variant": variant})
        except Exception:
            logger.exception("weights_download_failed", extra={"engine": engine.id, "variant": variant})
            self._downloads[key] = "failed"
            self._download_errors[key] = "No se pudo descargar el modelo. Comprueba la conexión y el espacio en disco."

    # ----- loading -----
    def resolve_device(self) -> str:
        pref = self.device_preference
        if pref == "cpu":
            return "cpu"
        if importlib.util.find_spec("torch") is not None:
            import torch

            if torch.cuda.is_available() and pref in ("auto", "cuda", "rocm"):
                return "cuda"
            if pref in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def ensure_loaded(self, engine_id: str, variant: str) -> TTSBackend:
        """Load `engine:variant`, unloading any other TTS engine first. Thread-safe; called from the worker."""
        engine = self.registry.get(engine_id)
        v = engine.variant(variant).id
        with self._lock:
            if self._loaded and (self._loaded.engine, self._loaded.variant) == (engine.id, v):
                self._loaded.last_used = time.time()
                return engine
            if self._in_use:
                raise AppError(ErrorCode.MODEL_BUSY, status_code=409)
            if not engine.is_installed():
                raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                               details={"paquetes": engine.missing_packages()})
            if not engine.weights_installed(v):
                raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                               message="Los pesos del modelo no están descargados. "
                                       "Descárgalos desde el panel del motor.",
                               details={"motor": engine.id, "variante": v})
            self._unload_current()
            device = self.resolve_device()
            if device == "cuda":
                self._ensure_vram(engine, v)
            started = time.perf_counter()
            try:
                engine.load(v, device, "auto")
            except AppError:
                raise
            except Exception as exc:
                code = classify_exception(exc)
                logger.exception("engine_load_failed", extra={"engine": engine.id, "variant": v})
                engine.unload()
                if code is ErrorCode.GPU_MEMORY_ERROR:
                    raise AppError(code, status_code=507) from exc
                raise AppError(ErrorCode.MODEL_LOAD_ERROR, status_code=500) from exc
            self._loaded = LoadedModel(engine=engine.id, variant=v, device=device, loaded_at=time.time(),
                                       last_used=time.time())
            logger.info("engine_loaded", extra={"engine": engine.id, "variant": v, "device": device,
                                                "seconds": round(time.perf_counter() - started, 2)})
            return engine

    def _ensure_vram(self, engine: TTSBackend, variant: str) -> None:
        required = engine.requirements(variant).vram_mb
        if not required:
            return
        try:
            self.gpu.check_can_load(required)
        except AppError:
            asr = get_asr_manager()
            if not asr.backend.loaded:
                raise
            logger.info("unloading_asr_for_vram", extra={"engine": engine.id})
            asr.unload()
            self.gpu.check_can_load(required)

    def _unload_current(self) -> None:
        if self._loaded is None:
            return
        current = self.registry.get(self._loaded.engine)
        try:
            current.unload()
        finally:
            logger.info("engine_unloaded", extra={"engine": self._loaded.engine, "variant": self._loaded.variant})
            self._loaded = None
            _empty_cuda_cache()

    def unload(self, force: bool = False) -> None:
        """Free the TTS model. Refuses while a generation is using it (unless `force`, used at shutdown)."""
        with self._lock:
            if self._in_use and not force:
                raise AppError(ErrorCode.MODEL_BUSY, status_code=409)
            self._unload_current()

    @contextmanager
    def use(self, engine_id: str, variant: str) -> Iterator[TTSBackend]:
        """Load (if needed) and hold the engine for the duration of a job, so nobody unloads it mid-generation."""
        with self._lock:
            engine = self.ensure_loaded(engine_id, variant)
            self._in_use += 1
        try:
            yield engine
        finally:
            with self._lock:
                self._in_use -= 1
                if self._loaded is not None:
                    self._loaded.last_used = time.time()

    def unload_if_idle(self, max_idle_s: float) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            loaded = self._loaded
            if loaded is None or self._in_use or time.time() - (loaded.last_used or loaded.loaded_at) < max_idle_s:
                return False
            logger.info("engine_unloaded_idle", extra={"engine": loaded.engine, "variant": loaded.variant})
            self._unload_current()
            return True
        finally:
            self._lock.release()

    def run_with_memory_retry(self, fn: Callable[[], T]) -> T:
        """Run a GPU call; on out-of-memory free the ASR model and cached blocks, then retry once."""
        try:
            return fn()
        except AppError:
            raise
        except Exception as exc:
            if classify_exception(exc) is not ErrorCode.GPU_MEMORY_ERROR:
                raise
            logger.warning("gpu_oom_retry", extra={"engine": self._loaded.engine if self._loaded else None})
            asr = get_asr_manager()
            if asr.backend.loaded:
                asr.unload()
            _empty_cuda_cache()
        try:
            return fn()
        except AppError:
            raise
        except Exception as exc:
            if classify_exception(exc) is ErrorCode.GPU_MEMORY_ERROR:
                _empty_cuda_cache()
                raise AppError(ErrorCode.GPU_MEMORY_ERROR, status_code=507) from exc
            raise


def _empty_cuda_cache() -> None:
    import gc

    gc.collect()
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@lru_cache
def get_model_manager() -> ModelManager:
    settings = get_settings()
    return ModelManager(get_engine_registry(), GPUManager(settings.device), settings.device)
