"""Memory status, manual release and automatic idle unloading of the TTS and ASR models."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.asr.manager import ASRManager
from app.core.config import Settings
from app.core.errors import AppError
from app.engines.manager import ModelManager, _empty_cuda_cache
from app.workers.asyncio_queue import AsyncioJobQueue

logger = logging.getLogger("voicelab.resources")

IDLE_CHECK_INTERVAL_S = 30


class GPUMemory(BaseModel):
    name: str
    total_mb: int
    free_mb: int
    used_mb: int = Field(description="Toda la memoria ocupada de la GPU (incluye otros programas)")
    torch_allocated_mb: int = Field(description="Memoria en uso por los tensores de VoiceLab")
    torch_reserved_mb: int = Field(description="Reservada por PyTorch (incluye bloques en caché reutilizables)")


class LoadedTTS(BaseModel):
    engine: str
    variant: str
    device: str
    loaded_at: datetime
    last_used_at: datetime | None
    in_use: bool
    idle_unload_in_s: int | None


class ASRMemory(BaseModel):
    model: str
    loaded: bool
    device: str | None
    last_used_at: datetime | None
    idle_unload_in_s: int | None


class MemoryStatus(BaseModel):
    gpu: GPUMemory | None
    process_ram_mb: int | None
    system_ram_total_mb: int | None
    system_ram_available_mb: int | None
    tts: LoadedTTS | None
    asr: ASRMemory
    idle_unload_minutes: dict[str, int]
    queue_active: int
    queue_waiting: int


class ReleaseRequest(BaseModel):
    tts: bool = True
    asr: bool = True


class ReleaseResult(BaseModel):
    released: list[str]
    skipped: dict[str, str]
    status: MemoryStatus


def _dt(ts: float | None) -> datetime | None:
    return datetime.fromtimestamp(ts, UTC) if ts else None


def _remaining(last: float | None, minutes: int) -> int | None:
    if not last or minutes <= 0:
        return None
    return max(0, int(last + minutes * 60 - time.time()))


class ResourcesService:
    def __init__(self, settings: Settings, models: ModelManager, asr: ASRManager, queue: AsyncioJobQueue) -> None:
        self.settings, self.models, self.asr, self.queue = settings, models, asr, queue

    def status(self) -> MemoryStatus:
        gpu = None
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info(0)
                    gpu = GPUMemory(name=torch.cuda.get_device_name(0), total_mb=total // 2**20,
                                    free_mb=free // 2**20, used_mb=(total - free) // 2**20,
                                    torch_allocated_mb=torch.cuda.memory_allocated(0) // 2**20,
                                    torch_reserved_mb=torch.cuda.memory_reserved(0) // 2**20)
            except Exception:  # broken CUDA must not break the status page
                logger.warning("gpu_memory_query_failed", exc_info=True)
        process_ram = total_ram = available_ram = None
        try:
            import psutil

            process_ram = psutil.Process(os.getpid()).memory_info().rss // 2**20
            vm = psutil.virtual_memory()
            total_ram, available_ram = vm.total // 2**20, vm.available // 2**20
        except Exception:
            pass
        loaded = self.models.loaded
        tts_minutes, asr_minutes = self.settings.model_idle_unload_minutes, self.settings.asr_idle_unload_minutes
        active, waiting = self.queue.counts()
        return MemoryStatus(
            gpu=gpu, process_ram_mb=process_ram, system_ram_total_mb=total_ram, system_ram_available_mb=available_ram,
            tts=LoadedTTS(engine=loaded.engine, variant=loaded.variant, device=loaded.device,
                          loaded_at=_dt(loaded.loaded_at), last_used_at=_dt(loaded.last_used), in_use=loaded.in_use,
                          idle_unload_in_s=None if loaded.in_use else _remaining(loaded.last_used or loaded.loaded_at,
                                                                                 tts_minutes))
            if loaded else None,
            asr=ASRMemory(model=self.asr.backend.model_name, loaded=self.asr.backend.loaded,
                          device=self.asr.backend.device, last_used_at=_dt(self.asr.last_used),
                          idle_unload_in_s=_remaining(self.asr.last_used, asr_minutes)
                          if self.asr.backend.loaded else None),
            idle_unload_minutes={"tts": tts_minutes, "asr": asr_minutes},
            queue_active=active, queue_waiting=waiting,
        )

    def release(self, body: ReleaseRequest) -> ReleaseResult:
        released: list[str] = []
        skipped: dict[str, str] = {}
        if body.tts and self.models.loaded is not None:
            try:
                self.models.unload()
                released.append("tts")
            except AppError as err:
                skipped["tts"] = err.message
        if body.asr and self.asr.backend.loaded:
            if self.asr.try_unload():
                released.append("asr")
            else:
                skipped["asr"] = "La transcripción está en uso. Inténtalo cuando termine."
        _empty_cuda_cache()
        return ReleaseResult(released=released, skipped=skipped, status=self.status())

    def unload_idle(self) -> list[str]:
        """One pass of the idle policy. Skipped entirely while jobs are running or waiting."""
        active, waiting = self.queue.counts()
        if active or waiting:
            return []
        done = []
        tts_s, asr_s = self.settings.model_idle_unload_minutes * 60, self.settings.asr_idle_unload_minutes * 60
        if tts_s > 0 and self.models.unload_if_idle(tts_s):
            done.append("tts")
        if asr_s > 0 and self.asr.unload_if_idle(asr_s):
            done.append("asr")
        if done:
            _empty_cuda_cache()
        return done


async def idle_monitor(service: ResourcesService, interval_s: float = IDLE_CHECK_INTERVAL_S) -> None:
    """Background task started with the app: frees models that nobody has used for a while."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            await asyncio.to_thread(service.unload_idle)
        except Exception:  # never let the monitor die
            logger.exception("idle_monitor_failed")
