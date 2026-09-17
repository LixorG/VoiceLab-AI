from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.asr.manager import ASRManager, get_asr_manager
from app.core.config import Settings, get_settings
from app.core.database import get_session
from app.core.environment import EnvironmentChecker, EnvironmentReport
from app.core.gpu import GPUInfo, GPUManager
from app.core.licenses import LICENSES, LicenseEntry
from app.engines.manager import ModelManager, get_model_manager
from app.services.generation_service import get_job_queue
from app.services.resources_service import MemoryStatus, ReleaseRequest, ReleaseResult, ResourcesService
from app.services.storage_service import CleanupReport, CleanupRequest, StorageService, StorageUsage
from app.workers.asyncio_queue import AsyncioJobQueue

router = APIRouter(prefix="/system", tags=["Sistema"])


class HealthResponse(BaseModel):
    status: str
    version: str


class SystemInfo(BaseModel):
    version: str
    app_env: str
    device: str
    default_model: str
    max_upload_mb: int
    max_audio_seconds: int
    data_dir: str
    model_dir: str


def get_gpu_manager(settings: Settings = Depends(get_settings)) -> GPUManager:
    return GPUManager(preferred=settings.device)


@router.get("/health", response_model=HealthResponse, summary="Estado del servidor")
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get("/info", response_model=SystemInfo, summary="Configuración activa")
def info(settings: Settings = Depends(get_settings)) -> SystemInfo:
    return SystemInfo(
        version=__version__, app_env=settings.app_env, device=settings.device,
        default_model=settings.default_model, max_upload_mb=settings.max_upload_mb,
        max_audio_seconds=settings.max_audio_seconds, data_dir=str(settings.data_dir),
        model_dir=str(settings.model_dir),
    )


@router.get("/gpu", response_model=GPUInfo, summary="Información de GPU")
def gpu(manager: GPUManager = Depends(get_gpu_manager)) -> GPUInfo:
    return manager.detect()


@router.get("/environment", response_model=EnvironmentReport, summary="Comprobación del entorno")
def environment(manager: GPUManager = Depends(get_gpu_manager)) -> EnvironmentReport:
    return EnvironmentChecker(manager).run()


@router.get("/licenses", response_model=list[LicenseEntry], summary="Licencias de modelos y dependencias")
def licenses() -> list[LicenseEntry]:
    return LICENSES


def get_resources_service(settings: Settings = Depends(get_settings), models: ModelManager = Depends(get_model_manager),
                          asr: ASRManager = Depends(get_asr_manager),
                          queue: AsyncioJobQueue = Depends(get_job_queue)) -> ResourcesService:
    return ResourcesService(settings, models, asr, queue)


def get_storage_service(session: Session = Depends(get_session), settings: Settings = Depends(get_settings),
                        queue: AsyncioJobQueue = Depends(get_job_queue)) -> StorageService:
    return StorageService(session, settings, queue)


@router.get("/memory", response_model=MemoryStatus, summary="Memoria de GPU/RAM y modelos cargados")
def memory(service: ResourcesService = Depends(get_resources_service)) -> MemoryStatus:
    return service.status()


@router.post("/memory/release", response_model=ReleaseResult,
             summary="Liberar de la memoria el modelo de voz y/o el de transcripción (si no están en uso)")
async def release_memory(body: ReleaseRequest,
                         service: ResourcesService = Depends(get_resources_service)) -> ReleaseResult:
    return await run_in_threadpool(service.release, body)


@router.get("/storage", response_model=StorageUsage, summary="Espacio en disco por categoría")
async def storage(service: StorageService = Depends(get_storage_service)) -> StorageUsage:
    return await run_in_threadpool(service.usage)


@router.post("/storage/cleanup", response_model=CleanupReport,
             summary="Borrar cachés regenerables, temporales y archivos huérfanos")
async def storage_cleanup(body: CleanupRequest,
                          service: StorageService = Depends(get_storage_service)) -> CleanupReport:
    return await run_in_threadpool(service.cleanup, body)
