"""/api/training — fine-tune a voice profile into a new Qwen3-TTS variant."""

import re

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.core.config import Settings, get_settings
from app.core.database import get_session
from app.core.errors import AppError, ErrorCode
from app.engines.manager import ModelManager, get_model_manager
from app.schemas.training import TrainingCreate, TrainingDatasetRead, TrainingRunRead
from app.services.generation_service import get_job_queue
from app.services.training_service import TrainingService
from app.training.evaluation import samples_dir
from app.workers.asyncio_queue import AsyncioJobQueue

router = APIRouter(prefix="/training", tags=["Entrenamiento"])
_SAMPLE = re.compile(r"^\d{1,2}_(real|trained|normal)\.wav$")


def get_training_service(session: Session = Depends(get_session), settings: Settings = Depends(get_settings),
                         models: ModelManager = Depends(get_model_manager),
                         queue: AsyncioJobQueue = Depends(get_job_queue)) -> TrainingService:
    return TrainingService(session, settings, models, queue)


@router.get("", response_model=list[TrainingRunRead], summary="Entrenamientos (de una voz o todos)")
def list_runs(profile_id: str | None = Query(default=None),
              service: TrainingService = Depends(get_training_service)) -> list[TrainingRunRead]:
    return service.list(profile_id)


@router.get("/dataset/{profile_id}", response_model=TrainingDatasetRead,
            summary="Audio de una voz disponible para entrenar")
def dataset(profile_id: str, service: TrainingService = Depends(get_training_service)) -> TrainingDatasetRead:
    return service.dataset(profile_id)


@router.post("", response_model=TrainingRunRead, status_code=202, summary="Entrenar una voz")
async def start(body: TrainingCreate, service: TrainingService = Depends(get_training_service)) -> TrainingRunRead:
    return await service.start(body)


@router.get("/{run_id}", response_model=TrainingRunRead, summary="Estado de un entrenamiento")
def get_run(run_id: str, service: TrainingService = Depends(get_training_service)) -> TrainingRunRead:
    return service.to_read(service.get(run_id))


@router.post("/{run_id}/cancel", response_model=TrainingRunRead, summary="Cancelar un entrenamiento")
async def cancel(run_id: str, service: TrainingService = Depends(get_training_service)) -> TrainingRunRead:
    return await service.cancel(run_id)


@router.delete("/{run_id}", status_code=204, summary="Borrar un entrenamiento y su modelo")
def delete(run_id: str, service: TrainingService = Depends(get_training_service)) -> Response:
    service.delete(run_id)
    return Response(status_code=204)


@router.get("/{run_id}/samples/{name}", summary="Muestra de la comparación (real, entrenada o normal)")
def sample(run_id: str, name: str, service: TrainingService = Depends(get_training_service)) -> FileResponse:
    service.get(run_id)
    path = samples_dir(run_id) / name
    if not _SAMPLE.match(name) or not path.is_file():
        raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="Esa muestra no existe.")
    return FileResponse(path, media_type="audio/wav")
