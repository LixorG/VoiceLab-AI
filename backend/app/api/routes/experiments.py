"""/api/experiments — same text and voice generated with several engines, compared side by side."""

from fastapi import APIRouter, Depends, Response
from sqlmodel import Session

from app.api.routes.generation import get_generation_service
from app.asr.manager import ASRManager, get_asr_manager
from app.core.database import get_session
from app.schemas.experiment import (
    ExperimentArmsAdd,
    ExperimentCreate,
    ExperimentRead,
    ExperimentSummary,
    ExperimentUpdate,
)
from app.services.evaluation_service import EvaluationService
from app.services.experiment_service import ExperimentService
from app.services.generation_service import GenerationService

router = APIRouter(prefix="/experiments", tags=["Experimentos"])


def get_experiment_service(session: Session = Depends(get_session),
                           generations: GenerationService = Depends(get_generation_service)) -> ExperimentService:
    return ExperimentService(session, generations)


@router.get("", response_model=list[ExperimentSummary], summary="Listar experimentos")
def list_experiments(service: ExperimentService = Depends(get_experiment_service)) -> list[ExperimentSummary]:
    return service.list()


@router.post("", response_model=ExperimentRead, status_code=202,
             summary="Crear un experimento y encolar una generación por motor")
async def create(body: ExperimentCreate,
                 service: ExperimentService = Depends(get_experiment_service)) -> ExperimentRead:
    return await service.to_read(await service.create(body))


@router.get("/{experiment_id}", response_model=ExperimentRead, summary="Detalle con sus generaciones")
async def get_experiment(experiment_id: str,
                         service: ExperimentService = Depends(get_experiment_service)) -> ExperimentRead:
    return await service.to_read(service.get(experiment_id))


@router.patch("/{experiment_id}", response_model=ExperimentRead, summary="Renombrar o añadir notas")
async def update(experiment_id: str, body: ExperimentUpdate,
                 service: ExperimentService = Depends(get_experiment_service)) -> ExperimentRead:
    return await service.to_read(service.update(experiment_id, body))


@router.post("/{experiment_id}/arms", response_model=ExperimentRead, status_code=202,
             summary="Añadir motores o configuraciones al experimento")
async def add_arms(experiment_id: str, body: ExperimentArmsAdd,
                   service: ExperimentService = Depends(get_experiment_service)) -> ExperimentRead:
    return await service.to_read(await service.add_arms(experiment_id, body.arms))


@router.post("/{experiment_id}/evaluate", response_model=ExperimentRead,
             summary="Evaluar automáticamente todas las generaciones terminadas")
async def evaluate_all(experiment_id: str, service: ExperimentService = Depends(get_experiment_service),
                       asr: ASRManager = Depends(get_asr_manager)) -> ExperimentRead:
    from starlette.concurrency import run_in_threadpool

    from app.models.enums import JobStatus

    evaluation = EvaluationService(service.session, service.generations, asr)
    for gen in service.generations_of(service.get(experiment_id).id):
        if gen.status == JobStatus.COMPLETED:
            await run_in_threadpool(evaluation.evaluate, gen.id)
    return await service.to_read(service.get(experiment_id))


@router.delete("/{experiment_id}", status_code=204, summary="Eliminar el experimento y sus audios")
async def delete(experiment_id: str, service: ExperimentService = Depends(get_experiment_service)) -> Response:
    await service.delete(experiment_id)
    return Response(status_code=204)
