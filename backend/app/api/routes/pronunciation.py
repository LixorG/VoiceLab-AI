"""/api/pronunciation — pronunciation dictionary and normalisation preview."""

from fastapi import APIRouter, Depends, Query, Response
from sqlmodel import Session

from app.core.database import get_session
from app.models.entities import PronunciationEntry
from app.schemas.pronunciation import NormalizePreview, NormalizePreviewRead, PronunciationInput, PronunciationRead
from app.services.pronunciation_service import PronunciationService

router = APIRouter(prefix="/pronunciation", tags=["Pronunciación"])


def get_service(session: Session = Depends(get_session)) -> PronunciationService:
    return PronunciationService(session)


@router.get("", response_model=list[PronunciationRead], summary="Listar el diccionario de pronunciación")
def list_entries(profile_id: str | None = Query(default=None), include_global: bool = True,
                 service: PronunciationService = Depends(get_service)) -> list[PronunciationEntry]:
    return service.list(profile_id, include_global)


@router.post("", response_model=PronunciationRead, status_code=201, summary="Añadir entrada")
def create(body: PronunciationInput, service: PronunciationService = Depends(get_service)) -> PronunciationEntry:
    return service.create(body)


@router.post("/preview", response_model=NormalizePreviewRead, summary="Probar la normalización de un texto")
def preview(body: NormalizePreview, service: PronunciationService = Depends(get_service)) -> NormalizePreviewRead:
    return service.preview(body)


@router.put("/{entry_id}", response_model=PronunciationRead, summary="Editar entrada")
def update(entry_id: str, body: PronunciationInput,
           service: PronunciationService = Depends(get_service)) -> PronunciationEntry:
    return service.update(entry_id, body)


@router.delete("/{entry_id}", status_code=204, summary="Eliminar entrada")
def delete(entry_id: str, service: PronunciationService = Depends(get_service)) -> Response:
    service.delete(entry_id)
    return Response(status_code=204)
