"""/api/references — upload, analysis and segment selection of reference audio."""

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from sqlmodel import Session

from app.core.config import Settings, get_settings
from app.core.database import get_session
from app.core.errors import AppError, ErrorCode
from app.schemas.references import ReferenceRead, ReferenceUpdate, UploadResponse
from app.services.reference_service import ReferenceService

router = APIRouter(prefix="/references", tags=["Referencias"])

MULTIPART_OVERHEAD = 1024 * 1024


def get_service(session: Session = Depends(get_session),
                settings: Settings = Depends(get_settings)) -> ReferenceService:
    return ReferenceService(session, settings)


@router.post("", response_model=UploadResponse, status_code=201, summary="Subir audio de referencia")
async def upload_reference(
    request: Request,
    file: UploadFile = File(..., description="WAV, MP3, FLAC, M4A, OGG, OPUS o AIFF"),
    profile_id: str | None = Form(default=None),
    service: ReferenceService = Depends(get_service),
) -> UploadResponse:
    declared = request.headers.get("content-length")
    limit = service.settings.max_upload_bytes + MULTIPART_OVERHEAD
    if declared and declared.isdigit() and int(declared) > limit:
        raise AppError(ErrorCode.FILE_TOO_LARGE, status_code=413,
                       details={"maximo_mb": service.settings.max_upload_mb})
    ref, duplicate = await service.upload(file, profile_id)
    return UploadResponse(reference=service.read(ref), duplicate=duplicate)


@router.get("", response_model=list[ReferenceRead], summary="Listar referencias")
def list_references(profile_id: str | None = None,
                    service: ReferenceService = Depends(get_service)) -> list[ReferenceRead]:
    refs = service.list(profile_id)
    best = service.recommended_id(refs)
    transcripts = service.transcripts_by_reference([r.id for r in refs])
    return [service.read(r, best, transcripts[r.id]) for r in refs]


@router.get("/{reference_id}", response_model=ReferenceRead, summary="Detalle de una referencia")
def get_reference(reference_id: str, service: ReferenceService = Depends(get_service)) -> ReferenceRead:
    return service.read(service.get(reference_id))


@router.patch("/{reference_id}", response_model=ReferenceRead,
              summary="Seleccionar segmento o etiquetar emoción")
def update_reference(reference_id: str, data: ReferenceUpdate,
                     service: ReferenceService = Depends(get_service)) -> ReferenceRead:
    return service.read(service.update(reference_id, data))


@router.post("/{reference_id}/reanalyze", response_model=ReferenceRead, summary="Volver a analizar")
async def reanalyze_reference(reference_id: str,
                              service: ReferenceService = Depends(get_service)) -> ReferenceRead:
    return service.read(await service.reanalyze(reference_id))


@router.delete("/{reference_id}", status_code=204, summary="Eliminar referencia")
def delete_reference(reference_id: str, service: ReferenceService = Depends(get_service)) -> Response:
    service.delete(reference_id)
    return Response(status_code=204)
