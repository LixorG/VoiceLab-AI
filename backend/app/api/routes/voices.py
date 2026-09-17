"""/api/voices — voice profiles (CRUD, recommended settings, export/import)."""

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from sqlmodel import Session

from app.core.config import Settings, get_settings
from app.core.database import get_session
from app.core.errors import AppError, ErrorCode
from app.engines.registry import EngineRegistry, get_engine_registry
from app.schemas.profiles import (
    EmotionOption,
    ProfileCreate,
    ProfileRead,
    ProfileUpdate,
    RecommendedSettingUpdate,
)
from app.services.profile_service import ProfileService
from app.voice_profiles.emotions import EMOTIONS

router = APIRouter(prefix="/voices", tags=["Voces"])


def get_profile_service(session: Session = Depends(get_session), settings: Settings = Depends(get_settings),
                        registry: EngineRegistry = Depends(get_engine_registry)) -> ProfileService:
    return ProfileService(session, settings, registry)


@router.get("/emotions", response_model=list[EmotionOption], summary="Etiquetas de emoción para referencias")
def emotions() -> list[EmotionOption]:
    return [EmotionOption(id=k, label=v) for k, v in EMOTIONS.items()]


@router.get("", response_model=list[ProfileRead], summary="Perfiles de voz")
def list_profiles(service: ProfileService = Depends(get_profile_service)) -> list[ProfileRead]:
    return [service.to_read(p) for p in service.list()]


@router.post("", response_model=ProfileRead, status_code=201, summary="Crear perfil de voz")
def create_profile(body: ProfileCreate, service: ProfileService = Depends(get_profile_service)) -> ProfileRead:
    return service.to_read(service.create(body))


@router.post("/import", response_model=ProfileRead, status_code=201, summary="Importar perfil (.voiceprofile)")
async def import_profile(file: UploadFile = File(...), settings: Settings = Depends(get_settings),
                         service: ProfileService = Depends(get_profile_service)) -> ProfileRead:
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise AppError(ErrorCode.FILE_TOO_LARGE, status_code=413)
    return service.to_read(await service.import_zip(data))


@router.get("/{profile_id}", response_model=ProfileRead, summary="Detalle de un perfil")
def get_profile(profile_id: str, service: ProfileService = Depends(get_profile_service)) -> ProfileRead:
    return service.to_read(service.get(profile_id))


@router.patch("/{profile_id}", response_model=ProfileRead, summary="Editar perfil")
def update_profile(profile_id: str, body: ProfileUpdate,
                   service: ProfileService = Depends(get_profile_service)) -> ProfileRead:
    return service.to_read(service.update(profile_id, body))


@router.delete("/{profile_id}", status_code=204, summary="Eliminar perfil")
def delete_profile(profile_id: str, delete_references: bool = Query(default=False),
                   service: ProfileService = Depends(get_profile_service)) -> Response:
    service.delete(profile_id, delete_references)
    return Response(status_code=204)


@router.put("/{profile_id}/settings/{engine_id}", response_model=ProfileRead,
            summary="Guardar configuración recomendada para un motor")
def set_recommended(profile_id: str, engine_id: str, body: RecommendedSettingUpdate,
                    service: ProfileService = Depends(get_profile_service)) -> ProfileRead:
    return service.to_read(service.set_recommended(profile_id, engine_id, body))


@router.delete("/{profile_id}/settings/{engine_id}", response_model=ProfileRead,
               summary="Quitar configuración recomendada de un motor")
def remove_recommended(profile_id: str, engine_id: str,
                       service: ProfileService = Depends(get_profile_service)) -> ProfileRead:
    return service.to_read(service.remove_recommended(profile_id, engine_id))


@router.get("/{profile_id}/export", summary="Exportar perfil (.voiceprofile = ZIP)")
def export_profile(profile_id: str, service: ProfileService = Depends(get_profile_service)) -> Response:
    data, filename = service.export_zip(profile_id)
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
