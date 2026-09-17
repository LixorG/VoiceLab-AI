"""/api/projects — scripts split into segments, generated with shared settings and exported as one audio."""

from urllib.parse import quote

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse
from sqlmodel import Session
from starlette.concurrency import run_in_threadpool

from app.api.routes.generation import get_generation_service
from app.core.database import get_session
from app.schemas.project import (
    ProjectCreate,
    ProjectGenerate,
    ProjectRead,
    ProjectSummary,
    ProjectUpdate,
    ScriptImport,
    SegmentOrder,
    SegmentsAdd,
    SegmentUpdate,
)
from app.services.generation_service import GenerationService
from app.services.project_service import ProjectService

router = APIRouter(prefix="/projects", tags=["Proyectos"])


def get_project_service(session: Session = Depends(get_session),
                        generations: GenerationService = Depends(get_generation_service)) -> ProjectService:
    return ProjectService(session, generations)


@router.get("", response_model=list[ProjectSummary], summary="Listar proyectos")
def list_projects(service: ProjectService = Depends(get_project_service)) -> list[ProjectSummary]:
    return service.list()


@router.post("", response_model=ProjectRead, status_code=201, summary="Crear proyecto")
async def create(body: ProjectCreate, service: ProjectService = Depends(get_project_service)) -> ProjectRead:
    return await service.to_read(service.create(body))


@router.get("/{project_id}", response_model=ProjectRead, summary="Detalle con segmentos y su estado")
async def get_project(project_id: str, service: ProjectService = Depends(get_project_service)) -> ProjectRead:
    return await service.to_read(service.get(project_id))


@router.patch("/{project_id}", response_model=ProjectRead, summary="Editar nombre, descripción o ajustes")
async def update(project_id: str, body: ProjectUpdate,
                 service: ProjectService = Depends(get_project_service)) -> ProjectRead:
    return await service.to_read(service.update(project_id, body))


@router.delete("/{project_id}", status_code=204, summary="Eliminar el proyecto, sus segmentos y audios")
async def delete(project_id: str, service: ProjectService = Depends(get_project_service)) -> Response:
    await service.delete(project_id)
    return Response(status_code=204)


@router.post("/{project_id}/segments", response_model=ProjectRead, summary="Añadir segmentos")
async def add_segments(project_id: str, body: SegmentsAdd,
                       service: ProjectService = Depends(get_project_service)) -> ProjectRead:
    return await service.to_read(service.add_segments(project_id, body))


@router.post("/{project_id}/import", response_model=ProjectRead, summary="Importar un guion (por párrafos o frases)")
async def import_script(project_id: str, body: ScriptImport,
                        service: ProjectService = Depends(get_project_service)) -> ProjectRead:
    return await service.to_read(service.import_script(project_id, body.text, body.split))


@router.put("/{project_id}/segments/order", response_model=ProjectRead, summary="Reordenar segmentos")
async def reorder(project_id: str, body: SegmentOrder,
                  service: ProjectService = Depends(get_project_service)) -> ProjectRead:
    return await service.to_read(service.reorder(project_id, body.ids))


@router.patch("/{project_id}/segments/{segment_id}", response_model=ProjectRead, summary="Editar un segmento")
async def update_segment(project_id: str, segment_id: str, body: SegmentUpdate,
                         service: ProjectService = Depends(get_project_service)) -> ProjectRead:
    return await service.to_read(service.update_segment(project_id, segment_id, body))


@router.delete("/{project_id}/segments/{segment_id}", response_model=ProjectRead, summary="Eliminar un segmento")
async def delete_segment(project_id: str, segment_id: str,
                         service: ProjectService = Depends(get_project_service)) -> ProjectRead:
    return await service.to_read(await service.delete_segment(project_id, segment_id))


@router.post("/{project_id}/generate", response_model=ProjectRead, status_code=202,
             summary="Generar los segmentos pendientes (o los indicados)")
async def generate(project_id: str, body: ProjectGenerate,
                   service: ProjectService = Depends(get_project_service)) -> ProjectRead:
    return await service.to_read(await service.generate(project_id, body))


@router.get("/{project_id}/export", summary="Exportar el audio completo (WAV) o un ZIP con segmentos y manifiesto")
async def export(project_id: str, format: str = Query(default="wav", pattern="^(wav|zip)$"),  # noqa: A002
                 allow_partial: bool = False, download: bool = True,
                 service: ProjectService = Depends(get_project_service)) -> Response:
    disposition = "attachment" if download else "inline"
    if format == "zip":
        data = await run_in_threadpool(service.export_zip, project_id, allow_partial)
        name = service.export_filename(project_id, "zip")
        ascii_name = name.encode("ascii", "replace").decode().replace("?", "_")
        header = f"{disposition}; filename=\"{ascii_name}\"; filename*=utf-8''{quote(name)}"
        return Response(content=data, media_type="application/zip", headers={"Content-Disposition": header})
    path, _ = await run_in_threadpool(service.render, project_id, allow_partial)
    return FileResponse(path, media_type="audio/wav", filename=service.export_filename(project_id, "wav"),
                        content_disposition_type=disposition)
