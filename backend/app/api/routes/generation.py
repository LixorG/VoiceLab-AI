"""/api/generation — create generations (full or preview), list, audio, delete."""

from typing import Literal

from fastapi import APIRouter, Depends, Path, Query, Response
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.asr.manager import ASRManager, get_asr_manager
from app.audio.export import FORMATS, ExportFormat, export_audio
from app.core.config import Settings, get_settings
from app.core.database import get_session
from app.core.errors import AppError, ErrorCode
from app.engines.manager import ModelManager, get_model_manager
from app.evaluation.base import EvaluatorInfo
from app.evaluation.speaker import SpeakerModelStatus, get_speaker_encoder
from app.generation.streaming import chunk_path
from app.postprocess.config import PostProcessCapabilities, PostProcessConfig
from app.postprocess.processor import capabilities as postprocess_capabilities
from app.schemas.experiment import RatingUpdate
from app.schemas.export import ExportFormatInfo
from app.schemas.generation import (
    BatchAccepted,
    BatchCreate,
    GenerationAccepted,
    GenerationCreate,
    GenerationPlanRead,
    GenerationRead,
    SegmentRegenerate,
    VariationsCreate,
)
from app.schemas.library import FavoriteUpdate, LibraryItem, TagsUpdate
from app.services import mastering
from app.services.evaluation_service import EvaluationService
from app.services.generation_service import GenerationService, get_job_queue
from app.workers.asyncio_queue import AsyncioJobQueue

router = APIRouter(prefix="/generation", tags=["Generación"])


def get_generation_service(session: Session = Depends(get_session), settings: Settings = Depends(get_settings),
                           models: ModelManager = Depends(get_model_manager),
                           queue: AsyncioJobQueue = Depends(get_job_queue)) -> GenerationService:
    return GenerationService(session, settings, models, queue)


@router.post("", response_model=GenerationAccepted, status_code=202, summary="Generar voz (o vista previa)")
async def create(body: GenerationCreate,
                 service: GenerationService = Depends(get_generation_service)) -> GenerationAccepted:
    gen = await service.create(body)
    return GenerationAccepted(generation=await service.to_read(gen), job_id=gen.id,
                              queue_position=service.queue.queued_position(gen.id))


@router.post("/plan", response_model=GenerationPlanRead,
             summary="Previsualizar cómo se aplicarán marcas, emociones y referencias (sin generar)")
def plan(body: GenerationCreate, service: GenerationService = Depends(get_generation_service)) -> GenerationPlanRead:
    return service.plan(body)


@router.post("/variations", response_model=list[GenerationAccepted], status_code=202,
             summary="Generar variaciones con semillas distintas")
async def variations(body: VariationsCreate,
                     service: GenerationService = Depends(get_generation_service)) -> list[GenerationAccepted]:
    created = await service.create_variations(GenerationCreate(**body.model_dump(exclude={"count"})), body.count)
    return [GenerationAccepted(generation=await service.to_read(g), job_id=g.id,
                               queue_position=service.queue.queued_position(g.id)) for g in created]


@router.post("/batch", response_model=BatchAccepted, status_code=202,
             summary="Encolar un lote (varios textos, variantes y repeticiones)")
async def batch(body: BatchCreate, service: GenerationService = Depends(get_generation_service)) -> BatchAccepted:
    created = await service.create_batch(body)
    items = [GenerationAccepted(generation=await service.to_read(g), job_id=g.id,
                                queue_position=service.queue.queued_position(g.id)) for g in created]
    return BatchAccepted(items=items, total=len(items))


@router.get("/postprocess/capabilities", response_model=PostProcessCapabilities,
            summary="Procesadores de posprocesado disponibles en esta instalación")
def postprocess_caps(settings: Settings = Depends(get_settings)) -> PostProcessCapabilities:
    return postprocess_capabilities(mastering.ffmpeg_or_none(settings))


@router.get("/export/formats", response_model=list[ExportFormatInfo], summary="Formatos de descarga disponibles")
def export_formats(service: GenerationService = Depends(get_generation_service)) -> list[ExportFormatInfo]:
    has_ffmpeg = mastering.ffmpeg_or_none(service.settings) is not None
    return [ExportFormatInfo(id=key, label=spec.label, available=key == "wav" or has_ffmpeg,
                             reason=None if key == "wav" or has_ffmpeg else "Requiere FFmpeg.")
            for key, spec in FORMATS.items()]


@router.get("/evaluators", response_model=list[EvaluatorInfo],
            summary="Evaluaciones automáticas disponibles (estimaciones)")
def evaluators(service: GenerationService = Depends(get_generation_service),
               asr: ASRManager = Depends(get_asr_manager)) -> list[EvaluatorInfo]:
    return EvaluationService(service.session, service, asr).available()


@router.get("/evaluators/speaker-model", response_model=SpeakerModelStatus,
            summary="Estado del modelo de similitud de voz")
def speaker_model_status() -> SpeakerModelStatus:
    return get_speaker_encoder().status()


@router.post("/evaluators/speaker-model/download", response_model=SpeakerModelStatus, status_code=202,
             summary="Descargar el modelo de similitud de voz (~405 MB)")
def speaker_model_download() -> SpeakerModelStatus:
    return get_speaker_encoder().start_download()


@router.get("", response_model=list[GenerationRead], summary="Generaciones recientes")
async def list_generations(limit: int = Query(default=30, ge=1, le=200),
                           service: GenerationService = Depends(get_generation_service)) -> list[GenerationRead]:
    return [await service.to_read(g) for g in service.list(limit)]


@router.get("/{generation_id}", response_model=GenerationRead, summary="Detalle de una generación")
async def get_generation(generation_id: str,
                         service: GenerationService = Depends(get_generation_service)) -> GenerationRead:
    return await service.to_read(service.get(generation_id))


@router.get("/{generation_id}/audio", summary="Audio generado (final o salida original del modelo)")
def audio(generation_id: str, download: bool = False, version: Literal["final", "raw"] = "final",
          format: ExportFormat = "wav",  # noqa: A002
          service: GenerationService = Depends(get_generation_service)) -> FileResponse:
    gen = service.get(generation_id)
    source = service.audio_path(gen, raw=version == "raw")
    path = export_audio(source, format, service.settings.data_dir / "cache" / "exports",
                        mastering.ffmpeg_or_none(service.settings))
    spec = FORMATS[format]
    suffix = "_original" if version == "raw" and gen.postprocess else ""
    filename = f"voicelab_{gen.engine}_{gen.created_at:%Y%m%d_%H%M%S}{suffix}.{spec.extension}"
    return FileResponse(path, media_type=spec.media_type, filename=filename,
                        content_disposition_type="attachment" if download else "inline")


@router.get("/{generation_id}/stream/{index}", summary="Trozo de audio de la vista previa en directo")
def stream_chunk(generation_id: str, index: int = Path(ge=0, le=9999),
                 service: GenerationService = Depends(get_generation_service)) -> FileResponse:
    gen = service.get(generation_id)
    path = chunk_path(mastering.generation_dir(service.settings, gen.id), index)
    if not path.exists():
        raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="Ese trozo de audio todavía no existe.")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.post("/{generation_id}/postprocess", response_model=GenerationRead,
             summary="Aplicar o cambiar el posprocesado (el audio original del modelo no se modifica)")
async def postprocess(generation_id: str, body: PostProcessConfig,
                      service: GenerationService = Depends(get_generation_service)) -> GenerationRead:
    from starlette.concurrency import run_in_threadpool

    return await service.to_read(await run_in_threadpool(service.postprocess, generation_id, body))


@router.delete("/{generation_id}/postprocess", response_model=GenerationRead,
               summary="Quitar el posprocesado y volver al audio original del modelo")
async def clear_postprocess(generation_id: str,
                            service: GenerationService = Depends(get_generation_service)) -> GenerationRead:
    from starlette.concurrency import run_in_threadpool

    return await service.to_read(await run_in_threadpool(service.clear_postprocess, generation_id))


@router.put("/{generation_id}/rating", response_model=GenerationRead, summary="Guardar valoración manual (1–5)")
async def rate(generation_id: str, body: RatingUpdate,
               service: GenerationService = Depends(get_generation_service)) -> GenerationRead:
    return await service.to_read(service.rate(generation_id, body.model_dump()))


@router.post("/{generation_id}/evaluate", response_model=GenerationRead,
             summary="Calcular estimaciones automáticas (inteligibilidad con ASR)")
async def evaluate(generation_id: str, service: GenerationService = Depends(get_generation_service),
                   asr: ASRManager = Depends(get_asr_manager)) -> GenerationRead:
    from starlette.concurrency import run_in_threadpool

    evaluation = EvaluationService(service.session, service, asr)
    return await service.to_read(await run_in_threadpool(evaluation.evaluate, generation_id))


@router.put("/{generation_id}/favorite", response_model=LibraryItem, summary="Marcar o desmarcar favorito")
def set_favorite(generation_id: str, body: FavoriteUpdate,
                 service: GenerationService = Depends(get_generation_service)) -> LibraryItem:
    from app.services.library_service import LibraryService

    return LibraryService(service.session, service).set_favorite(generation_id, body.favorite)


@router.put("/{generation_id}/tags", response_model=LibraryItem, summary="Editar las etiquetas")
def set_tags(generation_id: str, body: TagsUpdate,
             service: GenerationService = Depends(get_generation_service)) -> LibraryItem:
    from app.services.library_service import LibraryService

    return LibraryService(service.session, service).set_tags(generation_id, body.tags)


@router.get("/{generation_id}/segments/{index}/audio", summary="Audio de una frase suelta")
def segment_audio(generation_id: str, index: int = Path(ge=0, le=999),
                  service: GenerationService = Depends(get_generation_service)) -> FileResponse:
    from app.services.resegment_service import ResegmentService

    path = ResegmentService(service).segment_audio(generation_id, index)
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.post("/{generation_id}/segments/{index}/regenerate", response_model=GenerationAccepted, status_code=202,
             summary="Repetir solo una frase (el resto del audio no cambia)")
async def regenerate_segment(generation_id: str, body: SegmentRegenerate, index: int = Path(ge=0, le=999),
                             service: GenerationService = Depends(get_generation_service)) -> GenerationAccepted:
    from app.services.resegment_service import ResegmentService

    gen = await ResegmentService(service).regenerate(generation_id, index, body)
    return GenerationAccepted(generation=await service.to_read(gen), job_id=gen.id,
                              queue_position=service.queue.queued_position(gen.id))


@router.delete("/{generation_id}", status_code=204, summary="Eliminar generación")
async def delete(generation_id: str, service: GenerationService = Depends(get_generation_service)) -> Response:
    await service.delete(generation_id)
    return Response(status_code=204)
