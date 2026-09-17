"""/api/models — TTS engines, variants, capabilities and dynamic parameter schemas."""

from fastapi import APIRouter, Depends, Query

from app.engines.base import PRESET_LABELS, EngineCapabilities, ParameterSpec, TTSBackend
from app.engines.manager import EngineRuntimeStatus, LoadedModel, ModelManager, get_model_manager
from app.engines.registry import EngineRegistry, get_engine_registry
from app.schemas.engines import (
    EngineConfig,
    EngineSummary,
    PresetInfo,
    ValidateRequest,
    ValidateResponse,
)
from app.services.generation_service import MODEL_LOAD_JOB, get_job_queue
from app.workers.asyncio_queue import AsyncioJobQueue
from app.workers.base import JobSpec, JobState

router = APIRouter(prefix="/models", tags=["Modelos"])


def _summary(engine: TTSBackend) -> EngineSummary:
    return EngineSummary(
        id=engine.id, name=engine.display_name, description=engine.description,
        installed=engine.is_installed(), missing_packages=engine.missing_packages(),
        implemented=engine.implementation_phase is None, implementation_phase=engine.implementation_phase,
        license=engine.license, variants=engine.variants(), default_variant=engine.default_variant(),
    )


def _presets(engine: TTSBackend, variant: str | None) -> list[PresetInfo]:
    # Only offer presets when the engine defines real parameter changes; identical presets would mislead.
    if not any(spec.presets for spec in engine.get_parameters_schema(variant)):
        return []
    return [PresetInfo(id=pid, label=label, values=engine.resolve_preset(pid, variant))  # type: ignore[arg-type]
            for pid, label in PRESET_LABELS.items()]


@router.get("/runtime", response_model=LoadedModel | None, summary="Modelo cargado actualmente en memoria")
def runtime(models: ModelManager = Depends(get_model_manager)) -> LoadedModel | None:
    return models.loaded


@router.post("/unload", response_model=None, status_code=204,
             summary="Liberar el modelo de voz de la memoria (409 si se está usando)")
def unload(models: ModelManager = Depends(get_model_manager)) -> None:
    models.unload()


@router.get("", response_model=list[EngineSummary], summary="Motores de voz disponibles")
def list_engines(registry: EngineRegistry = Depends(get_engine_registry)) -> list[EngineSummary]:
    return [_summary(e) for e in registry.all()]


@router.get("/{engine_id}", response_model=EngineConfig,
            summary="Configuración completa de un motor (capacidades, parámetros y presets)")
def engine_config(engine_id: str, variant: str | None = Query(default=None),
                  registry: EngineRegistry = Depends(get_engine_registry)) -> EngineConfig:
    engine = registry.get(engine_id)
    v = engine.variant(variant)
    return EngineConfig(engine=_summary(engine), variant=v, capabilities=engine.capabilities(v.id),
                        parameters=engine.get_parameters_schema(v.id), presets=_presets(engine, v.id))


@router.get("/{engine_id}/capabilities", response_model=EngineCapabilities, summary="Capacidades del motor")
def capabilities(engine_id: str, variant: str | None = Query(default=None),
                 registry: EngineRegistry = Depends(get_engine_registry)) -> EngineCapabilities:
    return registry.get(engine_id).capabilities(variant)


@router.get("/{engine_id}/schema", response_model=list[ParameterSpec], summary="Esquema de parámetros")
def schema(engine_id: str, variant: str | None = Query(default=None),
           registry: EngineRegistry = Depends(get_engine_registry)) -> list[ParameterSpec]:
    return registry.get(engine_id).get_parameters_schema(variant)


@router.get("/{engine_id}/presets", response_model=list[PresetInfo], summary="Presets traducidos a parámetros")
def presets(engine_id: str, variant: str | None = Query(default=None),
            registry: EngineRegistry = Depends(get_engine_registry)) -> list[PresetInfo]:
    engine = registry.get(engine_id)
    return _presets(engine, engine.variant(variant).id)


@router.post("/{engine_id}/validate", response_model=ValidateResponse, summary="Validar parámetros")
def validate(engine_id: str, body: ValidateRequest,
             registry: EngineRegistry = Depends(get_engine_registry)) -> ValidateResponse:
    engine = registry.get(engine_id)
    v = engine.variant(body.variant)
    return ValidateResponse(variant=v.id, params=engine.validate_parameters(body.params, v.id))


@router.get("/{engine_id}/status", response_model=EngineRuntimeStatus,
            summary="Estado de instalación, pesos y carga de una variante")
def engine_status(engine_id: str, variant: str | None = Query(default=None),
                  models: ModelManager = Depends(get_model_manager)) -> EngineRuntimeStatus:
    return models.status(engine_id, variant)


@router.post("/{engine_id}/download", response_model=EngineRuntimeStatus, status_code=202,
             summary="Descargar los pesos de una variante")
def download_weights(engine_id: str, variant: str | None = Query(default=None),
                     models: ModelManager = Depends(get_model_manager)) -> EngineRuntimeStatus:
    return models.start_download(engine_id, variant)


@router.post("/{engine_id}/load", response_model=JobState, status_code=202,
             summary="Precargar una variante en memoria (pasa por la cola de la GPU)")
async def preload(engine_id: str, variant: str | None = Query(default=None),
                  models: ModelManager = Depends(get_model_manager),
                  queue: AsyncioJobQueue = Depends(get_job_queue)) -> JobState:
    from app.core.errors import AppError, ErrorCode

    if models.registry.get(engine_id).implementation_phase is not None:
        raise AppError(ErrorCode.NOT_IMPLEMENTED, status_code=409, details={"motor": engine_id})
    status = models.status(engine_id, variant)
    if not status.package_installed or not status.weights_installed:
        raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409, details={"motor": engine_id,
                                                                                "variante": status.variant})
    job_id = await queue.submit(JobSpec(kind=MODEL_LOAD_JOB, payload={"engine": engine_id,
                                                                      "variant": status.variant}))
    return await queue.get(job_id)
