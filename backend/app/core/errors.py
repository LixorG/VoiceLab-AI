"""Error catalog and global exception handlers.

User-facing messages are Spanish; tracebacks only go to the backend log.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("voicelab.errors")


class ErrorCode(StrEnum):
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    GPU_MEMORY_ERROR = "GPU_MEMORY_ERROR"
    GPU_NOT_AVAILABLE = "GPU_NOT_AVAILABLE"
    MODEL_NOT_INSTALLED = "MODEL_NOT_INSTALLED"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    ENGINE_NOT_FOUND = "ENGINE_NOT_FOUND"
    PARAMETER_ERROR = "PARAMETER_ERROR"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    AUDIO_DECODE_ERROR = "AUDIO_DECODE_ERROR"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    FFMPEG_NOT_FOUND = "FFMPEG_NOT_FOUND"
    ASR_NOT_AVAILABLE = "ASR_NOT_AVAILABLE"
    ASR_MODEL_NOT_INSTALLED = "ASR_MODEL_NOT_INSTALLED"
    ASR_ERROR = "ASR_ERROR"
    PROFILE_NOT_FOUND = "PROFILE_NOT_FOUND"
    PROFILE_IMPORT_ERROR = "PROFILE_IMPORT_ERROR"
    MARKUP_ERROR = "MARKUP_ERROR"
    REFERENCE_REQUIRED = "REFERENCE_REQUIRED"
    REFERENCE_TEXT_REQUIRED = "REFERENCE_TEXT_REQUIRED"
    REFERENCE_TOO_LONG = "REFERENCE_TOO_LONG"
    GENERATION_NOT_FOUND = "GENERATION_NOT_FOUND"
    GENERATION_NOT_READY = "GENERATION_NOT_READY"
    POSTPROCESS_ERROR = "POSTPROCESS_ERROR"
    EXPERIMENT_NOT_FOUND = "EXPERIMENT_NOT_FOUND"
    MODEL_BUSY = "MODEL_BUSY"
    JOB_INTERRUPTED = "JOB_INTERRUPTED"
    PROJECT_NOT_FOUND = "PROJECT_NOT_FOUND"
    PROJECT_NOT_READY = "PROJECT_NOT_READY"
    PRONUNCIATION_NOT_FOUND = "PRONUNCIATION_NOT_FOUND"
    EVALUATION_UNAVAILABLE = "EVALUATION_UNAVAILABLE"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_CANCELLED = "JOB_CANCELLED"


MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INTERNAL_ERROR: "Se produjo un error inesperado. Consulta los registros del servidor para más detalles.",
    ErrorCode.VALIDATION_ERROR: "Los datos enviados no son válidos.",
    ErrorCode.NOT_FOUND: "El recurso solicitado no existe.",
    ErrorCode.METHOD_NOT_ALLOWED: "Operación no permitida para este recurso.",
    ErrorCode.NOT_IMPLEMENTED: "Esta función todavía no está disponible.",
    ErrorCode.GPU_MEMORY_ERROR: "No hay suficiente memoria de GPU para ejecutar este modelo. Prueba con un modelo más pequeño o libera memoria.",
    ErrorCode.GPU_NOT_AVAILABLE: "No se detectó una GPU compatible. La generación en CPU será muy lenta.",
    ErrorCode.MODEL_NOT_INSTALLED: "El modelo seleccionado no está instalado.",
    ErrorCode.MODEL_LOAD_ERROR: "No se pudo cargar el modelo.",
    ErrorCode.ENGINE_NOT_FOUND: "El motor de voz solicitado no existe.",
    ErrorCode.PARAMETER_ERROR: "Uno o más parámetros no son válidos para este modelo.",
    ErrorCode.FILE_TOO_LARGE: "El archivo supera el tamaño máximo permitido.",
    ErrorCode.UNSUPPORTED_FORMAT: "Formato de audio no compatible.",
    ErrorCode.AUDIO_DECODE_ERROR: "No se pudo leer el archivo de audio. Puede estar dañado.",
    ErrorCode.AUDIO_TOO_LONG: "El audio supera la duración máxima permitida.",
    ErrorCode.FFMPEG_NOT_FOUND: "FFmpeg no está instalado o no se encuentra en el PATH.",
    ErrorCode.ASR_NOT_AVAILABLE: "La transcripción automática no está disponible: falta instalar faster-whisper.",
    ErrorCode.ASR_MODEL_NOT_INSTALLED: "El modelo de transcripción no está descargado. Descárgalo desde la tarjeta de la referencia o en Configuración.",
    ErrorCode.ASR_ERROR: "No se pudo transcribir el audio.",
    ErrorCode.PROFILE_NOT_FOUND: "El perfil de voz no existe.",
    ErrorCode.PROFILE_IMPORT_ERROR: "El archivo no es un perfil de voz válido.",
    ErrorCode.MARKUP_ERROR: "El texto contiene marcas de expresión no válidas.",
    ErrorCode.REFERENCE_REQUIRED: "Este modelo necesita un audio de referencia. Sube o selecciona una referencia.",
    ErrorCode.REFERENCE_TEXT_REQUIRED: "Este modelo necesita la transcripción exacta de la referencia. Transcríbela o escríbela antes de generar.",
    ErrorCode.REFERENCE_TOO_LONG: "La referencia es demasiado larga para este modelo. Selecciona un segmento más corto.",
    ErrorCode.GENERATION_NOT_FOUND: "La generación solicitada no existe.",
    ErrorCode.GENERATION_NOT_READY: "La generación todavía no ha terminado correctamente.",
    ErrorCode.POSTPROCESS_ERROR: "No se pudo aplicar el posprocesado al audio.",
    ErrorCode.EXPERIMENT_NOT_FOUND: "El experimento solicitado no existe.",
    ErrorCode.PROJECT_NOT_FOUND: "El proyecto solicitado no existe.",
    ErrorCode.PROJECT_NOT_READY: "Faltan segmentos por generar (o su audio está desactualizado) para exportar el proyecto.",
    ErrorCode.PRONUNCIATION_NOT_FOUND: "La entrada del diccionario de pronunciación no existe.",
    ErrorCode.JOB_INTERRUPTED: "La tarea se interrumpió porque el servidor se cerró o se reinició. Vuelve a lanzarla.",
    ErrorCode.MODEL_BUSY: "El modelo se está usando en una generación. Espera a que termine o cancélala.",
    ErrorCode.EVALUATION_UNAVAILABLE: "No hay ninguna evaluación automática disponible en esta instalación.",
    ErrorCode.JOB_NOT_FOUND: "La tarea solicitada no existe.",
    ErrorCode.JOB_CANCELLED: "La tarea fue cancelada.",
}


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Any | None = None


class AppError(Exception):
    """Domain error that maps to a friendly API response."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        status_code: int = 400,
        message: str | None = None,
        details: Any | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.message = message or MESSAGES[code]
        self.details = details
        super().__init__(f"{code}: {self.message}")


def _payload(code: ErrorCode, message: str | None = None, details: Any = None) -> dict:
    return ErrorResponse(
        error_code=code.value, message=message or MESSAGES[code], details=details
    ).model_dump()


def classify_exception(exc: BaseException) -> ErrorCode:
    """Map low-level exceptions (torch, OS) to catalog codes without importing torch."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "outofmemoryerror" in text:
        return ErrorCode.GPU_MEMORY_ERROR
    return ErrorCode.INTERNAL_ERROR


_HTTP_CODES = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.METHOD_NOT_ALLOWED, 501: ErrorCode.NOT_IMPLEMENTED}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("app_error", extra={"code": exc.code.value, "path": request.url.path})
        return JSONResponse(
            status_code=exc.status_code, content=_payload(exc.code, exc.message, exc.details)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"campo": ".".join(str(p) for p in err.get("loc", [])), "tipo": err.get("type")}
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content=_payload(ErrorCode.VALIDATION_ERROR, details=details))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        return JSONResponse(status_code=exc.status_code, content=_payload(code))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        code = classify_exception(exc)
        logger.exception("unhandled_error", extra={"code": code.value, "path": request.url.path})
        status = 507 if code is ErrorCode.GPU_MEMORY_ERROR else 500
        return JSONResponse(status_code=status, content=_payload(code))
