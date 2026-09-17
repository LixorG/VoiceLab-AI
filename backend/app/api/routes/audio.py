"""/api/audio — serve reference audio and waveform data."""

from typing import Literal

import soundfile as sf
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.api.routes.references import get_service
from app.audio.pipeline import INTERNAL_SAMPLE_RATE
from app.audio.processing import waveform_peaks
from app.audio.validation import ALLOWED_FORMATS
from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.models.enums import ReferenceStatus
from app.schemas.references import AudioFormats, PeaksResponse
from app.services.reference_service import ReferenceService

router = APIRouter(prefix="/audio", tags=["Audio"])

MEDIA_TYPES = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "m4a": "audio/mp4",
    "ogg": "audio/ogg", "opus": "audio/ogg", "aiff": "audio/aiff", "aif": "audio/aiff",
}


@router.get("/formats", response_model=AudioFormats, summary="Formatos y límites de subida")
def formats(settings: Settings = Depends(get_settings)) -> AudioFormats:
    return AudioFormats(extensions=sorted(ALLOWED_FORMATS), max_upload_mb=settings.max_upload_mb,
                        max_audio_seconds=settings.max_audio_seconds,
                        internal_sample_rate=INTERNAL_SAMPLE_RATE)


@router.get("/references/{reference_id}/peaks", response_model=PeaksResponse, summary="Datos de waveform")
def reference_peaks(reference_id: str, buckets: int = Query(default=1000, ge=50, le=8000),
                    service: ReferenceService = Depends(get_service)) -> PeaksResponse:
    ref = service.get(reference_id)
    path = service.storage.processed_path(ref.sha256)
    if ref.status != ReferenceStatus.ANALYZED or not path.exists():
        raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="La referencia aún no está procesada.")
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    peaks = waveform_peaks(samples, buckets)
    return PeaksResponse(buckets=len(peaks), duration_s=round(samples.size / sr, 3), sample_rate=sr,
                         peaks=peaks)


@router.get("/references/{reference_id}/{variant}", summary="Descargar audio original o procesado")
def reference_file(reference_id: str, variant: Literal["original", "processed"],
                   service: ReferenceService = Depends(get_service)) -> FileResponse:
    ref = service.get(reference_id)
    if variant == "original":
        path = service.storage.find_original(ref.sha256)
        media, filename = MEDIA_TYPES.get(ref.format, "application/octet-stream"), ref.original_name
    else:
        path = service.storage.processed_path(ref.sha256)
        media, filename = "audio/wav", f"{ref.original_name.rsplit('.', 1)[0]}_24k.wav"
    if path is None or not path.exists():
        raise AppError(ErrorCode.NOT_FOUND, status_code=404,
                       message="El archivo de audio no está disponible.")
    return FileResponse(path, media_type=media, filename=filename, content_disposition_type="inline")
