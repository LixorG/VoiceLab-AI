from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.enums import ReferenceStatus
from app.schemas.transcription import TranscriptRead


class ReferenceUrls(BaseModel):
    original: str
    processed: str | None
    peaks: str | None


class ReferenceRead(BaseModel):
    id: str
    profile_id: str | None
    original_name: str
    format: str
    size_bytes: int
    sample_rate: int | None
    channels: int | None
    duration_s: float | None
    status: ReferenceStatus
    quality_score: float | None
    quality_label: str | None
    analysis: dict[str, Any] | None
    segment_start_s: float | None
    segment_end_s: float | None
    emotion_tag: str | None
    is_recommended: bool
    is_primary: bool = False
    created_at: datetime
    urls: ReferenceUrls
    transcript: TranscriptRead | None = Field(default=None, description="Transcripción del audio completo")
    segment_transcript: TranscriptRead | None = Field(
        default=None, description="Transcripción del segmento guardado (si coincide exactamente)")
    segment_text_estimate: str | None = Field(
        default=None, description="Texto del segmento estimado a partir de los tiempos por palabra")


class UploadResponse(BaseModel):
    reference: ReferenceRead
    duplicate: bool = Field(description="El mismo archivo ya existía; no se volvió a procesar.")


class ReferenceUpdate(BaseModel):
    segment_start_s: float | None = Field(default=None, ge=0)
    segment_end_s: float | None = Field(default=None, ge=0)
    emotion_tag: str | None = Field(default=None, max_length=32)
    profile_id: str | None = Field(default=None, description="Asignar a un perfil (null = sin perfil)")
    snap_to_words: bool = Field(
        default=False, description="Ajustar los bordes a pausas entre palabras usando la transcripción completa")

    @model_validator(mode="after")
    def _segment_pair(self) -> ReferenceUpdate:
        from app.voice_profiles.emotions import EMOTIONS

        if self.emotion_tag is not None and self.emotion_tag not in EMOTIONS:
            raise ValueError("etiqueta de emoción no válida")
        start, end = self.segment_start_s, self.segment_end_s
        if (start is None) != (end is None):
            raise ValueError("segment_start_s y segment_end_s deben enviarse juntos")
        if start is not None and end is not None and end - start < 0.5:
            raise ValueError("el segmento debe durar al menos 0.5 s")
        return self


class PeaksResponse(BaseModel):
    buckets: int
    duration_s: float
    sample_rate: int
    peaks: list[float]


class AudioFormats(BaseModel):
    extensions: list[str]
    max_upload_mb: int
    max_audio_seconds: int
    internal_sample_rate: int
