from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.enums import TranscriptSource


class TranscriptWord(BaseModel):
    start: float
    end: float
    word: str
    probability: float


class TranscriptConfidence(BaseModel):
    language_probability: float | None = None
    avg_logprob: float | None = None
    max_no_speech_prob: float | None = None
    mean_word_probability: float | None = None


class TranscriptRead(BaseModel):
    id: str
    reference_id: str
    text: str
    language: str | None
    source: TranscriptSource
    edited: bool
    asr_model: str | None
    segment_start_s: float | None
    segment_end_s: float | None
    confidence: TranscriptConfidence
    warnings: list[str]
    words: list[TranscriptWord] | None = None
    updated_at: datetime


class TranscribeRequest(BaseModel):
    scope: Literal["full", "segment"] = "full"
    language: str | None = Field(default=None, description="Código ISO (es, en…). Vacío = detección automática.")
    force: bool = Field(default=False, description="Ignorar la caché y volver a ejecutar el modelo.")


class TranscriptUpdate(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class LanguageOption(BaseModel):
    code: str
    name: str
