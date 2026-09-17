from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.enums import JobStatus
from app.postprocess.config import PostProcessConfig, PostProcessReport

MAX_TEXT_CHARS = 5000


class GenerationCreate(BaseModel):
    engine: str
    variant: str | None = None
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    params: dict[str, Any] = {}
    reference_id: str | None = None
    profile_id: str | None = Field(default=None, description="Perfil de voz; si no hay referencia se usa la principal")
    preview: bool = Field(default=False, description="Genera solo una muestra corta (primera frase)")
    emotion: str | None = Field(default=None, description="Emoción global (se aplica según las capacidades del motor)")
    intensity: int = Field(default=50, ge=0, le=100, description="Intensidad de la emoción (0–100)")
    markup: bool = Field(default=True, description="Interpretar marcas como [pausa:500ms] o [emoción:feliz]")
    normalize: bool = Field(default=True, description="Escribir números, símbolos y abreviaturas como se leen y "
                                                       "aplicar el diccionario de pronunciación")
    postprocess: PostProcessConfig | None = Field(default=None,
                                                  description="Posprocesado opcional (desactivado por defecto)")

    @field_validator("emotion")
    @classmethod
    def _emotion(cls, value: str | None) -> str | None:
        from app.voice_profiles.emotions import EMOTIONS

        if value is not None and value not in EMOTIONS:
            raise ValueError("Emoción no válida.")
        return value

    @field_validator("text")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El texto no puede estar vacío.")
        return value


class VariationsCreate(GenerationCreate):
    count: int = Field(default=3, ge=2, le=8, description="Número de variaciones (semillas distintas)")


class PlannedSegmentRead(BaseModel):
    index: int
    text: str
    emotion: str | None
    emotion_via: str | None
    instruction: str | None
    pause_before_ms: int
    pause_after_ms: int
    reference_name: str | None
    seed: int | None = None
    duration_s: float | None = None


class TextChangeItem(BaseModel):
    original: str
    replacement: str
    kind: str


class GenerationPlanRead(BaseModel):
    segments: list[PlannedSegmentRead]
    warnings: list[str]
    segmented: bool
    text_changes: list[TextChangeItem] = []
    normalize_language: str | None = None


class GenerationReference(BaseModel):
    reference_id: str
    name: str | None
    start_s: float | None
    end_s: float | None
    text: str


class GenerationPostProcess(BaseModel):
    config: PostProcessConfig
    report: PostProcessReport | None = None


class GenerationRead(BaseModel):
    id: str
    kind: str
    engine: str
    variant: str | None
    text: str
    params: dict[str, Any] | None
    seed: int | None
    status: JobStatus
    progress: float
    progress_available: bool = True
    message: str | None
    error_code: str | None
    reference: GenerationReference | None
    duration_s: float | None
    audio_url: str | None
    metrics: dict[str, Any] | None
    warnings: list[str]
    expression: dict[str, Any] | None = None
    parent_id: str | None = None
    segments: list[PlannedSegmentRead] = []
    postprocess: GenerationPostProcess | None = None
    raw_audio_url: str | None = Field(default=None, description="Salida del modelo sin posprocesar")
    experiment_id: str | None = None
    label: str | None = None
    rating: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = Field(default=None, description="Estimaciones automáticas; «stale» si el "
                                                                      "audio cambió después de evaluarlo")
    created_at: datetime
    updated_at: datetime


class GenerationAccepted(BaseModel):
    generation: GenerationRead
    job_id: str
    queue_position: int
