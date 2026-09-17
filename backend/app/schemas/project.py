from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.postprocess.config import PostProcessConfig
from app.schemas.generation import MAX_TEXT_CHARS, GenerationRead

MAX_SEGMENTS = 500
SegmentStatus = Literal["empty", "queued", "generating", "ready", "stale", "failed", "cancelled"]


class ProjectSettings(BaseModel):
    """Defaults every segment inherits (a segment can override voice, emotion, pause and seed)."""

    engine: str | None = None
    variant: str | None = None
    params: dict[str, Any] = {}
    profile_id: str | None = None
    reference_id: str | None = None
    markup: bool = True
    normalize: bool = True  # números/abreviaturas en letra + diccionario de pronunciación
    default_pause_ms: int = Field(default=400, ge=0, le=10_000)
    export_postprocess: PostProcessConfig | None = Field(
        default=None, description="Masterización opcional aplicada al audio completo al exportar")


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    settings: ProjectSettings = ProjectSettings()


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    settings: ProjectSettings | None = None


class SegmentInput(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El texto no puede estar vacío.")
        return value


class SegmentsAdd(BaseModel):
    segments: list[SegmentInput] = Field(min_length=1, max_length=MAX_SEGMENTS)
    position: int | None = Field(default=None, ge=0, description="Índice donde insertar; vacío = al final")


class ScriptImport(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    split: Literal["paragraphs", "sentences"] = "paragraphs"


class SegmentUpdate(BaseModel):
    """Only the fields sent are changed; send null to go back to the project default."""

    text: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)
    profile_id: str | None = None
    reference_id: str | None = None
    emotion: str | None = None
    intensity: int | None = Field(default=None, ge=0, le=100)
    pause_after_ms: int | None = Field(default=None, ge=0, le=10_000)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class SegmentOrder(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=MAX_SEGMENTS)


class ProjectGenerate(BaseModel):
    segment_ids: list[str] | None = Field(default=None, description="Vacío = todo el proyecto")
    only_pending: bool = Field(default=True, description="Omitir segmentos listos y actualizados")


class SegmentRead(BaseModel):
    id: str
    position: int
    text: str
    profile_id: str | None
    reference_id: str | None
    emotion: str | None
    intensity: int | None
    pause_after_ms: int | None
    seed: int | None
    effective_pause_ms: int
    status: SegmentStatus
    generation: GenerationRead | None


class ProjectRead(BaseModel):
    id: str
    name: str
    description: str | None
    settings: ProjectSettings
    segments: list[SegmentRead]
    total_duration_s: float
    exported_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ProjectSummary(BaseModel):
    id: str
    name: str
    description: str | None
    segments: int
    ready: int
    pending: int
    running: int
    failed: int
    updated_at: datetime
