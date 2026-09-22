"""Voice training (fine-tuning a voice profile): dataset summary, launch options and run status."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Defaults measured on this machine (see app/training/qwen_lora.py and docs): LoRA rank 16, 10 epochs.
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4
MIN_MINUTES = 5.0
RECOMMENDED_MINUTES = (15.0, 30.0)

TrainingStatus = Literal["queued", "preparing", "training", "saving", "evaluating", "completed", "failed",
                         "cancelled"]


class TrainingReferenceSummary(BaseModel):
    reference_id: str
    name: str
    duration_s: float
    transcribed: bool = Field(description="Tiene transcripción completa con marcas de tiempo por palabra")
    clips: int = Field(description="Frases utilizables para entrenar")
    usable_s: float
    discarded: dict[str, int] = Field(default_factory=dict, description="Frases descartadas por motivo")


class TrainingDatasetRead(BaseModel):
    profile_id: str
    references: list[TrainingReferenceSummary]
    clips: int
    usable_minutes: float
    recorded_minutes: float
    missing_transcripts: list[str] = Field(description="Referencias sin transcripción completa")
    min_minutes: float = MIN_MINUTES
    recommended_minutes: tuple[float, float] = RECOMMENDED_MINUTES
    ready: bool
    warnings: list[str] = Field(default_factory=list)


class TrainingCreate(BaseModel):
    profile_id: str
    base_variant: Literal["base-1.7b", "base-0.6b"] = "base-1.7b"
    name: str | None = Field(default=None, max_length=60, description="Nombre del modelo en el selector de variantes")
    epochs: int = Field(default=DEFAULT_EPOCHS, ge=1, le=50)
    learning_rate: float = Field(default=DEFAULT_LEARNING_RATE, gt=0, le=1e-3)

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        value = (value or "").strip()
        return value or None


class TrainingRunRead(BaseModel):
    id: str
    profile_id: str
    profile_name: str | None
    engine: str
    base_variant: str
    name: str
    status: TrainingStatus
    progress: float
    message: str | None
    params: dict[str, Any] | None
    dataset: dict[str, Any] | None
    history: list[dict[str, Any]] | None
    evaluation: dict[str, Any] | None
    checkpoint_id: str | None
    variant: str | None = Field(description="Variante del motor con la voz entrenada (custom:<slug>)")
    error_code: str | None
    error_detail: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
