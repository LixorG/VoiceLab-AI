from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.postprocess.config import PostProcessConfig
from app.schemas.generation import MAX_TEXT_CHARS, GenerationRead

MAX_ARMS = 6


class ExperimentArm(BaseModel):
    engine: str
    variant: str | None = None
    params: dict[str, Any] = {}
    label: str | None = Field(default=None, max_length=80)


class ExperimentArmsAdd(BaseModel):
    arms: list[ExperimentArm] = Field(min_length=1, max_length=MAX_ARMS)


class ExperimentCreate(ExperimentArmsAdd):
    name: str | None = Field(default=None, max_length=120)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    profile_id: str | None = None
    reference_id: str | None = None
    emotion: str | None = None
    intensity: int = Field(default=50, ge=0, le=100)
    markup: bool = True
    postprocess: PostProcessConfig | None = None

    @field_validator("text")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El texto no puede estar vacío.")
        return value


class ExperimentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=5000)


class ExperimentSummary(BaseModel):
    id: str
    name: str
    text: str
    arms: int
    completed: int
    running: int
    failed: int
    engines: list[str]
    created_at: datetime
    updated_at: datetime


class ExperimentRead(BaseModel):
    id: str
    name: str
    text: str
    profile_id: str | None
    reference_id: str | None
    notes: str | None
    settings: dict[str, Any] | None
    generations: list[GenerationRead]
    created_at: datetime
    updated_at: datetime


class RatingUpdate(BaseModel):
    """Manual listening scores (1–5). Subjective by definition; empty = not rated."""

    timbre: int | None = Field(default=None, ge=1, le=5)
    naturalness: int | None = Field(default=None, ge=1, le=5)
    pronunciation: int | None = Field(default=None, ge=1, le=5)
    emotion: int | None = Field(default=None, ge=1, le=5)
    notes: str | None = Field(default=None, max_length=1000)
