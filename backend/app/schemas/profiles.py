from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RecommendedSetting(BaseModel):
    variant: str
    params: dict[str, Any] = {}
    note: str | None = Field(default=None, max_length=300)
    updated_at: datetime | None = None


class ProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=1000)
    language: str | None = Field(default=None, max_length=40)
    default_engine: str | None = None

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El nombre no puede estar vacío.")
        return value


class ProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=1000)
    language: str | None = Field(default=None, max_length=40)
    default_engine: str | None = None
    primary_reference_id: str | None = None


class RecommendedSettingUpdate(BaseModel):
    variant: str | None = None
    params: dict[str, Any] = {}
    note: str | None = Field(default=None, max_length=300)


class ProfileStats(BaseModel):
    reference_count: int
    analyzed_count: int
    transcribed_count: int
    total_duration_s: float
    total_speech_s: float
    average_quality: float | None
    quality_label: str | None
    emotions: list[str]


class ProfileRead(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None
    language: str | None
    default_engine: str | None
    primary_reference_id: str | None
    recommended_reference_id: str | None
    recommended_settings: dict[str, RecommendedSetting]
    stats: ProfileStats
    created_at: datetime
    updated_at: datetime


class EmotionOption(BaseModel):
    id: str
    label: str
