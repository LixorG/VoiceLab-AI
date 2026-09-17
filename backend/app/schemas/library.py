"""Library: search, filters, tags and favourites over the standalone generations of «Generar»."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.enums import JobStatus

MAX_TAGS = 10
MAX_TAG_CHARS = 30


def clean_tags(values: list[str]) -> list[str]:
    """Trimmed, lowercase, no duplicates, in order; empty strings dropped."""
    out: list[str] = []
    for raw in values:
        tag = " ".join(raw.split()).lower()[:MAX_TAG_CHARS]
        if tag and tag not in out:
            out.append(tag)
    return out[:MAX_TAGS]


class TagsUpdate(BaseModel):
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS * 2)

    @field_validator("tags")
    @classmethod
    def _clean(cls, values: list[str]) -> list[str]:
        return clean_tags(values)


class FavoriteUpdate(BaseModel):
    favorite: bool


class DeleteMany(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=200)


class DeleteReport(BaseModel):
    deleted: list[str]
    failed: list[str]


class LibraryItem(BaseModel):
    """Light row for the list: no segments and no live job state (see GenerationRead for the full detail)."""

    id: str
    kind: str
    engine: str
    variant: str | None
    text: str
    status: JobStatus
    seed: int | None
    duration_s: float | None
    audio_url: str | None
    favorite: bool
    tags: list[str]
    rating: dict[str, Any] | None
    profile_id: str | None
    profile_name: str | None
    reference_name: str | None
    created_at: datetime


class LibraryPage(BaseModel):
    items: list[LibraryItem]
    total: int
    limit: int
    offset: int


class TagCount(BaseModel):
    tag: str
    count: int


class FacetOption(BaseModel):
    id: str
    label: str
    count: int


class LibraryFacets(BaseModel):
    """Only values that actually exist in the history, so no filter can come back empty by construction."""

    tags: list[TagCount]
    engines: list[FacetOption]
    profiles: list[FacetOption]
    total: int
    favorites: int


class LibraryQuery(BaseModel):
    q: str | None = Field(default=None, max_length=200, description="Busca en el texto generado")
    engine: str | None = None
    profile_id: str | None = None
    status: JobStatus | None = None
    kind: str | None = None
    tag: str | None = None
    favorite: bool | None = None
    since: date | None = None
    until: date | None = None
    limit: int = Field(default=30, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
