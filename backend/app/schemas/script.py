"""Script preparation: the text made ready for speech (same words), what changed, alerts and proposed cuts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

MAX_SCRIPT_CHARS = 20_000


class ScriptPrepare(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_SCRIPT_CHARS)
    language: str | None = Field(default=None, description="Idioma del texto; si falta: motor → voz → el propio texto")
    params: dict[str, Any] = Field(default_factory=dict, description="Parámetros del motor (para su idioma)")
    profile_id: str | None = None


class ScriptChangeRead(BaseModel):
    kind: str
    before: str
    after: str
    reason: str


class ScriptAlertRead(BaseModel):
    kind: str
    message: str
    excerpt: str | None = None


class ScriptSuggestionRead(BaseModel):
    before: str
    after: str
    reason: str


class ScriptStatsRead(BaseModel):
    words: int
    sentences: int
    paragraphs: int
    average_words: float
    longest_words: int
    seconds: float = Field(description="Duración estimada (aproximada)")


class ScriptPrepared(BaseModel):
    text: str
    changed: bool
    language: str | None
    changes: list[ScriptChangeRead]
    alerts: list[ScriptAlertRead]
    suggestions: list[ScriptSuggestionRead]
    stats: ScriptStatsRead
