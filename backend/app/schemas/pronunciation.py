from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PronunciationInput(BaseModel):
    term: str = Field(min_length=1, max_length=80)
    replacement: str = Field(min_length=1, max_length=200)
    case_sensitive: bool = False
    profile_id: str | None = Field(default=None, description="Perfil de voz; vacío = diccionario global")

    @field_validator("term", "replacement")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("No puede estar vacío.")
        return value


class PronunciationRead(PronunciationInput):
    id: str
    created_at: datetime
    updated_at: datetime


class TextChangeRead(BaseModel):
    original: str
    replacement: str
    kind: str


class NormalizePreview(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str | None = Field(default="es", description="Código ISO (es, en…)")
    profile_id: str | None = None
    numbers: bool = True


class NormalizePreviewRead(BaseModel):
    text: str
    changes: list[TextChangeRead]
    language: str | None
    numbers_supported: bool
