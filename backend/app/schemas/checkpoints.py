"""Custom checkpoints: what the user declares when adding weights of their own."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class CheckpointCreate(BaseModel):
    name: str = Field(min_length=1, max_length=60, description="Nombre visible en el selector de variantes")
    base_variant: str = Field(description="Variante oficial de la que salió (arquitectura)")
    repo_id: str | None = Field(default=None, max_length=200, description="Repositorio de Hugging Face")
    ckpt_file: str | None = Field(default=None, max_length=300, description="Archivo dentro del repositorio")
    local_path: str | None = Field(default=None, max_length=500, description="Ruta del archivo en este equipo")
    vocab_file: str | None = Field(default=None, max_length=300,
                                   description="vocab.txt dentro del repositorio, si el checkpoint lo necesita")
    vocab_path: str | None = Field(default=None, max_length=500,
                                   description="vocab.txt en este equipo, si el checkpoint lo necesita")
    languages: list[str] = Field(default_factory=list, max_length=10,
                                 description="Idiomas en los que fue entrenado (los declaras tú)")
    notes: str | None = Field(default=None, max_length=500)

    @field_validator("name", "repo_id", "ckpt_file", "local_path", "vocab_file", "vocab_path", "notes")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("languages")
    @classmethod
    def _languages(cls, values: list[str]) -> list[str]:
        out = []
        for raw in values:
            code = raw.strip().lower()[:5]
            if code and code not in out:
                out.append(code)
        return out

    @model_validator(mode="after")
    def _source(self) -> CheckpointCreate:
        from_hub = bool(self.repo_id and self.ckpt_file)
        if from_hub == bool(self.local_path):
            raise ValueError("Indica un repositorio con su archivo o una ruta local, pero no las dos.")
        if self.vocab_file and not from_hub:
            raise ValueError("El vocabulario del repositorio solo se puede usar con un checkpoint de Hugging Face.")
        if self.vocab_file and self.vocab_path:
            raise ValueError("Indica el vocabulario del repositorio o uno local, pero no los dos.")
        return self


class CheckpointRead(BaseModel):
    id: str
    engine: str
    variant: str  # "custom:<slug>"
    name: str
    base_variant: str
    repo_id: str | None
    ckpt_file: str | None
    local_path: str | None
    vocab_file: str | None
    vocab_path: str | None
    languages: list[str]
    notes: str | None
    weights_installed: bool
    created_at: datetime
