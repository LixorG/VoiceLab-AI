"""Database entities (SQLModel). Portable types only, so SQLite → PostgreSQL stays a config change."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from app.models.enums import JobStatus, ReferenceStatus, TranscriptSource


def _uuid() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


def _json(**kwargs: Any) -> Any:
    return Field(sa_column=Column(JSON, nullable=True), **kwargs)


class TimestampMixin(SQLModel):
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class VoiceProfile(TimestampMixin, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    owner_id: str | None = Field(default=None, index=True)  # reserved for multi-user
    name: str
    slug: str = Field(index=True, unique=True)
    description: str | None = None
    language: str | None = None
    default_engine: str | None = None
    primary_reference_id: str | None = None  # no FK: avoids a cycle with ReferenceAudio.profile_id
    recommended_settings: dict[str, Any] | None = _json(default=None)
    extra_metadata: dict[str, Any] | None = _json(default=None)


class ReferenceAudio(TimestampMixin, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    profile_id: str | None = Field(default=None, foreign_key="voiceprofile.id", index=True)
    sha256: str = Field(index=True)
    original_name: str
    format: str
    size_bytes: int
    sample_rate: int | None = None
    channels: int | None = None
    duration_s: float | None = None
    status: ReferenceStatus = ReferenceStatus.UPLOADED
    quality_score: float | None = None
    analysis: dict[str, Any] | None = _json(default=None)
    emotion_tag: str | None = None
    segment_start_s: float | None = None
    segment_end_s: float | None = None
    is_recommended: bool = False


class Transcript(TimestampMixin, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    reference_id: str = Field(foreign_key="referenceaudio.id", index=True)
    text: str
    language: str | None = None
    source: TranscriptSource = TranscriptSource.ASR
    asr_model: str | None = None
    word_timestamps: list[dict[str, Any]] | None = _json(default=None)
    # None/None = whole reference; otherwise the transcribed range (seconds, processed timeline)
    segment_start_s: float | None = None
    segment_end_s: float | None = None
    asr_text: str | None = None  # untouched ASR output, kept when the user edits `text`
    asr_metadata: dict[str, Any] | None = _json(default=None)


class ModelRecord(TimestampMixin, table=True):
    __tablename__ = "model"

    id: str = Field(primary_key=True)  # "<engine>:<variant>"
    engine: str = Field(index=True)
    variant: str
    installed: bool = False
    size_bytes: int | None = None
    vram_estimate_mb: int | None = None
    license: str | None = None
    local_path: str | None = None


class Preset(TimestampMixin, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    engine: str = Field(index=True)
    name: str
    params: dict[str, Any] | None = _json(default=None)
    builtin: bool = False


class Project(TimestampMixin, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    owner_id: str | None = Field(default=None, index=True)
    name: str
    description: str | None = None
    script: dict[str, Any] | None = _json(default=None)  # legacy placeholder (unused)
    settings: dict[str, Any] | None = _json(default=None)  # ProjectSettings
    exported_at: datetime | None = None


class CustomCheckpoint(TimestampMixin, table=True):
    """A checkpoint added by the user (community fine-tune or own training) offered as an extra variant."""
    id: str = Field(default_factory=_uuid, primary_key=True)
    engine: str = Field(index=True)
    name: str
    slug: str = Field(index=True)  # variant id = "custom:<slug>"
    base_variant: str  # architecture it was fine-tuned from
    repo_id: str | None = None
    ckpt_file: str | None = None
    local_path: str | None = None
    vocab_file: str | None = None  # vocabulary inside the same repo
    vocab_path: str | None = None  # vocabulary on this machine
    languages: list[str] | None = _json(default=None)
    notes: str | None = None


class TrainingRun(TimestampMixin, table=True):
    """A fine-tune of a voice profile. `id` is also the id of its job in the GPU queue."""
    id: str = Field(default_factory=_uuid, primary_key=True)
    profile_id: str = Field(index=True)  # no FK: the run (and its model) outlive a deleted profile
    engine: str = "qwen3tts"
    base_variant: str  # base-1.7b | base-0.6b
    name: str
    status: str = "queued"  # queued | preparing | training | saving | evaluating | completed | failed | cancelled
    progress: float = 0.0
    message: str | None = None
    params: dict[str, Any] | None = _json(default=None)  # epochs, learning rate, LoRA rank…
    dataset: dict[str, Any] | None = _json(default=None)  # clips, minutes, held-out clips, reference used
    history: list[dict[str, Any]] | None = _json(default=None)  # losses per epoch
    evaluation: dict[str, Any] | None = _json(default=None)  # trained model vs normal cloning on held-out clips
    checkpoint_id: str | None = None  # CustomCheckpoint published when it finishes
    output_path: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class PronunciationEntry(TimestampMixin, table=True):
    """User pronunciation dictionary: `term` is written as `replacement` before sending text to the engine."""
    id: str = Field(default_factory=_uuid, primary_key=True)
    profile_id: str | None = Field(default=None, index=True)  # None = global
    term: str
    replacement: str
    case_sensitive: bool = False


class ProjectSegment(TimestampMixin, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    project_id: str = Field(foreign_key="project.id", index=True)
    position: int = Field(default=0, index=True)
    text: str
    # Overrides of the project defaults (None = inherit)
    profile_id: str | None = None
    reference_id: str | None = None
    emotion: str | None = None
    intensity: int | None = None
    pause_after_ms: int | None = None
    seed: int | None = None
    generation_id: str | None = Field(default=None, index=True)  # latest generation (no FK: deleted on regenerate)
    generated_signature: str | None = None  # settings hash used for that generation → "stale" detection


class Experiment(TimestampMixin, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    project_id: str | None = Field(default=None, foreign_key="project.id", index=True)
    name: str
    text: str
    profile_id: str | None = Field(default=None, index=True)  # no FK: experiments outlive deleted profiles
    reference_id: str | None = None
    settings: dict[str, Any] | None = _json(default=None)  # emotion, intensity, markup, postprocess shared by arms
    notes: str | None = None


class Generation(TimestampMixin, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    project_id: str | None = Field(default=None, foreign_key="project.id", index=True)
    profile_id: str | None = Field(default=None, foreign_key="voiceprofile.id", index=True)
    parent_id: str | None = Field(default=None, foreign_key="generation.id", index=True)
    reference_id: str | None = Field(default=None, index=True)  # no FK: generations outlive deleted references
    reference_snapshot: dict[str, Any] | None = _json(default=None)  # name, range and text actually used
    expression: dict[str, Any] | None = _json(default=None)  # emotion, intensity, markup flag requested
    kind: str = "single"  # single | preview | variation | experiment | script
    engine: str
    variant: str | None = None
    text: str
    params: dict[str, Any] | None = _json(default=None)
    seed: int | None = None
    status: JobStatus = Field(default=JobStatus.QUEUED, index=True)
    error_code: str | None = None
    output_path: str | None = None
    raw_output_path: str | None = None
    duration_s: float | None = None
    metrics: dict[str, Any] | None = _json(default=None)
    warnings: list[str] | None = _json(default=None)
    postprocess: dict[str, Any] | None = _json(default=None)  # {"config": PostProcessConfig, "report": ...}
    experiment_id: str | None = Field(default=None, foreign_key="experiment.id", index=True)
    label: str | None = None  # arm name inside an experiment
    favorite: bool = Field(default=False, index=True)  # user's library
    tags: list[str] | None = _json(default=None)  # user's library
    rating: dict[str, Any] | None = _json(default=None)  # manual 1–5 scores + notes
    evaluation: dict[str, Any] | None = _json(default=None)  # automatic estimates (see app/evaluation)


class GenerationSegment(TimestampMixin, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    generation_id: str = Field(foreign_key="generation.id", index=True)
    index: int
    text: str
    engine: str
    params: dict[str, Any] | None = _json(default=None)
    seed: int | None = None
    emotion: str | None = None
    pause_before_ms: int = 0
    pause_after_ms: int = 0
    instruction: str | None = None
    reference_snapshot: dict[str, Any] | None = _json(default=None)
    audio_path: str | None = None
    duration_s: float | None = None
