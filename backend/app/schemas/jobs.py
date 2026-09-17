"""Queue view: what is generating now and what is waiting."""

from pydantic import BaseModel

from app.models.enums import JobStatus


class QueuedJob(BaseModel):
    job_id: str
    kind: str  # job kind ("generation", "model_load"…)
    status: JobStatus
    progress: float
    message: str | None
    position: int  # 0 = already running
    text: str | None = None
    engine: str | None = None
    variant: str | None = None
    generation_kind: str | None = None  # single | preview | variation | experiment | project


class CancelReport(BaseModel):
    cancelled: list[str]
