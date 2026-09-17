"""Job queue contract. Implementations: AsyncioJobQueue (FASE 5); Celery/RQ later if needed."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.models.enums import JobStatus


class JobSpec(BaseModel):
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class JobState(BaseModel):
    job_id: str
    kind: str
    status: JobStatus
    progress: float = 0.0  # 0..1
    message: str | None = None  # Spanish, user-facing
    error_code: str | None = None
    result: dict[str, Any] | None = None


class JobEvent(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    message: str | None = None


class JobQueue(Protocol):
    async def submit(self, spec: JobSpec) -> str: ...

    async def cancel(self, job_id: str) -> None: ...

    async def get(self, job_id: str) -> JobState: ...

    def events(self, job_id: str) -> AsyncIterator[JobEvent]: ...
