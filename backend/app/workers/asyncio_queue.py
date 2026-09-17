"""In-process job queue: one serial GPU worker, progress events for SSE, cooperative cancellation.

Implements the `JobQueue` protocol so it can be swapped for Celery/RQ later without touching the API.
Handlers are synchronous (they run heavy inference) and execute in a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.core.errors import AppError, ErrorCode, classify_exception
from app.engines.base import CancelToken
from app.models.enums import JobStatus
from app.workers.base import JobEvent, JobSpec, JobState

logger = logging.getLogger("voicelab.jobs")

MAX_FINISHED_JOBS = 200


class JobContext:
    """Passed to handlers: report status/progress (thread-safe) and check cancellation."""

    def __init__(self, queue: AsyncioJobQueue, job_id: str, cancel: CancelToken) -> None:
        self._queue = queue
        self.job_id = job_id
        self.cancel = cancel

    def update(self, status: JobStatus | None = None, progress: float | None = None,
               message: str | None = None) -> None:
        self._queue._threadsafe_update(self.job_id, status, progress, message)

    def check_cancelled(self) -> None:
        if self.cancel.cancelled:
            raise AppError(ErrorCode.JOB_CANCELLED, status_code=409)


Handler = Callable[[JobSpec, JobContext], dict[str, Any] | None]
Hook = Callable[[JobState], None]


@dataclass
class _Job:
    spec: JobSpec
    state: JobState
    cancel: CancelToken = field(default_factory=CancelToken)
    subscribers: list[asyncio.Queue[JobEvent | None]] = field(default_factory=list)


class AsyncioJobQueue:
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}
        self._jobs: dict[str, _Job] = {}
        self._pending: asyncio.Queue[str] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self.on_finish: Hook | None = None  # persistence hook (e.g. mark Generation cancelled)

    # ----- lifecycle -----
    def register(self, kind: str, handler: Handler) -> None:
        self._handlers[kind] = handler

    async def start(self) -> None:
        if self._worker is None:
            self._loop = asyncio.get_running_loop()
            self._pending = asyncio.Queue()
            self._worker = asyncio.create_task(self._run(), name="voicelab-gpu-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            for job in self._jobs.values():
                job.cancel.cancel()
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    # ----- JobQueue protocol -----
    async def submit(self, spec: JobSpec, job_id: str | None = None) -> str:
        if spec.kind not in self._handlers:
            raise ValueError(f"no handler for job kind {spec.kind}")
        await self.start()
        job_id = job_id or uuid4().hex
        self._jobs[job_id] = _Job(spec=spec, state=JobState(job_id=job_id, kind=spec.kind, status=JobStatus.QUEUED,
                                                            message="En cola"))
        self._prune()
        assert self._pending is not None
        await self._pending.put(job_id)
        self._publish(job_id)
        return job_id

    async def cancel(self, job_id: str) -> None:
        job = self._get(job_id)
        if job.state.status.is_terminal:
            return
        job.cancel.cancel()
        if job.state.status == JobStatus.QUEUED:  # never started: finish immediately
            self._finish(job_id, JobStatus.CANCELLED, message="Cancelada", error_code=ErrorCode.JOB_CANCELLED.value)

    async def get(self, job_id: str) -> JobState:
        return self._get(job_id).state.model_copy()

    async def events(self, job_id: str) -> AsyncIterator[JobEvent]:
        job = self._get(job_id)
        queue: asyncio.Queue[JobEvent | None] = asyncio.Queue()
        yield self._event(job.state)
        if job.state.status.is_terminal:
            return
        job.subscribers.append(queue)
        try:
            while (event := await queue.get()) is not None:
                yield event
                if event.status.is_terminal:
                    break
        finally:
            if queue in job.subscribers:
                job.subscribers.remove(queue)

    def counts(self) -> tuple[int, int]:
        """(running, waiting) jobs."""
        running = sum(1 for j in self._jobs.values() if not j.state.status.is_terminal
                      and j.state.status != JobStatus.QUEUED)
        waiting = sum(1 for j in self._jobs.values() if j.state.status == JobStatus.QUEUED)
        return running, waiting

    def queued_position(self, job_id: str) -> int:
        queued = [jid for jid, j in self._jobs.items() if j.state.status == JobStatus.QUEUED]
        return queued.index(job_id) + 1 if job_id in queued else 0

    # ----- internals -----
    def _get(self, job_id: str) -> _Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise AppError(ErrorCode.JOB_NOT_FOUND, status_code=404)
        return job

    async def _run(self) -> None:
        assert self._pending is not None
        while True:
            job_id = await self._pending.get()
            job = self._jobs.get(job_id)
            if job is None or job.state.status.is_terminal:
                continue
            ctx = JobContext(self, job_id, job.cancel)
            try:
                result = await asyncio.to_thread(self._handlers[job.spec.kind], job.spec, ctx)
            except AppError as exc:
                status = JobStatus.CANCELLED if exc.code is ErrorCode.JOB_CANCELLED else JobStatus.FAILED
                self._finish(job_id, status, message=exc.message, error_code=exc.code.value)
            except Exception as exc:  # never let the worker die
                code = classify_exception(exc)
                logger.exception("job_failed", extra={"job_id": job_id, "kind": job.spec.kind})
                from app.core.errors import MESSAGES

                self._finish(job_id, JobStatus.FAILED, message=MESSAGES[code], error_code=code.value)
            else:
                self._finish(job_id, JobStatus.COMPLETED, message="Completada", result=result or {})

    def _finish(self, job_id: str, status: JobStatus, message: str | None = None, error_code: str | None = None,
                result: dict[str, Any] | None = None) -> None:
        job = self._jobs[job_id]
        job.state.status = status
        job.state.message = message
        job.state.error_code = error_code
        job.state.result = result
        if status == JobStatus.COMPLETED:
            job.state.progress = 1.0
        if self.on_finish:
            try:
                self.on_finish(job.state.model_copy())
            except Exception:
                logger.exception("job_finish_hook_failed", extra={"job_id": job_id})
        self._publish(job_id)

    def _threadsafe_update(self, job_id: str, status: JobStatus | None, progress: float | None,
                           message: str | None) -> None:
        def apply() -> None:
            job = self._jobs.get(job_id)
            if job is None or job.state.status.is_terminal:
                return
            if status is not None:
                job.state.status = status
            if progress is not None:
                job.state.progress = max(0.0, min(1.0, progress))
            if message is not None:
                job.state.message = message
            self._publish(job_id)

        if self._loop is not None:
            self._loop.call_soon_threadsafe(apply)

    @staticmethod
    def _event(state: JobState) -> JobEvent:
        return JobEvent(job_id=state.job_id, status=state.status, progress=state.progress, message=state.message)

    def _publish(self, job_id: str) -> None:
        job = self._jobs[job_id]
        event = self._event(job.state)
        for sub in list(job.subscribers):
            sub.put_nowait(event)

    def _prune(self) -> None:
        finished = [jid for jid, j in self._jobs.items() if j.state.status.is_terminal]
        for jid in finished[: max(0, len(finished) - MAX_FINISHED_JOBS)]:
            del self._jobs[jid]
