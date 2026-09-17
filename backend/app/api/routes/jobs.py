"""/api/jobs — job state, cancellation and Server-Sent Events progress stream."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlmodel import Session, col, select

from app.core.database import get_session
from app.models.entities import Generation
from app.schemas.jobs import CancelReport, QueuedJob
from app.services.generation_service import get_job_queue
from app.workers.asyncio_queue import AsyncioJobQueue
from app.workers.base import JobState

router = APIRouter(prefix="/jobs", tags=["Tareas"])


@router.get("", response_model=list[QueuedJob], summary="Tareas en curso y en cola")
async def list_jobs(session: Session = Depends(get_session),
                    queue: AsyncioJobQueue = Depends(get_job_queue)) -> list[QueuedJob]:
    jobs = queue.snapshot()
    texts = {g.id: (g.text, g.engine, g.variant, g.kind)
             for g in session.exec(select(Generation).where(
                 col(Generation.id).in_([state.job_id for state, _ in jobs])))} if jobs else {}
    out = []
    for state, position in jobs:
        text, engine, variant, kind = texts.get(state.job_id, (None, None, None, None))
        out.append(QueuedJob(job_id=state.job_id, kind=state.kind, status=state.status, progress=state.progress,
                             message=state.message, position=position, text=text, engine=engine, variant=variant,
                             generation_kind=kind))
    return out


@router.post("/cancel", response_model=CancelReport, summary="Cancelar las tareas pendientes")
async def cancel_all(only_queued: bool = Query(default=False, description="No tocar la que ya está generando"),
                     queue: AsyncioJobQueue = Depends(get_job_queue)) -> CancelReport:
    return CancelReport(cancelled=await queue.cancel_all(include_running=not only_queued))


@router.get("/{job_id}", response_model=JobState, summary="Estado de una tarea")
async def get_job(job_id: str, queue: AsyncioJobQueue = Depends(get_job_queue)) -> JobState:
    return await queue.get(job_id)


@router.post("/{job_id}/cancel", response_model=JobState, summary="Cancelar una tarea")
async def cancel_job(job_id: str, queue: AsyncioJobQueue = Depends(get_job_queue)) -> JobState:
    await queue.cancel(job_id)
    return await queue.get(job_id)


@router.get("/{job_id}/events", summary="Progreso en tiempo real (SSE)",
            response_class=StreamingResponse,
            responses={200: {"content": {"text/event-stream": {}}}})
async def job_events(job_id: str, queue: AsyncioJobQueue = Depends(get_job_queue)) -> StreamingResponse:
    await queue.get(job_id)  # 404 before opening the stream

    async def stream() -> AsyncIterator[str]:
        yield "retry: 3000\n\n"
        async for event in queue.events(job_id):
            yield f"event: progress\ndata: {json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
