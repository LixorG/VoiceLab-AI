"""/api/jobs — job state, cancellation and Server-Sent Events progress stream."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.services.generation_service import get_job_queue
from app.workers.asyncio_queue import AsyncioJobQueue
from app.workers.base import JobState

router = APIRouter(prefix="/jobs", tags=["Tareas"])


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
