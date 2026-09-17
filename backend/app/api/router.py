from fastapi import APIRouter

from app.api.routes import (
    audio,
    experiments,
    generation,
    jobs,
    library,
    models,
    projects,
    pronunciation,
    references,
    system,
    transcription,
    voices,
)

api_router = APIRouter(prefix="/api")
for module in (system, models, voices, references, transcription, generation, experiments, jobs, library, projects,
               pronunciation, audio):
    api_router.include_router(module.router)
