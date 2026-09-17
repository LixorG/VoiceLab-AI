from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from app import __version__
from app.api.router import api_router
from app.asr.manager import get_asr_manager
from app.core.config import Settings, get_settings
from app.core.database import get_engine, init_engine
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.engines.manager import get_model_manager
from app.services.checkpoint_service import refresh_store
from app.services.generation_service import get_job_queue, mark_interrupted_generations
from app.services.resources_service import ResourcesService, idle_monitor
from app.web import mount_frontend

logger = logging.getLogger("voicelab")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        settings.ensure_directories()
        settings.apply_process_environment()
        configure_logging(settings.log_level, settings.data_dir / "logs")
        init_engine(settings)
        with Session(get_engine()) as session:
            refresh_store(session)  # custom checkpoints become engine variants
        mark_interrupted_generations()
        queue = get_job_queue()
        await queue.start()
        monitor = asyncio.create_task(idle_monitor(ResourcesService(settings, get_model_manager(), get_asr_manager(),
                                                                    queue)), name="voicelab-idle-monitor")
        logger.info("startup", extra={"env": settings.app_env, "data_dir": str(settings.data_dir)})
        yield
        monitor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor
        await queue.stop()
        get_model_manager().unload(force=True)
        logger.info("shutdown")

    app = FastAPI(
        title="VoiceLab AI API",
        version=__version__,
        description="API local de clonación y generación de voz con motores TTS intercambiables.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    mount_frontend(app, settings.frontend_dist)
    return app


app = create_app()
