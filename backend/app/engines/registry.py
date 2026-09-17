"""Engine plugin discovery.

Built-in engines live in `app/engines/<name>/plugin.py` and expose `ENGINE_CLASS`. External packages can
register engines through the `voicelab.engines` entry-point group. Discovery only imports lightweight
descriptor modules; model packages are imported by each engine at load time.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import pkgutil
from functools import lru_cache

import app.engines as engines_pkg
from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.engines.base import TTSBackend

logger = logging.getLogger("voicelab.engines")

ENTRY_POINT_GROUP = "voicelab.engines"
DEV_ONLY = {"mock"}
ORDER = ["f5tts", "e2tts", "qwen3tts"]


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, TTSBackend] = {}

    def register(self, engine: TTSBackend) -> None:
        if engine.id in self._engines:
            raise ValueError(f"duplicate engine id: {engine.id}")
        self._engines[engine.id] = engine

    def discover(self, include_dev: bool = False) -> EngineRegistry:
        for module_info in pkgutil.iter_modules(engines_pkg.__path__):
            if not module_info.ispkg or (module_info.name in DEV_ONLY and not include_dev):
                continue
            try:
                module = importlib.import_module(f"app.engines.{module_info.name}.plugin")
            except ModuleNotFoundError as exc:
                if exc.name == f"app.engines.{module_info.name}.plugin":
                    continue  # package without a plugin descriptor
                logger.exception("engine_plugin_import_failed", extra={"plugin": module_info.name})
                continue
            self._register_class(getattr(module, "ENGINE_CLASS", None), module_info.name)

        for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            try:
                self._register_class(ep.load(), ep.name)
            except Exception:
                logger.exception("engine_entry_point_failed", extra={"entry_point": ep.name})
        return self

    def _register_class(self, cls: type[TTSBackend] | None, source: str) -> None:
        if cls is None or not issubclass(cls, TTSBackend):
            logger.warning("engine_plugin_without_class", extra={"plugin": source})
            return
        try:
            self.register(cls())
        except ValueError:
            logger.warning("engine_duplicate_ignored", extra={"plugin": source})

    def all(self) -> list[TTSBackend]:
        rank = {eid: i for i, eid in enumerate(ORDER)}
        return sorted(self._engines.values(), key=lambda e: (rank.get(e.id, len(ORDER)), e.id))

    def get(self, engine_id: str) -> TTSBackend:
        engine = self._engines.get(engine_id)
        if engine is None:
            raise AppError(ErrorCode.ENGINE_NOT_FOUND, status_code=404, details={"motor": engine_id})
        return engine


@lru_cache
def get_engine_registry() -> EngineRegistry:
    return EngineRegistry().discover(include_dev=get_settings().enable_mock_engine)
