from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("FRONTEND_DIST", "")  # API tests never depend on a local UI build
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _no_real_speaker_model(monkeypatch):
    """Keep the suite hermetic: a real WavLM download on this machine must not be picked up by the tests."""
    from app.evaluation.speaker import SpeakerEncoder, get_speaker_encoder

    monkeypatch.setattr(SpeakerEncoder, "model_installed", lambda self: False)
    yield
    get_speaker_encoder().unload()
