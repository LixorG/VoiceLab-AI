"""The backend serves the built UI (single port) without shadowing the API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def _dist(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>VoiceLab AI</title>", encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    return dist


def test_serves_spa_and_keeps_api(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTEND_DIST", str(_dist(tmp_path)))
    with TestClient(create_app(Settings())) as client:
        index = client.get("/")
        assert index.status_code == 200 and "VoiceLab AI" in index.text
        deep = client.get("/voces/perfil-1")  # client-side route → index.html
        assert deep.status_code == 200 and "VoiceLab AI" in deep.text and deep.headers["cache-control"] == "no-cache"
        asset = client.get("/assets/index-abc123.js")
        assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
        assert client.get("/api/system/health").json()["status"] == "ok"
        missing = client.get("/api/does-not-exist")
        assert missing.status_code == 404 and missing.json()["error_code"] == "NOT_FOUND"


def test_no_ui_without_build(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "missing"))
    with TestClient(create_app(Settings())) as client:
        assert client.get("/").status_code == 404
        assert client.get("/api/system/health").status_code == 200
    assert Settings(frontend_dist="").frontend_dist is None
