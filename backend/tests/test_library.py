"""Library: search, filters, facets, favourites, tags and bulk delete over the «Generar» history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session

from app.core.database import get_engine
from app.models.entities import Experiment, Generation, Project, VoiceProfile
from app.models.enums import JobStatus
from tests.test_generation import engines as engines  # noqa: F401  (fixture re-export)


def _seed() -> dict[str, str]:
    """Writes a small history directly: the library only reads, so no GPU/job is needed."""
    now = datetime.now(UTC).replace(tzinfo=None)
    with Session(get_engine()) as session:
        profile = VoiceProfile(name="Mateo", slug="mateo")
        experiment = Experiment(name="Comparativa", text="x")
        project = Project(name="Vídeo")
        session.add_all([profile, experiment, project])
        session.flush()
        rows = [
            Generation(engine="mock", variant="tone", text="Canción de prueba en español", status=JobStatus.COMPLETED,
                       kind="single", profile_id=profile.id, output_path="generated/a/final.wav", duration_s=2.0,
                       created_at=now, updated_at=now),
            Generation(engine="mock", variant="tone", text="Segundo intento sin acento", status=JobStatus.FAILED,
                       kind="single", created_at=now - timedelta(days=3), updated_at=now - timedelta(days=3)),
            Generation(engine="f5tts", variant="F5TTS_v1_Base", text="Prueba con otro motor",
                       status=JobStatus.COMPLETED, kind="preview", profile_id=profile.id,
                       created_at=now - timedelta(days=10), updated_at=now - timedelta(days=10)),
            # Excluded: they belong to an experiment / a project and have their own pages.
            Generation(engine="mock", text="De un experimento", status=JobStatus.COMPLETED,
                       experiment_id=experiment.id),
            Generation(engine="mock", text="De un proyecto", status=JobStatus.COMPLETED, project_id=project.id),
        ]
        for row in rows:
            session.add(row)
        session.commit()
        return {"profile": profile.id, "a": rows[0].id, "b": rows[1].id, "c": rows[2].id, "exp": rows[3].id}


@pytest.fixture
def history(client):
    return _seed()


def test_lists_only_standalone_generations(client, history):
    page = client.get("/api/library").json()
    assert page["total"] == 3 and page["offset"] == 0
    assert [i["text"] for i in page["items"]] == ["Canción de prueba en español", "Segundo intento sin acento",
                                                  "Prueba con otro motor"]
    first = page["items"][0]
    assert first["profile_name"] == "Mateo" and first["favorite"] is False and first["tags"] == []
    assert first["audio_url"].startswith("/api/generation/") and page["items"][1]["audio_url"] is None


def test_search_ignores_case_and_accents(client, history):
    assert client.get("/api/library?q=CANCION").json()["total"] == 1
    assert client.get("/api/library?q=prueba").json()["total"] == 2
    assert client.get("/api/library?q=nada de esto").json()["items"] == []


def test_filters_and_pagination(client, history):
    assert client.get("/api/library?engine=f5tts").json()["total"] == 1
    assert client.get(f"/api/library?profile_id={history['profile']}").json()["total"] == 2
    assert client.get("/api/library?status=FAILED").json()["total"] == 1
    assert client.get("/api/library?kind=preview").json()["total"] == 1

    since = (datetime.now(UTC) - timedelta(days=5)).date().isoformat()
    recent = client.get(f"/api/library?since={since}").json()
    assert recent["total"] == 2

    page = client.get("/api/library?limit=1&offset=1").json()
    assert page["total"] == 3 and len(page["items"]) == 1 and page["items"][0]["text"] == "Segundo intento sin acento"


def test_favorite_tags_and_facets(client, history):
    item = client.put(f"/api/generation/{history['a']}/favorite", json={"favorite": True}).json()
    assert item["favorite"] is True

    tagged = client.put(f"/api/generation/{history['a']}/tags",
                        json={"tags": ["  Buena  ", "buena", "PARA el vídeo", "x" * 40]}).json()
    assert tagged["tags"] == ["buena", "para el vídeo", "x" * 30]  # trimmed, lowercase, deduped, capped

    assert client.get("/api/library?favorite=true").json()["total"] == 1
    assert client.get("/api/library?tag=buena").json()["total"] == 1
    assert client.get("/api/library?tag=otra").json()["total"] == 0

    facets = client.get("/api/library/facets").json()
    assert facets["total"] == 3 and facets["favorites"] == 1
    assert [t["tag"] for t in facets["tags"]] == ["buena", "para el vídeo", "x" * 30]
    assert [e["id"] for e in facets["engines"]] == ["f5tts", "mock"]
    assert [p["label"] for p in facets["profiles"]] == ["Mateo"]

    cleared = client.put(f"/api/generation/{history['a']}/tags", json={"tags": []}).json()
    assert cleared["tags"] == []
    assert client.put("/api/generation/nope/favorite", json={"favorite": True}).json()["error_code"] == \
        "GENERATION_NOT_FOUND"


def test_bulk_delete_reports_failures(client, history, engines):  # noqa: F811
    res = client.post("/api/library/delete", json={"ids": [history["b"], "missing", history["c"]]})
    assert res.status_code == 200, res.text
    report = res.json()
    assert report["deleted"] == [history["b"], history["c"]] and report["failed"] == ["missing"]
    assert client.get("/api/library").json()["total"] == 1
    assert client.post("/api/library/delete", json={"ids": []}).status_code == 422


def test_generate_history_endpoint_is_unchanged(client, history):
    body = client.get("/api/generation").json()
    assert isinstance(body, list) and len(body) == 3  # experiments and projects still excluded
