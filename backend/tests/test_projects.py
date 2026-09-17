"""Projects: script import, segment editing/reordering, stale detection, batch generation and export (no GPU)."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from app.services.project_service import split_script
from tests.audio_fixtures import requires_ffmpeg
from tests.test_advanced_generation import engines as engines  # noqa: F401  (fixture re-export)
from tests.test_generation import wait_done

SCRIPT = """Bienvenidos al canal. Hoy hablamos de voces.

Primero, las referencias. Después, la generación.
¡Y al final, la exportación!"""


def _project(client, **settings):
    body = {"name": "Vídeo de prueba", "settings": {"engine": "instructmock", "params": {"seed": 10},
                                                   "default_pause_ms": 300, **settings}}
    res = client.post("/api/projects", json=body)
    assert res.status_code == 201, res.text
    return res.json()


def _wait_project(client, project_id):
    detail = client.get(f"/api/projects/{project_id}").json()
    for seg in detail["segments"]:
        if seg["generation"]:
            wait_done(client, seg["generation"]["id"])
    return client.get(f"/api/projects/{project_id}").json()


def test_split_script_by_paragraphs_and_sentences():
    assert split_script(SCRIPT, "paragraphs") == [
        "Bienvenidos al canal. Hoy hablamos de voces.",
        "Primero, las referencias. Después, la generación.",
        "¡Y al final, la exportación!",
    ]
    assert split_script(SCRIPT, "sentences") == [
        "Bienvenidos al canal.", "Hoy hablamos de voces.", "Primero, las referencias.", "Después, la generación.",
        "¡Y al final, la exportación!",
    ]
    long = " ".join(["Una frase muy larga sin puntos que sigue y sigue"] * 10)
    assert all(len(chunk) <= 280 for chunk in split_script(long, "sentences"))


def test_segments_crud_and_reorder(client, engines):  # noqa: F811
    project = _project(client)
    detail = client.post(f"/api/projects/{project['id']}/import", json={"text": SCRIPT, "split": "sentences"}).json()
    ids = [s["id"] for s in detail["segments"]]
    assert len(ids) == 5 and [s["status"] for s in detail["segments"]] == ["empty"] * 5
    assert detail["segments"][0]["effective_pause_ms"] == 300

    detail = client.post(f"/api/projects/{project['id']}/segments",
                         json={"segments": [{"text": "Intro"}], "position": 0}).json()
    assert [s["text"] for s in detail["segments"]][:2] == ["Intro", "Bienvenidos al canal."]

    new_order = [s["id"] for s in detail["segments"]][::-1]
    detail = client.put(f"/api/projects/{project['id']}/segments/order", json={"ids": new_order}).json()
    assert [s["id"] for s in detail["segments"]] == new_order and [s["position"] for s in detail["segments"]] == \
        list(range(6))
    assert client.put(f"/api/projects/{project['id']}/segments/order", json={"ids": new_order[:2]}).status_code == 422

    seg = detail["segments"][0]
    detail = client.patch(f"/api/projects/{project['id']}/segments/{seg['id']}",
                          json={"text": "Nuevo texto", "pause_after_ms": 1200, "emotion": "happy"}).json()
    edited = detail["segments"][0]
    assert (edited["text"], edited["effective_pause_ms"], edited["emotion"]) == ("Nuevo texto", 1200, "happy")
    assert client.patch(f"/api/projects/{project['id']}/segments/{seg['id']}",
                        json={"emotion": "furiosa"}).status_code == 422

    detail = client.delete(f"/api/projects/{project['id']}/segments/{seg['id']}").json()
    assert len(detail["segments"]) == 5 and [s["position"] for s in detail["segments"]] == list(range(5))

    summary = client.get("/api/projects").json()[0]
    assert (summary["name"], summary["segments"], summary["pending"]) == ("Vídeo de prueba", 5, 5)
    assert client.get("/api/projects/nope").json()["error_code"] == "PROJECT_NOT_FOUND"


@requires_ffmpeg
def test_generate_detect_stale_and_export(client, engines, settings):  # noqa: F811
    project = _project(client)
    pid = project["id"]
    client.post(f"/api/projects/{pid}/import", json={"text": SCRIPT, "split": "paragraphs"})

    assert client.get(f"/api/projects/{pid}/export").json()["error_code"] == "PROJECT_NOT_READY"

    detail = client.post(f"/api/projects/{pid}/generate", json={}).json()
    assert all(s["status"] in ("queued", "generating", "ready") for s in detail["segments"])
    detail = _wait_project(client, pid)
    assert [s["status"] for s in detail["segments"]] == ["ready"] * 3
    assert detail["total_duration_s"] > 0
    assert client.get("/api/generation").json() == []  # project generations stay out of the Generate history

    first_gen = detail["segments"][0]["generation"]["id"]
    again = client.post(f"/api/projects/{pid}/generate", json={}).json()  # nothing pending
    assert again["segments"][0]["generation"]["id"] == first_gen

    # editing the text (or a project default) makes the audio stale instead of silently exporting old audio
    seg = detail["segments"][1]
    detail = client.patch(f"/api/projects/{pid}/segments/{seg['id']}", json={"text": "Texto cambiado."}).json()
    assert detail["segments"][1]["status"] == "stale"
    assert client.get(f"/api/projects/{pid}/export").json()["details"]["segmentos"] == [2]
    assert client.get(f"/api/projects/{pid}/export?allow_partial=true").status_code == 200

    detail = client.post(f"/api/projects/{pid}/generate", json={}).json()  # regenerates only the stale one
    assert detail["segments"][0]["generation"]["id"] == first_gen
    detail = _wait_project(client, pid)
    assert [s["status"] for s in detail["segments"]] == ["ready"] * 3

    exported = client.get(f"/api/projects/{pid}/export")
    assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
    audio, sr = sf.read(io.BytesIO(exported.content))
    durations = [s["generation"]["duration_s"] for s in detail["segments"]]
    assert audio.size / sr == pytest.approx(sum(durations) + 2 * 0.3, abs=0.03)  # pauses between segments only

    zipped = zipfile.ZipFile(io.BytesIO(client.get(f"/api/projects/{pid}/export?format=zip").content))
    names = zipped.namelist()
    assert "project.json" in names and "Vídeo_de_prueba.wav" in names and len([n for n in names
                                                                              if n.startswith("segmentos/")]) == 3
    manifest = json.loads(zipped.read("project.json"))
    assert manifest["segments"][0]["seed"] == 10 and manifest["segments"][1]["text"] == "Texto cambiado."

    # mastering on export only affects the exported file
    client.patch(f"/api/projects/{pid}", json={"settings": {**detail["settings"],
                                                            "export_postprocess": {"peak": {"enabled": True,
                                                                                            "target_dbfs": -6}}}})
    mastered, _ = sf.read(io.BytesIO(client.get(f"/api/projects/{pid}/export").content))
    assert 20 * np.log10(np.max(np.abs(mastered))) == pytest.approx(-6, abs=0.1)
    assert [s["status"] for s in client.get(f"/api/projects/{pid}").json()["segments"]] == ["ready"] * 3

    gen_ids = [s["generation"]["id"] for s in detail["segments"]]
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert all(client.get(f"/api/generation/{g}").status_code == 404 for g in gen_ids)
    assert not (settings.data_dir / "projects" / pid).exists()


def test_generation_errors_point_to_the_segment(client, engines):  # noqa: F811
    project = _project(client, engine=None)
    client.post(f"/api/projects/{project['id']}/import", json={"text": "Hola.", "split": "sentences"})
    res = client.post(f"/api/projects/{project['id']}/generate", json={})
    assert res.status_code == 422 and "motor" in res.json()["message"]

    project = _project(client, params={"nope": 1})
    client.post(f"/api/projects/{project['id']}/import", json={"text": "Uno.\nDos.", "split": "paragraphs"})
    res = client.post(f"/api/projects/{project['id']}/generate", json={})
    assert res.status_code == 422 and res.json()["details"]["segmento"] == 1
    assert all(s["status"] == "empty" for s in client.get(f"/api/projects/{project['id']}").json()["segments"])
