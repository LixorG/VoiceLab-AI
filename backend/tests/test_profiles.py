from __future__ import annotations

import io
import json
import zipfile

import pytest

from app.engines.manager import get_model_manager
from app.services.profile_service import slugify
from tests.audio_fixtures import requires_ffmpeg, speechlike, write_wav


@pytest.fixture
def profile(client):
    res = client.post("/api/voices", json={"name": "Mi Voz Cálida", "language": "Español", "default_engine": "f5tts"})
    assert res.status_code == 201, res.text
    return res.json()


def upload(client, tmp_path, name, profile_id=None, words=10, seed=0, noise_db=-70.0):
    wav = write_wav(tmp_path / name, speechlike(words=words, seed=seed, noise_db=noise_db))
    with wav.open("rb") as fh:
        data = {"profile_id": profile_id} if profile_id else None
        res = client.post("/api/references", files={"file": (name, fh, "audio/wav")}, data=data)
    assert res.status_code == 201, res.text
    return res.json()["reference"]


@pytest.mark.parametrize(("name", "slug"), [("Mi Voz Cálida", "mi-voz-calida"), ("  ¡Ñandú 2!  ", "nandu-2"),
                                            ("日本語", "voz")])
def test_slugify(name, slug):
    assert slugify(name) == slug


def test_crud_and_unique_slugs(client, profile, settings):
    assert profile["slug"] == "mi-voz-calida" and profile["stats"]["reference_count"] == 0
    assert (settings.data_dir / "voices" / "mi-voz-calida" / "profile.json").exists()
    dup = client.post("/api/voices", json={"name": "Mi voz cálida"}).json()
    assert dup["slug"] == "mi-voz-calida-2"

    updated = client.patch(f"/api/voices/{profile['id']}",
                           json={"description": "Narración", "name": "Narradora"}).json()
    assert updated["name"] == "Narradora" and updated["slug"] == "mi-voz-calida"  # slug stays stable
    assert client.patch(f"/api/voices/{profile['id']}", json={"default_engine": "xtts"}).status_code == 404
    assert client.post("/api/voices", json={"name": "   "}).status_code == 422
    assert [p["name"] for p in client.get("/api/voices").json()] == ["Mi voz cálida", "Narradora"]

    assert client.delete(f"/api/voices/{profile['id']}").status_code == 204
    assert client.get(f"/api/voices/{profile['id']}").json()["error_code"] == "PROFILE_NOT_FOUND"
    assert not (settings.data_dir / "voices" / "mi-voz-calida" / "profile.json").exists()


def test_emotions_catalog(client):
    emotions = {e["id"]: e["label"] for e in client.get("/api/voices/emotions").json()}
    assert emotions["excited"] == "Entusiasmado" and len(emotions) == 11


@requires_ffmpeg
def test_references_primary_stats_and_moves(client, profile, tmp_path, settings):
    pid = profile["id"]
    a = upload(client, tmp_path, "a.wav", pid, seed=1)
    b = upload(client, tmp_path, "b.wav", pid, seed=2, noise_db=-30)
    loose = upload(client, tmp_path, "suelta.wav", seed=3)

    detail = client.get(f"/api/voices/{pid}").json()
    assert detail["stats"]["reference_count"] == 2 and detail["stats"]["analyzed_count"] == 2
    assert detail["stats"]["total_speech_s"] > 0 and detail["stats"]["quality_label"]
    assert detail["recommended_reference_id"] == a["id"]  # cleaner one
    assert [r["id"] for r in client.get(f"/api/references?profile_id={pid}").json()] == [a["id"], b["id"]]

    # primary reference must belong to the profile
    assert client.patch(f"/api/voices/{pid}", json={"primary_reference_id": loose["id"]}).status_code == 422
    primary = client.patch(f"/api/voices/{pid}", json={"primary_reference_id": b["id"]}).json()
    assert primary["primary_reference_id"] == b["id"]
    assert client.get(f"/api/references/{b['id']}").json()["is_primary"] is True

    # emotion tags are validated
    assert client.patch(f"/api/references/{a['id']}", json={"emotion_tag": "excited"}).status_code == 200
    assert client.patch(f"/api/references/{a['id']}", json={"emotion_tag": "furioso"}).status_code == 422
    assert client.get(f"/api/voices/{pid}").json()["stats"]["emotions"] == ["excited"]

    # moving the primary reference out clears it; moving into the profile works
    client.patch(f"/api/references/{b['id']}", json={"profile_id": None})
    assert client.get(f"/api/voices/{pid}").json()["primary_reference_id"] is None
    client.patch(f"/api/references/{loose['id']}", json={"profile_id": pid})
    assert client.get(f"/api/voices/{pid}").json()["stats"]["reference_count"] == 2
    manifest = json.loads((settings.data_dir / "voices" / "mi-voz-calida" / "profile.json").read_text("utf-8"))
    assert {r["original_name"] for r in manifest["references"]} == {"a.wav", "suelta.wav"}

    # same audio twice in a profile is rejected
    copy = upload(client, tmp_path, "a.wav", seed=1)  # identical content, no profile
    moved = client.patch(f"/api/references/{copy['id']}", json={"profile_id": pid})
    assert moved.status_code == 409

    # deleting the profile keeps references by default
    assert client.delete(f"/api/voices/{pid}").status_code == 204
    assert client.get(f"/api/references/{a['id']}").json()["profile_id"] is None


def test_recommended_settings_are_validated(client, profile):
    pid = profile["id"]
    bad = client.put(f"/api/voices/{pid}/settings/f5tts", json={"params": {"nfe_steps": 999}})
    assert bad.status_code == 422 and bad.json()["error_code"] == "PARAMETER_ERROR"
    assert client.put(f"/api/voices/{pid}/settings/f5tts", json={"params": {"temperature": 1}}).status_code == 422
    ok = client.put(f"/api/voices/{pid}/settings/f5tts", json={"params": {"nfe_steps": 48, "speed": 0.95},
                                                              "note": "Más pausada"}).json()
    setting = ok["recommended_settings"]["f5tts"]
    assert setting["variant"] == "F5TTS_v1_Base" and setting["params"]["nfe_steps"] == 48
    assert setting["params"]["cfg_strength"] == 2.0  # defaults filled in: stored config is complete
    qwen = client.put(f"/api/voices/{pid}/settings/qwen3tts", json={"variant": "base-0.6b",
                                                                   "params": {"language": "Spanish"}}).json()
    assert set(qwen["recommended_settings"]) == {"f5tts", "qwen3tts"}
    removed = client.delete(f"/api/voices/{pid}/settings/f5tts").json()
    assert set(removed["recommended_settings"]) == {"qwen3tts"}


@requires_ffmpeg
def test_generation_uses_profile_reference(client, app, profile, tmp_path, monkeypatch):
    from app.core.gpu import GPUManager
    from app.engines.manager import ModelManager
    from app.engines.registry import EngineRegistry
    from app.services import generation_service
    from tests.test_generation import RefMock, wait_done

    registry = EngineRegistry()
    registry.register(RefMock())
    manager = ModelManager(registry, GPUManager("cpu"), "cpu")
    app.dependency_overrides[get_model_manager] = lambda: manager
    monkeypatch.setattr(generation_service, "get_model_manager", lambda: manager)
    try:
        pid = profile["id"]
        ref = upload(client, tmp_path, "perfil.wav", pid)
        other = upload(client, tmp_path, "otra.wav", seed=5)
        from tests.test_generation import _add_transcript

        _add_transcript(ref["id"], "texto de la referencia")
        body = {"engine": "refmock", "text": "hola", "profile_id": pid}
        done = wait_done(client, client.post("/api/generation", json=body).json()["job_id"])
        assert done["reference"]["reference_id"] == ref["id"]
        mismatch = client.post("/api/generation", json={**body, "reference_id": other["id"]})
        assert mismatch.status_code == 422
        assert client.post("/api/generation", json={**body, "profile_id": "nope"}).status_code == 404
    finally:
        app.dependency_overrides.clear()


@requires_ffmpeg
def test_export_import_roundtrip(client, profile, tmp_path):
    pid = profile["id"]
    a = upload(client, tmp_path, "a.wav", pid, seed=1)
    upload(client, tmp_path, "b.wav", pid, seed=2)
    client.patch(f"/api/references/{a['id']}", json={"emotion_tag": "calm", "segment_start_s": 0.5,
                                                     "segment_end_s": 3.5})
    client.patch(f"/api/voices/{pid}", json={"primary_reference_id": a["id"]})
    client.put(f"/api/voices/{pid}/settings/f5tts", json={"params": {"nfe_steps": 40}})
    from tests.test_generation import _add_transcript

    _add_transcript(a["id"], "texto del segmento", 0.5, 3.5)

    res = client.get(f"/api/voices/{pid}/export")
    assert res.status_code == 200 and "mi-voz-calida.voiceprofile" in res.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        names = sorted(zf.namelist())
        manifest = json.loads(zf.read("profile.json"))
    assert names[0] == "profile.json" and len(names) == 3
    assert manifest["format"] == "voicelab.voiceprofile" and len(manifest["references"]) == 2

    imported = client.post("/api/voices/import", files={"file": ("perfil.voiceprofile", res.content,
                                                                   "application/zip")})
    assert imported.status_code == 201, imported.text
    new = imported.json()
    assert new["slug"] == "mi-voz-calida-importado" and new["stats"]["reference_count"] == 2
    assert new["name"] == "Mi Voz Cálida (importado)"
    assert new["recommended_settings"]["f5tts"]["params"]["nfe_steps"] == 40
    refs = {r["original_name"]: r for r in client.get(f"/api/references?profile_id={new['id']}").json()}
    assert new["primary_reference_id"] == refs["a.wav"]["id"]
    assert refs["a.wav"]["emotion_tag"] == "calm" and refs["a.wav"]["segment_start_s"] == 0.5
    assert refs["a.wav"]["segment_transcript"]["text"] == "texto del segmento"
    assert refs["a.wav"]["status"] == "ANALYZED"


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _import(client, data: bytes):
    return client.post("/api/voices/import", files={"file": ("x.voiceprofile", data, "application/zip")})


def test_import_rejects_unsafe_or_invalid_archives(client):
    manifest = json.dumps({"format": "voicelab.voiceprofile", "version": 1, "profile": {"name": "X"},
                           "references": []}).encode()
    assert _import(client, b"no es un zip").json()["error_code"] == "PROFILE_IMPORT_ERROR"
    traversal = _import(client, _zip({"profile.json": manifest, "../../evil.py": b"x"}))
    assert traversal.status_code == 422 and "no permitidas" in traversal.json()["message"]
    assert _import(client, _zip({"profile.json": manifest, "references/abc.exe": b"x"})).status_code == 422
    assert _import(client, _zip({"references/" + "a" * 64 + ".wav": b"x"})).status_code == 422  # no manifest
    wrong = json.dumps({"format": "otro", "version": 1}).encode()
    assert "no compatible" in _import(client, _zip({"profile.json": wrong})).json()["message"]
    sha = "a" * 64
    tampered = json.dumps({"format": "voicelab.voiceprofile", "version": 1, "profile": {"name": "X"},
                           "references": [{"file": f"references/{sha}.wav"}]}).encode()
    bad_hash = _import(client, _zip({"profile.json": tampered, f"references/{sha}.wav": b"RIFF"}))
    assert bad_hash.status_code == 422 and "hash" in bad_hash.json()["message"]
    assert client.get("/api/voices").json() == []  # failed imports leave nothing behind
    empty = _import(client, _zip({"profile.json": manifest}))
    assert empty.status_code == 201 and empty.json()["name"] == "X"
