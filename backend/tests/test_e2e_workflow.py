"""Integration: the whole user journey through the HTTP API, with simulated engines and ASR (no GPU).

upload references → transcribe + edit → voice profile (emotions, primary, recommended settings) → plan with markup
→ segmented generation with post-processing → A/B original vs processed → variations → experiment with two engines
→ automatic evaluation + rating → export/import profile → delete everything → no orphaned files left.
"""

from __future__ import annotations

import io
import os
import time
import zipfile

import numpy as np
import soundfile as sf

from app.asr.manager import ASRManager, get_asr_manager
from tests.audio_fixtures import requires_ffmpeg, speechlike, write_wav
from tests.test_advanced_generation import engines as engines  # noqa: F401  (fixture re-export)
from tests.test_generation import wait_done
from tests.test_transcription_api import FakeBackend


def _upload(client, tmp_path, name, profile_id, seed):
    wav = write_wav(tmp_path / name, speechlike(words=8, seed=seed))
    with wav.open("rb") as fh:
        res = client.post("/api/references", files={"file": (name, fh, "audio/wav")}, data={"profile_id": profile_id})
    assert res.status_code == 201, res.text
    return res.json()["reference"]


@requires_ffmpeg
def test_full_user_journey(client, app, engines, tmp_path, settings):  # noqa: F811
    asr_backend = FakeBackend()
    app.dependency_overrides[get_asr_manager] = lambda: ASRManager(asr_backend)

    # 1. profile + references + transcripts ------------------------------------------------------------------
    profile = client.post("/api/voices", json={"name": "Narradora", "language": "Español"}).json()
    neutral = _upload(client, tmp_path, "neutral.wav", profile["id"], 1)
    happy = _upload(client, tmp_path, "feliz.wav", profile["id"], 2)
    for ref, text in ((neutral, "texto neutral de referencia"), (happy, "texto alegre de referencia")):
        transcript = client.post(f"/api/transcription/references/{ref['id']}", json={"language": "es"}).json()
        assert transcript["source"] == "asr"
        edited = client.patch(f"/api/transcription/transcripts/{transcript['id']}", json={"text": text}).json()
        assert edited["text"] == text and edited["source"] == "manual"
    client.patch(f"/api/references/{happy['id']}", json={"emotion_tag": "happy"})
    client.patch(f"/api/voices/{profile['id']}", json={"primary_reference_id": neutral["id"]})
    saved = client.put(f"/api/voices/{profile['id']}/settings/segmock",
                       json={"variant": None, "params": {"seed": 5}}).json()
    assert saved["recommended_settings"]["segmock"]["params"]["seed"] == 5
    detail = client.get(f"/api/voices/{profile['id']}").json()
    assert detail["stats"]["transcribed_count"] == 2 and detail["stats"]["emotions"] == ["happy"]

    # 2. plan + segmented generation with post-processing -------------------------------------------------------
    body = {"engine": "segmock", "profile_id": profile["id"], "params": {"seed": 5},
            "text": "Buenos días. [emoción:feliz]¡Qué alegría![/emoción] [pausa:400ms] Hasta luego.",
            "postprocess": {"crossfade_ms": 80, "peak": {"enabled": True, "target_dbfs": -3},
                            "fades": {"enabled": True}}}
    plan = client.post("/api/generation/plan", json=body).json()
    assert [s["reference_name"] for s in plan["segments"]] == ["neutral.wav", "feliz.wav", "neutral.wav"]
    gen = wait_done(client, client.post("/api/generation", json=body).json()["job_id"])
    assert gen["status"] == "COMPLETED" and len(gen["segments"]) == 3
    steps = [s["id"] for s in gen["postprocess"]["report"]["steps"]]
    assert steps == ["crossfade", "peak", "fades"]
    final, _ = sf.read(io.BytesIO(client.get(gen["audio_url"]).content))
    raw, _ = sf.read(io.BytesIO(client.get(gen["raw_audio_url"]).content))
    assert abs(20 * np.log10(np.max(np.abs(final))) + 3) < 0.1
    assert raw.size != final.size or not np.allclose(raw, final)  # original kept apart from the processed file

    # 3. variations -----------------------------------------------------------------------------------------------
    variations = client.post("/api/generation/variations",
                             json={"engine": "instructmock", "text": "Hola", "count": 2}).json()
    finished = [wait_done(client, v["job_id"]) for v in variations]
    assert len({g["seed"] for g in finished}) == 2

    # 4. experiment + evaluation + rating ----------------------------------------------------------------------
    exp = client.post("/api/experiments", json={
        "name": "Comparativa", "text": "Hola a todos", "profile_id": profile["id"],
        "arms": [{"engine": "segmock", "params": {"seed": 1}}, {"engine": "instructmock", "params": {"seed": 1}}],
    }).json()
    for g in exp["generations"]:
        wait_done(client, g["id"])
    evaluated = client.post(f"/api/experiments/{exp['id']}/evaluate").json()
    assert all(g["evaluation"]["metrics"] for g in evaluated["generations"])
    arm = evaluated["generations"][0]
    rated = client.put(f"/api/generation/{arm['id']}/rating", json={"timbre": 5, "notes": "La mejor"}).json()
    assert rated["rating"]["timbre"] == 5
    assert {g["id"] for g in client.get("/api/generation?limit=50").json()}.isdisjoint(
        {g["id"] for g in exp["generations"]})

    # 5. export → import the profile -------------------------------------------------------------------------------
    exported = client.get(f"/api/voices/{profile['id']}/export")
    assert exported.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(exported.content)).namelist()
    assert "profile.json" in names and sum(n.startswith("references/") for n in names) == 2
    imported = client.post("/api/voices/import", files={"file": ("n.voiceprofile", exported.content,
                                                                 "application/zip")}).json()
    assert imported["name"] == "Narradora (importado)" and imported["stats"]["transcribed_count"] == 2
    assert imported["recommended_settings"]["segmock"]["params"]["seed"] == 5

    # 6. delete everything; storage has nothing orphaned ---------------------------------------------------------
    assert client.delete(f"/api/experiments/{exp['id']}").status_code == 204
    for g in [gen, *finished]:
        assert client.delete(f"/api/generation/{g['id']}").status_code == 204
    for p in (profile, imported):
        assert client.delete(f"/api/voices/{p['id']}?delete_references=true").status_code == 204
    assert client.get("/api/references").json() == []
    assert client.get("/api/voices").json() == []

    old = time.time() - 3600
    for path in (settings.data_dir / "references").rglob("*"):
        if path.is_file():
            os.utime(path, (old, old))  # past the orphan age threshold, so leftovers would be reported
    usage = client.get("/api/system/storage").json()
    assert usage["orphans_files"] == 0, usage
    assert not any((settings.data_dir / "generated").iterdir())
