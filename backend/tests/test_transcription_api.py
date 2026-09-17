"""Transcription API with a fake ASR backend (no GPU / model download)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.asr.base import ASRSegment, ASRWord, TranscriptionResult
from app.asr.manager import ASRManager, get_asr_manager
from app.core.errors import AppError, ErrorCode
from app.services.transcription_service import words_in_range
from tests.audio_fixtures import requires_ffmpeg, speechlike, write_wav


class FakeBackend:
    model_name = "fake-whisper"

    def __init__(self) -> None:
        self.installed = True
        self.calls: list[tuple[float, str | None]] = []
        self._loaded = False
        self.fail_with: Exception | None = None

    def package_available(self) -> bool:
        return True

    def model_installed(self) -> bool:
        return self.installed

    def download_model(self) -> None:
        self.installed = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self):
        return "cuda" if self._loaded else None

    @property
    def compute_type(self):
        return "float16" if self._loaded else None

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def transcribe(self, audio_16k, language, initial_prompt=None):
        if self.fail_with:
            raise self.fail_with
        self.load()
        duration = audio_16k.size / 16_000
        self.calls.append((round(duration, 2), language))
        words = [ASRWord(start=0.2 + i, end=0.8 + i, word=f" palabra{i}", probability=0.95)
                 for i in range(int(duration))]
        text = "".join(w.word for w in words).strip()
        return TranscriptionResult(
            text=text, language=language or "es", language_probability=0.98, duration_s=duration,
            segments=[ASRSegment(start=0, end=duration, text=" " + text, avg_logprob=-0.2,
                                 no_speech_prob=0.01, compression_ratio=1.2)],
            words=words, model=self.model_name, device="cuda", compute_type="float16", elapsed_s=0.01)


@pytest.fixture
def fake(app):
    backend = FakeBackend()
    app.dependency_overrides[get_asr_manager] = lambda: manager
    manager = ASRManager(backend)
    yield backend
    app.dependency_overrides.clear()


@pytest.fixture
def reference(client, tmp_path):
    wav = write_wav(tmp_path / "voz.wav", speechlike(words=12))
    with wav.open("rb") as fh:
        res = client.post("/api/references", files={"file": ("voz.wav", fh, "audio/wav")})
    assert res.status_code == 201, res.text
    return res.json()["reference"]


def test_words_in_range_uses_word_midpoints():
    words = [{"start": 0.0, "end": 0.5, "word": " Hola"}, {"start": 0.6, "end": 1.2, "word": " a"},
             {"start": 1.9, "end": 2.6, "word": " todos."}]
    assert words_in_range(words, 0.0, 1.0) == "Hola a"
    assert words_in_range(words, 2.0, 3.0) == "todos."


def test_status_and_languages(client, fake):
    status = client.get("/api/transcription/status").json()
    assert status["installed"] is True and status["loaded"] is False and status["model"] == "fake-whisper"
    langs = {lang["code"]: lang["name"] for lang in client.get("/api/transcription/languages").json()}
    assert langs["es"] == "Español" and langs["ja"] == "Japonés"


@requires_ffmpeg
def test_transcribe_full_reference(client, fake, reference):
    res = client.post(f"/api/transcription/references/{reference['id']}", json={"language": "es"})
    assert res.status_code == 200, res.text
    tr = res.json()
    assert tr["text"].startswith("palabra0") and tr["language"] == "es"
    assert tr["source"] == "asr" and tr["edited"] is False and tr["segment_start_s"] is None
    assert tr["words"] and tr["warnings"] == []
    assert fake.calls[0][1] == "es"
    assert fake.calls[0][0] == pytest.approx(reference["analysis"]["duration_s"], abs=0.05)  # 16 kHz input

    ref = client.get(f"/api/references/{reference['id']}").json()
    assert ref["transcript"]["id"] == tr["id"]
    assert ref["transcript"]["words"] is None  # words only on transcription endpoints
    assert client.get("/api/transcription/status").json()["loaded"] is True


@requires_ffmpeg
def test_cache_avoids_second_inference_unless_forced(client, fake, reference):
    url = f"/api/transcription/references/{reference['id']}"
    first = client.post(url, json={}).json()
    second = client.post(url, json={}).json()
    assert len(fake.calls) == 1 and second["id"] == first["id"]
    client.post(url, json={"force": True})
    assert len(fake.calls) == 2
    client.post(url, json={"language": "en"})  # different language → different cache key
    assert len(fake.calls) == 3


@requires_ffmpeg
def test_segment_transcription_offsets_words(client, fake, reference):
    rid = reference["id"]
    url = f"/api/transcription/references/{rid}"
    assert client.post(url, json={"scope": "segment"}).status_code == 422  # no saved segment yet
    client.post(url, json={})
    client.patch(f"/api/references/{rid}", json={"segment_start_s": 2.0, "segment_end_s": 5.0})

    ref = client.get(f"/api/references/{rid}").json()
    assert ref["segment_transcript"] is None
    assert ref["segment_text_estimate"] == "palabra2 palabra3 palabra4"

    seg = client.post(url, json={"scope": "segment"}).json()
    assert (seg["segment_start_s"], seg["segment_end_s"]) == (2.0, 5.0)
    assert fake.calls[-1][0] == pytest.approx(3.0, abs=0.01)
    assert seg["words"][0]["start"] == pytest.approx(2.2)  # shifted to reference timeline
    ref = client.get(f"/api/references/{rid}").json()
    assert ref["segment_transcript"]["id"] == seg["id"] and ref["transcript"]["id"] != seg["id"]


@requires_ffmpeg
def test_manual_edit_and_revert(client, fake, reference):
    tr = client.post(f"/api/transcription/references/{reference['id']}", json={}).json()
    edited = client.patch(f"/api/transcription/transcripts/{tr['id']}",
                          json={"text": "  Hola a todos, hoy hablamos de IA.  "}).json()
    assert edited["text"] == "Hola a todos, hoy hablamos de IA."
    assert edited["source"] == "manual" and edited["edited"] is True
    assert client.patch(f"/api/transcription/transcripts/{tr['id']}", json={"text": ""}).status_code == 422
    reverted = client.post(f"/api/transcription/transcripts/{tr['id']}/revert").json()
    assert reverted["text"] == tr["text"] and reverted["edited"] is False and reverted["source"] == "asr"
    # re-transcribing replaces the manual text with the fresh ASR output
    client.patch(f"/api/transcription/transcripts/{tr['id']}", json={"text": "manual"})
    again = client.post(f"/api/transcription/references/{reference['id']}", json={"force": True}).json()
    assert again["id"] == tr["id"] and again["source"] == "asr"


@requires_ffmpeg
def test_model_not_installed_and_download(client, fake, reference):
    fake.installed = False
    res = client.post(f"/api/transcription/references/{reference['id']}", json={})
    assert res.status_code == 409 and res.json()["error_code"] == "ASR_MODEL_NOT_INSTALLED"
    assert client.get("/api/transcription/status").json()["installed"] is False
    assert client.post("/api/transcription/model/download").status_code == 202
    for _ in range(50):
        if client.get("/api/transcription/status").json()["installed"]:
            break
    assert client.get("/api/transcription/status").json()["installed"] is True


@requires_ffmpeg
def test_validation_and_errors(client, fake, reference):
    url = f"/api/transcription/references/{reference['id']}"
    assert client.post(url, json={"language": "klingon"}).json()["error_code"] == "VALIDATION_ERROR"
    assert client.post("/api/transcription/references/nope", json={}).status_code == 404
    fake.fail_with = RuntimeError("CUDA out of memory")
    res = client.post(url, json={"force": True})
    assert res.status_code == 507 and res.json()["error_code"] == "GPU_MEMORY_ERROR"
    fake.fail_with = RuntimeError("boom")
    res = client.post(url, json={"force": True})
    assert res.status_code == 500 and res.json()["error_code"] == "ASR_ERROR"
    assert "boom" not in res.text


@requires_ffmpeg
def test_unload_and_delete_reference_removes_transcripts(client, fake, reference):
    client.post(f"/api/transcription/references/{reference['id']}", json={})
    assert client.post("/api/transcription/model/unload").json()["loaded"] is False
    client.delete(f"/api/references/{reference['id']}")
    assert client.get(f"/api/transcription/references/{reference['id']}").status_code == 404


def test_manager_rejects_when_package_missing():
    class Missing(FakeBackend):
        def package_available(self) -> bool:
            return False

    with pytest.raises(AppError) as exc:
        ASRManager(Missing()).transcribe(np.zeros(16000, dtype=np.float32), None)
    assert exc.value.code is ErrorCode.ASR_NOT_AVAILABLE


def test_sqlite_schema_sync_adds_new_columns(tmp_path):
    import sqlite3

    from sqlalchemy import inspect

    from app.core.config import Settings
    from app.core.database import init_engine

    db = tmp_path / "database" / "voicelab.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE transcript (id VARCHAR PRIMARY KEY, reference_id VARCHAR, text VARCHAR, "
                "language VARCHAR, source VARCHAR, asr_model VARCHAR, word_timestamps JSON, "
                "created_at DATETIME, updated_at DATETIME)")
    con.commit()
    con.close()
    engine = init_engine(Settings(_env_file=None, data_dir=tmp_path))
    cols = {c["name"] for c in inspect(engine).get_columns("transcript")}
    assert {"segment_start_s", "segment_end_s", "asr_text", "asr_metadata"} <= cols


@pytest.mark.skipif(os.environ.get("VOICELAB_GPU_TESTS") != "1", reason="Requiere modelo real (VOICELAB_GPU_TESTS=1)")
def test_real_faster_whisper_smoke():
    from app.asr.faster_whisper_backend import FasterWhisperBackend

    backend = FasterWhisperBackend(os.environ.get("VOICELAB_ASR_MODEL", "large-v3-turbo"))
    assert backend.model_installed()
    result = backend.transcribe(np.zeros(16_000 * 2, dtype=np.float32), language="es")
    assert result.language == "es" and result.device in ("cuda", "cpu")
    backend.unload()


@requires_ffmpeg
def test_segment_snaps_to_word_pauses_when_requested(client, fake, reference):
    rid = reference["id"]
    url = f"/api/references/{rid}"
    unsnapped = client.patch(url, json={"segment_start_s": 2.5, "segment_end_s": 5.6, "snap_to_words": True}).json()
    assert (unsnapped["segment_start_s"], unsnapped["segment_end_s"]) == (2.5, 5.6)  # no transcript yet
    client.post(f"/api/transcription/references/{rid}", json={})
    snapped = client.patch(url, json={"segment_start_s": 2.5, "segment_end_s": 5.6, "snap_to_words": True}).json()
    assert (snapped["segment_start_s"], snapped["segment_end_s"]) == (2.0, 6.0)
    raw = client.patch(url, json={"segment_start_s": 2.5, "segment_end_s": 5.6}).json()
    assert (raw["segment_start_s"], raw["segment_end_s"]) == (2.5, 5.6)
