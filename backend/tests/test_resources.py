"""Memory management (model leases, idle unload, OOM retry, preload) and disk cleanup — no GPU needed."""

from __future__ import annotations

import os
import time

import pytest

from app.asr.manager import ASRManager, get_asr_manager
from app.core.errors import AppError, ErrorCode
from app.core.gpu import GPUManager
from app.engines import manager as manager_module
from app.engines.manager import ModelManager
from app.engines.mock.plugin import MockBackend
from app.engines.registry import EngineRegistry
from app.services.resources_service import ResourcesService
from tests.audio_fixtures import requires_ffmpeg, speechlike, write_wav
from tests.test_generation import _add_transcript, wait_done, wait_status, wait_until
from tests.test_generation import engines as engines  # noqa: F401  (fixture re-export)


class FakeASRBackend:
    model_name = "fake"

    def __init__(self) -> None:
        self.loaded = True
        self.device, self.compute_type = "cpu", None
        self.unloads = 0

    def package_available(self):
        return True

    def model_installed(self):
        return True

    def unload(self):
        self.loaded = False
        self.unloads += 1


class FakeQueue:
    def __init__(self, running=0, waiting=0):
        self.running, self.waiting = running, waiting

    def counts(self):
        return self.running, self.waiting


@pytest.fixture
def manager(monkeypatch):
    registry = EngineRegistry()
    registry.register(MockBackend())
    asr = ASRManager(FakeASRBackend())
    monkeypatch.setattr(manager_module, "get_asr_manager", lambda: asr)
    return ModelManager(registry, GPUManager("cpu"), "cpu"), asr


# ---------------------------------------------------------------- model manager
def test_model_in_use_cannot_be_unloaded(manager):
    models, _ = manager
    with models.use("mock", "") as engine:
        assert engine.id == "mock" and models.loaded.in_use
        with pytest.raises(AppError) as err:
            models.unload()
        assert err.value.code is ErrorCode.MODEL_BUSY
        assert not models.unload_if_idle(0)
    assert not models.loaded.in_use and models.loaded.last_used is not None
    models.unload()
    assert models.loaded is None


def test_idle_unload_waits_for_idle_time(manager):
    models, _ = manager
    with models.use("mock", ""):
        pass
    assert not models.unload_if_idle(60)
    models._loaded.last_used = time.time() - 120
    assert models.unload_if_idle(60) and models.loaded is None


def test_memory_retry_frees_asr_and_retries_once(manager):
    models, asr = manager
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
        return "ok"

    assert models.run_with_memory_retry(flaky) == "ok" and len(calls) == 2 and asr.backend.unloads == 1

    def always_oom():
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(AppError) as err:
        models.run_with_memory_retry(always_oom)
    assert err.value.code is ErrorCode.GPU_MEMORY_ERROR and err.value.status_code == 507

    with pytest.raises(ValueError):
        models.run_with_memory_retry(lambda: (_ for _ in ()).throw(ValueError("otro error")))


def test_asr_idle_unload_skips_running_transcription(manager):
    _, asr = manager
    asr.last_used = time.time() - 700
    asr._lock.acquire()
    try:
        assert not asr.unload_if_idle(600)  # transcribing right now
    finally:
        asr._lock.release()
    assert asr.unload_if_idle(600) and not asr.backend.loaded


def test_idle_policy_does_nothing_while_jobs_are_pending(manager, settings):
    models, asr = manager
    with models.use("mock", ""):
        pass
    models._loaded.last_used = time.time() - 10_000
    asr.last_used = time.time() - 10_000
    busy = ResourcesService(settings, models, asr, FakeQueue(waiting=1))
    assert busy.unload_idle() == [] and models.loaded is not None
    idle = ResourcesService(settings, models, asr, FakeQueue())
    assert idle.unload_idle() == ["tts", "asr"] and models.loaded is None and not asr.backend.loaded


# ---------------------------------------------------------------- API
def test_memory_endpoints(client, engines, app):  # noqa: F811
    asr = ASRManager(FakeASRBackend())
    app.dependency_overrides[get_asr_manager] = lambda: asr

    status = client.get("/api/system/memory").json()
    assert status["tts"] is None and status["asr"]["loaded"] is True
    assert status["idle_unload_minutes"] == {"tts": 15, "asr": 10}
    assert status["queue_active"] == 0 and status["queue_waiting"] == 0

    done = wait_done(client, client.post("/api/generation", json={"engine": "mock", "text": "Hola"}).json()["job_id"])
    status = client.get("/api/system/memory").json()
    assert done["status"] == "COMPLETED" and status["tts"]["engine"] == "mock" and not status["tts"]["in_use"]
    assert 800 < status["tts"]["idle_unload_in_s"] <= 900

    released = client.post("/api/system/memory/release", json={"tts": True, "asr": True}).json()
    assert set(released["released"]) == {"tts", "asr"} and released["status"]["tts"] is None


@requires_ffmpeg
def test_unload_refused_during_generation(client, engines, tmp_path):  # noqa: F811
    models, _mock, refmock = engines
    refmock.release.clear()
    wav = write_wav(tmp_path / "r.wav", speechlike(words=8))
    with wav.open("rb") as fh:
        ref = client.post("/api/references", files={"file": ("r.wav", fh, "audio/wav")}).json()["reference"]
    _add_transcript(ref["id"], "hola a todos")
    gen_id = client.post("/api/generation", json={"engine": "refmock", "text": "LENTO uno",
                                                  "reference_id": ref["id"]}).json()["job_id"]
    wait_status(client, gen_id, "GENERATING")
    res = client.post("/api/models/unload")
    assert res.status_code == 409 and res.json()["error_code"] == "MODEL_BUSY"
    assert client.get("/api/system/memory").json()["tts"]["in_use"] is True
    refmock.release.set()
    assert wait_done(client, gen_id)["status"] == "COMPLETED"
    assert client.post("/api/models/unload").status_code == 204 and models.loaded is None


def test_preload_goes_through_the_queue(client, engines):  # noqa: F811
    models, *_ = engines
    job = client.post("/api/models/mock/load").json()

    def finished():
        state = client.get(f"/api/jobs/{job['job_id']}").json()
        return state if state["status"] in ("COMPLETED", "FAILED") else None

    state = wait_until(finished)
    assert state["status"] == "COMPLETED" and models.loaded.engine == "mock" and not models.loaded.in_use
    assert client.post("/api/models/planned/load").json()["error_code"] == "NOT_IMPLEMENTED"


# ---------------------------------------------------------------- storage
def _old(path, age_s=3600):
    past = time.time() - age_s
    os.utime(path, (past, past))
    return path


def test_storage_cleanup_only_removes_unreferenced_old_files(client, settings, engines):  # noqa: F811
    data = settings.data_dir
    done = wait_done(client, client.post("/api/generation", json={"engine": "mock", "text": "Hola"}).json()["job_id"])
    gen_dir = data / "generated" / done["id"]
    for f in gen_dir.iterdir():
        _old(f)

    orphan_dir = data / "generated" / "deadbeef"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "raw.wav").write_bytes(b"x" * 1000)
    _old(orphan_dir / "raw.wav")
    _old(orphan_dir)
    stale_final = gen_dir / "final_old.wav"
    stale_final.write_bytes(b"y" * 500)
    _old(stale_final)
    fresh_orphan = data / "references" / "processed" / f"{'a' * 64}.wav"
    fresh_orphan.parent.mkdir(parents=True, exist_ok=True)
    fresh_orphan.write_bytes(b"z" * 10)  # too recent: kept
    old_orphan = data / "references" / "processed" / f"{'b' * 64}.wav"
    old_orphan.write_bytes(b"z" * 10)
    _old(old_orphan)
    tmp = data / "cache" / "uploads" / "leftover.part"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(b"t" * 20)
    _old(tmp, 7200)
    transcript_cache = data / "cache" / "transcripts" / "c.json"
    transcript_cache.parent.mkdir(parents=True, exist_ok=True)
    transcript_cache.write_text("{}")

    usage = client.get("/api/system/storage").json()
    assert usage["orphans_files"] == 3 and {c["id"] for c in usage["categories"]} >= {"generated", "models"}

    report = client.post("/api/system/storage/cleanup", json={}).json()
    assert report["details"] == {"orphans": 3, "temp": 1}
    assert not orphan_dir.exists() and not stale_final.exists() and not old_orphan.exists() and not tmp.exists()
    assert fresh_orphan.exists() and transcript_cache.exists()
    assert client.get(done["audio_url"]).status_code == 200 and client.get(done["raw_audio_url"]).status_code == 200

    report = client.post("/api/system/storage/cleanup", json={"orphans": False, "temp": False,
                                                              "transcript_cache": True}).json()
    assert report["details"] == {"transcript_cache": 1} and not transcript_cache.exists()
