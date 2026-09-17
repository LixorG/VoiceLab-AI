"""Generation pipeline (queue, model manager, API, SSE) with simulated engines — no GPU needed."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
import soundfile as sf

from app.core.errors import AppError, ErrorCode
from app.core.gpu import GPUDevice, GPUInfo, GPUManager
from app.engines.base import CancelToken, EngineCapabilities, EngineRequest, EngineResult
from app.engines.manager import ModelManager, get_model_manager
from app.engines.mock.plugin import MockBackend
from app.engines.registry import EngineRegistry
from app.models.enums import JobStatus
from app.services import generation_service
from app.services.generation_service import get_job_queue, preview_text
from app.workers.asyncio_queue import AsyncioJobQueue
from app.workers.base import JobSpec
from tests.audio_fixtures import requires_ffmpeg, speechlike, write_wav


# ---------------------------------------------------------------------------
# Simulated engines
# ---------------------------------------------------------------------------
class RefMock(MockBackend):
    """Clone-style engine: needs reference audio + exact text, 3–12 s."""

    id = "refmock"
    display_name = "Clonación simulada"

    def __init__(self) -> None:
        super().__init__()
        self.prepared: list = []
        self.release = threading.Event()
        self.loads = 0
        self.weights = True

    def weights_installed(self, variant: str) -> bool:
        return self.weights

    def load(self, variant: str, device: str, dtype: str) -> None:
        self.loads += 1
        super().load(variant, device, dtype)

    def capabilities(self, variant=None) -> EngineCapabilities:
        caps = super().capabilities(variant)
        return caps.model_copy(update={"requires_reference_audio": True, "requires_reference_text": True,
                                       "reference_duration_s": (3.0, 12.0),
                                       "reference_text_not_required_when": {"tone_hz": 880}})

    def prepare_reference(self, reference, variant):
        self.prepared.append(reference)
        return super().prepare_reference(reference, variant)

    def generate(self, request: EngineRequest, progress, cancel: CancelToken) -> EngineResult:
        if "FALLA" in request.text:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        if "LENTO" in request.text:
            progress(0.5, "a mitad")
            while not cancel.cancelled and not self.release.wait(0.02):
                pass
            if cancel.cancelled:
                raise AppError(ErrorCode.JOB_CANCELLED, status_code=409)
        return super().generate(request, progress, cancel)


class Planned(MockBackend):
    id = "planned"
    display_name = "Motor planificado"
    implementation_phase = 42


@pytest.fixture
def engines(app, monkeypatch):
    registry = EngineRegistry()
    mock, refmock = MockBackend(), RefMock()
    registry.register(mock)
    registry.register(refmock)
    registry.register(Planned())
    gpu = GPUManager("cpu")
    manager = ModelManager(registry, gpu, "cpu")
    app.dependency_overrides[get_model_manager] = lambda: manager
    monkeypatch.setattr(generation_service, "get_model_manager", lambda: manager)
    yield manager, mock, refmock
    refmock.release.set()
    app.dependency_overrides.clear()


def wait_until(predicate, timeout: float = 30.0, interval: float = 0.02):
    """Poll until `predicate()` returns a truthy value. Generous deadline: the suite may run on a busy machine."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def wait_done(client, gen_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/generation/{gen_id}").json()
        if body["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"generation {gen_id} did not finish: {body}")


def wait_status(client, gen_id: str, status: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get(f"/api/generation/{gen_id}").json()["status"] == status:
            return
        time.sleep(0.02)
    raise AssertionError(f"{gen_id} never reached {status}")


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("text", "expected"), [
    ("Hola a todos. Esto es una prueba larga.", "Hola a todos."),
    ("¿Qué tal? Bien.", "¿Qué tal?"),
    ("Sin puntuación final", "Sin puntuación final"),
    ("palabra " * 60, ("palabra " * 20).strip()),
])
def test_preview_text(text, expected):
    assert preview_text(text) == expected


def test_queue_runs_serially_and_reports_events():
    async def scenario():
        queue = AsyncioJobQueue()
        order: list[str] = []

        def handler(spec, ctx):
            ctx.update(JobStatus.GENERATING, 0.5, "mitad")
            order.append(spec.payload["n"])
            if spec.payload["n"] == "boom":
                raise RuntimeError("secret")
            return {"n": spec.payload["n"]}

        queue.register("t", handler)
        a = await queue.submit(JobSpec(kind="t", payload={"n": "a"}))
        b = await queue.submit(JobSpec(kind="t", payload={"n": "boom"}))
        c = await queue.submit(JobSpec(kind="t", payload={"n": "c"}))
        events = [e.status async for e in queue.events(c)]
        assert order == ["a", "boom", "c"]
        assert (await queue.get(a)).status == JobStatus.COMPLETED
        failed = await queue.get(b)
        assert failed.status == JobStatus.FAILED and failed.error_code == "INTERNAL_ERROR"
        assert "secret" not in (failed.message or "")
        assert events[-1] == JobStatus.COMPLETED
        await queue.stop()

    asyncio.run(scenario())


def test_queue_cancels_queued_job_immediately():
    async def scenario():
        queue = AsyncioJobQueue()
        gate = threading.Event()
        queue.register("t", lambda spec, ctx: gate.wait(2))
        first = await queue.submit(JobSpec(kind="t"))
        second = await queue.submit(JobSpec(kind="t"))
        await queue.cancel(second)
        assert (await queue.get(second)).status == JobStatus.CANCELLED
        gate.set()
        async for _ in queue.events(first):
            pass
        assert (await queue.get(first)).status == JobStatus.COMPLETED
        with pytest.raises(AppError):
            await queue.get("nope")
        await queue.stop()

    asyncio.run(scenario())


def test_model_manager_keeps_one_engine_loaded():
    registry = EngineRegistry()
    a, b = MockBackend(), RefMock()
    registry.register(a)
    registry.register(b)
    manager = ModelManager(registry, GPUManager("cpu"), "cpu")
    manager.ensure_loaded("mock", "tone")
    manager.ensure_loaded("mock", "tone")
    assert a.loaded_variant == "tone" and manager.loaded.engine == "mock"
    manager.ensure_loaded("refmock", "tone")
    assert a.loaded_variant is None and b.loaded_variant == "tone" and b.loads == 1
    b.weights = False
    manager.unload()
    with pytest.raises(AppError) as exc:
        manager.ensure_loaded("refmock", "tone")
    assert exc.value.code is ErrorCode.MODEL_NOT_INSTALLED
    assert manager.status("refmock", None).weights_installed is False


def test_model_manager_frees_asr_when_vram_is_short(monkeypatch):
    class Asr:
        class backend:
            loaded = True

        unloaded = False

        def unload(self):
            self.unloaded = True
            Asr.backend.loaded = False

    asr = Asr()
    free = {"mb": 1000}

    class Gpu(GPUManager):
        def detect(self):
            return GPUInfo(backend="cuda", torch_available=True, source="torch",
                           devices=[GPUDevice(index=0, name="Mock", total_memory_mb=16000, free_memory_mb=free["mb"])])

    class Big(MockBackend):
        def requirements(self, variant=None):
            r = super().requirements(variant)
            return r.model_copy(update={"vram_mb": 4000})

    registry = EngineRegistry()
    registry.register(Big())
    manager = ModelManager(registry, Gpu(), "cuda")
    monkeypatch.setattr(manager, "resolve_device", lambda: "cuda")
    monkeypatch.setattr("app.engines.manager.get_asr_manager", lambda: asr)
    with pytest.raises(AppError) as exc:  # freeing ASR is not enough
        manager.ensure_loaded("mock", "tone")
    assert exc.value.code is ErrorCode.GPU_MEMORY_ERROR and asr.unloaded

    Asr.backend.loaded = True
    asr.unloaded = False

    def unload_and_free():
        Asr.unload(asr)
        free["mb"] = 6000

    asr.unload = unload_and_free
    manager.ensure_loaded("mock", "tone")
    assert manager.loaded.engine == "mock"


# ---------------------------------------------------------------------------
# API with the plain mock engine (no reference)
# ---------------------------------------------------------------------------
def test_generate_completes_and_serves_wav(client, engines):
    manager, mock, _ = engines
    res = client.post("/api/generation", json={"engine": "mock", "text": "Hola a todos. Segunda frase.",
                                               "params": {"seed": 42, "speed": 1.5}})
    assert res.status_code == 202, res.text
    accepted = res.json()
    assert accepted["job_id"] == accepted["generation"]["id"]
    body = wait_done(client, accepted["job_id"])
    assert body["status"] == "COMPLETED" and body["progress"] == 1.0
    assert body["seed"] == 42 and body["params"]["speed"] == 1.5 and body["params"]["tone_hz"] == 220
    assert body["metrics"]["device"] == "cpu" and body["duration_s"] > 0
    assert manager.loaded.engine == "mock"

    audio = client.get(body["audio_url"])
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    download = client.get(body["audio_url"] + "&download=true")
    assert "attachment" in download.headers["content-disposition"]

    import io

    info = sf.info(io.BytesIO(audio.content))
    assert (info.samplerate, info.subtype) == (24000, "PCM_24")

    again = wait_done(client, client.post("/api/generation", json={
        "engine": "mock", "text": "Hola a todos. Segunda frase.", "params": body["params"]}).json()["job_id"])
    assert client.get(again["audio_url"]).content == audio.content  # same seed → same audio


def test_preview_uses_first_sentence(client, engines):
    res = client.post("/api/generation", json={"engine": "mock", "text": "Primera frase. Segunda frase larga.",
                                               "preview": True})
    body = wait_done(client, res.json()["job_id"])
    assert body["kind"] == "preview" and body["text"] == "Primera frase."


def test_request_validation_errors(client, engines):
    bad = client.post("/api/generation", json={"engine": "mock", "text": "hola", "params": {"nfe_steps": 32}})
    assert bad.status_code == 422 and bad.json()["error_code"] == "PARAMETER_ERROR"
    assert client.post("/api/generation", json={"engine": "mock", "text": "   "}).status_code == 422
    missing = client.post("/api/generation", json={"engine": "xtts", "text": "hola"})
    assert missing.status_code == 404 and missing.json()["error_code"] == "ENGINE_NOT_FOUND"
    later = client.post("/api/generation", json={"engine": "planned", "text": "hola"})
    assert later.status_code == 501 and "fase 42" in later.json()["message"]


def test_list_and_delete(client, engines, settings):
    gen = wait_done(client, client.post("/api/generation", json={"engine": "mock", "text": "hola"}).json()["job_id"])
    assert [g["id"] for g in client.get("/api/generation").json()][0] == gen["id"]
    assert (settings.data_dir / "generated" / gen["id"] / "raw.wav").exists()
    assert client.delete(f"/api/generation/{gen['id']}").status_code == 204
    assert not (settings.data_dir / "generated" / gen["id"]).exists()
    assert client.get(f"/api/generation/{gen['id']}").json()["error_code"] == "GENERATION_NOT_FOUND"


def test_runtime_and_unload_endpoints(client, engines):
    wait_done(client, client.post("/api/generation", json={"engine": "mock", "text": "hola"}).json()["job_id"])
    assert client.get("/api/models/runtime").json()["engine"] == "mock"
    assert client.get("/api/models/mock/status").json()["loaded"] is True
    assert client.post("/api/models/unload").status_code == 204
    assert client.get("/api/models/runtime").json() is None


# ---------------------------------------------------------------------------
# Reference handling, failures, cancellation, SSE
# ---------------------------------------------------------------------------
@pytest.fixture
def reference(client, tmp_path):
    wav = write_wav(tmp_path / "voz.wav", speechlike(words=10))
    with wav.open("rb") as fh:
        ref = client.post("/api/references", files={"file": ("voz.wav", fh, "audio/wav")}).json()["reference"]
    return ref


def _add_transcript(ref_id: str, text: str, start=None, end=None):
    from sqlmodel import Session

    from app.core.database import get_engine
    from app.models.entities import Transcript

    with Session(get_engine()) as s:
        s.add(Transcript(reference_id=ref_id, text=text, asr_text=text, segment_start_s=start, segment_end_s=end))
        s.commit()


@requires_ffmpeg
def test_reference_rules(client, engines, reference):
    _, _, refmock = engines
    url = "/api/generation"
    body = {"engine": "refmock", "text": "hola"}
    assert client.post(url, json=body).json()["error_code"] == "REFERENCE_REQUIRED"
    body["reference_id"] = reference["id"]
    assert client.post(url, json=body).json()["error_code"] == "REFERENCE_TEXT_REQUIRED"
    # engines may declare parameter values that make the transcript optional (e.g. Qwen x-vector mode)
    no_text = client.post(url, json={**body, "params": {"tone_hz": 880}})
    assert no_text.status_code == 202, no_text.text
    assert wait_done(client, no_text.json()["job_id"])["reference"]["text"] == ""

    _add_transcript(reference["id"], "texto completo")
    too_long = client.post(url, json=body).json()  # ~7-9 s is fine; force a long one via segment check below
    if reference["analysis"]["duration_s"] > 12.5:
        assert too_long["error_code"] == "REFERENCE_TOO_LONG"
    else:
        wait_done(client, too_long["job_id"])

    client.patch(f"/api/references/{reference['id']}", json={"segment_start_s": 1.0, "segment_end_s": 5.0})
    seg_missing = client.post(url, json=body).json()
    assert seg_missing["error_code"] == "REFERENCE_TEXT_REQUIRED" and "segmento" in seg_missing["message"]
    _add_transcript(reference["id"], "texto del segmento", 1.0, 5.0)
    done = wait_done(client, client.post(url, json=body).json()["job_id"])
    assert done["status"] == "COMPLETED"
    assert done["reference"] == {"reference_id": reference["id"], "name": "voz.wav", "start_s": 1.0, "end_s": 5.0,
                                 "text": "texto del segmento"}
    prepared = refmock.prepared[-1]
    assert (prepared.start_s, prepared.end_s, prepared.text) == (1.0, 5.0, "texto del segmento")
    assert prepared.audio_path.name == f"{prepared.sha256}.wav"


@requires_ffmpeg
def test_failure_is_persisted_with_friendly_code(client, engines, reference):
    _add_transcript(reference["id"], "texto")
    res = client.post("/api/generation", json={"engine": "refmock", "text": "FALLA aquí",
                                               "reference_id": reference["id"]})
    body = wait_done(client, res.json()["job_id"])
    assert body["status"] == "FAILED" and body["error_code"] == "GPU_MEMORY_ERROR"
    assert "memoria de GPU" in body["message"] and "allocate" not in body["message"]


@requires_ffmpeg
def test_cancel_running_job_and_sse_stream(client, engines, reference):
    _add_transcript(reference["id"], "texto")
    gen_id = client.post("/api/generation", json={"engine": "refmock", "text": "LENTO",
                                                  "reference_id": reference["id"]}).json()["job_id"]
    wait_status(client, gen_id, "GENERATING")
    cancelled = client.post(f"/api/jobs/{gen_id}/cancel").json()
    assert cancelled["job_id"] == gen_id
    body = wait_done(client, gen_id)
    assert body["status"] == "CANCELLED" and body["audio_url"] is None

    with client.stream("GET", f"/api/jobs/{gen_id}/events") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in stream.iter_lines() if line.startswith("data:")]
    assert json.loads(lines[-1][5:])["status"] == "CANCELLED"

    ok_id = client.post("/api/generation", json={"engine": "mock", "text": "hola"}).json()["job_id"]
    with client.stream("GET", f"/api/jobs/{ok_id}/events") as stream:
        statuses = [json.loads(line[5:])["status"] for line in stream.iter_lines() if line.startswith("data:")]
    assert statuses[-1] == "COMPLETED"
    assert client.get("/api/jobs/nope").json()["error_code"] == "JOB_NOT_FOUND"


def test_job_queue_singleton_is_registered():
    assert "generation" in get_job_queue()._handlers


def test_restart_marks_orphaned_generations_as_interrupted(client, engines):
    """Jobs live in memory: after a restart, queued/running generations must not stay "En cola" forever."""
    from sqlmodel import Session

    from app.core.database import get_engine
    from app.models.entities import Generation
    from app.services.generation_service import mark_interrupted_generations

    with Session(get_engine()) as session:
        for status in (JobStatus.QUEUED, JobStatus.GENERATING, JobStatus.COMPLETED):
            session.add(Generation(id=f"orphan-{status.value}", engine="mock", text="Hola", status=status))
        session.commit()
    assert mark_interrupted_generations() == 2
    queued = client.get("/api/generation/orphan-QUEUED").json()
    assert queued["status"] == "FAILED" and queued["error_code"] == "JOB_INTERRUPTED"
    assert "reinició" in queued["message"]
    assert client.get("/api/generation/orphan-COMPLETED").json()["status"] == "COMPLETED"

