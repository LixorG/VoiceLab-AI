"""Voice training API and job with a simulated trainer process (no GPU, no model weights)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sqlmodel import Session, select

from app.core.database import get_engine
from app.engines.custom import get_custom_variants
from app.engines.manager import ModelManager
from app.models.entities import ReferenceAudio, TrainingRun, Transcript, VoiceProfile
from app.models.enums import ReferenceStatus
from app.services import training_service
from app.training import evaluation

FAKE_TRAINER = r'''
import json, os, sys, time
spec = json.load(open(sys.argv[1], encoding="utf-8"))
def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)
mode = os.environ.get("FAKE_TRAINER_MODE", "ok")
emit("stage", stage="codes", message="Preparando")
emit("stage", stage="load", message="Cargando")
emit("stage", stage="train", message="Entrenando", clips=len(spec["clips"]), epochs=spec["epochs"])
if mode == "slow":
    time.sleep(60)
if mode == "oom":
    emit("error", code="OUT_OF_MEMORY", detail="CUDA out of memory")
    sys.exit(1)
if mode == "crash":
    print("Traceback: boom", file=sys.stderr)
    sys.exit(3)
for e in range(spec["epochs"]):
    emit("progress", epoch=e + 1, epochs=spec["epochs"], step=e + 1, total_steps=spec["epochs"], talker_loss=1.0,
         predictor_loss=5.0, eta_s=1)
    emit("epoch", epoch=e + 1, talker_loss=1.0 - e * 0.1, predictor_loss=5.0)
emit("stage", stage="save", message="Guardando")
out = spec["output_dir"]
os.makedirs(os.path.join(out, "speech_tokenizer"), exist_ok=True)
for f in ("config.json", "generation_config.json", "model.safetensors", "speech_tokenizer/config.json",
          "speech_tokenizer/model.safetensors"):
    open(os.path.join(out, f), "w").write("{}")
emit("done", output_dir=out, history=[{"epoch": 1, "talker_loss": 1.0, "predictor_loss": 5.0}], seconds=1)
'''


@pytest.fixture
def trainer(tmp_path, monkeypatch, settings):
    script = tmp_path / "fake_trainer.py"
    script.write_text(FAKE_TRAINER, encoding="utf-8")
    monkeypatch.setattr(training_service, "_trainer_command", lambda spec: [sys.executable, str(script), str(spec)])
    monkeypatch.setattr(training_service, "MIN_MINUTES", 0.05)
    monkeypatch.setattr(ModelManager, "resolve_device", lambda self: "cuda")
    from app.engines.qwen3tts.plugin import Qwen3TTSBackend

    monkeypatch.setattr(Qwen3TTSBackend, "is_installed", lambda self: True)
    real_weights = Qwen3TTSBackend.weights_installed
    monkeypatch.setattr(Qwen3TTSBackend, "weights_installed",
                        lambda self, v: True if v.startswith("base-") else real_weights(self, v))
    base = tmp_path / "base_snapshot"
    base.mkdir()
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(base))
    compared: list = []

    def fake_compare(session, models, run, variant, ctx):
        compared.append(variant)
        return {"held_out": 2, "systems": {"trained": {"wer": 0.02}, "normal": {"wer": 0.03}},
                "verdict": "La voz entrenada se entiende igual (estimación automática)."}

    monkeypatch.setattr(evaluation, "compare", fake_compare)
    yield compared
    get_custom_variants().clear()


def _profile(settings, seconds: float = 12.0, transcribed: bool = True, name: str = "Daniela") -> str:
    """A profile with one recording: sentences of 2 s separated by 0.5 s pauses, with word timestamps."""
    with Session(get_engine()) as s:
        profile = VoiceProfile(name=name, slug=name.lower(), language="Español")
        s.add(profile)
        s.flush()
        sha = f"{len(name):02d}" + "a" * 62
        path = Path(settings.data_dir) / "references" / "processed" / f"{sha}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        sf.write(path, (0.1 * rng.standard_normal(int(24_000 * seconds))).astype(np.float32), 24_000)
        ref = ReferenceAudio(profile_id=profile.id, sha256=sha, original_name="charla.wav", format="wav",
                             size_bytes=1, duration_s=seconds, status=ReferenceStatus.ANALYZED)
        s.add(ref)
        if transcribed:
            words, t, n = [], 0.2, 0
            while t + 2.0 < seconds:
                for k in range(4):
                    words.append({"word": f" palabra{n}" + ("." if k == 3 else ""), "start": round(t, 2),
                                  "end": round(t + 0.45, 2), "probability": 0.95})
                    t += 0.5
                    n += 1
                t += 0.5
            text = " ".join(w["word"].strip() for w in words)
            s.add(Transcript(reference_id=ref.id, text=text, asr_text=text, word_timestamps=words))
        s.commit()
        return profile.id


def _wait(client, run_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/training/{run_id}").json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"training {run_id} did not finish: {body}")


def test_dataset_summary_counts_usable_speech_and_missing_transcripts(client, settings, trainer):
    pid = _profile(settings, seconds=12.0)
    with Session(get_engine()) as s:
        s.add(ReferenceAudio(profile_id=pid, sha256="b" * 64, original_name="sin_texto.wav", format="wav",
                             size_bytes=1, duration_s=30.0, status=ReferenceStatus.ANALYZED))
        s.commit()
    body = client.get(f"/api/training/dataset/{pid}").json()
    assert body["clips"] == 4 and 0.1 < body["usable_minutes"] < 0.2
    first = next(r for r in body["references"] if r["name"] == "charla.wav")
    assert first["transcribed"] and first["clips"] == 4
    assert len(body["missing_transcripts"]) == 1
    assert any("sin transcripción" in w for w in body["warnings"])
    assert client.get("/api/training/dataset/nope").status_code == 404


def test_training_runs_publishes_a_variant_and_compares(client, settings, trainer):
    pid = _profile(settings, seconds=12.0)
    res = client.post("/api/training", json={"profile_id": pid, "epochs": 3})
    assert res.status_code == 202, res.text
    run = res.json()
    assert run["status"] == "queued" and run["name"] == "Daniela (entrenada)" and run["params"]["epochs"] == 3

    done = _wait(client, run["id"])
    assert done["status"] == "completed", done
    assert done["variant"] == "custom:daniela-entrenada" and done["checkpoint_id"]
    assert [h["epoch"] for h in done["history"]] == [1]
    assert done["evaluation"]["verdict"].startswith("La voz entrenada") and trainer == [done["variant"]]
    assert done["dataset"]["reference_name"] == "charla.wav"
    spec = json.loads((Path(settings.data_dir) / "training" / run["id"] / "spec.json").read_text(encoding="utf-8"))
    assert spec["epochs"] == 3 and spec["speaker_name"] == "voicelab" and len(spec["clips"]) == 4
    assert all(c["end_s"] > c["start_s"] for c in spec["clips"])

    # the trained voice is now a Qwen3-TTS variant that needs no reference and has no speaker/instruction params
    from app.engines.qwen3tts.plugin import Qwen3TTSBackend

    qwen = Qwen3TTSBackend()
    variant = qwen.variant(done["variant"])
    assert variant.mode == "custom_voice" and variant.source == "custom" and variant.base_variant == "base-1.7b"
    caps = qwen.capabilities(done["variant"])
    assert not caps.requires_reference_audio and caps.controls["instruction"].source == "unavailable"
    assert {p.id for p in qwen.get_parameters_schema(done["variant"])}.isdisjoint({"speaker", "instruct"})
    assert qwen.weights_installed(done["variant"])

    listed = client.get("/api/training", params={"profile_id": pid}).json()
    assert [r["id"] for r in listed] == [run["id"]]

    # delete: variant and folder go away
    with Session(get_engine()) as s:
        folder = Path(s.get(TrainingRun, run["id"]).output_path)
    assert folder.is_dir()
    assert client.delete(f"/api/training/{run['id']}").status_code == 204
    assert not folder.exists() and not get_custom_variants().for_engine("qwen3tts")
    assert client.get(f"/api/training/{run['id']}").status_code == 404


def test_training_needs_enough_speech_and_one_run_at_a_time(client, settings, trainer, monkeypatch):
    pid = _profile(settings, seconds=12.0)
    monkeypatch.setattr(training_service, "MIN_MINUTES", 5.0)
    res = client.post("/api/training", json={"profile_id": pid})
    assert res.status_code == 422 and res.json()["error_code"] == "TRAINING_DATA_INSUFFICIENT"
    assert "al menos 5 minutos" in res.json()["message"]
    monkeypatch.setattr(training_service, "MIN_MINUTES", 0.05)

    monkeypatch.setenv("FAKE_TRAINER_MODE", "slow")
    first = client.post("/api/training", json={"profile_id": pid}).json()
    busy = client.post("/api/training", json={"profile_id": pid})
    assert busy.status_code == 409 and busy.json()["error_code"] == "TRAINING_BUSY"
    assert client.delete(f"/api/training/{first['id']}").status_code == 409  # cancel first
    deadline = time.time() + 30
    while client.get(f"/api/training/{first['id']}").json()["status"] != "training" and time.time() < deadline:
        time.sleep(0.05)
    assert client.post(f"/api/training/{first['id']}/cancel").status_code == 200
    done = _wait(client, first["id"])
    assert done["status"] == "cancelled" and done["variant"] is None


@pytest.mark.parametrize(("mode", "code", "text"), [
    ("oom", "GPU_MEMORY_ERROR", "sin memoria"),
    ("crash", "TRAINING_FAILED", "falló"),
])
def test_trainer_failures_are_reported_in_spanish(client, settings, trainer, monkeypatch, mode, code, text):
    pid = _profile(settings, seconds=12.0)
    monkeypatch.setenv("FAKE_TRAINER_MODE", mode)
    run = client.post("/api/training", json={"profile_id": pid}).json()
    done = _wait(client, run["id"])
    assert done["status"] == "failed" and done["error_code"] == code and text in done["message"]
    assert done["variant"] is None


def test_training_requires_gpu_weights_and_profile(client, settings, trainer, monkeypatch):
    pid = _profile(settings)
    assert client.post("/api/training", json={"profile_id": "nope"}).status_code == 404
    monkeypatch.setattr(ModelManager, "resolve_device", lambda self: "cpu")
    res = client.post("/api/training", json={"profile_id": pid})
    assert res.status_code == 409 and "GPU" in res.json()["message"]
    monkeypatch.setattr(ModelManager, "resolve_device", lambda self: "cuda")
    from app.engines.qwen3tts.plugin import Qwen3TTSBackend

    monkeypatch.setattr(Qwen3TTSBackend, "weights_installed", lambda self, v: False)
    res = client.post("/api/training", json={"profile_id": pid, "base_variant": "base-0.6b"})
    assert res.status_code == 409 and "pesos" in res.json()["message"]
    assert client.post("/api/training", json={"profile_id": pid, "base_variant": "otro"}).status_code == 422


def test_samples_route_only_serves_comparison_files(client, settings, trainer):
    pid = _profile(settings)
    run = _wait(client, client.post("/api/training", json={"profile_id": pid}).json()["id"])
    folder = evaluation.samples_dir(run["id"])
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / "0_trained.wav", np.zeros(2400, np.float32), 24_000)
    ok = client.get(f"/api/training/{run['id']}/samples/0_trained.wav")
    assert ok.status_code == 200 and ok.headers["content-type"] == "audio/wav"
    assert client.get(f"/api/training/{run['id']}/samples/1_trained.wav").status_code == 404
    assert client.get(f"/api/training/{run['id']}/samples/..%2Fspec.json").status_code == 404


def test_interrupted_runs_are_marked_at_startup(settings, app):
    from fastapi.testclient import TestClient

    with TestClient(app):
        with Session(get_engine()) as s:
            s.add(TrainingRun(profile_id="p", base_variant="base-1.7b", name="x", status="training"))
            s.commit()
    assert training_service.mark_interrupted_trainings() == 1
    with Session(get_engine()) as s:
        run = s.exec(select(TrainingRun)).one()
        assert run.status == "failed" and run.error_code == "JOB_INTERRUPTED"


def test_deleting_the_variant_from_models_removes_its_folder(client, settings, trainer):
    pid = _profile(settings)
    run = _wait(client, client.post("/api/training", json={"profile_id": pid}).json()["id"])
    with Session(get_engine()) as s:
        folder = Path(s.get(TrainingRun, run["id"]).output_path)
    assert folder.is_dir()
    assert client.delete(f"/api/models/checkpoints/{run['checkpoint_id']}").status_code == 204
    assert not folder.exists()
    after = client.get(f"/api/training/{run['id']}").json()
    assert after["status"] == "completed" and after["variant"] is None and after["message"] == "Modelo borrado."
