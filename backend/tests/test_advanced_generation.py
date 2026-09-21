"""Segmented generation, emotion-aware references/instructions, variations — simulated engines, no GPU."""

from __future__ import annotations

import pytest
import soundfile as sf

from app.core.gpu import GPUManager
from app.engines.base import (
    ControlCapability,
    ControlSource,
    EngineCapabilities,
    GenericControl,
    ParameterSpec,
    ParameterTooltip,
)
from app.engines.manager import ModelManager, get_model_manager
from app.engines.mock.plugin import SR, MockBackend
from app.engines.registry import EngineRegistry, get_engine_registry
from app.services import generation_service
from tests.audio_fixtures import requires_ffmpeg, speechlike, write_wav
from tests.test_generation import RefMock, _add_transcript, wait_done, wait_status

TIP = ParameterTooltip(what="x", effect="x", cost="x", typical="x")


class SegMock(RefMock):
    """Clone engine whose emotion is applied by switching references (like F5/E2/Qwen clone)."""

    id = "segmock"

    def capabilities(self, variant=None) -> EngineCapabilities:
        caps = super().capabilities(variant)
        controls = dict(caps.controls)
        controls[GenericControl.EMOTION] = ControlCapability(
            source=ControlSource.SEGMENTATION, reason="Por referencia."
        )
        return caps.model_copy(update={"controls": controls})


class InstructMock(MockBackend):
    """Non-cloning engine with a native style instruction (like Qwen CustomVoice)."""

    id = "instructmock"

    def __init__(self) -> None:
        super().__init__()
        self.requests: list = []

    def capabilities(self, variant=None) -> EngineCapabilities:
        caps = super().capabilities(variant)
        controls = dict(caps.controls)
        by_instruction = ControlCapability(source=ControlSource.INSTRUCTION, reason="Por instrucción.")
        controls[GenericControl.INSTRUCTION] = ControlCapability(source=ControlSource.NATIVE, parameter="instruct")
        controls[GenericControl.EMOTION] = by_instruction
        controls[GenericControl.PITCH] = by_instruction
        return caps.model_copy(update={"controls": controls})

    def get_parameters_schema(self, variant=None) -> list[ParameterSpec]:
        return [*super().get_parameters_schema(variant),
                ParameterSpec(id="instruct", type="text", value_type="str", label="Instrucción", nullable=True,
                              max_length=2048, tooltip=TIP)]

    def generate(self, request, progress, cancel):
        self.requests.append(request)
        return super().generate(request, progress, cancel)


@pytest.fixture
def engines(app, monkeypatch):
    registry = EngineRegistry()
    seg, instr = SegMock(), InstructMock()
    registry.register(seg)
    registry.register(instr)
    manager = ModelManager(registry, GPUManager("cpu"), "cpu")
    app.dependency_overrides[get_model_manager] = lambda: manager
    app.dependency_overrides[get_engine_registry] = lambda: registry
    monkeypatch.setattr(generation_service, "get_model_manager", lambda: manager)
    yield seg, instr
    seg.release.set()
    app.dependency_overrides.clear()


def _upload(client, tmp_path, name, profile_id=None, seed=0):
    wav = write_wav(tmp_path / name, speechlike(words=8, seed=seed))
    with wav.open("rb") as fh:
        res = client.post("/api/references", files={"file": (name, fh, "audio/wav")},
                          data={"profile_id": profile_id} if profile_id else None)
    return res.json()["reference"]


@pytest.fixture
def profile_with_emotions(client, tmp_path, engines):
    pid = client.post("/api/voices", json={"name": "Actriz"}).json()["id"]
    neutral = _upload(client, tmp_path, "neutral.wav", pid, seed=1)
    happy = _upload(client, tmp_path, "feliz.wav", pid, seed=2)
    sad = _upload(client, tmp_path, "triste.wav", pid, seed=3)  # tagged but without transcript: not usable
    for ref, text in ((neutral, "texto neutral"), (happy, "texto feliz")):
        _add_transcript(ref["id"], text)
    client.patch(f"/api/references/{happy['id']}", json={"emotion_tag": "happy"})
    client.patch(f"/api/references/{sad['id']}", json={"emotion_tag": "sad"})
    client.patch(f"/api/voices/{pid}", json={"primary_reference_id": neutral["id"]})
    return pid, neutral, happy, sad


# ---------------------------------------------------------------- plan endpoint
def test_plan_reports_segments_and_markup_errors(client, engines):
    body = {"engine": "instructmock", "text": "Hola. [pausa:700ms] [susurro]Secreto[/susurro]", "emotion": "calm",
            "intensity": 10}
    plan = client.post("/api/generation/plan", json=body).json()
    assert plan["segmented"] is True and plan["warnings"] == []
    assert [(s["text"], s["pause_after_ms"]) for s in plan["segments"]] == [("Hola.", 700), ("Secreto", 0)]
    assert plan["segments"][1]["instruction"] == "Speak in a slightly calm tone. Whisper softly."

    bad = client.post("/api/generation/plan", json={**body, "text": "Hola [pausa:mucho]"})
    assert bad.status_code == 422 and bad.json()["error_code"] == "MARKUP_ERROR"
    assert bad.json()["details"]["posicion"] == 5
    assert client.post("/api/generation/plan", json={**body, "emotion": "furiosa"}).status_code == 422
    raw = client.post("/api/generation/plan", json={**body, "markup": False}).json()
    assert raw["segmented"] is True and "[pausa:700ms]" in raw["segments"][0]["text"]  # calm emotion still applies


@requires_ffmpeg
def test_segmentation_engine_switches_reference_by_emotion(client, engines, profile_with_emotions):
    seg_engine, _ = engines
    pid, neutral, happy, sad = profile_with_emotions
    body = {"engine": "segmock", "profile_id": pid, "params": {"seed": 100},
            "text": "Hola a todos. [emoción:feliz]¡Qué gran noticia![/emoción] [pausa:500ms]"
                    "[emoción:triste]Aunque no todo es bueno.[/emoción]"}
    plan = client.post("/api/generation/plan", json=body).json()
    assert [s["reference_name"] for s in plan["segments"]] == ["neutral.wav", "feliz.wav", "neutral.wav"]
    assert any("triste" in w for w in plan["warnings"])  # tagged reference has no transcript

    gen_id = client.post("/api/generation", json=body).json()["job_id"]
    done = wait_done(client, gen_id)
    assert done["status"] == "COMPLETED" and done["metrics"]["segments"] == 3
    assert [s["seed"] for s in done["segments"]] == [100, 101, 102]
    assert [s["reference_name"] for s in done["segments"]] == ["neutral.wav", "feliz.wav", "neutral.wav"]
    assert done["expression"] == {"emotion": None, "intensity": 50, "markup": True, "normalize": True,
                                 "text_changes": []}
    assert [r.text for r in seg_engine.prepared] == ["texto neutral", "texto feliz"]  # prepared once per reference

    audio = client.get(done["audio_url"])
    import io

    data, sr = sf.read(io.BytesIO(audio.content))
    seg_durations = sum(s["duration_s"] for s in done["segments"])
    expected = seg_durations - 0.04 + 0.5  # one crossfade (segments 1-2 contiguous) + one pause
    assert sr == SR and abs(data.size / sr - expected) < 0.02


@requires_ffmpeg
def test_instruction_engine_merges_user_instruction(client, engines):
    _, instr = engines
    body = {"engine": "instructmock", "params": {"instruct": "Narrador de documentales", "seed": 5},
            "text": "Empieza tranquilo. [emoción:entusiasmado]¡Y termina con energía![/emoción]", "emotion": "calm",
            "intensity": 80}
    done = wait_done(client, client.post("/api/generation", json=body).json()["job_id"])
    assert done["status"] == "COMPLETED"
    instructs = [r.params["instruct"] for r in instr.requests]
    assert instructs == ["Narrador de documentales. Speak in a very calm tone.",
                         "Narrador de documentales. Speak in a very excited tone."]
    assert done["params"]["instruct"] == "Narrador de documentales"  # base params kept clean for "repeat"


@requires_ffmpeg
def test_preview_generates_only_first_segment(client, engines):
    body = {"engine": "instructmock", "text": "Primera frase. Segunda. [pausa:1s] Otro segmento.", "preview": True,
            "emotion": "happy"}
    done = wait_done(client, client.post("/api/generation", json=body).json()["job_id"])
    assert done["kind"] == "preview" and done["text"] == "Primera frase."
    assert len(done["segments"]) == 1 and done["segments"][0]["pause_after_ms"] == 0


@requires_ffmpeg
def test_simple_text_keeps_single_call_path(client, engines):
    _, instr = engines
    done = wait_done(client, client.post("/api/generation", json={"engine": "instructmock",
                                                                  "text": "Texto  sin\nmarcas"}).json()["job_id"])
    assert done["segments"] == [] and done["text"] == "Texto sin marcas" and len(instr.requests) == 1


@requires_ffmpeg
def test_variations_use_distinct_seeds_and_group(client, engines):
    res = client.post("/api/generation/variations", json={"engine": "instructmock", "text": "Hola", "count": 3})
    assert res.status_code == 202
    accepted = res.json()
    ids = [a["job_id"] for a in accepted]
    finished = [wait_done(client, gid) for gid in ids]
    seeds = {g["seed"] for g in finished}
    assert len(seeds) == 3 and all(g["kind"] == "variation" for g in finished)
    assert finished[0]["parent_id"] is None and {g["parent_id"] for g in finished[1:]} == {ids[0]}
    assert client.post("/api/generation/variations", json={"engine": "instructmock", "text": "x",
                                                           "count": 20}).status_code == 422
    # deleting the first variation detaches the others instead of failing
    assert client.delete(f"/api/generation/{ids[0]}").status_code == 204
    assert client.get(f"/api/generation/{ids[1]}").json()["parent_id"] is None


@requires_ffmpeg
def test_cancel_between_segments(client, engines, profile_with_emotions):
    seg_engine, _ = engines
    pid, *_ = profile_with_emotions
    body = {"engine": "segmock", "profile_id": pid, "text": "LENTO uno. [pausa:300ms] dos. [pausa:300ms] tres."}
    gen_id = client.post("/api/generation", json=body).json()["job_id"]
    wait_status(client, gen_id, "GENERATING")
    client.post(f"/api/jobs/{gen_id}/cancel")
    done = wait_done(client, gen_id)
    assert done["status"] == "CANCELLED" and done["audio_url"] is None


class SentenceMock(MockBackend):
    """Engine that sounds better sentence by sentence (like Qwen clone) and pads its output with silence."""

    id = "sentencemock"

    def capabilities(self, variant=None) -> EngineCapabilities:
        return super().capabilities(variant).model_copy(update={"sentence_chunks": True, "max_chars_per_call": 300})

    def generate(self, request, progress, cancel, on_audio=None):
        import numpy as np

        result = super().generate(request, progress, cancel)
        pad = np.zeros(int(0.3 * result.sample_rate), dtype=np.float32)
        return result.model_copy(update={"audio": np.concatenate([pad, result.audio, pad])})


@requires_ffmpeg
def test_sentence_engines_get_natural_pauses_of_exact_length(client, app, monkeypatch):
    import io

    registry = EngineRegistry()
    registry.register(SentenceMock())
    manager = ModelManager(registry, GPUManager("cpu"), "cpu")
    app.dependency_overrides[get_model_manager] = lambda: manager
    app.dependency_overrides[get_engine_registry] = lambda: registry
    monkeypatch.setattr(generation_service, "get_model_manager", lambda: manager)
    first = "Primera frase del guion, bastante larga para ir sola."
    rest = ["Segunda frase, también larga para ir sola.", "Tercera frase del mismo párrafo, larga también."]
    text = f"{first}\n\n{rest[0]} {rest[1]}"

    done = wait_done(client, client.post("/api/generation", json={"engine": "sentencemock", "text": text,
                                                                  "normalize": False}).json()["job_id"])
    assert done["status"] == "COMPLETED"
    assert [(s["text"], s["pause_after_ms"]) for s in done["segments"]] == [(first, 750), (rest[0], 350), (rest[1], 0)]
    # the engine's own 0.3 s of silence is trimmed wherever a pause is inserted (a margin of 40/80 ms stays)
    speech = [len(s["text"]) * 0.06 for s in done["segments"]]
    trimmed = [s["duration_s"] for s in done["segments"]]
    assert abs(trimmed[0] - (0.3 + speech[0] + 0.08)) < 0.02  # start untouched, end trimmed
    assert abs(trimmed[1] - (0.04 + speech[1] + 0.08)) < 0.02
    assert abs(trimmed[2] - (0.04 + speech[2] + 0.3)) < 0.02  # the final silence is not next to a pause
    data, sr = sf.read(io.BytesIO(client.get(done["audio_url"]).content))
    assert abs(data.size / sr - (sum(trimmed) + 0.75 + 0.35)) < 0.02
    app.dependency_overrides.clear()
