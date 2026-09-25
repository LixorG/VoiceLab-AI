"""Dialogues with several voices: reading the script, one voice per character and the pause between turns."""

from __future__ import annotations

import hashlib

import pytest
from sqlmodel import Session, select

from app.core.database import get_engine
from app.core.gpu import GPUManager
from app.engines.manager import ModelManager, get_model_manager
from app.engines.mock.plugin import MockBackend
from app.engines.registry import EngineRegistry, get_engine_registry
from app.generation import dialogue
from app.models.entities import GenerationSegment, ReferenceAudio, VoiceProfile
from app.models.enums import ReferenceStatus
from app.services import generation_service
from tests.audio_fixtures import requires_ffmpeg
from tests.test_generation import wait_done

SCRIPT = "Ana: Hola, ¿qué tal?\nLuis: Muy bien, gracias.\nAna: Me alegro."


def test_a_script_becomes_turns():
    turns = dialogue.parse(SCRIPT)
    assert [(t.speaker, t.text) for t in turns] == [
        ("Ana", "Hola, ¿qué tal?"), ("Luis", "Muy bien, gracias."), ("Ana", "Me alegro.")]
    assert dialogue.speakers(SCRIPT) == ["Ana", "Luis"]  # in the order they first speak


def test_a_line_without_a_label_continues_the_same_turn():
    turns = dialogue.parse("Ana: Primera línea.\nSigue hablando ella.\nLuis: Ahora yo.")
    assert [(t.speaker, t.text) for t in turns] == [
        ("Ana", "Primera línea.\nSigue hablando ella."), ("Luis", "Ahora yo.")]


def test_text_before_the_first_character_keeps_the_chosen_voice():
    turns = dialogue.parse("Una introducción.\nAna: Hola.")
    assert turns[0].speaker is None and turns[0].text == "Una introducción."


@pytest.mark.parametrize("text", [
    "Son las 12:30 y hace calor.",              # a time is not a character
    "Mira esto: https://ejemplo.com/cosa",      # neither is a url
    "Nota: esto es una sola línea con nota.",   # one label alone is not a dialogue
    "Esto es normal. Sin dos puntos.",
])
def test_what_is_not_a_dialogue_is_left_alone(text):
    assert dialogue.looks_like_dialogue(text) is False


def test_a_real_script_is_recognised():
    assert dialogue.looks_like_dialogue(SCRIPT) is True


# ---------------------------------------------------------------------------- through the API
class CastMock(MockBackend):
    """Clone-style engine: it needs reference audio, so each turn must bring its own speaker's voice."""

    id = "castmock"

    def __init__(self) -> None:
        super().__init__()
        self.prepared: list = []

    def capabilities(self, variant=None):
        return super().capabilities(variant).model_copy(update={"requires_reference_audio": True})

    def prepare_reference(self, reference, variant):
        self.prepared.append(reference)
        return super().prepare_reference(reference, variant)


@pytest.fixture
def castmock(app, monkeypatch):
    registry = EngineRegistry()
    engine = CastMock()
    registry.register(engine)
    manager = ModelManager(registry, GPUManager("cpu"), "cpu")
    app.dependency_overrides[get_model_manager] = lambda: manager
    app.dependency_overrides[get_engine_registry] = lambda: registry
    monkeypatch.setattr(generation_service, "get_model_manager", lambda: manager)
    yield engine
    app.dependency_overrides.clear()


def voice(session: Session, name: str, slug: str) -> VoiceProfile:
    profile = VoiceProfile(name=name, slug=slug)
    session.add(profile)
    session.flush()
    ref = ReferenceAudio(profile_id=profile.id, sha256=hashlib.sha256(slug.encode()).hexdigest(),
                         original_name=f"{slug}.wav", format="wav",
                         duration_s=6.0, sample_rate=24000, channels=1, size_bytes=10, status=ReferenceStatus.ANALYZED)
    session.add(ref)
    session.flush()
    profile.primary_reference_id = ref.id
    session.add(profile)
    session.commit()
    return profile


@pytest.fixture
def cast(client, castmock):
    with Session(get_engine()) as session:
        ana, luis = voice(session, "Ana", "ana"), voice(session, "Luis", "luis")
        return {"Ana": ana.id, "Luis": luis.id}


def test_the_plan_says_which_characters_it_found(client, castmock, cast):
    plan = client.post("/api/generation/plan", json={
        "engine": "castmock", "text": SCRIPT, "profile_id": cast["Ana"]}).json()
    assert plan["detected_speakers"] == ["Ana", "Luis"]
    assert any("parece un diálogo" in w for w in plan["warnings"])  # nobody assigned a voice yet

    plan = client.post("/api/generation/plan", json={
        "engine": "castmock", "text": SCRIPT, "speakers": cast}).json()
    assert [s["speaker"] for s in plan["segments"]] == ["Ana", "Luis", "Ana"]
    assert [s["text"] for s in plan["segments"]] == ["Hola, ¿qué tal?", "Muy bien, gracias.", "Me alegro."]
    assert not any("parece un diálogo" in w for w in plan["warnings"])


def test_the_names_are_never_read_out_loud(client, castmock, cast):
    plan = client.post("/api/generation/plan", json={
        "engine": "castmock", "text": SCRIPT, "speakers": cast}).json()
    assert not any(s["text"].startswith(("Ana:", "Luis:")) for s in plan["segments"])


def test_each_character_uses_its_own_voice_and_the_turns_are_separated(client, castmock, cast):
    plan = client.post("/api/generation/plan", json={
        "engine": "castmock", "text": SCRIPT, "speakers": cast, "turn_pause_ms": 700}).json()
    assert [s["reference_name"] for s in plan["segments"]] == ["ana.wav", "luis.wav", "ana.wav"]
    assert [s["pause_after_ms"] for s in plan["segments"]] == [700, 700, 0]  # no pause after the last one


def test_a_character_without_a_voice_is_explained(client, castmock, cast):
    response = client.post("/api/generation/plan", json={
        "engine": "castmock", "text": SCRIPT, "speakers": {"Ana": cast["Ana"]}})
    assert response.status_code == 422
    body = response.json()
    assert "Luis" in body["message"] and body["details"]["personajes"] == ["Luis"]


def test_a_voice_that_no_longer_exists_is_explained(client, castmock, cast):
    response = client.post("/api/generation/plan", json={
        "engine": "castmock", "text": SCRIPT, "speakers": {**cast, "Luis": "no-existe"}})
    assert response.status_code == 404 and "Luis" in response.json()["message"]


def test_a_text_without_characters_is_rejected_in_dialogue_mode(client, castmock, cast):
    response = client.post("/api/generation/plan", json={
        "engine": "castmock", "text": "Solo un texto normal.", "speakers": cast})
    assert response.status_code == 422 and "ningún personaje" in response.json()["message"]


def test_more_characters_than_allowed_is_refused(client, castmock):
    response = client.post("/api/generation/plan", json={
        "engine": "castmock", "text": SCRIPT, "speakers": {f"P{i}": "x" for i in range(dialogue.MAX_SPEAKERS + 1)}})
    assert response.status_code == 422


@requires_ffmpeg
def test_the_dialogue_is_generated_turn_by_turn(client, castmock, cast):
    accepted = client.post("/api/generation", json={
        "engine": "castmock", "text": SCRIPT, "speakers": cast, "params": {"seed": 100}}).json()
    done = wait_done(client, accepted["job_id"])

    assert done["status"] == "COMPLETED"
    assert [s["speaker"] for s in done["segments"]] == ["Ana", "Luis", "Ana"]
    with Session(get_engine()) as session:
        segments = list(session.exec(select(GenerationSegment).where(
            GenerationSegment.generation_id == done["id"]).order_by(GenerationSegment.index)))  # type: ignore[arg-type]
        assert [s.speaker for s in segments] == ["Ana", "Luis", "Ana"]
        assert [(s.reference_snapshot or {}).get("name") for s in segments] == ["ana.wav", "luis.wav", "ana.wav"]
    assert done["text"] == SCRIPT  # the history keeps the script as it was written


@requires_ffmpeg
def test_a_preview_of_a_dialogue_is_only_the_first_turn(client, castmock, cast):
    accepted = client.post("/api/generation", json={
        "engine": "castmock", "text": SCRIPT, "speakers": cast, "preview": True}).json()
    done = wait_done(client, accepted["job_id"])
    assert [s["speaker"] for s in done["segments"]] == ["Ana"]
