"""Experiments, manual ratings and automatic evaluation (fake ASR) — simulated engines, no GPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlmodel import Session

from app.asr.manager import get_asr_manager
from app.core.database import get_engine
from app.evaluation.intelligibility import error_rates, normalize
from app.models.entities import Generation, Transcript
from app.services.evaluation_service import target_language, target_text
from tests.audio_fixtures import requires_ffmpeg
from tests.test_advanced_generation import _upload
from tests.test_advanced_generation import engines as engines  # noqa: F401  (fixture re-export)
from tests.test_generation import wait_done


class FakeASR:
    def __init__(self, text: str = "hola a todos", installed: bool = True) -> None:
        self.text = text
        self.calls: list = []
        self.backend = SimpleNamespace(package_available=lambda: True, model_installed=lambda: installed,
                                       model_name="fake")

    def transcribe(self, audio, language, initial_prompt=None):
        self.calls.append((audio.size, language))
        return SimpleNamespace(text=self.text, language="es", model="fake")


@pytest.fixture
def asr(app):
    fake = FakeASR()
    app.dependency_overrides[get_asr_manager] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_asr_manager, None)


# ---------------------------------------------------------------- pure helpers
def test_error_rates_ignore_case_and_punctuation():
    assert normalize("¡Hola, Mundo!  ¿Qué tal?") == "hola mundo qué tal"
    assert error_rates("Hola, mundo.", "hola mundo") == (0.0, 0.0)
    wer, cer = error_rates("uno dos tres cuatro", "uno dos tres")
    assert wer == 0.25 and 0 < cer < 0.4
    assert error_rates("1,000 personas", "1000 personas") == (0.0, 0.0)


def test_target_text_removes_markup():
    gen = Generation(engine="x", text="Hola [pausa:1s] [emoción:feliz]mundo[/emoción]", expression={"markup": True})
    assert target_text(gen, []) == "Hola mundo"
    raw = Generation(engine="x", text="Hola [pausa:1s]", expression={"markup": False})
    assert target_text(raw, []) == "Hola [pausa:1s]"


# ---------------------------------------------------------------- experiments
@requires_ffmpeg
def test_experiment_generates_one_arm_per_engine(client, engines):  # noqa: F811
    body = {"text": "Hola a todos", "name": "Comparativa", "emotion": "calm",
            "arms": [{"engine": "instructmock", "params": {"seed": 1}},
                     {"engine": "instructmock", "params": {"seed": 2}},
                     {"engine": "instructmock", "label": "Mi prueba"}]}
    res = client.post("/api/experiments", json=body)
    assert res.status_code == 202, res.text
    exp = res.json()
    labels = [g["label"] for g in exp["generations"]]
    assert labels[0] != labels[1] and labels[1].endswith("(2)") and labels[2] == "Mi prueba"
    assert all(g["kind"] == "experiment" and g["experiment_id"] == exp["id"] for g in exp["generations"])

    for g in exp["generations"]:
        assert wait_done(client, g["id"])["status"] == "COMPLETED"
    detail = client.get(f"/api/experiments/{exp['id']}").json()
    assert [g["seed"] for g in detail["generations"]][:2] == [1, 2]
    assert detail["generations"][0]["expression"]["emotion"] == "calm"

    summary = client.get("/api/experiments").json()[0]
    assert (summary["name"], summary["arms"], summary["completed"], summary["engines"]) == \
        ("Comparativa", 3, 3, ["instructmock"])
    assert client.get("/api/generation").json() == []  # experiment runs stay out of the Generate history

    added = client.post(f"/api/experiments/{exp['id']}/arms", json={"arms": [{"engine": "instructmock"}]}).json()
    assert len(added["generations"]) == 4
    renamed = client.patch(f"/api/experiments/{exp['id']}", json={"name": "Nueva", "notes": "F5 suena mejor"}).json()
    assert (renamed["name"], renamed["notes"]) == ("Nueva", "F5 suena mejor")

    for g in added["generations"]:
        wait_done(client, g["id"])
    assert client.delete(f"/api/experiments/{exp['id']}").status_code == 204
    assert client.get(f"/api/experiments/{exp['id']}").status_code == 404
    assert client.get(f"/api/generation/{exp['generations'][0]['id']}").status_code == 404


def test_invalid_arm_creates_nothing(client, engines):  # noqa: F811
    res = client.post("/api/experiments", json={"text": "Hola", "arms": [
        {"engine": "instructmock"}, {"engine": "instructmock", "params": {"nope": 1}}]})
    assert res.status_code == 422 and res.json()["details"]["posicion"] == 2
    assert client.get("/api/experiments").json() == []
    assert client.post("/api/experiments", json={"text": "Hola", "arms": []}).status_code == 422
    assert client.get("/api/experiments/nope").json()["error_code"] == "EXPERIMENT_NOT_FOUND"


# ---------------------------------------------------------------- rating & evaluation
@requires_ffmpeg
def test_rating_and_evaluation(client, engines, asr):  # noqa: F811
    gen = wait_done(client, client.post("/api/generation", json={
        "engine": "instructmock", "text": "Hola [pausa:300ms] a todos, amigos"}).json()["job_id"])

    rated = client.put(f"/api/generation/{gen['id']}/rating", json={"timbre": 4, "naturalness": 5,
                                                                    "notes": "Buena"}).json()
    assert rated["rating"]["timbre"] == 4 and "emotion" not in rated["rating"]
    assert client.put(f"/api/generation/{gen['id']}/rating", json={"timbre": 9}).status_code == 422
    assert client.put(f"/api/generation/{gen['id']}/rating", json={}).json()["rating"] is None

    infos = {i["id"]: i for i in client.get("/api/generation/evaluators").json()}
    assert infos["intelligibility"]["available"] and not infos["speaker_similarity"]["available"]
    assert infos["speaker_similarity"]["reason"]

    evaluated = client.post(f"/api/generation/{gen['id']}/evaluate").json()
    ev = evaluated["evaluation"]
    metrics = {m["id"]: m for m in ev["metrics"]}
    assert ev["target_text"] == "Hola a todos, amigos" and ev["stale"] is False
    assert metrics["wer"]["value"] == 0.25 and metrics["wer"]["better"] == "lower"
    assert ev["details"]["intelligibility"]["transcript"] == "hola a todos"
    assert {u["id"] for u in ev["unavailable"]} == {"speaker_similarity", "naturalness"}
    assert asr.calls and asr.calls[0][1] is None  # 16 kHz audio, language auto-detected

    remastered = client.post(f"/api/generation/{gen['id']}/postprocess", json={"peak": {"enabled": True}}).json()
    assert remastered["evaluation"]["stale"] is True


def test_evaluation_unavailable_without_asr_model(client, engines, app):  # noqa: F811
    app.dependency_overrides[get_asr_manager] = lambda: FakeASR(installed=False)
    infos = client.get("/api/generation/evaluators").json()
    assert not any(i["available"] for i in infos)
    assert "no está descargado" in infos[0]["reason"]


@requires_ffmpeg
def test_evaluation_forces_the_text_language(client, engines, asr, tmp_path):  # noqa: F811
    """An accented voice must not be auto-detected (and translated) as another language by the ASR."""
    auto = wait_done(client, client.post("/api/generation", json={"engine": "instructmock",
                                                                  "text": "Hola"}).json()["job_id"])
    ev = client.post(f"/api/generation/{auto['id']}/evaluate").json()["evaluation"]
    assert (ev["language"], ev["language_source"]) == (None, "auto") and asr.calls[-1][1] is None

    ref = _upload(client, tmp_path, "es.wav")
    with Session(get_engine()) as session:  # transcript whose language Whisper detected as Spanish
        session.add(Transcript(reference_id=ref["id"], text="hola a todos", asr_text="hola a todos", language="es"))
        session.commit()
    cloned = wait_done(client, client.post("/api/generation", json={"engine": "segmock", "text": "Hola",
                                                                    "reference_id": ref["id"]}).json()["job_id"])
    ev = client.post(f"/api/generation/{cloned['id']}/evaluate").json()["evaluation"]
    assert (ev["language"], ev["language_source"]) == ("es", "reference") and asr.calls[-1][1] == "es"


def test_engine_language_names_map_to_asr_codes():
    gen = Generation(engine="qwen3tts", text="Hola", params={"language": "Spanish"})
    assert target_language(None, gen) == ("es", "engine")  # type: ignore[arg-type]
    assert target_language(None, Generation(engine="x", text="Hi", params={"language": "Auto"}))[0] is None  # type: ignore[arg-type]
