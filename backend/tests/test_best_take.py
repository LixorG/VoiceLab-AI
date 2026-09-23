"""Best-of-N takes: the score, the checks that need no model, and the job keeping the winner."""

from __future__ import annotations

import numpy as np
import pytest

from app.generation import best_take
from app.services.generation_service import take_seeds

SR = 24_000


def audio(seconds: float = 1.0, level: float = 0.3, silence_s: float = 0.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    voice = (level * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    return np.concatenate([voice, np.zeros(int(SR * silence_s), np.float32)])


def score(take: int, seconds=1.0, wer=None, similarity=None, level=0.3, silence_s=0.0) -> best_take.TakeScore:
    s = best_take.measure(take, 100 + take, audio(seconds, level, silence_s), SR)
    s.wer, s.similarity = wer, similarity
    return s


def test_measure_reports_duration_speech_and_peak():
    s = best_take.measure(0, 7, audio(2.0, 0.5, silence_s=2.0), SR)
    assert s.duration_s == 4.0 and s.peak == pytest.approx(0.5, abs=0.01)
    assert 0.45 < s.speech_ratio < 0.55  # half of it is silence
    assert best_take.measure(1, None, np.zeros(10, np.float32), SR).speech_ratio == 0.0


def test_the_take_that_reads_the_text_best_wins():
    takes = [score(0, wer=0.25, similarity=0.95), score(1, wer=0.0, similarity=0.9), score(2, wer=0.1, similarity=0.92)]
    ranked = best_take.rank(takes)
    assert [t.take for t in ranked] == [1, 2, 0]
    assert "25 % de palabras distintas" in " ".join(ranked[-1].notes)
    assert "lee 0 % de palabras distintas del texto (la peor, 25 %)" in best_take.reason(ranked[0], takes)


def test_similarity_decides_when_the_words_are_the_same():
    takes = [score(0, wer=0.0, similarity=0.80), score(1, wer=0.0, similarity=0.93)]
    winner = best_take.rank(takes)[0]
    assert winner.take == 1 and "se parece 0.93" in best_take.reason(winner, takes)


def test_checks_that_need_no_model_discard_runaway_silent_or_clipped_takes():
    takes = [score(0, seconds=1.0), score(1, seconds=3.0), score(2, seconds=1.0, silence_s=2.0),
             score(3, seconds=1.0, level=1.0)]
    ranked = best_take.rank(takes)
    assert ranked[0].take == 0
    notes = {t.take: " ".join(t.notes) for t in ranked}
    assert "más larga" in notes[1] and "silencio" in notes[2] and "satura" in notes[3]
    assert best_take.reason(ranked[0], takes).startswith("Mejor toma de 4")


def test_a_tie_keeps_the_first_take_and_says_so():
    takes = [score(0, wer=0.0, similarity=0.9), score(1, wer=0.0, similarity=0.9)]
    ranked = best_take.rank(takes)
    assert ranked[0].take == 0 and "se conserva la primera" in best_take.reason(ranked[0], takes)
    assert best_take.reason(ranked[0], takes[:1]) is None  # a single take is not a choice


def test_summary_keeps_every_take_in_order():
    takes = best_take.rank([score(1, wer=0.2), score(0, wer=0.0)])
    data = best_take.summary(takes, takes[0], segment=3)
    assert data["chosen"] == 0 and data["segment"] == 3
    assert [t["take"] for t in data["takes"]] == [0, 1] and data["takes"][0]["seed"] == 100


@pytest.mark.parametrize(("params", "takes", "expected"), [
    ({"seed": 7}, 1, [7]),
    ({"seed": 7}, 3, [7, 8, 9]),
    ({"seed": None}, 3, [None, None, None]),
    ({}, 2, [None, None]),
])
def test_take_seeds_are_reproducible(params, takes, expected):
    assert take_seeds(params, takes) == expected


# ---------------------------------------------------------------------------- through the API
from app.core.gpu import GPUManager  # noqa: E402
from app.engines.manager import ModelManager, get_model_manager  # noqa: E402
from app.engines.mock.plugin import MockBackend  # noqa: E402
from app.engines.registry import EngineRegistry, get_engine_registry  # noqa: E402
from app.services import generation_service  # noqa: E402
from tests.audio_fixtures import requires_ffmpeg  # noqa: E402
from tests.test_generation import wait_done  # noqa: E402


class TakeMock(MockBackend):
    """Every third seed rambles on (double length): the kind of take best-of-N is meant to drop."""

    id = "takemock"

    def __init__(self) -> None:
        super().__init__()
        self.seeds: list[int] = []

    def generate(self, request, progress, cancel, on_audio=None):
        result = super().generate(request, progress, cancel)
        self.seeds.append(result.seed)
        if result.seed % 3 == 1:
            return result.model_copy(update={"audio": np.concatenate([result.audio, result.audio])})
        return result


@pytest.fixture
def takemock(app, monkeypatch):
    registry = EngineRegistry()
    engine = TakeMock()
    registry.register(engine)
    manager = ModelManager(registry, GPUManager("cpu"), "cpu")
    app.dependency_overrides[get_model_manager] = lambda: manager
    app.dependency_overrides[get_engine_registry] = lambda: registry
    monkeypatch.setattr(generation_service, "get_model_manager", lambda: manager)
    yield engine
    app.dependency_overrides.clear()


@requires_ffmpeg
def test_the_job_keeps_the_best_take_and_records_the_others(client, takemock):
    body = {"engine": "takemock", "text": "Una frase de prueba.", "params": {"seed": 100}, "takes": 3}
    done = wait_done(client, client.post("/api/generation", json=body).json()["job_id"])

    assert done["status"] == "COMPLETED" and done["takes"] == 3
    assert takemock.seeds == [100, 101, 102]  # consecutive seeds from the one that was asked for
    best = done["metrics"]["best_take"]
    assert done["metrics"]["takes"] == 3 and len(best) == 1
    assert best[0]["chosen"] == 1  # seed 100 rambles on; of the two good takes the first one wins the tie
    assert [t["seed"] for t in best[0]["takes"]] == [100, 101, 102]
    assert "más larga que las demás" in " ".join(best[0]["takes"][0]["notes"])
    assert any("Mejor toma de 3" in w for w in done["warnings"])
    assert all(t["wer"] is not None for t in best[0]["takes"])  # the words were checked with the ASR
    assert abs(done["duration_s"] - len("Una frase de prueba.") * 0.06) < 0.3  # the take that rambles was dropped


@requires_ffmpeg
def test_previews_and_variations_use_a_single_take(client, takemock):
    preview = wait_done(client, client.post("/api/generation", json={
        "engine": "takemock", "text": "Una frase de prueba.", "takes": 3, "preview": True}).json()["job_id"])
    assert preview["takes"] == 1 and preview["metrics"].get("best_take") is None
    accepted = client.post("/api/generation/variations", json={
        "engine": "takemock", "text": "Otra frase.", "takes": 3, "count": 2}).json()
    for item in accepted:
        assert wait_done(client, item["job_id"])["takes"] == 1


@requires_ffmpeg
def test_without_evaluators_it_says_how_it_chose(client, takemock, monkeypatch):
    from app.asr.manager import get_asr_manager

    monkeypatch.setattr(get_asr_manager().backend, "model_installed", lambda: False)
    done = wait_done(client, client.post("/api/generation", json={
        "engine": "takemock", "text": "Una frase de prueba.", "params": {"seed": 100}, "takes": 2}).json()["job_id"])
    assert any("no hay evaluadores instalados" in w for w in done["warnings"])
    takes = done["metrics"]["best_take"][0]["takes"]
    assert all(t["wer"] is None and t["similarity"] is None for t in takes)
    assert done["metrics"]["best_take"][0]["chosen"] == 1  # only the length check could tell them apart
