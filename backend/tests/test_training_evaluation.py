"""Trained voice vs normal cloning on held-out recordings: generation, metrics, samples and verdict (all simulated)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from app.core.errors import AppError
from app.engines.base import CancelToken, EngineResult, PreparedReference
from app.models.entities import TrainingRun
from app.training import evaluation

SR = 24_000


def tone(freq: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeEngine:
    """The trained voice sounds like the real recordings (220 Hz); normal cloning a bit off (260 Hz)."""

    def __init__(self, freq: float, needs_reference: bool = False) -> None:
        self.freq, self.requests, self.prepared = freq, [], []
        self.needs_reference = needs_reference

    def capabilities(self, variant=None):
        return SimpleNamespace(requires_reference_audio=self.needs_reference)

    def validate_parameters(self, params, variant):
        return dict(params)

    def prepare_reference(self, reference, variant):
        self.prepared.append(reference)
        return PreparedReference(engine="qwen3tts", variant=variant, cache_key="k")

    def generate(self, request, progress, cancel):
        self.requests.append(request)
        return EngineResult(audio=tone(self.freq), sample_rate=48_000 if self.freq == 260 else SR, seed=1,
                            effective_params={})


class FakeModels:
    def __init__(self) -> None:
        self.engines = {"custom:v": FakeEngine(220.0), "base-1.7b": FakeEngine(260.0)}
        self.used: list[str] = []

    @contextmanager
    def use(self, engine_id, variant):
        self.used.append(variant)
        yield self.engines[variant]


class FakeEncoder:
    def model_installed(self):
        return True

    def embed(self, audio, sample_rate):
        spectrum = np.abs(np.fft.rfft(np.asarray(audio)[: sample_rate // 2]))[:2000]
        return spectrum / (np.linalg.norm(spectrum) or 1.0)


class FakeASR:
    def __init__(self, installed=True) -> None:
        self.backend = SimpleNamespace(package_available=lambda: True, model_installed=lambda: installed)

    def transcribe(self, audio, language):
        return SimpleNamespace(text="hola a todos", language=language or "es", model="fake")


@pytest.fixture
def run(tmp_path, settings):
    path = tmp_path / "real.wav"
    sf.write(path, np.concatenate([tone(220.0, 4.0), tone(220.0, 4.0)]), SR)
    clip = {"audio_path": str(path), "text": "hola a todos", "start_s": 0.0, "end_s": 4.0}
    return TrainingRun(id="r1", profile_id="p", base_variant="base-1.7b", name="x", dataset={
        "held_out": [clip, {**clip, "start_s": 4.0, "end_s": 8.0, "text": "hola amigos"}],
        "reference": {**clip, "end_s": 3.0}})


def _patch(monkeypatch, encoder=True, asr=True):
    from app.asr import manager as asr_manager
    from app.evaluation import speaker

    enc = FakeEncoder()
    if not encoder:
        enc.model_installed = lambda: False
    monkeypatch.setattr(speaker, "get_speaker_encoder", lambda: enc)
    monkeypatch.setattr(asr_manager, "get_asr_manager", lambda: FakeASR(installed=asr))


def test_compare_generates_both_systems_and_saves_samples(run, monkeypatch):
    _patch(monkeypatch)
    models = FakeModels()
    session = SimpleNamespace(get=lambda *_: SimpleNamespace(language="Español"))
    result = evaluation.compare(session, models, run, "custom:v", SimpleNamespace(cancel=CancelToken()))

    assert models.used == ["custom:v", "base-1.7b"]
    trained, normal = models.engines["custom:v"], models.engines["base-1.7b"]
    assert [r.text for r in trained.requests] == ["hola a todos", "hola amigos"]
    assert [r.params["seed"] for r in normal.requests] == [1000, 1001]  # same seeds for both systems
    assert trained.requests[0].reference is None and normal.requests[0].reference.cache_key == "k"
    assert normal.prepared[0].end_s == 3.0 and normal.prepared[0].text == "hola a todos"

    systems = result["systems"]
    assert systems["trained"]["similarity"] > systems["normal"]["similarity"]
    assert systems["real"]["similarity"] == pytest.approx(1.0, abs=1e-3)
    assert systems["trained"]["wer"] == systems["normal"]["wer"] == 0.5  # the fake ASR always hears the first text
    assert result["verdict"] == ("La voz entrenada se parece más a tu voz real que la clonación normal y se entiende "
                                 "igual (estimación automática).")
    assert result["held_out"] == 2 and result["missing"] == []
    files = sorted(p.name for p in evaluation.samples_dir("r1").iterdir())
    assert files == ["0_normal.wav", "0_real.wav", "0_trained.wav", "1_normal.wav", "1_real.wav", "1_trained.wav"]
    assert sf.info(evaluation.samples_dir("r1") / "0_normal.wav").samplerate == SR  # resampled from 48 kHz


def test_compare_without_evaluators_or_held_out_clips(run, monkeypatch):
    _patch(monkeypatch, encoder=False, asr=False)
    session = SimpleNamespace(get=lambda *_: None)
    result = evaluation.compare(session, FakeModels(), run, "custom:v", SimpleNamespace(cancel=CancelToken()))
    assert result["systems"] == {"trained": {}, "normal": {}}
    assert result["missing"] == ["similarity", "wer"] and "escucha las muestras" in result["verdict"]

    run.dataset = {"held_out": []}
    assert "no se reservan" in evaluation.compare(session, FakeModels(), run, "custom:v", None)["skipped"]


def test_compare_stops_when_cancelled(run, monkeypatch):
    _patch(monkeypatch)
    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(AppError):
        evaluation.compare(SimpleNamespace(get=lambda *_: None), FakeModels(), run, "custom:v",
                           SimpleNamespace(cancel=cancel))


@pytest.mark.parametrize(("trained", "normal", "text"), [
    ({"similarity": 0.90, "wer": 0.10}, {"similarity": 0.95, "wer": 0.02}, "se parece menos"),
    ({"similarity": 0.95, "wer": 0.10}, {"similarity": 0.95, "wer": 0.02},
     "igual que la clonación normal y se entiende peor"),
    ({"wer": 0.01}, {"wer": 0.10}, "La voz entrenada se entiende mejor"),
])
def test_verdict_wording(trained, normal, text):
    assert text in evaluation._verdict({"trained": trained, "normal": normal})
