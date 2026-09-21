"""Speaker similarity: the evaluator math, availability, skips, the reference clip it compares against and the API."""

from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.base import EvaluationInput
from app.evaluation.speaker import (
    SAME_SPEAKER_THRESHOLD,
    SpeakerEncoder,
    SpeakerSimilarityEvaluator,
    cosine,
    get_speaker_encoder,
    to_16k,
)
from tests.fakes import tone


class FakeEncoder(SpeakerEncoder):
    """Embeds by pitch: tones of the same frequency are «the same speaker»."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []

    def model_installed(self) -> bool:
        return True

    def embed(self, audio, sample_rate):
        wav = to_16k(audio, sample_rate)
        self.calls.append(wav.size)
        spectrum = np.abs(np.fft.rfft(wav[:16_000]))
        return spectrum / (np.linalg.norm(spectrum) or 1.0)


def _input(gen_freq=220.0, ref_freq=220.0, ref_seconds=3.0, reference=True):
    return EvaluationInput(audio=tone(3.0, gen_freq), sample_rate=24_000, target_text="hola",
                           reference_audio=tone(ref_seconds, ref_freq) if reference else None,
                           reference_sample_rate=24_000 if reference else None)


def test_cosine_and_resampling():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 0]) == 0.0
    assert to_16k(np.zeros(24_000), 24_000).size == 16_000
    assert to_16k(np.zeros(16_000), 16_000).size == 16_000


def test_similar_and_different_voices_score_accordingly():
    evaluator = SpeakerSimilarityEvaluator(FakeEncoder())
    same = evaluator.evaluate(_input(220, 220)).metrics[0]
    other = evaluator.evaluate(_input(220, 3000)).metrics[0]
    assert same.id == "speaker_similarity" and same.better == "higher"
    assert same.value > SAME_SPEAKER_THRESHOLD > other.value
    assert "por encima" in same.description and "por debajo" in other.description
    assert "acento" in same.description  # says what it does not measure


def test_skips_without_reference_or_with_too_little_audio():
    evaluator = SpeakerSimilarityEvaluator(FakeEncoder())
    no_ref = evaluator.evaluate(_input(reference=False))
    assert no_ref.metrics == [] and "no usó una grabación de referencia" in no_ref.details["skipped"]
    short = evaluator.evaluate(_input(ref_seconds=0.4))
    assert short.metrics == [] and "al menos" in short.details["skipped"]


def test_unavailable_until_downloaded(monkeypatch):
    encoder = SpeakerEncoder()
    info = SpeakerSimilarityEvaluator(encoder).info()
    assert info.available is False and "no está descargado" in info.reason and "MB" in info.reason
    monkeypatch.setattr(encoder, "_download_state", "downloading")
    assert "Descargando" in SpeakerSimilarityEvaluator(encoder).info().reason


def test_download_runs_in_background(monkeypatch, tmp_path):
    from tests.fakes import FakeHub

    hub = FakeHub(root=tmp_path / "hub")
    hub.install(monkeypatch)
    encoder = SpeakerEncoder()
    encoder._download()  # the thread body, run inline
    assert {f for _, f in hub.downloads} == {"config.json", "preprocessor_config.json", "pytorch_model.bin"}
    assert encoder.status().download_state == "idle"

    def boom(**_):
        raise OSError("sin red")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    encoder._download()
    assert encoder.status().download_state == "failed" and "conexión" in encoder.status().download_error


def test_api_status_and_evaluators_listing(client):
    status = client.get("/api/generation/evaluators/speaker-model").json()
    assert status["model"] == "microsoft/wavlm-base-plus-sv" and status["installed"] is False
    assert status["download_size_mb"] == 405 and "MIT" in status["license"]

    evaluators = {e["id"]: e for e in client.get("/api/generation/evaluators").json()}
    assert evaluators["speaker_similarity"]["available"] is False
    assert evaluators["naturalness"]["available"] is False  # still planned, not simulated


def test_evaluation_compares_against_the_cloned_reference_range(client, monkeypatch, settings):
    """End to end: the generation's reference range is read from the processed WAV and compared."""
    import soundfile as sf
    from sqlmodel import Session

    from app.core.database import get_engine
    from app.models.entities import Generation
    from app.models.enums import JobStatus

    fake = FakeEncoder()
    monkeypatch.setattr("app.evaluation.registry.get_speaker_encoder", lambda: fake)

    sha = "a" * 64
    processed = settings.data_dir / "references" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    sf.write(processed / f"{sha}.wav", np.concatenate([tone(2.0, 3000), tone(3.0, 220)]), 24_000)
    out_dir = settings.data_dir / "generated" / "g1"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "final.wav", tone(3.0, 220), 24_000, subtype="PCM_24")

    with Session(get_engine()) as session:
        gen = Generation(id="g1", engine="mock", text="hola", status=JobStatus.COMPLETED,
                         output_path="generated/g1/final.wav",
                         reference_snapshot={"reference_id": "r1", "sha256": sha, "start_s": 2.0, "end_s": 5.0,
                                             "name": "ref.wav", "text": "hola"})
        session.add(gen)
        session.commit()

    res = client.post("/api/generation/g1/evaluate")
    assert res.status_code == 200, res.text
    metrics = {m["id"]: m for m in res.json()["evaluation"]["metrics"]}
    assert metrics["speaker_similarity"]["value"] > SAME_SPEAKER_THRESHOLD  # compared with the 220 Hz range only
    assert fake.calls == [48_000, 48_000]  # 3 s at 16 kHz on both sides: the cut range, not the whole file


def test_real_encoder_refuses_without_weights():
    from app.core.errors import AppError

    with pytest.raises(AppError) as exc:
        get_speaker_encoder().embed(tone(2.0), 24_000)
    assert "no está descargado" in exc.value.message


def test_long_audio_is_embedded_in_windows(monkeypatch):
    import torch

    from app.evaluation import speaker

    sizes: list[int] = []

    class Model:
        def __call__(self, input_values):
            sizes.append(input_values.shape[-1])
            return type("Out", (), {"embeddings": torch.tensor([[1.0, float(len(sizes))]])})()

    encoder = SpeakerEncoder()
    monkeypatch.setattr(encoder, "_load", lambda: None)
    encoder._model = Model()
    encoder._extractor = lambda wav, **_: {"input_values": torch.tensor(wav)[None]}
    emb = encoder.embed(tone(45.5, sr=16_000), 16_000)  # 20 s + 20 s + 5.5 s
    assert sizes == [20 * 16_000, 20 * 16_000, int(5.5 * 16_000)]
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-6
    sizes.clear()
    encoder.embed(tone(20.5, sr=16_000), 16_000)  # a sub-second tail is dropped
    assert sizes == [20 * 16_000]
    assert speaker.WINDOW_SECONDS == 20
