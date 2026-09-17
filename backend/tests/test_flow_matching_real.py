"""Real F5-TTS / E2-TTS inference (needs f5-tts, downloaded weights and ideally a GPU). Opt-in: VOICELAB_GPU_TESTS=1."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("VOICELAB_GPU_TESTS") != "1",
                                reason="Requiere pesos reales de F5-TTS (VOICELAB_GPU_TESTS=1)")


@pytest.mark.parametrize(("engine_id", "variant"), [("f5tts", "F5TTS_v1_Base"), ("e2tts", "E2TTS_Base")])
def test_flow_matching_generates_reproducible_audio(engine_id, variant, tmp_path, monkeypatch):
    import soundfile as sf
    from f5_tts.api import F5TTS  # noqa: F401  (fails fast if the package is broken)

    from app.core.config import get_settings
    from app.engines.base import CancelToken, EngineRequest, ReferenceInput
    from app.engines.registry import EngineRegistry
    from tests.audio_fixtures import speechlike

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    engine = EngineRegistry().discover().get(engine_id)
    if not engine.weights_installed(variant):
        pytest.skip(f"pesos de {variant} no descargados")

    ref_path = tmp_path / "ref.wav"
    sf.write(ref_path, speechlike(words=10), 24_000, subtype="FLOAT")
    import torch

    engine.load(variant, "cuda" if torch.cuda.is_available() else "cpu", "auto")
    try:
        prepared = engine.prepare_reference(
            ReferenceInput(audio_path=ref_path, text="Hello everyone, this is a test.", sha256="a" * 64,
                           start_s=0.0, end_s=4.0), variant)
        request = EngineRequest(variant=variant, text="This is VoiceLab.", params={"seed": 7, "nfe_steps": 8},
                                reference=prepared)
        progress: list[float] = []
        first = engine.generate(request, lambda p, _m: progress.append(p), CancelToken())
        second = engine.generate(request, lambda *_: None, CancelToken())
    finally:
        engine.unload()
        get_settings.cache_clear()

    assert first.sample_rate == 24_000 and first.seed == 7 and first.audio.size > 24_000 * 0.3
    assert np.allclose(first.audio, second.audio)
    assert progress and progress[-1] == 1.0
    assert first.effective_params["nfe_steps"] == 8
