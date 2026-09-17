"""Real Qwen3-TTS voice cloning (needs qwen-tts + downloaded Base weights + GPU). Opt-in: VOICELAB_GPU_TESTS=1.

Run with `-s` to print timing and VRAM measurements.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("VOICELAB_GPU_TESTS") != "1",
                                reason="Requiere pesos reales de Qwen3-TTS (VOICELAB_GPU_TESTS=1)")

REF_TEXT = os.environ.get("VOICELAB_REF_TEXT", "")
REF_WAV = os.environ.get("VOICELAB_REF_WAV", "")


@pytest.mark.parametrize("variant", ["base-0.6b", "base-1.7b"])
def test_qwen_voice_clone(variant, tmp_path, monkeypatch):
    import soundfile as sf
    import torch

    from app.core.config import get_settings
    from app.engines.base import CancelToken, EngineRequest, ReferenceInput
    from app.engines.qwen3tts.plugin import Qwen3TTSBackend
    from tests.audio_fixtures import speechlike

    engine = Qwen3TTSBackend()
    if not engine.weights_installed(variant):
        pytest.skip(f"pesos de {variant} no descargados")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    if REF_WAV:
        ref_path, ref_text = REF_WAV, REF_TEXT
    else:
        ref_path, ref_text = tmp_path / "ref.wav", "Hello everyone, this is a test."
        sf.write(ref_path, speechlike(words=10), 24_000, subtype="FLOAT")
    info = sf.info(ref_path)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    started = time.perf_counter()
    engine.load(variant, "cuda", "auto")
    load_s = time.perf_counter() - started
    weights_mb = (torch.cuda.memory_allocated() - base_alloc) / 2**20
    try:
        prepared = engine.prepare_reference(
            ReferenceInput(audio_path=ref_path, text=ref_text, sha256="b" * 64, start_s=0.0,
                           end_s=min(info.duration, 12.0)), variant)
        text = "The future belongs to those who prepare for it."
        results = {}
        for mode in ("icl", "x_vector"):
            req = EngineRequest(variant=variant, text=text, params={"seed": 11, "clone_mode": mode,
                                                                   "language": "English"}, reference=prepared)
            t0 = time.perf_counter()
            first = engine.generate(req, lambda *_: None, CancelToken())
            gen_s = time.perf_counter() - t0
            second = engine.generate(req, lambda *_: None, CancelToken())
            duration = first.audio.size / first.sample_rate
            same = first.audio.size == second.audio.size and bool(np.allclose(first.audio, second.audio, atol=1e-4))
            results[mode] = (duration, gen_s, same)
            sf.write(tmp_path / f"{variant}_{mode}.wav", first.audio, first.sample_rate)
            assert first.sample_rate == 24_000 and 0.8 < duration < 15, (mode, duration)
            assert first.details["attention"] in ("sdpa", "flash_attention_2")
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
        print(f"\n[{variant}] carga {load_s:.1f}s · pesos {weights_mb:.0f} MB · pico {peak_mb:.0f} MB · "
              f"dtype {first.details['dtype']} · atención {first.details['attention']}")
        for mode, (duration, gen_s, same) in results.items():
            print(f"  {mode}: {duration:.2f}s de audio en {gen_s:.2f}s (RTF {gen_s / duration:.2f}) · "
                  f"misma semilla idéntica: {same}")
        if os.environ.get("VOICELAB_KEEP_AUDIO"):
            import shutil

            out = os.environ["VOICELAB_KEEP_AUDIO"]
            for f in tmp_path.glob(f"{variant}_*.wav"):
                shutil.copy(f, out)
    finally:
        engine.unload()
        torch.cuda.empty_cache()
        get_settings.cache_clear()
