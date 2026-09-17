"""Real engine adapters (F5/E2, Qwen3, faster-whisper) driven by fake libraries — "mock GPU" tests.

These run everywhere (no CUDA, weights or model packages needed) and check what the adapters send to the libraries.
The opt-in GPU tests (test_*_real.py) cover the libraries themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.asr import faster_whisper_backend as fw_module
from app.asr.faster_whisper_backend import FasterWhisperBackend
from app.core.errors import AppError, ErrorCode
from app.engines.base import CancelToken, EngineRequest, ReferenceInput
from app.engines.e2tts.plugin import E2TTSBackend
from app.engines.f5tts.plugin import F5TTSBackend
from app.engines.qwen3tts.plugin import Qwen3TTSBackend
from tests.fakes import SR, FakeF5TTS, FakeHub, FakeQwenModel, FakeWhisperModel, install_module, tone


@pytest.fixture
def reference(settings, tmp_path):
    path = tmp_path / "processed.wav"
    sf.write(path, tone(6.0), SR, subtype="FLOAT")
    return ReferenceInput(audio_path=path, text=" Texto de referencia. ", sha256="c" * 64, start_s=1.0, end_s=4.0)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Fake Hub; also stops the adapters from touching the real Hugging Face cache on this machine."""
    from app.engines import flow_matching
    from app.engines.qwen3tts import plugin as qwen_plugin

    fake = FakeHub(tmp_path / "hub")
    fake.install(monkeypatch)
    monkeypatch.setattr(flow_matching, "prepare_hf_cache", lambda: None)
    monkeypatch.setattr(qwen_plugin, "prepare_hf_cache", lambda: None)
    return fake


def _no_progress(p, msg):  # noqa: ARG001
    return None


# ---------------------------------------------------------------- F5 / E2
@pytest.fixture
def f5(monkeypatch, hub):
    FakeF5TTS.instances.clear()
    install_module(monkeypatch, "f5_tts.api", F5TTS=FakeF5TTS)
    removed = []
    install_module(monkeypatch, "f5_tts.infer.utils_infer",
                   remove_silence_for_generated_wav=lambda path: removed.append(path) or sf.write(
                       path, tone(0.5), SR))
    engine = F5TTSBackend()
    monkeypatch.setattr(engine, "is_installed", lambda: True)
    return engine, hub, removed


def test_flow_matching_weights_and_download(f5):
    engine, hub, _ = f5
    variant = engine.default_variant()
    assert not engine.weights_installed(variant)
    engine.download_weights(variant)
    repo, filename = engine.checkpoints[variant]
    assert (repo, filename) in hub.downloads and ("charactr/vocos-mel-24khz", "config.yaml") in hub.downloads
    assert engine.weights_installed(variant)

    e2 = E2TTSBackend()
    assert e2.checkpoints["E2TTS_Base"][0] == "SWivid/E2-TTS"


def test_flow_matching_load_generate_maps_parameters(f5, reference):
    engine, hub, removed = f5
    variant = engine.default_variant()
    engine.download_weights(variant)
    engine.load(variant, "cuda", "auto")
    engine.load(variant, "cuda", "auto")  # already loaded: no second model
    assert len(FakeF5TTS.instances) == 1
    model = FakeF5TTS.instances[0]
    assert model.device == "cuda" and model.model == variant
    assert Path(model.ckpt_file).as_posix().endswith(engine.checkpoints[variant][1])

    prepared = engine.prepare_reference(reference, variant)
    assert sf.info(prepared.payload["audio_path"]).duration == pytest.approx(3.0, abs=0.01)
    assert prepared.payload["text"] == "Texto de referencia."
    assert engine.prepare_reference(reference, variant).cache_key == prepared.cache_key  # cached clip

    fractions = []
    result = engine.generate(
        EngineRequest(variant=variant, text="Hola", reference=prepared,
                      params={"speed": 0.5, "nfe_steps": 16, "seed": 7}),
        progress=lambda p, msg: fractions.append(p), cancel=CancelToken())
    call = model.calls[-1]
    assert (call["nfe_step"], call["speed"], call["seed"], call["remove_silence"]) == (16, 0.5, 7, False)
    assert call["ref_text"] == "Texto de referencia." and call["gen_text"] == "Hola"
    assert result.seed == 7 and result.audio.size == SR * 2 and fractions == pytest.approx([1 / 3, 2 / 3, 1.0])

    random_seed = engine.generate(EngineRequest(variant=variant, text="Hola", reference=prepared, params={}),
                                  progress=_no_progress, cancel=CancelToken())
    assert isinstance(random_seed.seed, int) and random_seed.effective_params["seed"] == random_seed.seed

    trimmed = engine.generate(EngineRequest(variant=variant, text="Hola", reference=prepared,
                                            params={"remove_silence": True}), progress=_no_progress,
                              cancel=CancelToken())
    assert removed and trimmed.audio.size == SR // 2  # library routine applied on a temporary file


def test_flow_matching_errors_and_cancellation(f5, reference):
    engine, *_ = f5
    variant = engine.default_variant()
    request = EngineRequest(variant=variant, text="Hola", reference=None, params={})
    with pytest.raises(AppError) as err:
        engine.generate(request, _no_progress, CancelToken())
    assert err.value.code is ErrorCode.MODEL_LOAD_ERROR

    engine.download_weights(variant)
    engine.load(variant, "cpu", "auto")
    with pytest.raises(AppError) as err:
        engine.generate(request, _no_progress, CancelToken())
    assert err.value.code is ErrorCode.REFERENCE_REQUIRED

    with pytest.raises(AppError) as err:
        engine.prepare_reference(reference.model_copy(update={"text": "  "}), variant)
    assert err.value.code is ErrorCode.REFERENCE_TEXT_REQUIRED

    prepared = engine.prepare_reference(reference, variant)
    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(AppError) as err:
        engine.generate(request.model_copy(update={"reference": prepared}), _no_progress, cancel)
    assert err.value.code is ErrorCode.JOB_CANCELLED

    engine.unload()
    assert engine.loaded_variant is None


# ---------------------------------------------------------------- Qwen3
@pytest.fixture
def qwen(monkeypatch, hub):
    import torch

    FakeQwenModel.loaded.clear()
    module = install_module(monkeypatch, "qwen_tts", Qwen3TTSModel=type("Q", (), {}))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    def use(model_type: str) -> Qwen3TTSBackend:
        module.Qwen3TTSModel.from_pretrained = staticmethod(FakeQwenModel.factory(model_type))
        engine = Qwen3TTSBackend()
        monkeypatch.setattr(engine, "is_installed", lambda: True)
        return engine

    return use, hub


def test_qwen_load_uses_bf16_sdpa_and_checks_model_type(qwen):
    use, hub = qwen
    engine = use("base")
    engine.load("base-0.6b", "cuda", "auto")
    loaded = FakeQwenModel.loaded[-1]
    assert str(loaded["dtype"]) == "torch.bfloat16" and loaded["device_map"] == "cuda:0"
    assert loaded["attn_implementation"] in ("sdpa", "flash_attention_2")
    assert hub.snapshots[-1].endswith("|local") and engine.loaded_variant == "base-0.6b"
    assert engine.supported_speakers() == ["Vivian", "ryan"]

    mismatch = use("custom_voice")
    with pytest.raises(AppError) as err:
        mismatch.load("base-1.7b", "cpu", "auto")
    assert err.value.code is ErrorCode.MODEL_LOAD_ERROR and err.value.details["encontrado"] == "custom_voice"
    assert mismatch.loaded_variant is None


def test_qwen_clone_caches_prompt_and_maps_sampling(qwen, reference):
    use, _ = qwen
    engine = use("base")
    engine.load("base-0.6b", "cpu", "auto")
    prepared = engine.prepare_reference(reference, "base-0.6b")
    params = {"seed": 11, "temperature": 0.7, "top_k": 30, "language": "Spanish"}
    request = EngineRequest(variant="base-0.6b", text="Hola", reference=prepared, params=params)
    first = engine.generate(request, _no_progress, CancelToken())
    engine.generate(request, _no_progress, CancelToken())
    model = engine._model
    assert len(model.prompt_calls) == 1 and model.prompt_calls[0]["text"] == "Texto de referencia."
    name, kwargs = model.calls[-1]
    assert name == "clone" and kwargs["voice_clone_prompt"] == ["prompt-1"]
    sampling = (kwargs["temperature"], kwargs["top_k"], kwargs["language"], kwargs["max_new_tokens"])
    assert sampling[:3] == (0.7, 30, "Spanish") and 60 < sampling[3] < 8192  # runaway limit from the text length
    assert "seed" not in kwargs and first.seed == 11 and first.details["dtype"] == "float32"

    engine.generate(request.model_copy(update={"params": {**params, "clone_mode": "x_vector"}}), _no_progress,
                    CancelToken())
    assert model.prompt_calls[-1]["x_vector"] is True and model.prompt_calls[-1]["text"] is None

    no_text = engine.prepare_reference(reference.model_copy(update={"text": None}), "base-0.6b")
    with pytest.raises(AppError) as err:
        engine.generate(request.model_copy(update={"reference": no_text}), _no_progress, CancelToken())
    assert err.value.code is ErrorCode.REFERENCE_TEXT_REQUIRED

    model.seconds = 8192 / 12.5  # the model never ends the sentence: stopped by the runaway limit, not after 11 min
    looping = engine.generate(request, _no_progress, CancelToken())
    assert any("posible bucle" in w for w in looping.warnings) and looping.audio.size / SR < 20


def test_qwen_reports_estimated_progress_and_stops_mid_generation(qwen, reference):
    use, _ = qwen
    engine = use("base")
    engine.load("base-0.6b", "cpu", "auto")
    prepared = engine.prepare_reference(reference, "base-0.6b")
    text = "Una frase de prueba con suficiente longitud para varios segundos de audio generado."
    request = EngineRequest(variant="base-0.6b", text=text, reference=prepared, params={"seed": 1})
    engine._model.seconds = 5.0
    updates: list[tuple[float, str | None]] = []
    engine.generate(request, lambda p, m: updates.append((p, m)), CancelToken())
    fractions = [p for p, _ in updates]
    assert fractions == sorted(fractions) and max(fractions) <= 1.0
    assert any("s de audio" in (m or "") for _, m in updates)
    assert engine._model.model.talker.hooks == []  # hook removed after generating

    cancel = CancelToken()
    engine._model.seconds = 60.0

    def cancel_after_two_seconds(p, m):
        if m and "~2 s" in m:
            cancel.cancel()

    with pytest.raises(AppError) as err:
        engine.generate(request, cancel_after_two_seconds, cancel)
    assert err.value.code is ErrorCode.JOB_CANCELLED and engine._model.model.talker.hooks == []


def test_qwen_custom_voice_and_design_modes(qwen):
    use, _ = qwen
    custom = use("custom_voice")
    custom.load("custom-voice-0.6b", "cpu", "auto")
    custom.generate(EngineRequest(variant="custom-voice-0.6b", text="Hola",
                                  params={"speaker": "Vivian", "instruct": "Alegre"}), _no_progress, CancelToken())
    name, kwargs = custom._model.calls[-1]
    assert name == "custom_voice" and kwargs["speaker"] == "Vivian" and kwargs["instruct"] == "Alegre"

    design = use("voice_design")
    design.load("voice-design-1.7b", "cpu", "auto")
    design.generate(EngineRequest(variant="voice-design-1.7b", text="Hola",
                                  params={"instruct": "Voz grave y pausada"}), _no_progress, CancelToken())
    assert design._model.calls[-1][0] == "voice_design"

    clone = use("base")
    clone.load("base-1.7b", "cpu", "auto")
    with pytest.raises(AppError) as err:
        clone.generate(EngineRequest(variant="base-1.7b", text="Hola", params={}), _no_progress, CancelToken())
    assert err.value.code is ErrorCode.REFERENCE_REQUIRED
    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(AppError) as err:
        custom.generate(EngineRequest(variant="custom-voice-0.6b", text="Hola", params={}), _no_progress, cancel)
    assert err.value.code is ErrorCode.JOB_CANCELLED


def test_qwen_weights_download(qwen):
    use, hub = qwen
    engine = use("base")
    assert not engine.weights_installed("base-0.6b")
    engine.download_weights("base-0.6b")
    assert hub.snapshots[-1] == "Qwen/Qwen3-TTS-12Hz-0.6B-Base|remote"


# ---------------------------------------------------------------- faster-whisper
@pytest.fixture
def whisper(monkeypatch):
    FakeWhisperModel.created.clear()
    FakeWhisperModel.fail_on_cuda = False
    downloads = []
    installed = {"value": True}

    def download_model(name, local_files_only=False, cache_dir=None):
        if local_files_only and not installed["value"]:
            raise FileNotFoundError(name)
        if not local_files_only:
            downloads.append(name)
        return "path"

    install_module(monkeypatch, "faster_whisper", WhisperModel=FakeWhisperModel)
    install_module(monkeypatch, "faster_whisper.utils", download_model=download_model)
    cuda = {"count": 1}
    install_module(monkeypatch, "ctranslate2", get_cuda_device_count=lambda: cuda["count"])
    monkeypatch.setattr(fw_module.importlib.util, "find_spec",
                        lambda name: object() if name == "faster_whisper" else None)
    monkeypatch.setattr(fw_module, "_dll_dirs_added", True)
    return FasterWhisperBackend("large-v3-turbo"), installed, downloads, cuda


def test_whisper_loads_on_cuda_float16_and_parses_words(whisper):
    backend, _installed, downloads, _cuda = whisper
    assert backend.package_available() and backend.model_installed()
    backend.download_model()
    assert downloads == ["large-v3-turbo"]

    result = backend.transcribe(np.zeros(32_000, dtype=np.float32), "es", initial_prompt="Nombres")
    created = FakeWhisperModel.created[-1]
    assert (created["device"], created["compute_type"], created["local_files_only"]) == ("cuda", "float16", True)
    assert backend.loaded and (backend.device, backend.compute_type) == ("cuda", "float16")
    assert result.text == "Hola mundo." and result.language == "es" and result.duration_s == 2.0
    assert [w.word for w in result.words] == [" Hola", " mundo."] and result.words[0].probability == 0.9877
    assert backend._model.kwargs["vad_filter"] and backend._model.kwargs["word_timestamps"]
    assert backend._model.kwargs["initial_prompt"] == "Nombres"

    backend.unload()
    assert not backend.loaded
    backend.unload()  # idempotent


def test_whisper_falls_back_to_cpu_and_reports_missing_model(whisper):
    backend, installed, _downloads, cuda = whisper
    FakeWhisperModel.fail_on_cuda = True
    backend.load()
    assert [c["device"] for c in FakeWhisperModel.created] == ["cuda", "cpu"]
    assert (backend.device, backend.compute_type) == ("cpu", "int8")

    cpu_only = FasterWhisperBackend("small", device="auto", compute_type="int8_float32")
    cuda["count"] = 0
    cpu_only.load()
    assert (cpu_only.device, cpu_only.compute_type) == ("cpu", "int8_float32")

    installed["value"] = False
    missing = FasterWhisperBackend("medium")
    with pytest.raises(AppError) as err:
        missing.load()
    assert err.value.code is ErrorCode.ASR_MODEL_NOT_INSTALLED and not missing.model_installed()
