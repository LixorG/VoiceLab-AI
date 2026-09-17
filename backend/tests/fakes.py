"""Stand-ins for heavy model packages (f5_tts, qwen_tts, faster_whisper, huggingface_hub, CUDA).

They record how the adapters call the real libraries, so the adapter logic (parameter mapping, caching, seeds,
cancellation, device/dtype choice, error paths) is tested without GPU, weights or the packages installed.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SR = 24_000


def install_module(monkeypatch, name: str, **attrs: Any) -> types.ModuleType:
    """Register a fake module (and its parents) in sys.modules for the duration of a test."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules or not getattr(sys.modules[parent], "__fake__", False):
            pkg = types.ModuleType(parent)
            pkg.__fake__ = True  # type: ignore[attr-defined]
            pkg.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, parent, pkg)
        setattr(sys.modules[parent], parts[i], module) if i == len(parts) - 1 else None
    module.__fake__ = True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    return module


def tone(seconds: float = 1.0, freq: float = 220.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------- huggingface_hub
@dataclass
class FakeHub:
    """Local cache lookups and downloads."""

    root: Any
    cached: set[tuple[str, str]] = field(default_factory=set)
    downloads: list[tuple[str, str]] = field(default_factory=list)
    snapshots: list[str] = field(default_factory=list)

    def hf_hub_download(self, repo_id: str, filename: str, local_files_only: bool = False, **_: Any) -> str:
        if local_files_only and (repo_id, filename) not in self.cached:
            raise FileNotFoundError(filename)
        if not local_files_only:
            self.downloads.append((repo_id, filename))
            self.cached.add((repo_id, filename))
        path = self.root / repo_id.replace("/", "__") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def try_to_load_from_cache(self, repo_id: str, filename: str) -> str | None:
        return str(self.root / filename) if (repo_id, filename) in self.cached else None

    def snapshot_download(self, repo_id: str, local_files_only: bool = False, **_: Any) -> str:
        self.snapshots.append(f"{repo_id}|{'local' if local_files_only else 'remote'}")
        return str(self.root / repo_id.replace("/", "__"))

    def install(self, monkeypatch) -> None:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", self.hf_hub_download)
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", self.try_to_load_from_cache)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", self.snapshot_download)


# ---------------------------------------------------------------- f5_tts
class FakeF5TTS:
    instances: list[FakeF5TTS] = []

    def __init__(self, model: str, ckpt_file: str, vocoder_local_path: str, device: str) -> None:
        self.model, self.ckpt_file, self.vocoder_local_path, self.device = model, ckpt_file, vocoder_local_path, device
        self.calls: list[dict[str, Any]] = []
        FakeF5TTS.instances.append(self)

    def infer(self, **kwargs: Any):
        self.calls.append(kwargs)
        batches = kwargs["progress"].tqdm(range(3))
        for _ in batches:  # consume like the real library does (cancellation is checked here)
            pass
        seconds = 1.0 / kwargs["speed"]
        return tone(seconds), SR, None


# ---------------------------------------------------------------- qwen_tts
class FakeTalker:
    """Decoder stand-in: runs its forward pre-hooks once per generated audio token, like the real talker."""

    def __init__(self) -> None:
        self.hooks: list = []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return types.SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def decode(self, tokens: int) -> None:
        for _ in range(tokens):
            for hook in list(self.hooks):
                hook(self, ())


class FakeQwenModel:
    loaded: list[dict[str, Any]] = []

    def __init__(self, model_type: str) -> None:
        self.model = types.SimpleNamespace(tts_model_type=model_type, talker=FakeTalker())
        self.prompt_calls: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.seconds = 2.0

    @classmethod
    def factory(cls, model_type: str):
        def from_pretrained(path: str, **kwargs: Any) -> FakeQwenModel:
            cls.loaded.append({"path": path, **kwargs})
            return cls(model_type)

        return from_pretrained

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.prompt_calls.append({"sr": ref_audio[1], "text": ref_text, "x_vector": x_vector_only_mode})
        return [f"prompt-{len(self.prompt_calls)}"]

    def _out(self, name: str, kwargs: dict[str, Any]):
        self.calls.append((name, kwargs))
        tokens = min(int(self.seconds * 12.5), kwargs["max_new_tokens"])  # the real model stops at max_new_tokens
        self.model.talker.decode(tokens)
        return [tone(tokens / 12.5)], SR

    def generate_voice_clone(self, **kwargs: Any):
        return self._out("clone", kwargs)

    def generate_custom_voice(self, **kwargs: Any):
        return self._out("custom_voice", kwargs)

    def generate_voice_design(self, **kwargs: Any):
        return self._out("voice_design", kwargs)

    def get_supported_speakers(self):
        return ["ryan", "Vivian"]


# ---------------------------------------------------------------- faster_whisper
@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float


class FakeWhisperModel:
    created: list[dict[str, Any]] = []
    fail_on_cuda = False

    def __init__(self, model_name: str, device: str, compute_type: str, **kwargs: Any) -> None:
        FakeWhisperModel.created.append({"model": model_name, "device": device, "compute_type": compute_type,
                                         **kwargs})
        if device == "cuda" and FakeWhisperModel.fail_on_cuda:
            raise RuntimeError("Library cublas64_12.dll is not found")
        self.kwargs: dict[str, Any] = {}

    def transcribe(self, audio, **kwargs: Any):
        self.kwargs = kwargs
        seg = types.SimpleNamespace(start=0.0, end=1.2, text=" Hola mundo.", avg_logprob=-0.2, no_speech_prob=0.01,
                                    compression_ratio=1.1,
                                    words=[FakeWord(0.0, 0.5, " Hola", 0.98765), FakeWord(0.6, 1.2, " mundo.", 0.9)])
        info = types.SimpleNamespace(language="es", language_probability=0.991234, duration=audio.size / 16_000)
        return iter([seg]), info
