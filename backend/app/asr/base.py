"""ASR backend contract. Implementations must keep heavy imports inside methods."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel

ASR_SAMPLE_RATE = 16_000


class ASRWord(BaseModel):
    start: float
    end: float
    word: str
    probability: float


class ASRSegment(BaseModel):
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


class TranscriptionResult(BaseModel):
    text: str
    language: str
    language_probability: float
    duration_s: float
    segments: list[ASRSegment]
    words: list[ASRWord]
    model: str
    device: str
    compute_type: str
    elapsed_s: float


class ASRBackend(Protocol):
    model_name: str

    def package_available(self) -> bool: ...

    def model_installed(self) -> bool: ...

    def download_model(self) -> None: ...

    @property
    def loaded(self) -> bool: ...

    @property
    def device(self) -> str | None: ...

    @property
    def compute_type(self) -> str | None: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def transcribe(self, audio_16k: np.ndarray, language: str | None,
                   initial_prompt: str | None = None) -> TranscriptionResult: ...
