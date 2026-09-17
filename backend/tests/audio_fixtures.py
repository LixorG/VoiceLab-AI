"""Synthetic audio generators for tests (no real voices, deterministic)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None or shutil.which("ffprobe") is None,
                                     reason="FFmpeg/ffprobe no están en el PATH")


def speechlike(sr: int = 24_000, words: int = 12, noise_db: float = -70.0, seed: int = 0,
               gain: float = 0.3) -> np.ndarray:
    """Harmonic bursts (~voice pitch, syllable envelope) separated by pauses, plus white noise."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(0.4 * sr), dtype=np.float32)]
    for _ in range(words):
        dur = rng.uniform(0.35, 0.8)
        t = np.arange(int(dur * sr)) / sr
        f0 = rng.uniform(110, 220)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        env = np.sin(np.pi * t / dur) ** 0.5
        parts.append((gain * tone * env / 2.3).astype(np.float32))
        parts.append(np.zeros(int(rng.uniform(0.15, 0.45) * sr), dtype=np.float32))
    signal = np.concatenate(parts)
    noise = rng.normal(0, 10 ** (noise_db / 20), signal.size).astype(np.float32)
    return signal + noise


def write_wav(path: Path, samples: np.ndarray, sr: int = 24_000, channels: int = 1) -> Path:
    data = np.stack([samples] * channels, axis=1) if channels > 1 else samples
    sf.write(path, data, sr, subtype="PCM_16")
    return path


def transcode(src: Path, dst: Path, *args: str) -> Path:
    subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", str(src), *args, str(dst)],
                   check=True, timeout=60)
    return dst
