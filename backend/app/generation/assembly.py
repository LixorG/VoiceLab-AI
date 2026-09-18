"""Join segment audio: inserted silences for pauses, linear crossfade between contiguous segments.

Linear (not equal-power) because adjacent TTS segments of the same voice can be correlated: equal-power gains
would exceed the input level (up to +3 dB) and could clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CROSSFADE_MS = 40
EDGE_FADE_MS = 8


@dataclass
class SegmentAudio:
    audio: np.ndarray
    sample_rate: int
    pause_before_ms: int = 0
    pause_after_ms: int = 0


def _resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, target)
    return resample_poly(audio, target // g, sr // g).astype(np.float32)


def _fade_in(audio: np.ndarray, samples: int) -> np.ndarray:
    """Fade-in after a silence (the fade-out is applied when the following pause is inserted)."""
    n = min(samples, audio.size)
    if n <= 0:
        return audio
    out = audio.copy()
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


def timeline(parts: list[SegmentAudio], crossfade_ms: int = CROSSFADE_MS) -> list[tuple[float, float]]:
    """(start_s, end_s) of every part inside the audio `assemble()` builds from the same parts (for subtitles)."""
    if not parts:
        return []
    sr = parts[0].sample_rate
    xfade = int(sr * crossfade_ms / 1000)
    cursor = int(sr * parts[0].pause_before_ms / 1000)
    spans = []
    for i, part in enumerate(parts):
        length = int(round(np.asarray(part.audio).size * sr / part.sample_rate))
        prev_pause = parts[i - 1].pause_after_ms if i > 0 else part.pause_before_ms
        if i > 0 and prev_pause == 0 and cursor >= xfade and length >= xfade and xfade > 0:
            cursor -= xfade  # the crossfade overlaps the previous part
        spans.append((cursor / sr, (cursor + length) / sr))
        cursor += length
        if part.pause_after_ms:
            cursor += int(sr * part.pause_after_ms / 1000)
    return spans


def assemble(parts: list[SegmentAudio], crossfade_ms: int = CROSSFADE_MS) -> tuple[np.ndarray, int]:
    if not parts:
        return np.zeros(0, dtype=np.float32), 24_000
    sr = parts[0].sample_rate
    xfade = int(sr * crossfade_ms / 1000)
    edge = int(sr * EDGE_FADE_MS / 1000)
    out = np.zeros(int(sr * parts[0].pause_before_ms / 1000), dtype=np.float32)

    for i, part in enumerate(parts):
        audio = _resample(np.asarray(part.audio, dtype=np.float32).reshape(-1), part.sample_rate, sr)
        prev_pause = parts[i - 1].pause_after_ms if i > 0 else part.pause_before_ms
        contiguous = i > 0 and prev_pause == 0 and out.size >= xfade and audio.size >= xfade and xfade > 0
        if contiguous:
            ramp = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
            overlap = out[-xfade:] * (1.0 - ramp) + audio[:xfade] * ramp
            out = np.concatenate([out[:-xfade], overlap, audio[xfade:]])
        else:
            out = np.concatenate([out, _fade_in(audio, edge) if (i > 0 or part.pause_before_ms) else audio])
        if part.pause_after_ms:
            if i < len(parts) - 1:
                out[-edge:] *= np.linspace(1.0, 0.0, min(edge, out.size), dtype=np.float32)
            out = np.concatenate([out, np.zeros(int(sr * part.pause_after_ms / 1000), dtype=np.float32)])
    return out, sr
