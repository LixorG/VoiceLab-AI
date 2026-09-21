"""Live audio for Qwen3-TTS: decode the codec frames as the talker produces them.

The library decodes everything at the end with `decoder.chunked_decode(chunk_size=300, left_context_size=25)` over
the reference codes followed by the generated ones, then cuts the reference part. The decoder's output depends on
its context, so live chunks reproduce the library's own windows: each chunk is decoded from the start of the
300-frame window it falls in (plus that window's 25 frames of left context), capped at 100 frames of context to
keep each decode cheap. Measured against the final file: SNR ~35 dB with the cap, ~37–42 dB with the whole window
(for scale: the library itself with 150- or 100-frame windows differs from its 300-frame output by 23–24 dB; a
fixed 25-frame context gave 17 dB). Streaming costs about RTF +0.1 (0.62 → 0.73 measured). The final file is
still the library's own decode.

Chunk schedule: 8 frames (~0.64 s), 8 more, then 16 at a time; the player starts on the second chunk so the
first seconds do not run dry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

WINDOW = 300  # frames per window in the library's chunked_decode
WINDOW_CONTEXT = 25  # left context the library adds to every window after the first
SCHEDULE = (8, 8)  # first chunks (frames), for a fast start
MAX_CONTEXT = 100  # frames of left context per decode (see the module docstring)
EVERY_FRAMES = 16  # then ~1.28 s per chunk


class CodeStreamer:
    """Accumulates codec frames and emits decoded audio in chunks that match the library's final decode."""

    def __init__(self, decode: Callable[[Any], np.ndarray], emit: Callable[[np.ndarray, int], None],
                 prefix: Any = None, samples_per_frame: int = 1920, sample_rate: int = 24_000,
                 schedule: tuple[int, ...] = SCHEDULE, every: int = EVERY_FRAMES, window: int = WINDOW,
                 window_context: int = WINDOW_CONTEXT, max_context: int | None = MAX_CONTEXT) -> None:
        import torch

        self._torch = torch
        self.decode = decode
        self.emit = emit
        self.prefix = prefix if prefix is not None and len(prefix) else None
        self.samples_per_frame = samples_per_frame
        self.sample_rate = sample_rate
        self.schedule, self.every = schedule, every
        self.window, self.window_context = window, window_context
        self.max_context = max_context  # cap on left context (None = the whole library window)
        self.frames: list[Any] = []
        self.done = 0  # generated frames already emitted
        self.emitted = 0  # chunks emitted

    def add(self, codes: Any) -> None:
        """One frame of codes, shape (num_quantizers,) or (1, num_quantizers)."""
        self.frames.append(codes.reshape(-1))
        threshold = self.schedule[self.emitted] if self.emitted < len(self.schedule) else self.every
        if len(self.frames) - self.done >= threshold:
            self._emit(len(self.frames))

    def flush(self) -> None:
        if len(self.frames) > self.done:
            self._emit(len(self.frames))

    def _emit(self, end: int) -> None:
        torch = self._torch
        generated = torch.stack(self.frames[:end])
        full = torch.cat([self.prefix.to(generated.device), generated]) if self.prefix is not None else generated
        offset = 0 if self.prefix is None else self.prefix.shape[0]
        first = offset + self.done  # absolute index of the first frame to emit
        window_start = (first // self.window) * self.window
        low = window_start - min(self.window_context, window_start)
        if self.max_context is not None:
            low = max(low, first - self.max_context)
        audio = self.decode(full[low:offset + end])
        self.emit(audio[(first - low) * self.samples_per_frame:], self.sample_rate)
        self.done = end
        self.emitted += 1


def decoder_of(model: Any) -> tuple[Callable[[Any], np.ndarray], int, int] | None:
    """(decode frames → samples, samples per frame, sample rate) from a loaded Qwen3TTSModel, or None."""
    tokenizer = getattr(getattr(model, "model", None), "speech_tokenizer", None)
    inner = getattr(tokenizer, "model", None)
    decoder = getattr(inner, "decoder", None)
    if decoder is None or not hasattr(decoder, "total_upsample"):
        return None
    import torch

    def decode(codes: Any) -> np.ndarray:
        with torch.no_grad():
            wav = decoder(torch.clamp(codes, min=0).T.unsqueeze(0))  # (1, quantizers, frames) → (1, 1, samples)
        return wav.reshape(-1).float().cpu().numpy()

    rate = int(tokenizer.get_output_sample_rate()) if hasattr(tokenizer, "get_output_sample_rate") else 24_000
    return decode, int(decoder.total_upsample), rate
