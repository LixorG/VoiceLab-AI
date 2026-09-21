"""Live preview while a generation runs: audio is written in small WAV chunks the browser plays back to back.

A chunk is a preview, not the result: the final file is still assembled, post-processed and saved at the end
(crossfades and mastering are not applied to chunks). Chunks come from two places:
- engines that can stream (`supports_streaming`, e.g. Qwen3-TTS) call `on_audio` while they generate;
- for any engine, every finished segment of a segmented generation (plus its pause) is one chunk.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

AudioCallback = Callable[[np.ndarray, int], None]

CHUNK_PREFIX = "stream_"


def chunk_path(folder: Path, index: int) -> Path:
    return folder / f"{CHUNK_PREFIX}{index:04d}.wav"


def silence(ms: int, sample_rate: int) -> np.ndarray:
    return np.zeros(int(sample_rate * max(0, ms) / 1000), dtype=np.float32)


class StreamWriter:
    """Writes numbered chunks atomically and reports how many are ready (the reader never sees a partial file)."""

    def __init__(self, folder: Path, notify: Callable[[int], None]) -> None:
        self.folder = folder
        self.notify = notify
        self.count = 0
        self._lock = threading.Lock()

    def write(self, audio: np.ndarray, sample_rate: int) -> None:
        import soundfile as sf

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        with self._lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            target = chunk_path(self.folder, self.count)
            tmp = target.with_name(f"{target.stem}.tmp.wav")
            sf.write(tmp, np.clip(samples, -1.0, 1.0), sample_rate, subtype="PCM_16")
            tmp.replace(target)
            self.count += 1
            count = self.count
        self.notify(count)
