"""Non-destructive processing for the internal working copy (gain only, no compression/EQ)."""

from __future__ import annotations

import numpy as np

TARGET_LUFS = -20.0
PEAK_CEILING_DBFS = -1.0
MAX_GAIN_DB = 20.0


def normalization_gain_db(loudness_lufs: float | None, rms_dbfs: float, peak_dbfs: float) -> float:
    """Gain toward TARGET_LUFS, capped so the peak stays below the ceiling and noise isn't boosted wildly."""
    current = loudness_lufs if loudness_lufs is not None else rms_dbfs
    gain = TARGET_LUFS - current
    gain = min(gain, PEAK_CEILING_DBFS - peak_dbfs, MAX_GAIN_DB)
    return float(round(gain, 2))


def apply_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    return (samples * np.float32(10 ** (gain_db / 20.0))).astype(np.float32, copy=False)


def waveform_peaks(samples: np.ndarray, buckets: int) -> list[float]:
    """Max absolute amplitude per bucket (WaveSurfer-compatible peaks, 0..1)."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return []
    buckets = max(1, min(buckets, samples.size))
    usable = samples.size - samples.size % buckets
    chunks = np.abs(samples[:usable]).reshape(buckets, -1)
    peaks = chunks.max(axis=1)
    return np.round(peaks, 4).tolist()
