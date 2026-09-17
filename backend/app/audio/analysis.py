"""Heuristic reference-audio analysis.

All values are technical indicators. The quality score is NOT a measurement of cloning quality.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

FRAME_S = 0.02
HOP_S = 0.01
EPS = 1e-10
CLIP_THRESHOLD = 0.999
MIN_SPEECH_S = 0.1
MAX_GAP_S = 0.3
SEGMENT_MIN_S = 3.0
SEGMENT_MAX_S = 12.0
SEGMENT_IDEAL_S = 8.0


class Region(BaseModel):
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class SuggestedSegment(Region):
    score: float  # 0..1, higher = better candidate (heuristic)


class QualityComponent(BaseModel):
    id: str
    label: str
    score: float  # 0..1
    weight: float


class AudioAnalysis(BaseModel):
    duration_s: float
    sample_rate: int
    peak_dbfs: float
    rms_dbfs: float
    loudness_lufs: float | None
    clipping_ratio: float
    clipped_samples: int
    noise_floor_dbfs: float
    speech_level_dbfs: float
    snr_db: float | None
    speech_ratio: float
    silence_ratio: float
    effective_speech_s: float
    speech_detected: bool
    speech_regions: list[Region]
    suggested_segments: list[SuggestedSegment]
    quality_score: float  # 0..100 heuristic
    quality_label: str
    quality_components: list[QualityComponent]
    warnings: list[str]


def _db(power: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(power, EPS))


def frame_power(samples: np.ndarray, sr: int) -> np.ndarray:
    frame, hop = max(1, int(FRAME_S * sr)), max(1, int(HOP_S * sr))
    if samples.size < frame:
        return np.array([float(np.mean(samples**2))]) if samples.size else np.array([0.0])
    n = 1 + (samples.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.mean(samples[idx] ** 2, axis=1)


def detect_speech(frame_db: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Energy VAD with adaptive threshold. Returns (mask, noise_floor_db, speech_level_db)."""
    noise_floor = float(np.percentile(frame_db, 10))
    speech_level = float(np.percentile(frame_db, 95))
    dynamic = speech_level - noise_floor
    threshold = max(noise_floor + max(6.0, 0.35 * dynamic), -60.0)
    mask = frame_db > threshold
    if dynamic < 6.0:  # flat signal: either all silence or constant noise/tone
        mask[:] = speech_level > -45.0

    # close short gaps, then drop very short bursts
    mask = _fill_runs(mask, value=False, max_len=int(MAX_GAP_S / HOP_S))
    mask = _fill_runs(mask, value=True, max_len=int(MIN_SPEECH_S / HOP_S), replacement=False)
    return mask, noise_floor, speech_level


def _fill_runs(mask: np.ndarray, value: bool, max_len: int, replacement: bool | None = None) -> np.ndarray:
    out = mask.copy()
    repl = (not value) if replacement is None else replacement
    n, i = out.size, 0
    while i < n:
        if out[i] == value:
            j = i
            while j < n and out[j] == value:
                j += 1
            interior = i > 0 and j < n
            if (j - i) <= max_len and (interior or value):
                out[i:j] = repl
            i = j
        else:
            i += 1
    return out


def mask_to_regions(mask: np.ndarray, duration_s: float) -> list[Region]:
    regions: list[Region] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append(Region(start_s=start * HOP_S, end_s=min(duration_s, i * HOP_S + FRAME_S)))
            start = None
    if start is not None:
        regions.append(Region(start_s=start * HOP_S, end_s=duration_s))
    return [Region(start_s=round(r.start_s, 3), end_s=round(r.end_s, 3)) for r in regions]


def suggest_segments(regions: list[Region], frame_db: np.ndarray, limit: int = 5) -> list[SuggestedSegment]:
    """Group consecutive speech regions into 3–12 s windows that start/end at pauses."""
    candidates: list[SuggestedSegment] = []
    for i, first in enumerate(regions):
        end_idx = i
        while end_idx + 1 < len(regions) and regions[end_idx + 1].end_s - first.start_s <= SEGMENT_MAX_S:
            end_idx += 1
        span_start, span_end = first.start_s, regions[end_idx].end_s
        span = span_end - span_start
        if span < SEGMENT_MIN_S:
            continue
        span_end = min(span_end, span_start + SEGMENT_MAX_S)
        speech = sum(min(r.end_s, span_end) - r.start_s for r in regions[i : end_idx + 1])
        density = speech / (span_end - span_start)
        a, b = int(span_start / HOP_S), max(int(span_start / HOP_S) + 1, int(span_end / HOP_S))
        level = float(np.mean(frame_db[a:b])) if b <= frame_db.size else float(np.mean(frame_db[a:]))
        length_fit = 1.0 - min(1.0, abs((span_end - span_start) - SEGMENT_IDEAL_S) / SEGMENT_IDEAL_S)
        level_fit = float(np.clip((level + 50.0) / 30.0, 0.0, 1.0))
        score = 0.45 * density + 0.35 * length_fit + 0.20 * level_fit
        candidates.append(SuggestedSegment(start_s=round(span_start, 3), end_s=round(span_end, 3),
                                           score=round(score, 3)))

    chosen: list[SuggestedSegment] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        if all(cand.end_s <= c.start_s or cand.start_s >= c.end_s for c in chosen):
            chosen.append(cand)
        if len(chosen) == limit:
            break
    return sorted(chosen, key=lambda c: c.start_s)


def _loudness(samples: np.ndarray, sr: int) -> float | None:
    if samples.size < int(0.4 * sr):
        return None
    import pyloudnorm

    value = float(pyloudnorm.Meter(sr).integrated_loudness(samples.astype(np.float64)))
    return None if math.isinf(value) or math.isnan(value) else round(value, 2)


def _quality(a: dict, source_sample_rate: int) -> tuple[float, str, list[QualityComponent], list[str]]:
    warnings: list[str] = []
    comps: list[QualityComponent] = []

    def add(cid: str, label: str, score: float, weight: float) -> None:
        comps.append(QualityComponent(id=cid, label=label, score=round(float(np.clip(score, 0, 1)), 3),
                                      weight=weight))

    snr = a["snr_db"]
    if snr is None:
        add("snr", "Relación señal/ruido", 0.7, 0.30)
        warnings.append("No hay pausas suficientes para estimar el ruido de fondo.")
    else:
        add("snr", "Relación señal/ruido", (snr - 5.0) / 25.0, 0.30)
        if snr < 15:
            warnings.append("Ruido de fondo apreciable: puede transferirse a la voz generada.")

    eff = a["effective_speech_s"]
    add("duration", "Duración de habla", eff / 6.0, 0.20)
    if eff < 3.0:
        warnings.append("Menos de 3 s de habla efectiva: la referencia puede ser insuficiente.")

    clip = a["clipping_ratio"]
    add("clipping", "Saturación", 1.0 - min(1.0, clip / 0.001) * 0.8, 0.20)
    if a["clipped_samples"] > 0 and clip >= 0.0001:
        warnings.append("Se detectó saturación (clipping) en la grabación.")

    level = a["speech_level_dbfs"]
    add("level", "Nivel de grabación", (level + 45.0) / 20.0, 0.10)
    if level < -35.0:
        warnings.append("Nivel de grabación muy bajo.")

    add("speech_ratio", "Proporción de habla", a["speech_ratio"] / 0.5, 0.10)
    if not a["speech_detected"]:
        warnings.append("No se detectó voz en el audio.")
    elif a["speech_ratio"] < 0.3:
        warnings.append("El audio contiene mucho silencio o partes sin voz.")

    sr_score = 1.0 if source_sample_rate >= 22050 else 0.85 if source_sample_rate >= 16000 else 0.5
    add("sample_rate", "Frecuencia de muestreo original", sr_score, 0.10)
    if source_sample_rate < 16000:
        warnings.append("Frecuencia de muestreo original baja (calidad telefónica).")

    total = sum(c.score * c.weight for c in comps) / sum(c.weight for c in comps)
    if not a["speech_detected"]:
        total = min(total, 0.2)
    score = round(total * 100, 1)
    label = "Excelente" if score >= 85 else "Buena" if score >= 70 else "Aceptable" if score >= 50 else "Baja"
    return score, label, comps, warnings


def analyze(samples: np.ndarray, sr: int, source_sample_rate: int | None = None) -> AudioAnalysis:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    duration = samples.size / sr
    abs_s = np.abs(samples)
    peak = float(abs_s.max()) if samples.size else 0.0
    clipped = int(np.count_nonzero(abs_s >= CLIP_THRESHOLD))

    power = frame_power(samples, sr)
    frame_db = np.asarray(_db(power))
    mask, noise_floor, speech_level = detect_speech(frame_db)
    regions = mask_to_regions(mask, duration)
    speech_frames = int(mask.sum())
    effective = sum(r.end_s - r.start_s for r in regions)

    snr: float | None = None
    non_speech = ~mask
    if speech_frames and non_speech.sum() * HOP_S >= 0.3:
        snr = round(float(_db(power[mask].mean()) - _db(power[non_speech].mean())), 2)

    base = {
        "snr_db": snr,
        "effective_speech_s": round(effective, 3),
        "clipping_ratio": clipped / max(1, samples.size),
        "clipped_samples": clipped,
        "speech_level_dbfs": round(speech_level, 2),
        "speech_ratio": round(speech_frames / max(1, mask.size), 4),
        "speech_detected": effective >= 0.3,
    }
    score, label, comps, warnings = _quality(base, source_sample_rate or sr)

    return AudioAnalysis(
        duration_s=round(duration, 3),
        sample_rate=sr,
        peak_dbfs=round(float(_db(peak**2)), 2),
        rms_dbfs=round(float(_db(float(np.mean(samples**2)) if samples.size else 0.0)), 2),
        loudness_lufs=_loudness(samples, sr),
        noise_floor_dbfs=round(noise_floor, 2),
        silence_ratio=round(1.0 - base["speech_ratio"], 4),
        speech_regions=regions,
        suggested_segments=suggest_segments(regions, frame_db),
        quality_score=score,
        quality_label=label,
        quality_components=comps,
        warnings=warnings,
        **base,
    )
