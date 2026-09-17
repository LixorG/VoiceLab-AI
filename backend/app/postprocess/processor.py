"""PostProcessor: MODEL OUTPUT → optional DSP chain → MASTERED AUDIO.

Pure numpy for levels, silence and fades; FFmpeg filters for denoise (afftdn) and time stretch / pitch shift
(rubberband, formant preservation optional). Never applied unless enabled; every step is reported.
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.errors import AppError, ErrorCode
from app.postprocess.config import (
    LevelStats,
    PostProcessCapabilities,
    PostProcessConfig,
    PostProcessReport,
    ProcessingStep,
)

logger = logging.getLogger("voicelab.postprocess")

FFMPEG_TIMEOUT_S = 180
DENOISE_REDUCTION_DB = {"light": 6, "medium": 12, "strong": 20}
DENOISE_LABEL = {"light": "suave", "medium": "media", "strong": "fuerte"}
LOUDNESS_MIN_S = 0.5
SAFETY_CEILING_DBFS = -1.0
LARGE_PITCH_SEMITONES = 4.0
LARGE_STRETCH = 0.25


# ---------------------------------------------------------------------------- capabilities
@lru_cache(maxsize=4)
def ffmpeg_filters(ffmpeg: str) -> frozenset[str]:
    try:
        proc = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    names = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return frozenset(names)


def capabilities(ffmpeg: str | None) -> PostProcessCapabilities:
    filters = ffmpeg_filters(ffmpeg) if ffmpeg else frozenset()
    has_rubberband, has_afftdn = "rubberband" in filters, "afftdn" in filters
    processors = {"trim_silence": True, "loudness": True, "peak": True, "fades": True, "crossfade": True,
                  "denoise": has_afftdn, "time_stretch": has_rubberband, "pitch_shift": has_rubberband}
    reasons: dict[str, str] = {}
    if not ffmpeg:
        missing = "FFmpeg no está disponible."
    else:
        missing = "La versión de FFmpeg instalada no incluye el filtro «{}»."
    if not has_afftdn:
        reasons["denoise"] = missing.format("afftdn")
    if not has_rubberband:
        reasons["time_stretch"] = reasons["pitch_shift"] = missing.format("rubberband")
    return PostProcessCapabilities(processors=processors, reasons=reasons)


# ---------------------------------------------------------------------------- measurements
def _db(value: float) -> float | None:
    return round(20 * math.log10(value), 2) if value > 0 else None


def loudness_lufs(audio: np.ndarray, sr: int) -> float | None:
    if audio.size < sr * LOUDNESS_MIN_S:
        return None
    import pyloudnorm

    value = pyloudnorm.Meter(sr).integrated_loudness(audio.astype(np.float64))
    return round(float(value), 2) if math.isfinite(value) else None


def level_stats(audio: np.ndarray, sr: int) -> LevelStats:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return LevelStats(duration_s=round(audio.size / sr, 3) if sr else 0.0, peak_dbfs=_db(peak),
                      loudness_lufs=loudness_lufs(audio, sr))


# ---------------------------------------------------------------------------- processors
def _run_ffmpeg(audio: np.ndarray, sr: int, filters: str, ffmpeg: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="voicelab_pp_") as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        sf.write(src, audio, sr, subtype="FLOAT")
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
               "-af", filters, "-ar", str(sr), "-ac", "1", "-c:a", "pcm_f32le", str(dst)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AppError(ErrorCode.POSTPROCESS_ERROR, status_code=500) from exc
        if proc.returncode != 0 or not dst.exists():
            logger.warning("postprocess_ffmpeg_failed", extra={"filters": filters, "stderr": proc.stderr[-800:]})
            raise AppError(ErrorCode.POSTPROCESS_ERROR, status_code=500)
        out, _ = sf.read(dst, dtype="float32", always_2d=False)
    return np.asarray(out, dtype=np.float32).reshape(-1)


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float, padding_ms: int) -> tuple[np.ndarray, float, float]:
    """Remove leading/trailing audio below the threshold (10 ms RMS frames). Returns (audio, cut_start, cut_end)."""
    frame = max(1, sr // 100)
    n = audio.size // frame
    if n == 0:
        return audio, 0.0, 0.0
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.flatnonzero(20 * np.log10(rms) > threshold_db)
    if loud.size == 0:
        return audio, 0.0, 0.0
    pad = int(sr * padding_ms / 1000)
    start = max(0, loud[0] * frame - pad)
    end = min(audio.size, (loud[-1] + 1) * frame + pad)
    return audio[start:end], start / sr, (audio.size - end) / sr


def apply_fades(audio: np.ndarray, sr: int, fade_in_ms: int, fade_out_ms: int) -> np.ndarray:
    out = audio.copy()
    half = out.size // 2
    for ms, is_in in ((fade_in_ms, True), (fade_out_ms, False)):
        n = min(half, int(sr * ms / 1000))
        if n <= 0:
            continue
        curve = np.sin(np.linspace(0.0, np.pi / 2, n, dtype=np.float32)) ** 2  # smooth, no click
        if is_in:
            out[:n] *= curve
        else:
            out[-n:] *= curve[::-1]
    return out


# ---------------------------------------------------------------------------- chain
def process(audio: np.ndarray, sr: int, config: PostProcessConfig, ffmpeg: str | None,
            native_speed_parameter: str | None = None) -> tuple[np.ndarray, PostProcessReport]:
    """Apply the enabled processors in a fixed order and describe exactly what was done."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    before = level_stats(audio, sr)
    steps: list[ProcessingStep] = []
    warnings: list[str] = []
    caps = capabilities(ffmpeg)

    def unavailable(processor: str) -> bool:
        if caps.processors.get(processor, False):
            return False
        warnings.append(f"No se aplicó «{processor}»: {caps.reasons.get(processor, 'no disponible.')}")
        return True

    if config.denoise.enabled and not unavailable("denoise"):
        reduction = DENOISE_REDUCTION_DB[config.denoise.strength]
        audio = _run_ffmpeg(audio, sr, f"afftdn=nr={reduction}:nf=-50:tn=1", ffmpeg or "")
        steps.append(ProcessingStep(id="denoise", label="Limpieza de ruido",
                                    detail=f"Intensidad {DENOISE_LABEL[config.denoise.strength]} "
                                           f"(reducción {reduction} dB, FFmpeg afftdn)."))
        if config.denoise.strength == "strong":
            warnings.append("La limpieza de ruido fuerte puede producir artefactos metálicos en la voz.")

    if config.trim_silence.enabled:
        audio, cut_start, cut_end = trim_silence(audio, sr, config.trim_silence.threshold_db,
                                                 config.trim_silence.padding_ms)
        steps.append(ProcessingStep(id="trim_silence", label="Recorte de silencios",
                                    detail=f"Inicio −{cut_start:.2f} s, final −{cut_end:.2f} s "
                                           f"(umbral {config.trim_silence.threshold_db:.0f} dBFS)."))

    rubberband: list[str] = []
    ts, ps = config.time_stretch, config.pitch_shift
    if ts.enabled and ts.rate != 1.0 and not unavailable("time_stretch"):
        rubberband.append(f"tempo={ts.rate:.4f}")
        steps.append(ProcessingStep(id="time_stretch", label="Cambio de velocidad (DSP)",
                                    detail=f"×{ts.rate:.2f} sin cambiar el tono (rubberband)."))
        if native_speed_parameter:
            warnings.append(f"Este motor controla la velocidad de forma nativa (parámetro «{native_speed_parameter}»), "
                            "que suele sonar más natural que el cambio de velocidad posterior.")
        if abs(ts.rate - 1.0) > LARGE_STRETCH:
            warnings.append("Cambios de velocidad grandes pueden sonar artificiales.")
    if ps.enabled and ps.semitones != 0 and not unavailable("pitch_shift"):
        rubberband.append(f"pitch={2 ** (ps.semitones / 12):.6f}")
        if ps.preserve_formants:
            rubberband.append("formant=preserved")
        steps.append(ProcessingStep(id="pitch_shift", label="Cambio de tono (DSP)",
                                    detail=f"{ps.semitones:+.1f} semitonos, formantes "
                                           f"{'conservados' if ps.preserve_formants else 'desplazados'} (rubberband)."))
        if abs(ps.semitones) > LARGE_PITCH_SEMITONES:
            warnings.append("Desplazamientos de tono grandes alteran notablemente el timbre de la voz.")
    if rubberband:
        audio = _run_ffmpeg(audio, sr, "rubberband=" + ":".join([*rubberband, "pitchq=quality"]), ffmpeg or "")

    if config.loudness.enabled:
        measured = loudness_lufs(audio, sr)
        if measured is None:
            warnings.append("No se normalizó la sonoridad: el audio es demasiado corto o silencioso para medirla.")
        else:
            gain_db = config.loudness.target_lufs - measured
            audio = audio * np.float32(10 ** (gain_db / 20))
            detail = f"{measured:.1f} → {config.loudness.target_lufs:.1f} LUFS ({gain_db:+.1f} dB)."
            ceiling = config.peak.target_dbfs if config.peak.enabled else SAFETY_CEILING_DBFS
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 10 ** (ceiling / 20):
                reduce_db = 20 * math.log10(peak) - ceiling
                audio = audio * np.float32(10 ** (-reduce_db / 20))
                warnings.append(f"Para no saturar, la sonoridad quedó {reduce_db:.1f} dB por debajo del objetivo "
                                f"(pico limitado a {ceiling:.1f} dBFS).")
                detail += f" Reducido {reduce_db:.1f} dB por el techo de pico."
            steps.append(ProcessingStep(id="loudness", label="Normalización de sonoridad", detail=detail))

    if config.peak.enabled:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        target = 10 ** (config.peak.target_dbfs / 20)
        if peak <= 0:
            warnings.append("No se normalizó el pico: el audio está en silencio.")
        elif config.loudness.enabled:
            steps.append(ProcessingStep(id="peak", label="Techo de pico",
                                        detail=f"Pico máximo {config.peak.target_dbfs:.1f} dBFS "
                                               "(con normalización de sonoridad actúa solo como límite)."))
        else:
            gain_db = 20 * math.log10(target / peak)
            audio = audio * np.float32(target / peak)
            steps.append(ProcessingStep(id="peak", label="Normalización de pico",
                                        detail=f"Pico a {config.peak.target_dbfs:.1f} dBFS ({gain_db:+.1f} dB)."))

    if config.fades.enabled and (config.fades.fade_in_ms or config.fades.fade_out_ms):
        audio = apply_fades(audio, sr, config.fades.fade_in_ms, config.fades.fade_out_ms)
        steps.append(ProcessingStep(id="fades", label="Fundidos",
                                    detail=f"Entrada {config.fades.fade_in_ms} ms, "
                                           f"salida {config.fades.fade_out_ms} ms."))

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = np.clip(audio, -1.0, 1.0)
        warnings.append("El audio procesado superaba 0 dBFS y se recortó. Activa la normalización de pico.")
    after = level_stats(audio, sr)
    return audio.astype(np.float32), PostProcessReport(steps=steps, warnings=warnings, before=before, after=after)

