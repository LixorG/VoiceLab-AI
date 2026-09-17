"""Reference preprocessing pipeline (pure file work, no DB). Cached by sha256 + PIPELINE_VERSION."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import soundfile as sf

from app.audio.analysis import AudioAnalysis, analyze
from app.audio.ffmpeg import ProbeResult, decode_to_wav
from app.audio.processing import apply_gain, normalization_gain_db
from app.audio.storage import AudioStorage

logger = logging.getLogger("voicelab.audio.pipeline")

PIPELINE_VERSION = 1
INTERNAL_SAMPLE_RATE = 24_000  # native rate of F5-TTS, E2-TTS and Qwen3-TTS


class PipelineResult:
    def __init__(self, analysis: AudioAnalysis, probe: ProbeResult, gain_db: float, cached: bool) -> None:
        self.analysis = analysis
        self.probe = probe
        self.gain_db = gain_db
        self.cached = cached


def _load_cached(path: Path, processed: Path) -> PipelineResult | None:
    if not (path.exists() and processed.exists()):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("pipeline_version") != PIPELINE_VERSION:
            return None
        return PipelineResult(AudioAnalysis.model_validate(data["analysis"]),
                              ProbeResult.model_validate(data["source"]), data["normalization_gain_db"], True)
    except (ValueError, KeyError):
        logger.warning("analysis_cache_invalid", extra={"path": str(path)})
        return None


def run_pipeline(storage: AudioStorage, sha: str, original: Path, probe: ProbeResult, ffmpeg: str,
                 force: bool = False) -> PipelineResult:
    processed = storage.processed_path(sha)
    analysis_file = storage.analysis_path(sha)
    if not force and (cached := _load_cached(analysis_file, processed)):
        return cached

    decoded = storage.temp_path(".decoded.wav")
    try:
        decode_to_wav(original, decoded, ffmpeg, INTERNAL_SAMPLE_RATE, channels=1)
        samples, sr = sf.read(decoded, dtype="float32", always_2d=False)
    finally:
        decoded.unlink(missing_ok=True)

    # Analysis runs on the un-normalised signal so levels/clipping reflect the recording.
    result = analyze(samples, sr, source_sample_rate=probe.sample_rate)
    gain = normalization_gain_db(result.loudness_lufs, result.rms_dbfs, result.peak_dbfs)

    tmp_processed = storage.temp_path(".processed.wav")
    sf.write(tmp_processed, apply_gain(samples, gain), sr, subtype="FLOAT")
    tmp_processed.replace(processed)

    payload: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "sha256": sha,
        "source": probe.model_dump(),
        "internal_sample_rate": sr,
        "normalization_gain_db": gain,
        "analysis": result.model_dump(),
    }
    analysis_file.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return PipelineResult(result, probe, gain, cached=False)
