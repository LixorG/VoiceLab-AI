"""Model output files and optional mastering for a generation.

Layout under ``data/generated/<generation_id>/``:
- ``raw.wav``       model output (float32, never modified; segmented outputs joined with the default crossfade)
- ``seg_NNN.wav``   each segment as generated (float32), used to re-assemble with another crossfade
- ``final_*.wav``   what the user plays/downloads (PCM 24 bits): raw, or raw + post-processing

Older generations (before phase 10) only have ``generated/<id>.wav``; it is treated as the raw output.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.audio.ffmpeg import resolve_binary
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.generation.assembly import SegmentAudio, assemble
from app.models.entities import Generation, GenerationSegment
from app.postprocess.config import DEFAULT_CROSSFADE_MS, PostProcessConfig, ProcessingStep
from app.postprocess.processor import process

logger = logging.getLogger("voicelab.mastering")

FINAL_SUBTYPE = "PCM_24"
RAW_SUBTYPE = "FLOAT"
CLIP_WARNING = ("El audio generado superaba 0 dBFS y se recortó al guardarlo. "
                "Puedes corregirlo con la normalización de pico.")


def generation_dir(settings: Settings, generation_id: str) -> Path:
    path = settings.data_dir / "generated" / generation_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relative(settings: Settings, path: Path) -> str:
    return path.relative_to(settings.data_dir).as_posix()


def _safe_path(settings: Settings, relative: str | None) -> Path | None:
    if not relative:
        return None
    path = (settings.data_dir / relative).resolve()
    return path if settings.data_dir.resolve() in path.parents else None


def ffmpeg_or_none(settings: Settings) -> str | None:
    try:
        return resolve_binary("ffmpeg", settings.ffmpeg_path)
    except AppError:
        return None


def save_raw(settings: Settings, gen: Generation, audio: np.ndarray, sample_rate: int) -> None:
    path = generation_dir(settings, gen.id) / "raw.wav"
    sf.write(path, audio, sample_rate, subtype=RAW_SUBTYPE)
    gen.raw_output_path = _relative(settings, path)


def save_segment(settings: Settings, seg: GenerationSegment, audio: np.ndarray, sample_rate: int) -> None:
    path = generation_dir(settings, seg.generation_id) / f"seg_{seg.index:03d}.wav"
    sf.write(path, audio, sample_rate, subtype=RAW_SUBTYPE)
    seg.audio_path = _relative(settings, path)


def raw_path(settings: Settings, gen: Generation) -> Path:
    path = _safe_path(settings, gen.raw_output_path or gen.output_path)
    if path is None or not path.exists():
        raise AppError(ErrorCode.GENERATION_NOT_FOUND, status_code=404,
                       message="El audio original de esta generación no está disponible.")
    return path


def write_final(settings: Settings, gen: Generation, audio: np.ndarray, sample_rate: int) -> list[str]:
    """Write the playable file (new name each time so open players/caches never see a half-written file)."""
    warnings: list[str] = []
    if audio.size and float(np.max(np.abs(audio))) > 1.0:
        audio = np.clip(audio, -1.0, 1.0)
        warnings.append(CLIP_WARNING)
    if gen.raw_output_path is None and gen.output_path:
        gen.raw_output_path = gen.output_path  # legacy generation: its only file becomes the raw output
    previous = _safe_path(settings, gen.output_path)
    path = generation_dir(settings, gen.id) / f"final_{uuid.uuid4().hex[:8]}.wav"
    sf.write(path, audio, sample_rate, subtype=FINAL_SUBTYPE)
    gen.output_path = _relative(settings, path)
    gen.duration_s = round(audio.size / sample_rate, 3) if sample_rate else None
    if previous is not None and previous.name.startswith("final_"):
        try:
            previous.unlink(missing_ok=True)
        except OSError:  # still open by a player on Windows: harmless leftover, removed with the generation
            logger.info("final_file_in_use", extra={"path": str(previous)})
    return warnings


def delete_files(settings: Settings, gen: Generation) -> None:
    for relative in (gen.output_path, gen.raw_output_path):
        path = _safe_path(settings, relative)
        if path is not None:
            path.unlink(missing_ok=True)
    folder = settings.data_dir / "generated" / gen.id
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def _source(settings: Settings, gen: Generation, segments: list[GenerationSegment],
            crossfade_ms: int | None) -> tuple[np.ndarray, int, list[ProcessingStep], list[str]]:
    steps: list[ProcessingStep] = []
    warnings: list[str] = []
    if crossfade_ms is not None:
        seg_paths = [_safe_path(settings, s.audio_path) for s in segments]
        if len(segments) > 1 and all(p is not None and p.exists() for p in seg_paths):
            parts = []
            for seg, path in zip(segments, seg_paths, strict=True):
                data, sr = sf.read(path, dtype="float32", always_2d=False)  # type: ignore[arg-type]
                parts.append(SegmentAudio(np.asarray(data).reshape(-1), sr, seg.pause_before_ms, seg.pause_after_ms))
            audio, sr = assemble(parts, crossfade_ms=crossfade_ms)
            steps.append(ProcessingStep(id="crossfade", label="Crossfade entre segmentos",
                                        detail=f"{crossfade_ms} ms entre {len(segments)} segmentos contiguos "
                                               f"(por defecto {DEFAULT_CROSSFADE_MS} ms)."))
            return audio, sr, steps, warnings
        if len(segments) > 1:
            warnings.append("Esta generación no conserva sus segmentos por separado: el crossfade no se puede "
                            "cambiar sin volver a generarla.")
        else:
            warnings.append("El crossfade solo se aplica a generaciones con varios segmentos.")
    data, sr = sf.read(raw_path(settings, gen), dtype="float32", always_2d=False)
    return np.asarray(data).reshape(-1), sr, steps, warnings


def master(settings: Settings, gen: Generation, segments: list[GenerationSegment], config: PostProcessConfig,
           native_speed_parameter: str | None) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Raw output (or re-assembled segments) → post-processing chain. Returns audio and the stored record."""
    audio, sr, steps, warnings = _source(settings, gen, segments, config.crossfade_ms)
    audio, report = process(audio, sr, config, ffmpeg_or_none(settings), native_speed_parameter)
    report.steps[:0] = steps
    report.warnings[:0] = warnings
    return audio, sr, {"config": config.model_dump(), "report": report.model_dump()}
