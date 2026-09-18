"""Delivery formats for finished audio: WAV (as stored) or a transcoded copy (MP3, OGG/Opus, FLAC) via FFmpeg.

Stored audio is never replaced: every other format is a cached copy under `data/cache/exports`, keyed by the source
file's content signature, so asking twice costs one encode. Lossy formats are for sharing/editing in a video; the
WAV stays the reference quality.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.errors import AppError, ErrorCode

ExportFormat = Literal["wav", "mp3", "ogg", "flac"]


@dataclass(frozen=True)
class FormatSpec:
    extension: str
    media_type: str
    args: tuple[str, ...]
    label: str


FORMATS: dict[str, FormatSpec] = {
    "wav": FormatSpec("wav", "audio/wav", (), "WAV (sin pérdida)"),
    # VBR quality 2: the best LAME preset; for 24 kHz mono it lands around 60–70 kbps. Every video editor reads it.
    "mp3": FormatSpec("mp3", "audio/mpeg", ("-c:a", "libmp3lame", "-q:a", "2"), "MP3 (para vídeo y web)"),
    # Opus at 64 kbps is near-transparent for a mono voice; size is similar to the MP3 (measured: 8 s → ~65 KB).
    "ogg": FormatSpec("ogg", "audio/ogg", ("-c:a", "libopus", "-b:a", "64k", "-application", "audio"),
                      "OGG Opus (web y apps)"),
    "flac": FormatSpec("flac", "audio/flac", ("-c:a", "flac", "-compression_level", "8"),
                       "FLAC (sin pérdida, comprimido)"),
}


def signature(path: Path, spec: FormatSpec | None = None) -> str:
    """Source content signature plus the encoder settings, so changing a codec setting never serves an old copy."""
    stat = path.stat()
    settings = " ".join(spec.args) if spec else ""
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{settings}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def export_audio(source: Path, fmt: str, cache_dir: Path, ffmpeg: str | None) -> Path:
    """Returns `source` for WAV, otherwise a cached transcoded copy (created on first request)."""
    spec = FORMATS.get(fmt)
    if spec is None:
        raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422, message="Formato de exportación no válido.",
                       details={"formatos": sorted(FORMATS)})
    if fmt == "wav":
        return source
    if ffmpeg is None:
        raise AppError(ErrorCode.FFMPEG_NOT_FOUND, status_code=503,
                       message=f"Para exportar en {spec.label.split(' ')[0]} hace falta FFmpeg.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{signature(source, spec)}.{spec.extension}"
    if target.exists():
        return target
    tmp = target.with_name(f"{target.stem}.tmp.{spec.extension}")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1", *spec.args, str(tmp)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)  # noqa: S603
    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise AppError(ErrorCode.EXPORT_ERROR, status_code=500,
                       message=f"No se pudo convertir el audio a {spec.label.split(' ')[0]}.")
    tmp.replace(target)
    return target
