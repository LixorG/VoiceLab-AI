"""Upload validation: extension whitelist, filename sanitising, container checks."""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePath

from app.audio.ffmpeg import ProbeResult
from app.core.errors import AppError, ErrorCode

# extension -> ffprobe format_name tokens that are acceptable for it
ALLOWED_FORMATS: dict[str, set[str]] = {
    "wav": {"wav"},
    "mp3": {"mp3"},
    "flac": {"flac"},
    "m4a": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    "ogg": {"ogg"},
    "opus": {"ogg"},
    "aiff": {"aiff"},
    "aif": {"aiff"},
}

_UNSAFE = re.compile(r"[^\w.\- ]+", re.UNICODE)


def sanitize_filename(name: str, max_length: int = 120) -> str:
    """Keep only a display-safe basename. Storage never uses this value as a path."""
    base = PurePath(name.replace("\\", "/")).name
    base = unicodedata.normalize("NFC", base).strip().strip(".")
    base = _UNSAFE.sub("_", base)
    base = re.sub(r"_+", "_", base)
    if not base:
        base = "audio"
    if len(base) > max_length:
        stem, dot, ext = base.rpartition(".")
        base = (stem[: max_length - len(ext) - 1] + dot + ext) if dot else base[:max_length]
    return base


def extension_of(name: str) -> str:
    ext = PurePath(name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_FORMATS:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, status_code=415, details={
            "extension": ext or None, "permitidas": sorted(ALLOWED_FORMATS)})
    return ext


def validate_probe(ext: str, probe: ProbeResult, max_duration_s: float) -> None:
    tokens = set(probe.format_name.split(","))
    if not tokens & ALLOWED_FORMATS[ext]:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, status_code=415,
                       message="El contenido del archivo no coincide con su extensión.",
                       details={"extension": ext, "detectado": probe.format_name})
    if probe.duration_s <= 0.1:
        raise AppError(ErrorCode.AUDIO_DECODE_ERROR, status_code=422,
                       message="El audio está vacío o es demasiado corto.")
    if probe.duration_s > max_duration_s:
        raise AppError(ErrorCode.AUDIO_TOO_LONG, status_code=422, details={
            "duracion_s": round(probe.duration_s, 1), "maximo_s": max_duration_s})
