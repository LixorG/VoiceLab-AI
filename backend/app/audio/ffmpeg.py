"""Thin, safe wrappers around ffprobe/ffmpeg (argument lists only, never a shell)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from app.core.errors import AppError, ErrorCode

logger = logging.getLogger("voicelab.audio.ffmpeg")

PROBE_TIMEOUT_S = 30
DECODE_TIMEOUT_S = 600


class ProbeResult(BaseModel):
    format_name: str
    codec: str
    sample_rate: int
    channels: int
    duration_s: float
    bit_rate: int | None = None


def resolve_binary(name: str, configured: str | None = None) -> str:
    exe = shutil.which(configured or name)
    if not exe:
        raise AppError(ErrorCode.FFMPEG_NOT_FOUND, status_code=500)
    return exe


def probe(path: Path, ffprobe: str) -> ProbeResult:
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise AppError(ErrorCode.AUDIO_DECODE_ERROR, status_code=422) from exc
    if proc.returncode != 0:
        logger.info("ffprobe_rejected", extra={"stderr": proc.stderr[-500:]})
        raise AppError(ErrorCode.AUDIO_DECODE_ERROR, status_code=422)
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AppError(ErrorCode.AUDIO_DECODE_ERROR, status_code=422) from exc

    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    if audio is None:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, status_code=422,
                       message="El archivo no contiene una pista de audio.")
    duration = audio.get("duration") or fmt.get("duration")
    try:
        return ProbeResult(
            format_name=fmt.get("format_name", ""),
            codec=audio.get("codec_name", ""),
            sample_rate=int(audio["sample_rate"]),
            channels=int(audio["channels"]),
            duration_s=float(duration),
            bit_rate=int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AppError(ErrorCode.AUDIO_DECODE_ERROR, status_code=422) from exc


def decode_to_wav(src: Path, dst: Path, ffmpeg: str, sample_rate: int, channels: int = 1) -> None:
    """Decode any supported input to float32 WAV. Writes atomically via a temp file."""
    tmp = dst.with_suffix(".tmp.wav")
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", str(channels), "-ar", str(sample_rate),
        "-af", "aresample=resampler=soxr",
        "-c:a", "pcm_f32le", str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DECODE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise AppError(ErrorCode.AUDIO_DECODE_ERROR, status_code=422) from exc
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        # soxr may be missing from some builds; retry with the default resampler.
        if "soxr" in proc.stderr.lower():
            cmd.remove("-af")
            cmd.remove("aresample=resampler=soxr")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DECODE_TIMEOUT_S)
        if proc.returncode != 0:
            logger.warning("ffmpeg_decode_failed", extra={"stderr": proc.stderr[-800:]})
            tmp.unlink(missing_ok=True)
            raise AppError(ErrorCode.AUDIO_DECODE_ERROR, status_code=422)
    tmp.replace(dst)
