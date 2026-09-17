"""EnvironmentChecker: reports runtime readiness without importing heavy packages."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from typing import Literal

from pydantic import BaseModel

from app.core.gpu import GPUInfo, GPUManager

Status = Literal["ok", "warning", "error", "missing"]


class Check(BaseModel):
    id: str
    label: str
    status: Status
    detail: str | None = None


class EnvironmentReport(BaseModel):
    platform: str
    python_version: str
    gpu: GPUInfo
    checks: list[Check]


# (check id, distribution name, label)
OPTIONAL_PACKAGES = (
    ("engine_f5tts", "f5-tts", "F5-TTS / E2-TTS"),
    ("engine_qwen3tts", "qwen-tts", "Qwen3-TTS"),
    ("asr_faster_whisper", "faster-whisper", "faster-whisper (transcripción)"),
    ("flash_attn", "flash-attn", "FlashAttention 2 (opcional)"),
)


def _package_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ffmpeg_version(binary: str) -> str | None:
    exe = shutil.which(binary)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.splitlines()[0] if out else ""
    return first.split(" ")[2] if first.count(" ") >= 2 else "desconocida"


class EnvironmentChecker:
    def __init__(self, gpu_manager: GPUManager) -> None:
        self.gpu_manager = gpu_manager

    def run(self) -> EnvironmentReport:
        gpu = self.gpu_manager.detect()
        checks: list[Check] = []

        py_ok = sys.version_info[:2] in ((3, 11), (3, 12))
        checks.append(Check(
            id="python", label="Python", status="ok" if py_ok else "warning",
            detail=platform.python_version() + ("" if py_ok else " — se recomienda 3.11 o 3.12"),
        ))

        if gpu.torch_available:
            checks.append(Check(id="pytorch", label="PyTorch", status="ok", detail=gpu.torch_version))
        elif gpu.torch_error:
            checks.append(Check(id="pytorch", label="PyTorch", status="error", detail=gpu.torch_error))
        else:
            checks.append(Check(id="pytorch", label="PyTorch", status="missing",
                                detail="No instalado. Necesario para los motores de voz."))

        if gpu.devices:
            names = ", ".join(f"{d.name} ({(d.total_memory_mb or 0) / 1024:.0f} GB)" for d in gpu.devices)
            checks.append(Check(id="gpu", label="GPU detectada", status="ok", detail=names))
        else:
            checks.append(Check(id="gpu", label="GPU detectada", status="warning",
                                detail="No se detectó GPU. Se usará CPU (muy lento)."))

        if gpu.backend in ("cuda", "rocm") and gpu.torch_available:
            label = "ROCm disponible" if gpu.backend == "rocm" else "CUDA disponible"
            checks.append(Check(id="cuda", label=label, status="ok",
                                detail=gpu.rocm_version or gpu.cuda_version))
        elif gpu.devices:
            checks.append(Check(id="cuda", label="CUDA disponible", status="warning",
                                detail="Driver presente, pero PyTorch con CUDA no está instalado."))
        else:
            checks.append(Check(id="cuda", label="CUDA disponible", status="missing"))

        for binary in ("ffmpeg", "ffprobe"):
            version = _ffmpeg_version(binary)
            checks.append(Check(
                id=binary, label=f"{'FFmpeg' if binary == 'ffmpeg' else 'FFprobe'} disponible",
                status="ok" if version else "error",
                detail=version or "No encontrado en el PATH. Es obligatorio para procesar audio.",
            ))

        if shutil.which("ffmpeg"):
            from app.postprocess.processor import ffmpeg_filters

            filters = ffmpeg_filters(shutil.which("ffmpeg") or "ffmpeg")
            has = "rubberband" in filters
            checks.append(Check(
                id="ffmpeg_rubberband", label="FFmpeg con rubberband (tono y velocidad)",
                status="ok" if has else "warning",
                detail="Disponible" if has else "Esta versión de FFmpeg no lo incluye: el posprocesado de tono y "
                                                 "velocidad no estará disponible (usa una build «full»).",
            ))

        for check_id, dist, label in OPTIONAL_PACKAGES:
            version = _package_version(dist)
            checks.append(Check(
                id=check_id, label=label, status="ok" if version else "missing",
                detail=version or "No instalado",
            ))

        return EnvironmentReport(platform=platform.platform(),
                                 python_version=platform.python_version(), gpu=gpu, checks=checks)
