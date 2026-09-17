"""GPU detection and memory checks.

Works without PyTorch installed (falls back to nvidia-smi) so the app can start
before any engine dependencies are present.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
from typing import Literal

from pydantic import BaseModel

from app.core.errors import AppError, ErrorCode

logger = logging.getLogger("voicelab.gpu")

Backend = Literal["cuda", "rocm", "mps", "cpu"]


class GPUDevice(BaseModel):
    index: int
    name: str
    total_memory_mb: int | None = None
    free_memory_mb: int | None = None


class GPUInfo(BaseModel):
    backend: Backend
    devices: list[GPUDevice] = []
    torch_available: bool
    torch_version: str | None = None
    torch_error: str | None = None
    cuda_version: str | None = None
    rocm_version: str | None = None
    driver_version: str | None = None
    bf16_supported: bool | None = None
    source: Literal["torch", "nvidia-smi", "none"]


def _query_nvidia_smi() -> tuple[list[GPUDevice], str | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return [], None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        logger.warning("nvidia_smi_failed", exc_info=True)
        return [], None
    devices, driver = [], None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        devices.append(GPUDevice(index=int(parts[0]), name=parts[1],
                                 total_memory_mb=int(float(parts[2])),
                                 free_memory_mb=int(float(parts[3]))))
        driver = parts[4]
    return devices, driver


class GPUManager:
    def __init__(self, preferred: str = "auto") -> None:
        self.preferred = preferred

    def detect(self) -> GPUInfo:
        smi_devices, driver = _query_nvidia_smi()
        if importlib.util.find_spec("torch") is None:
            backend: Backend = "cuda" if smi_devices and self.preferred in ("auto", "cuda") else "cpu"
            return GPUInfo(backend=backend, devices=smi_devices, torch_available=False,
                           driver_version=driver, source="nvidia-smi" if smi_devices else "none")
        try:
            return self._detect_with_torch(driver)
        except Exception:  # broken/partial torch install (e.g. missing CUDA DLLs) must not break the app
            logger.warning("torch_detection_failed", exc_info=True)
            return GPUInfo(backend="cuda" if smi_devices else "cpu", devices=smi_devices, torch_available=False,
                           torch_error="PyTorch está instalado pero no se pudo cargar.",
                           driver_version=driver, source="nvidia-smi" if smi_devices else "none")

    def _detect_with_torch(self, driver: str | None) -> GPUInfo:
        import torch  # heavy import, only when installed

        info = GPUInfo(backend="cpu", torch_available=True, torch_version=torch.__version__,
                       driver_version=driver, source="torch")
        if torch.cuda.is_available() and self.preferred in ("auto", "cuda", "rocm"):
            hip = getattr(torch.version, "hip", None)
            info.backend = "rocm" if hip else "cuda"
            info.cuda_version = torch.version.cuda
            info.rocm_version = hip
            info.bf16_supported = bool(torch.cuda.is_bf16_supported())
            for idx in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(idx)
                info.devices.append(GPUDevice(index=idx, name=torch.cuda.get_device_name(idx),
                                              total_memory_mb=total // 2**20,
                                              free_memory_mb=free // 2**20))
        elif (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              and self.preferred in ("auto", "mps")):
            info.backend = "mps"
        return info

    def check_can_load(self, required_mb: int, device_index: int = 0) -> None:
        """Raise a friendly error if the selected device lacks free memory."""
        info = self.detect()
        if info.backend == "cpu":
            return
        device = next((d for d in info.devices if d.index == device_index), None)
        if device is None or device.free_memory_mb is None:
            return
        if device.free_memory_mb < required_mb:
            raise AppError(ErrorCode.GPU_MEMORY_ERROR, status_code=507, details={
                "requerido_mb": required_mb, "disponible_mb": device.free_memory_mb})
