"""GPU-dependent code paths with a simulated CUDA device: detection, VRAM checks, loading, downloads, discovery."""

from __future__ import annotations

import subprocess
import threading
import time
import types

import pytest

from app.asr.manager import ASRManager
from app.core import gpu as gpu_module
from app.core.errors import AppError, ErrorCode
from app.core.gpu import GPUDevice, GPUManager
from app.engines import manager as manager_module
from app.engines import registry as registry_module
from app.engines.base import EngineVariant
from app.engines.manager import ModelManager
from app.engines.mock.plugin import MockBackend
from app.engines.registry import EngineRegistry
from tests.test_generation import wait_until
from tests.test_resources import FakeASRBackend


# ---------------------------------------------------------------- fake CUDA
class FakeCuda:
    """Minimal torch.cuda surface used by GPUManager and ResourcesService."""

    def __init__(self, total_mb=16384, free_mb=12000, bf16=True) -> None:
        self.total, self.free, self.bf16 = total_mb * 2**20, free_mb * 2**20, bf16
        self.emptied = 0

    def install(self, monkeypatch) -> None:
        import torch

        for name, value in {
            "is_available": lambda: True, "device_count": lambda: 1,
            "mem_get_info": lambda idx=0: (self.free, self.total),
            "get_device_name": lambda idx=0: "RTX 3080 Laptop (simulada)",
            "is_bf16_supported": lambda: self.bf16, "memory_allocated": lambda idx=0: 700 * 2**20,
            "memory_reserved": lambda idx=0: 800 * 2**20, "empty_cache": self._empty,
        }.items():
            monkeypatch.setattr(torch.cuda, name, value)
        monkeypatch.setattr(torch.version, "cuda", "12.8")
        monkeypatch.setattr(torch.version, "hip", None, raising=False)

    def _empty(self) -> None:
        self.emptied += 1


def test_nvidia_smi_parsing(monkeypatch):
    monkeypatch.setattr(gpu_module.shutil, "which", lambda name: "nvidia-smi")
    out = "0, NVIDIA GeForce RTX 3080 Laptop GPU, 16384, 15000, 581.29\n1, broken line\n"
    monkeypatch.setattr(gpu_module.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    devices, driver = gpu_module._query_nvidia_smi()
    assert devices == [GPUDevice(index=0, name="NVIDIA GeForce RTX 3080 Laptop GPU", total_memory_mb=16384,
                                 free_memory_mb=15000)] and driver == "581.29"

    def fail(*a, **k):
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    monkeypatch.setattr(gpu_module.subprocess, "run", fail)
    assert gpu_module._query_nvidia_smi() == ([], None)
    monkeypatch.setattr(gpu_module.shutil, "which", lambda name: None)
    assert gpu_module._query_nvidia_smi() == ([], None)


def test_detection_with_simulated_cuda(monkeypatch):
    FakeCuda(free_mb=9000).install(monkeypatch)
    monkeypatch.setattr(gpu_module, "_query_nvidia_smi", lambda: ([], "581.29"))
    info = GPUManager("auto").detect()
    assert (info.backend, info.cuda_version, info.bf16_supported, info.source) == ("cuda", "12.8", True, "torch")
    assert info.devices[0].free_memory_mb == 9000 and info.driver_version == "581.29"

    cpu = GPUManager("cpu").detect()
    assert cpu.backend == "cpu" and cpu.devices == []

    GPUManager("auto").check_can_load(8000)
    with pytest.raises(AppError) as err:
        GPUManager("auto").check_can_load(9500)
    assert err.value.details == {"requerido_mb": 9500, "disponible_mb": 9000}


def test_detection_without_torch_uses_nvidia_smi(monkeypatch):
    smi = [GPUDevice(index=0, name="RTX", total_memory_mb=8192, free_memory_mb=4000)]
    monkeypatch.setattr(gpu_module, "_query_nvidia_smi", lambda: (smi, "550"))
    monkeypatch.setattr(gpu_module.importlib.util, "find_spec", lambda name: None)
    info = GPUManager("auto").detect()
    assert (info.backend, info.torch_available, info.source) == ("cuda", False, "nvidia-smi")
    assert GPUManager("cpu").detect().backend == "cpu"


# ---------------------------------------------------------------- ModelManager on a simulated GPU
class HeavyMock(MockBackend):
    id = "heavy"

    def __init__(self, vram_mb: int = 6000) -> None:
        super().__init__()
        self.vram_mb = vram_mb
        self.downloaded: list[str] = []
        self.fail_download = False
        self.load_error: Exception | None = None
        self.weights = True

    def variants(self) -> list[EngineVariant]:
        return [v.model_copy(update={"vram_estimate_mb": self.vram_mb}) for v in super().variants()]

    def weights_installed(self, variant: str) -> bool:
        return self.weights

    def download_weights(self, variant: str) -> None:
        time.sleep(0.05)
        if self.fail_download:
            raise OSError("disk full")
        self.downloaded.append(variant)
        self.weights = True

    def load(self, variant, device, dtype):
        if self.load_error:
            raise self.load_error
        super().load(variant, device, dtype)


@pytest.fixture
def gpu_models(monkeypatch):
    cuda = FakeCuda(free_mb=12000)
    cuda.install(monkeypatch)
    monkeypatch.setattr(gpu_module, "_query_nvidia_smi", lambda: ([], None))
    asr = ASRManager(FakeASRBackend())
    monkeypatch.setattr(manager_module, "get_asr_manager", lambda: asr)
    registry = EngineRegistry()
    engine = HeavyMock()
    registry.register(engine)
    return ModelManager(registry, GPUManager("auto"), "auto"), engine, cuda, asr


def test_loads_on_cuda_when_vram_is_enough(gpu_models):
    models, engine, cuda, asr = gpu_models
    assert models.resolve_device() == "cuda"
    models.ensure_loaded("heavy", "")
    assert models.loaded.device == "cuda" and asr.backend.loaded  # ASR untouched when memory suffices
    models.unload()
    assert cuda.emptied >= 1


def test_unloads_asr_when_vram_is_short_then_fails_friendly(gpu_models):
    models, engine, cuda, asr = gpu_models
    engine.vram_mb = 13000  # more than the simulated 12 GB free

    def free_after_asr_unload():
        FakeASRBackend.unload(asr.backend)
        cuda.free += 2000 * 2**20

    asr.backend.unload = free_after_asr_unload
    models.ensure_loaded("heavy", "")
    assert not asr.backend.loaded and models.loaded.engine == "heavy"

    models.unload()
    engine.vram_mb = 20000
    with pytest.raises(AppError) as err:
        models.ensure_loaded("heavy", "")
    assert err.value.code is ErrorCode.GPU_MEMORY_ERROR and err.value.status_code == 507


def test_load_errors_are_classified(gpu_models):
    models, engine, *_ = gpu_models
    engine.load_error = RuntimeError("CUDA out of memory. Tried to allocate 3 GiB")
    with pytest.raises(AppError) as err:
        models.ensure_loaded("heavy", "")
    assert err.value.code is ErrorCode.GPU_MEMORY_ERROR and models.loaded is None

    engine.load_error = ValueError("corrupted checkpoint")
    with pytest.raises(AppError) as err:
        models.ensure_loaded("heavy", "")
    assert err.value.code is ErrorCode.MODEL_LOAD_ERROR

    engine.load_error, engine.weights = None, False
    with pytest.raises(AppError) as err:
        models.ensure_loaded("heavy", "")
    assert err.value.code is ErrorCode.MODEL_NOT_INSTALLED and err.value.details["motor"] == "heavy"


def test_weight_downloads_run_in_background(gpu_models):
    models, engine, *_ = gpu_models
    engine.weights = False
    status = models.start_download("heavy", None)
    assert status.download_state == "downloading" and not status.weights_installed
    assert models.start_download("heavy", None).download_state == "downloading"  # no duplicate thread
    wait_until(lambda: models.status("heavy", None).download_state == "idle")
    status = models.status("heavy", None)
    assert status.weights_installed and engine.downloaded == [engine.default_variant()]

    engine.weights, engine.fail_download = False, True
    models.start_download("heavy", None)
    wait_until(lambda: models.status("heavy", None).download_state == "failed")
    failed = models.status("heavy", None)
    assert failed.download_state == "failed" and "espacio en disco" in failed.download_error


def test_download_requires_package(gpu_models, monkeypatch):
    models, engine, *_ = gpu_models
    monkeypatch.setattr(engine, "is_installed", lambda: False)
    with pytest.raises(AppError) as err:
        models.start_download("heavy", None)
    assert err.value.code is ErrorCode.MODEL_NOT_INSTALLED


def test_concurrent_jobs_share_one_loaded_model(gpu_models):
    models, *_ = gpu_models
    errors: list[Exception] = []

    def job():
        try:
            with models.use("heavy", ""):
                time.sleep(0.02)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=job) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and not models.loaded.in_use


# ---------------------------------------------------------------- plugin discovery
def test_discovery_skips_broken_plugins_and_loads_entry_points(monkeypatch):
    class External(MockBackend):
        id = "external"

    good = types.SimpleNamespace(name="external", load=lambda: External)
    broken = types.SimpleNamespace(name="broken", load=lambda: (_ for _ in ()).throw(ImportError("missing dep")))
    not_engine = types.SimpleNamespace(name="junk", load=lambda: dict)
    monkeypatch.setattr(registry_module.importlib.metadata, "entry_points",
                        lambda group: [good, broken, not_engine])
    registry = EngineRegistry().discover(include_dev=True)
    ids = [e.id for e in registry.all()]
    assert ids[:3] == ["f5tts", "e2tts", "qwen3tts"] and "external" in ids and "mock" in ids
    assert "junk" not in ids and "broken" not in ids

    duplicate = types.SimpleNamespace(name="dup", load=lambda: External)
    monkeypatch.setattr(registry_module.importlib.metadata, "entry_points", lambda group: [good, duplicate])
    assert [e.id for e in EngineRegistry().discover().all()].count("external") == 1
