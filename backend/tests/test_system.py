from app.core import environment as env_module
from app.core.gpu import GPUDevice, GPUInfo, GPUManager


def test_health(client):
    res = client.get("/api/system/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_info_uses_configured_dirs(client, settings):
    body = client.get("/api/system/info").json()
    assert body["data_dir"] == str(settings.data_dir)
    assert body["app_env"] == "test"


def test_startup_creates_data_layout(client, settings):
    for sub in ("references/original", "references/processed", "generated", "cache", "database"):
        assert (settings.data_dir / sub).is_dir()
    assert (settings.data_dir / "database" / "voicelab.db").exists()


def test_environment_report_is_spanish_and_complete(client):
    body = client.get("/api/system/environment").json()
    ids = {c["id"] for c in body["checks"]}
    assert {"python", "pytorch", "gpu", "ffmpeg", "engine_f5tts", "engine_qwen3tts"} <= ids
    assert all(c["status"] in {"ok", "warning", "error", "missing"} for c in body["checks"])
    assert any(c["label"] == "GPU detectada" for c in body["checks"])


def test_environment_without_gpu_or_ffmpeg(monkeypatch):
    monkeypatch.setattr(GPUManager, "detect", lambda self: GPUInfo(
        backend="cpu", torch_available=False, source="none"))
    monkeypatch.setattr(env_module, "_ffmpeg_version", lambda binary: None)
    report = env_module.EnvironmentChecker(GPUManager()).run()
    by_id = {c.id: c for c in report.checks}
    assert by_id["gpu"].status == "warning"
    assert by_id["ffmpeg"].status == "error"
    assert by_id["pytorch"].status == "missing"


def test_check_can_load_raises_friendly_error(monkeypatch):
    monkeypatch.setattr(GPUManager, "detect", lambda self: GPUInfo(
        backend="cuda", torch_available=True, source="torch",
        devices=[GPUDevice(index=0, name="Mock GPU", total_memory_mb=8192, free_memory_mb=1024)]))
    import pytest

    from app.core.errors import AppError, ErrorCode

    with pytest.raises(AppError) as exc:
        GPUManager().check_can_load(required_mb=4096)
    assert exc.value.code is ErrorCode.GPU_MEMORY_ERROR
    assert "memoria de GPU" in exc.value.message


def test_licenses_flag_non_commercial_weights(client):
    body = client.get("/api/system/licenses").json()
    f5 = next(e for e in body if e["component"] == "F5-TTS")
    assert f5["weights_license"] == "CC-BY-NC-4.0"
    assert f5["commercial_use"] == "no_permitido"


def test_openapi_lists_all_route_groups(client):
    tags = {t for path in client.get("/openapi.json").json()["paths"].values()
            for op in path.values() for t in op.get("tags", [])}
    assert "Sistema" in tags


def test_broken_torch_does_not_break_environment(monkeypatch):
    import importlib.util

    from app.core import gpu as gpu_module

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(gpu_module.importlib.util, "find_spec",
                        lambda name, *a: object() if name == "torch" else real_find_spec(name, *a))

    def boom(self, driver):
        raise OSError("Error loading caffe2_nvrtc.dll")

    monkeypatch.setattr(GPUManager, "_detect_with_torch", boom)
    info = GPUManager().detect()
    assert info.torch_available is False and info.torch_error
    report = env_module.EnvironmentChecker(GPUManager()).run()
    assert next(c for c in report.checks if c.id == "pytorch").status == "error"
